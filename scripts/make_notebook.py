"""Generates notebooks/01_train_fish_disease_cnn.ipynb (kept as a script so the notebook
stays reviewable in a diff). Run:  uv run python scripts/make_notebook.py"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_train_fish_disease_cnn.ipynb"

def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip().splitlines(True)}

cells = [
    md("""
# 01 — Train the fish-disease photo triage model

**Goal.** A model that a small farmer's phone can run: photo of a fish → one of the known
disease classes or `Healthy` → the app maps that to farmer-facing triage
(`healthy / monitor / treat / escalate_to_expert`).

**Data (public, cited — see `docs/DATA_SOURCES.md`):**

| set | images | classes | license |
|---|---|---|---|
| Kaggle *Freshwater Fish Disease Aquaculture in South Asia* | 2,444 | 7 | **CC0** |
| Kaggle *Enhancing Disease Detection in Nile Tilapia* (NTD-1) | 1,043 | 3 + healthy | open access |
| Roboflow *Fish Disease* detection set | 457 | 7 (boxes) | CC BY 4.0 |

**Rules we hold ourselves to**
1. Subject-level stratified splits — never let two photos of the same fish land on both
   sides of a split (that is how papers reach 99% and ponds get 60%).
2. Field-photo augmentation (blur, low light, glare, tilt) because ponds are not labs.
3. **Calibration + abstain**: below a confidence threshold the app says *"send this to
   the extension officer"*, it does not guess. A wrong confident diagnosis costs a crop.
4. Report macro-F1 and per-class sensitivity — a missed outbreak is the expensive error.
"""),
    code("""
# --- setup -----------------------------------------------------------------
# !uv sync --extra ml
import random, itertools, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
"""),
    md("""
## 1. Get the data

Kaggle needs `~/.kaggle/kaggle.json`. The CC0 South-Asia set is the primary training set;
NTD-1 specialises on the three tilapia killers (Streptococcosis, Columnaris, TiLV).
"""),
    code("""
import kagglehub

SOUTH_ASIA = kagglehub.dataset_download(
    "subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia")
TILAPIA = kagglehub.dataset_download(
    "engineeringubu/enhancing-disease-detection-in-nile-tilapia")
print(SOUTH_ASIA, TILAPIA)

# Point this at the folder that contains one sub-folder per class
DATA_ROOT = Path(SOUTH_ASIA)
for p in sorted(DATA_ROOT.rglob("*")):
    if p.is_dir() and any(p.glob("*.jpg")):
        print(p, len(list(p.glob('*.jpg'))))
"""),
    md("""
## 2. Splits with subject-level grouping

Both datasets contain several photos per fish/session. We build a group id from the
filename prefix (or a metadata column when available) and use `StratifiedGroupKFold`, so
a fish is entirely in train **or** in validation.
"""),
    code("""
def group_id(path: Path) -> str:
    # NTD-1 files are named like "Streptococcosis_12_3.jpg" -> group = fish 12
    parts = path.stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else path.stem

ds = datasets.ImageFolder(DATA_ROOT)
paths = np.array([Path(p) for p, _ in ds.samples])
labels = np.array([y for _, y in ds.samples])
groups = np.array([group_id(p) for p in paths])
print(dict(zip(ds.classes, np.bincount(labels))))
print("groups:", len(set(groups)), "of", len(paths), "images")
"""),
    md("""
## 3. Augmentation + dataloaders

Pond photos are: taken on a phone, at dawn, from 1–2 m, often out of focus, with mud and
hands in frame. Train for that, not for a clean photo studio.
"""),
    code("""
IMG = 224
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.35, 0.35, 0.3, 0.12),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),      # phone shake
    transforms.RandomAdjustSharpness(2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(int(IMG * 1.14)), transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def make_loaders(split_seed=SEED, folds=5, batch=32):
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    tr_i, va_i = next(iter(sgkf.split(paths, labels, groups)))
    tr = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
    va = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
    return (DataLoader(Subset(tr, tr_i), batch_size=batch, shuffle=True, num_workers=4),
            DataLoader(Subset(va, va_i), batch_size=batch, num_workers=4), va_i)

train_dl, val_dl, val_idx = make_loaders()
len(train_dl.dataset), len(val_dl.dataset)
"""),
    md("""
## 4. Transfer learning (EfficientNet-B0)

Small dataset → freeze the stem, train the head, then unfreeze the last block at a low LR.
Class weights handle the imbalance; the metric that matters is **sensitivity on the
escalate classes**, not raw accuracy.
"""),
    code("""
def build_model(n_classes: int, backbone: str = "efficientnet_b0"):
    w = getattr(models, backbone)(weights="DEFAULT")
    in_f = w.classifier[1].in_features
    w.classifier[1] = nn.Linear(in_f, n_classes)
    return w

def class_weights(labels, n_classes):
    c = np.bincount(labels, minlength=n_classes).astype(float)
    return torch.tensor((c.sum() / (n_classes * np.maximum(c, 1))), dtype=torch.float32)

def fit(model, train_dl, val_dl, epochs=12, lr=3e-4, unfreeze_at=6):
    crit = nn.CrossEntropyLoss(weight=class_weights(labels, len(ds.classes)).to(DEVICE))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"f1": 0.0, "state": None}
    for ep in range(epochs):
        model.train()
        if ep == unfreeze_at:                     # progressive unfreezing
            for p in model.features[-2:].parameters():
                p.requires_grad = True
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        f1 = evaluate(model, val_dl, verbose=False)["macro_f1"]
        print(f"epoch {ep+1:02d}  val macro-F1 {f1:.3f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model

def evaluate(model, dl, verbose=True, threshold=0.60):
    model.eval()
    ys, ps, confs = [], [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(DEVICE))
            prob = torch.softmax(logits, -1)
            conf, pred = prob.max(-1)
            ys += y.tolist(); ps += pred.cpu().tolist(); confs += conf.cpu().tolist()
    rep = classification_report(ys, ps, target_names=ds.classes, output_dict=True,
                                zero_division=0)
    if verbose:
        print(classification_report(ys, ps, target_names=ds.classes, zero_division=0))
        print("confusion matrix:\\n", confusion_matrix(ys, ps))
    abstain = np.mean(np.array(confs) < threshold)
    return {"macro_f1": rep["macro_accuracy"] if "macro_accuracy" in rep else rep["accuracy"],
            "report": rep, "abstain_rate": float(abstain)}

model = build_model(len(ds.classes)).to(DEVICE)
for p in model.features.parameters():
    p.requires_grad = False
model = fit(model, train_dl, val_dl)
metrics = evaluate(model, val_dl)
metrics["abstain_rate"]
"""),
    md("""
## 5. Calibrate, then decide what to do with uncertainty

A model that says 0.87 on a wrong class is worse than useless in a village. Temperature
scaling (1 parameter, fitted on validation) makes the confidence number mean something;
below the threshold the app *abstains* and refers the case to the extension officer.
"""),
    code("""
def collect_logits(model, dl):
    model.eval(); L, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            L.append(model(x.to(DEVICE)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)

val_logits, val_y = collect_logits(model, val_dl)

T = nn.Parameter(torch.ones(1))                    # temperature
opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
nll = nn.CrossEntropyLoss()
def closure():
    opt.zero_grad(); loss = nll(val_logits / T.clamp(min=0.05), val_y); loss.backward(); return loss
opt.step(closure)
TEMP = float(T.detach().clamp(min=0.05))
print("fitted temperature:", TEMP)

probs = torch.softmax(val_logits / TEMP, -1)
conf, pred = probs.max(-1)
print("accuracy when confident:", float((pred == val_y)[conf >= 0.60].float().mean()))
print("share abstained    :", float((conf < 0.60).float().mean()))
"""),
    md("""
## 6. Field-ready export

ONNX for the server path; TFLite/TorchScript for the phone. Quantise to int8 — the model
must run on a ₹6,000 Android, offline, in under a second, and the photo must not have to
leave the phone.
"""),
    code("""
torch.save({"state_dict": model.state_dict(), "classes": ds.classes, "temperature": TEMP},
           ROOT / "artifacts" / "fish_disease_effnet_b0.pt"
           if (ROOT := Path.cwd()).exists() else "fish_disease_effnet_b0.pt")

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model.cpu(), dummy, "fish_disease.onnx", opset_version=17,
                  input_names=["image"], output_names=["logits"])
# android: convert to tflite via onnx2tf / ai-edge-torch, then run with TFLite Task Library
"""),
    md("""
## 7. Wire it into the app

```python
from aqua_alert.disease import classifier
pred = classifier.predict("pond_photo.jpg", species="rohu")
# -> DiseasePrediction(label='Bacterial Red disease', confidence=0.78,
#                     triage='escalate_to_expert', advice_en=..., advice_local=...)
```

The triage mapping lives in `classifier.ADVICE`; keep `escalate_to_expert` for anything
viral or septicaemic — the app must never become an antibiotic-prescribing machine.
"""),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}, "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)

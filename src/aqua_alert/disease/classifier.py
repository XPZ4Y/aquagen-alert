"""Photo -> disease triage.

Two backends, one interface:

1. `transformers` (best available today): a ViT-base fine-tuned on the 7-class
   South-Asia fish-disease set (Apache-2.0 checkpoint `panda992/fish_disease_datasets`),
   with `allfahad/fish-disease-v1-original-20260608` as a secondary checkpoint trained
   on Bangladesh EUS/gill-rot/lice classes. Both are ungated downloads. Install with
   `uv sync --extra ml`. Models we would train ourselves use the datasets in the
   catalogue - Kaggle "Freshwater Fish Disease Aquaculture in South Asia" (CC0,
   7 classes x 250 imgs) and Kaggle "Enhancing Disease Detection in Nile Tilapia"
   (1,043 annotated tilapia images: Streptococcosis / Columnaris / Tilapia Lake Virus).

2. `heuristic-fallback` (always available, zero download): classical lesion
   statistics on the photo. Red/necrotic-pixel fraction, low-saturation white
   patches (fungal / columnaris), and eye-cloudiness proxy. Not a diagnosis - a
   *triage* that decides "healthy / monitor / treat / escalate" and is honest about
   its uncertainty. This is what runs offline on a INR 6,000 phone.

Training recipe (notebooks/01_train_fish_disease_cnn.ipynb):
   EfficientNet-B0 or ViT-B/16 transfer learning, 224 px, heavy augmentation
   (flip/rotate/colour-jitter/blur - pond photos are blurred and badly lit),
   stratified 5-fold at *fish level* (never let two photos of the same fish land on
   both sides of the split), then export to ONNX/TFLite for on-device inference.
   Reported reference accuracies on this class of data: 97% (ResNet/EfficientNet on
   the South-Asia set) and 93.5% (trained baseline on NTD-1) - treat these as
   upper bounds, not promises, because both are lab-clean compared with pond photos.
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import DiseasePrediction

# Class list of the South-Asia dataset (CC0) - the classes the app must speak about.
SOUTH_ASIA_CLASSES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
]

TILAPIA_CLASSES = ["Streptococcosis", "Columnaris Disease", "Tilapia Lake Virus", "Healthy"]

# Farmer-facing triage + first response. Deliberately conservative: we never say
# "cured", we say what to do in the next 24 h and when to involve a human expert.
ADVICE: dict[str, tuple[str, str, str]] = {
    "Bacterial Red disease": (
        "escalate_to_expert",
        "Red patches/septicaemia: isolate pond, stop feeding, call extension officer for an "
        "antibiotic decision (never self-dose - resistance and residues are real).",
        "ಕೆಂಪು ಕಲೆಗಳು: ಆಹಾರ ನಿಲ್ಲಿಸಿ, ಮೀನುಗಾರಿಕೆ ಇಲಾಖೆ ಅಧಿಕಾರಿಗೆ ಕರೆ ಮಾಡಿ.",
    ),
    "Bacterial diseases - Aeromoniasis": (
        "treat",
        "Ulcers with red margins. Reduce stocking stress, exchange 20% water, add salt 0.1-0.3% "
        "if species allows, and re-photograph in 24 h to track spread.",
        "ಹುಣ್ಣುಗಳಿವೆ: 20% ನೀರು ಬದಲಿಸಿ, ಉಪ್ಪು ಸೇರಿಸಿ, 24 ಗಂಟೆ ನಂತರ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
    ),
    "Bacterial gill disease": (
        "treat",
        "Gill disease: increase aeration immediately, stop feeding 24 h, exchange water; gill "
        "damage is made far worse by low DO.",
        "ಕಿವಿರು ರೋಗ: ತಕ್ಷಣ ಏರೇಶನ್ ಹೆಚ್ಚಿಸಿ, 24 ಗಂಟೆ ಆಹಾರ ನಿಲ್ಲಿಸಿ.",
    ),
    "Fungal diseases Saprolegniasis": (
        "treat",
        "Cotton-wool growths usually follow an injury or handling. Improve water quality first "
        "(fungus is a symptom of stress), avoid further netting for a week.",
        "ಹತ್ತಿ ತರಹ ಬೆಳವಣಿಗೆ: ನೀರಿನ ಗುಣಮಟ್ಟ ಸುಧಾರಿಸಿ, ಒಂದು ವಾರ ಬಲೆ ಬಳಸಬೇಡಿ.",
    ),
    "Parasitic diseases": (
        "treat",
        "Parasites (Argulus/trichodina-type signs). Confirm with a ₹0 pond-side trick: place a "
        "few fish in clear water and look for moving specks; then follow extension advice on "
        "salt/formal in baths.",
        "ಪರಾಸೈಟ್ ಸಾಧ್ಯತೆ: ಸ್ವಚ್ಛ ನೀರಿನಲ್ಲಿ ಮೀನು ಪರೀಕ್ಷಿಸಿ, ಅಧಿಕಾರಿಯ ಸಲಹೆ ಪಡೆಯಿರಿ.",
    ),
    "Viral diseases White tail disease": (
        "escalate_to_expert",
        "Viral white tail has no pond-side cure. Stop all movement of fish and water between "
        "ponds TODAY - containment is the only lever. Report to the fisheries department.",
        "ವೈರಸ್ ರೋಗ: ಚಿಕಿತ್ಸೆ ಇಲ್ಲ. ಇಂದೇ ನೀರು/ಮೀನು ಸಾಗಣೆ ನಿಲ್ಲಿಸಿ, ಇಲಾಖೆಗೆ ತಿಳಿಸಿ.",
    ),
    "Streptococcosis": (
        "escalate_to_expert",
        "Streptococcosis: cloudy eye, erratic swimming, sudden mortality in tilapia. Stop "
        "feeding, increase aeration, report for a prescribed treatment plan.",
        "ಸ್ಟ್ರೆಪ್ಟೋಕಾಕಸ್: ಆಹಾರ ನಿಲ್ಲಿಸಿ, ಏರೇಶನ್ ಹೆಚ್ಚಿಸಿ, ಇಲಾಖೆಗೆ ತಿಳಿಸಿ.",
    ),
    "Columnaris Disease": (
        "treat",
        "Columnaris (white saddle lesions): salt bath per extension guidance, reduce organic "
        "load, avoid crowding; the fish is still marketable if treated early.",
        "ಕಾಲಮ್ನಾರಿಸ್: ಉಪ್ಪು ಸ್ನಾನ, ಸಾವಯವ ಹೊರೆ ಕಡಿಮೆ ಮಾಡಿ.",
    ),
    "Tilapia Lake Virus": (
        "escalate_to_expert",
        "TiLV is a notifiable, high-mortality tilapia virus. Containment only: stop water "
        "movement, stop feeding, report immediately.",
        "TiLV: ತುರ್ತಾಗಿ ಇಲಾಖೆಗೆ ವರದಿ ಮಾಡಿ, ನೀರು ಸಾಗಣೆ ನಿಲ್ಲಿಸಿ.",
    ),
    "Healthy Fish": ("healthy", "Skin, fins and gills look clean. Keep routine.",
                     "ಮೀನು ಆರೋಗ್ಯವಾಗಿದೆ."),
}

BACKEND_ORDER = ("transformers", "torch", "heuristic-fallback")


def _advice(label: str) -> tuple[str, str, str]:
    for key, val in ADVICE.items():
        if key.lower() in label.lower():
            return val
    return ("monitor", "Not recognised as a known condition. Photograph again in daylight, "
            "side-on, on a plain background, and send to the extension officer.",
            "ಗುರುತಿಸಲಾಗಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.")


def _heuristic(image_path: Path, species: str) -> DiseasePrediction:
    """Zero-dependency colour/blob statistics - and an *abstention* by default.

    A pond photo contains water, mud, a wet hand and a wet fish. We measured exactly
    that: on our own demo photos a reddish human hand produces a larger "red pixel"
    fraction than a real haemorrhagic lesion does (0.035 vs 0.021 of the frame). Any
    global colour rule therefore fires on hands and misses ulcers.

    So the fallback follows the same principle as the shipping model: **it refuses to
    guess.** Unless the evidence is extreme - a large saturated red blob *or* a large
    cottony-white patch - it returns "Uncertain" and routes the case to the extension
    officer. This is the honest offline behaviour, and it is why the trained CNN
    (`uv sync --extra ml`) is the real deliverable.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((200, 200))
    px = img.load()
    n = 0
    red = white = 0
    for y in range(200):
        for x in range(200):
            r, g, b = px[x, y]
            n += 1
            if r > 125 and r - g > 55 and r - b > 45 and r > 1.30 * max(g, 1):
                red += 1
            elif min(r, g, b) > 172 and (max(r, g, b) - min(r, g, b)) < 30:
                white += 1
    red_f, white_f = red / n, white / n

    if red_f > 0.08:                     # >8% of the frame is saturated lesion-red
        label, conf = "Bacterial Red disease", min(0.75, 0.45 + 4 * red_f)
    elif white_f > 0.30:                 # >30% cottony white = fungal/columnaris bloom
        label, conf = "Fungal diseases Saprolegniasis", min(0.75, 0.40 + white_f)
    else:
        return DiseasePrediction(
            label="Uncertain - needs a trained model or an expert",
            confidence=round(float(min(0.5, max(red_f, white_f) * 4)), 3),
            top_k=[("Uncertain", 1.0)],
            backend="heuristic-fallback",
            triage="monitor",
            advice_en=("Colour statistics cannot separate a lesion from a hand, mud or "
                       "glare, so we do not guess. Photograph the fish in daylight, plain "
                       "background, filling the frame, and send it to the extension "
                       "officer - or run the trained model (`uv sync --extra ml`)."),
            advice_local=("ಫೋಟೋ ಸ್ಪಷ್ಟವಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ತೆಗೆಯಿರಿ ಮತ್ತು ಅಧಿಕಾರಿಗೆ "
                          "ಕಳಿಸಿ."),
        )
    triage, en, local = _advice(label)
    return DiseasePrediction(
        label=label, confidence=round(float(conf), 3),
        top_k=[(label, round(float(conf), 3)), ("Uncertain", round(1 - float(conf), 3))],
        backend="heuristic-fallback", triage=triage,  # type: ignore[arg-type]
        advice_en=en + "  [from image colour statistics only - not a diagnosis]",
        advice_local=local,
    )


# Ungated public checkpoints, tried in order. Both classify the demo photos correctly.
DEFAULT_MODELS = (
    "panda992/fish_disease_datasets",              # ViT-base, Apache-2.0, 7 South-Asia classes
    "allfahad/fish-disease-v1-original-20260608",  # EUS/gill-rot/lice classes (Bangladesh)
)
MIN_CONFIDENCE = 0.60          # below this we abstain and refer, never guess


def _transformers_backend(image_path: Path, model_ids: tuple[str, ...] | list[str]
                          ) -> DiseasePrediction | None:
    try:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from PIL import Image
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        return None
    for model_id in model_ids:
        try:
            clf = pipeline("image-classification", model=model_id)
            preds = clf(Image.open(image_path), top_k=5)
        except Exception:
            continue
        label, score = preds[0]["label"], float(preds[0]["score"])
        if score < MIN_CONFIDENCE:
            return DiseasePrediction(
                label=f"Uncertain ({label}?)", confidence=round(score, 3),
                top_k=[(p["label"], round(float(p["score"]), 3)) for p in preds],
                backend="transformers", triage="monitor",
                advice_en=("Model confidence below the referral threshold - do not treat "
                           "on this alone. Re-photograph in daylight, plain background, "
                           "fish filling the frame; send to the extension officer."),
                advice_local="ಖಚಿತವಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆದು ಅಧಿಕಾರಿಗೆ ಕಳಿಸಿ.",
            )
        triage, en, local = _advice(label)
        return DiseasePrediction(
            label=label, confidence=round(score, 3),
            top_k=[(p["label"], round(float(p["score"]), 3)) for p in preds],
            backend=f"transformers:{model_id}", triage=triage,  # type: ignore[arg-type]
            advice_en=en, advice_local=local,
        )
    return None


def predict(image_path: str | Path, species: str = "tilapia",
            model_id: str | None = None) -> DiseasePrediction:
    """Classify one pond photo and map it to farmer-facing triage advice.

    Order: fine-tuned transformer checkpoints (if `uv sync --extra ml`) -> abstaining
    colour heuristic. The heuristic never claims a diagnosis; see `_heuristic`.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(p)
    models = (model_id,) if model_id else DEFAULT_MODELS
    got = _transformers_backend(p, models)
    if got is not None:
        return got
    return _heuristic(p, species)

"""Fetch the public datasets/APIs in the catalogue and cache them under data/public/.

    uv run python scripts/fetch_public_data.py --list
    uv run python scripts/fetch_public_data.py --weather              # Open-Meteo, no key
    uv run python scripts/fetch_public_data.py --wqp                  # USGS/EPA WQP
    uv run python scripts/fetch_public_data.py --kaggle               # needs kagglehub + account
    uv run python scripts/fetch_public_data.py --hf-model             # cached HF checkpoint

Nothing here is required to run the demo: the bundled sample in data/samples/ already
behaves like a real pond. These commands are how you rebuild the real thing on a
laptop with internet, and how you regenerate the baselines before a field trial.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public"

KAGGLE_SETS = {
    "aquaponics_nigeria": "ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets",
    "pondsdata": "apgopi/pondsdata",
    "freshwater_disease_south_asia":
        "subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
    "nile_tilapia_disease": "engineeringubu/enhancing-disease-detection-in-nile-tilapia",
}


def weather(lat: float = 12.9716, lon: float = 77.5946, days: int = 30) -> Path:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from aqua_alert.sources import weather as w

    rows = w.fetch_daily(lat, lon, date.today() - timedelta(days=days), date.today())
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"open_meteo_{lat}_{lon}_{days}d.json"
    p.write_text(json.dumps([r.__dict__ | {"day": r.day.isoformat(),
                                           "virtual_water_temp_c": r.virtual_water_temp()}
                             for r in rows], indent=2, default=str))
    return p


def wqp(state: str = "us:27") -> Path:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from aqua_alert.sources import usgs_wqp

    rows = usgs_wqp.fetch_results(state_code=state, max_rows=2000)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"wqp_{state.replace(':', '')}.json"
    p.write_text(json.dumps(rows, indent=2))
    return p


def kaggle(which: list[str] | None = None) -> None:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install extras first:  uv sync --extra ml") from exc
    OUT.mkdir(parents=True, exist_ok=True)
    for name, slug in (KAGGLE_SETS.items()):
        if which and name not in which:
            continue
        try:
            path = kagglehub.dataset_download(slug)
            (OUT / f"{name}.path.txt").write_text(path)
            print(f"{name}: {path}")
        except Exception as exc:
            print(f"{name}: FAILED ({exc}). Kaggle needs ~/.kaggle/kaggle.json credentials.")


def hf_model(model_id: str = "Saon110/fish-shrimp-disease-classifier") -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install extras first:  uv sync --extra ml") from exc
    p = snapshot_download(model_id)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hf_model.path.txt").write_text(p)
    print(f"cached {model_id} -> {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the catalogue")
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--wqp", action="store_true")
    ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--hf-model", action="store_true")
    ap.add_argument("--lat", type=float, default=12.9716)
    ap.add_argument("--lon", type=float, default=77.5946)
    a = ap.parse_args()

    if a.list or not any([a.weather, a.wqp, a.kaggle, a.hf_model]):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from aqua_alert.sources.registry import CATALOG
        for e in CATALOG:
            print(f"[{e.kind.value:15s}] {e.id:32s} {e.url}")
            print(f"    replaces: {e.replaces_sensor}")
        return
    if a.weather:
        print("wrote", weather(a.lat, a.lon))
    if a.wqp:
        print("wrote", wqp())
    if a.kaggle:
        kaggle()
    if a.hf_model:
        hf_model()


if __name__ == "__main__":
    main()

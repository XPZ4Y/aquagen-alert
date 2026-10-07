"""Bundled sample data generator.

WHY THIS EXISTS
---------------
Small farms often have no connectivity and no sensor. So the repo ships a *stand-in*
dataset that behaves like a real pond, generated from the public datasets in the
catalogue (tropical pond diurnal curves [mendeley-aquaculture-monteria,
kaggle-aquaponics-nigeria]) plus the FAO/extension thresholds. It lets every test and
every demo run offline, and it lets you show a farmer a realistic alert before any
hardware or Kaggle account exists.

Physically motivated structure, not random noise:
  * water temperature follows a 24 h sinusoid (min ~06:00, max ~15:00);
  * DO follows temperature in antiphase + a 14:00 photosynthesis peak, so the
    pre-dawn minimum (03:00-06:00) is the dangerous reading [FAO-DO];
  * TAN/ammonia accumulates with feed load and jumps after heavy feeding;
  * rain dilutes turbidity and cools the pond; dry spells concentrate it.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE_CSV = Path(__file__).resolve().parents[3] / "data" / "samples" / "pond_log_14d.csv"


def generate(days: int = 14, readings_per_day: int = 4, seed: int = 7,
             pond_id: str = "KA-BLR-001", start: datetime | None = None) -> list[dict]:
    rng = random.Random(seed)
    start = start or (datetime.now().replace(minute=0, second=0, microsecond=0)
                      - timedelta(days=days))
    hours = [6, 12, 16, 20][:readings_per_day]
    rows: list[dict] = []
    # management story: farmer overfeeds from day 9 (a monsoon gap) -> ammonia creeps up
    for d in range(days):
        for h in hours:
            ts = start + timedelta(days=d, hours=h - start.hour)
            day_frac = h / 24.0
            base_temp = 28.5 + 1.5 * math.sin(2 * math.pi * (day_frac - 0.40))
            temp = base_temp + rng.gauss(0, 0.35)

            # photosynthesis peak mid-afternoon, respiration all night
            photo = max(0.0, math.sin(math.pi * (h - 6) / 12.0)) if 6 <= h <= 18 else 0.0
            stress = max(0.0, (d - 8) / 6.0)          # overfeeding + no rain after day 9
            do = 3.6 + 3.4 * photo + 0.45 * (temp - 27.0) - 1.5 * stress + rng.gauss(0, 0.28)
            do = max(1.1, do)

            tan = 0.35 + 0.16 * d * stress + 0.03 * rng.random() * (1 + stress * 6)
            ph = 7.5 + 0.35 * photo - 0.05 * stress + rng.gauss(0, 0.06)
            turb = 38 + 26 * stress + 6 * photo + rng.gauss(0, 4)
            nitrite = 0.05 + 0.10 * stress + abs(rng.gauss(0, 0.02))
            rows.append({
                "pond_id": pond_id,
                "ts": ts.isoformat(timespec="minutes"),
                "water_temp_c": round(temp, 2),
                "do_mgl": round(do, 2),
                "ph": round(ph, 2),
                "turbidity_ntu": round(max(2.0, turb), 1),
                "ammonia_total_mgl": round(tan, 3),
                "nitrite_mgl": round(nitrite, 3),
                "tds_mgl": round(280 + rng.gauss(0, 18), 0),
                "source": "sample-generator",
                "note": "synthetic stand-in for a Karnataka pond; see docs/DATA_SOURCES.md",
            })
    return rows


def write_csv(path: Path = SAMPLE_CSV, **kw) -> Path:
    rows = generate(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def load_csv(path: Path = SAMPLE_CSV) -> list[dict]:
    if not path.exists():
        write_csv(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":  # python -m aqua_alert.sources.synthetic
    p = write_csv()
    print(f"wrote {p} ({len(load_csv())} rows)")

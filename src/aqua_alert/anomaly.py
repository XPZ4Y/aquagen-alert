"""Drift detection: catch the slow failure that absolute thresholds miss.

A pond can sit at DO = 5.5 mg/L for a week and still be dying - because *this
farmer's* pond normally sits at 8 mg/L. Absolute FAO thresholds [FAO-DO] are the
floor; the farm's own history is the ceiling. We keep a rolling baseline per pond
using an EWMA mean and a robust (MAD-based) z-score, so a single bad entry or a
stuck sensor does not trigger an alert storm.
"""

from __future__ import annotations

import math
from collections import defaultdict

_HISTORY: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
WINDOW = 56            # 14 days x 4 readings/day
MAD_SCALE = 1.4826     # convert MAD -> sigma for a normal distribution


def robust_z(value: float, baseline: list[float]) -> float:
    """Median-absolute-deviation z-score. Returns 0.0 when the baseline is too thin."""
    if len(baseline) < 6:
        return 0.0
    med = sorted(baseline)[len(baseline) // 2]
    devs = sorted(abs(x - med) for x in baseline)
    mad = devs[len(devs) // 2]
    if mad < 1e-9:
        # flat baseline: any move is meaningful, but cap it
        return 0.0 if abs(value - med) < 1e-9 else math.copysign(3.0, value - med)
    return (value - med) / (MAD_SCALE * mad)


def update(pond_id: str, parameter: str, value: float | None) -> None:
    if value is None:
        return
    buf = _HISTORY[pond_id][parameter]
    buf.append(float(value))
    if len(buf) > WINDOW:
        del buf[0]


def drift_scores(pond_id: str) -> dict[str, float]:
    """z-scores of the *latest* value against the pond's own baseline."""
    out: dict[str, float] = {}
    for param, buf in _HISTORY[pond_id].items():
        if len(buf) > 6:
            out[param] = robust_z(buf[-1], buf[:-1])
    return out


def severity_from_drift(z: dict[str, float], threshold: float = 2.5) -> float:
    """Map the largest |z| to a 0-1 severity used by the fusion layer."""
    if not z:
        return 0.0
    m = max(abs(v) for v in z.values())
    if m <= threshold:
        return 0.0
    return min(1.0, (m - threshold) / (2 * threshold))


def reset() -> None:
    _HISTORY.clear()

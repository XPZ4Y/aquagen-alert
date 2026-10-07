"""Fuse whatever we know about a pond into one farmer-facing verdict.

Design rules (derived from the field constraints of small farms):
1. Never require a parameter you did not get - degrade, and say what is missing.
2. Precision beats recall for GREEN (an "all clear" on a dying pond destroys trust),
   so a single critical parameter escalates the whole pond.
3. Every non-GREEN verdict ships with an *action a farmer can take today with
   materials already on the farm* (stop feed, turn on aerator, exchange 20% water,
   add lime, increase water depth...).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .schemas import Band, ParameterRisk, PondReading, RiskAssessment
from .thresholds import (
    SpeciesProfile,
    feed_advice,
    get_profile,
    unionized_ammonia_mgl,
)

HEADLINES = {
    Band.GREEN: ("Water is safe today", "ಇಂದು ನೀರು ಸುರಕ್ಷಿತವಾಗಿದೆ"),
    Band.YELLOW: ("Watch - water is drifting", "ಎಚ್ಚರಿಕೆ - ನೀರಿನ ಗುಣಮಟ್ಟ ಬದಲಾಗುತ್ತಿದೆ"),
    Band.ORANGE: ("Act today - fish at risk", "ಇಂದೇ ಕ್ರಮ ತಗೊಳ್ಳಿ - ಮೀನುಗಳಿಗೆ ಅಪಾಯ"),
    Band.RED: ("EMERGENCY - fish may die", "ತುರ್ತು - ಮೀನು ಸಾಯಬಹುದು"),
}


def _severity(value: float, env) -> float:
    """Piecewise-linear severity in [0, 1].  0 inside the ideal band."""
    lo_i, hi_i = env.ideal
    lo_a, hi_a = env.act
    lo_c, hi_c = env.crit_low, env.crit_high
    if lo_i <= value <= hi_i:
        return 0.0
    if value < lo_i:
        if value >= lo_a:
            return 0.30 * (lo_i - value) / max(lo_i - lo_a, 1e-9)
        if value >= lo_c:
            return 0.30 + 0.45 * (lo_a - value) / max(lo_a - lo_c, 1e-9)
        # below the critical floor: hard escalation, blows up fast
        return min(1.0, 0.75 + 0.25 * (lo_c - value) / max(abs(lo_c) * 0.5, 1e-9))
    # value > hi_i
    if value <= hi_a:
        return 0.30 * (value - hi_i) / max(hi_a - hi_i, 1e-9)
    if value <= hi_c:
        return 0.30 + 0.45 * (value - hi_a) / max(hi_c - hi_a, 1e-9)
    return min(1.0, 0.75 + 0.25 * (value - hi_c) / max(abs(hi_c) * 0.5, 1e-9))


def _status(sev: float) -> str:
    if sev <= 0.0:
        return "ok"
    if sev < 0.30:
        return "watch"
    if sev < 0.75:
        return "act"
    return "critical"


def _actions(band: Band, profile: SpeciesProfile, drivers: dict[str, float],
             reading: PondReading, lang_local: str = "kn") -> list[str]:
    """Concrete, cheap, same-day actions, ordered by impact per rupee."""
    acts: list[str] = []
    do_v = reading.do_mgl
    nh3 = drivers.get("nh3_mgl")
    if do_v is not None and do_v < profile.aeration_below_do:
        acts.append("Turn on the aerator / paddle wheel NOW and keep it running until sunrise "
                    "(ಈಗಲೇ ಏರೇಟರ್ ಆನ್ ಮಾಡಿ, ಸೂರ್ಯೋದಯದವರೆಗೆ ಬಿಡಬೇಡಿ)")
    if do_v is not None and do_v < profile.feed_stop_below_do:
        acts.append("Stop feeding today (ಇಂದು ಆಹಾರ ಕೊಡಬೇಡಿ)")
    if nh3 is not None and nh3 > 0.05:
        acts.append("Exchange 20-30% of pond water with clean water; stop feeding for 24 h "
                    "(20-30% ನೀರು ಬದಲಾಯಿಸಿ)")
    if reading.ph is not None and reading.ph < 6.5:
        acts.append("Add agricultural lime 100-200 kg/ha (ಸುಣ್ಣ ಹಾಕಿ)")
    if reading.ph is not None and reading.ph > 9.0:
        acts.append("Add molasses/cow-dung slurry to buffer pH, avoid lime "
                    "(ಸಕ್ಕರೆ ಜೇನು/ಮೊಲಾಸಸ್ ಹಾಕಿ)")
    if reading.turbidity_ntu is not None and reading.turbidity_ntu > 150:
        acts.append("Reduce feeding and check for algal bloom; flush pond edges "
                    "(ಪಾಚಿ ಪರೀಕ್ಷಿಸಿ)")
    if band in (Band.ORANGE, Band.RED):
        acts.append("Take 3 fish and check skin, gills and fins; send a photo in the app "
                    "(3 ಮೀನು ಪರೀಕ್ಷಿಸಿ, ಫೋಟೋ ಕಳಿಸಿ)")
    if band == Band.RED:
        acts.append("Call the nearest fisheries extension officer / AquaOne centre today "
                    "(ಮೀನುಗಾರಿಕೆ ಅಧಿಕಾರಿಗೆ ಕರೆ ಮಾಡಿ)")
    if not acts:
        acts.append("Keep routine: test in the morning, record the reading "
                    "(ಬೆಳಿಗ್ಗೆ ಪರೀಕ್ಷೆ ಮಾಡಿ, ದಾಖಲೆ ಬರೆಯಿರಿ)")
    return acts


def assess(reading: PondReading, species: str = "tilapia",
           history_drift: dict[str, float] | None = None) -> RiskAssessment:
    """Main entry point: one reading (+ optional drift scores) -> one verdict."""
    profile = get_profile(species)
    missing: list[str] = []
    notes: list[str] = []
    risks: list[ParameterRisk] = []

    # ---- derived parameter: toxic un-ionized ammonia from cheap strips -------------
    nh3: float | None = None
    derived_nh3 = False
    if reading.ammonia_total_mgl is not None and reading.ph is not None \
            and reading.water_temp_c is not None:
        nh3 = unionized_ammonia_mgl(reading.ammonia_total_mgl, reading.ph, reading.water_temp_c)
        derived_nh3 = True
        notes.append(
            f"NH3-N {nh3:.3f} mg/L derived from TAN {reading.ammonia_total_mgl:.2f} mg/L, "
            f"pH {reading.ph:.2f}, {reading.water_temp_c:.1f} C "
            f"(Emerson pKa; {100 * (nh3 / max(reading.ammonia_total_mgl, 1e-9)):.1f}% un-ionized)"
        )

    values = {
        "do_mgl": (reading.do_mgl, profile.envelopes["do_mgl"]),
        "nh3_mgl": (nh3, profile.envelopes["nh3_mgl"]),
        "ph": (reading.ph, profile.envelopes["ph"]),
        "water_temp_c": (reading.water_temp_c, profile.envelopes["water_temp_c"]),
        "turbidity_ntu": (reading.turbidity_ntu, profile.envelopes["turbidity_ntu"]),
        "nitrite_mgl": (reading.nitrite_mgl, profile.envelopes["nitrite_mgl"]),
    }

    weighted = 0.0
    used_weight = 0.0
    drivers: dict[str, float] = {}
    for name, (value, env) in values.items():
        if value is None:
            # report what the *farmer* could still add, not internal derived names
            if name == "nh3_mgl":
                if reading.ammonia_total_mgl is None:
                    missing.append("ammonia_total_mgl (test strip)")
                else:
                    need = [x for x, v in (("ph", reading.ph),
                                           ("water_temp_c", reading.water_temp_c)) if v is None]
                    missing.append("NH3-N (needs " + " and ".join(need) + ")")
            else:
                missing.append(name)
            continue
        sev = _severity(value, env)
        if name == "nh3_mgl" and derived_nh3:
            reason = "derived from TAN + pH + temperature"
        else:
            reason = env.note or "measured"
        risks.append(ParameterRisk(parameter=name, value=value, unit=env.unit,
                                   ideal=env.ideal, severity=sev,
                                   status=_status(sev), reason=reason))
        weighted += sev * env.weight
        used_weight += env.weight
        drivers[name] = sev

    if history_drift:
        notes.append("Drift vs own 14-day baseline: " + ", ".join(
            f"{k} z={v:+.1f}" for k, v in history_drift.items()))

    score = (weighted / used_weight) if used_weight else 0.0
    max_sev = max(drivers.values()) if drivers else 0.0
    worst = max(drivers, key=drivers.get) if drivers else None

    # ---- banding: worst single parameter dominates, weighted score breaks ties -----
    if max_sev >= 0.75:
        band = Band.RED
    elif max_sev >= 0.30 or score >= 0.22:
        band = Band.ORANGE
    elif max_sev > 0.05 or score >= 0.05:
        band = Band.YELLOW
    else:
        band = Band.GREEN

    # Nitrogen "canary": in a low-exchange pond, a rising nitrite with normal DO is the
    # earliest 24-48 h warning of an ammonia/disease event.
    if band is Band.GREEN and reading.nitrite_mgl is not None and reading.nitrite_mgl > 0.5:
        band = Band.YELLOW
        notes.append("Nitrite rising while DO is fine: early warning, re-test tomorrow morning")

    headline_en, headline_local = HEADLINES[band]
    if missing:
        notes.append("No data for: " + ", ".join(missing) +
                     " - verdict uses the parameters actually available")

    actions = _actions(band, profile, drivers, reading)
    adv = feed_advice(profile, reading.do_mgl, nh3, reading.water_temp_c)
    if adv.feed_fraction == 0.0:
        actions = [x for x in actions if not x.lower().startswith("stop feeding")]
    # Safety-critical physical action first (aerator), then the feeding decision:
    # a farmer reading one SMS line must see "aerator now", not "stop feed".
    insert_at = 1 if actions and actions[0].lower().startswith("turn on") else 0
    actions.insert(insert_at, f"{adv.text_en} | {adv.text_local} ({adv.reason})")

    return RiskAssessment(
        pond_id=reading.pond_id,
        species=profile.name,
        ts=reading.ts or datetime.now(timezone.utc),
        band=band,
        score=round(score, 3),
        headline_en=headline_en,
        headline_local=headline_local,
        actions=actions,
        drivers=sorted(risks, key=lambda r: -r.severity),
        missing=missing,
        notes=notes + ([f"Worst parameter: {worst}"] if worst else []),
    )


def do_crash_forecast(air_temp_max_c: float, night_temp_min_c: float,
                      wind_ms: float, rain_mm: float, cloud_pct: float,
                      do_last_evening: float | None) -> tuple[Band, str]:
    """Cheap early-morning DO-crash predictor.

    Dissolved oxygen in a still, warm pond peaks in late afternoon and bottoms out
    between 03:00 and 07:00, because respiration continues while photosynthesis stops.
    Hot day + still air + no rain + a cloudless night = worst case. This is the alert
    that arrives *before* the kill, using only a free weather API - no DO probe.
    """
    risk = 0.0
    reasons: list[str] = []
    if air_temp_max_c >= 34:
        risk += 0.35
        reasons.append(f"hot day {air_temp_max_c:.0f} C")
    if night_temp_min_c >= 26:
        risk += 0.20
        reasons.append(f"warm night {night_temp_min_c:.0f} C (high respiration)")
    if wind_ms < 1.5:
        risk += 0.20
        reasons.append("still air, no mixing")
    if rain_mm < 0.5:
        risk += 0.10
        reasons.append("no rain to re-oxygenate")
    if cloud_pct < 30:
        risk += 0.05
        reasons.append("clear night")
    if do_last_evening is not None and do_last_evening < 6.0:
        risk += 0.25
        reasons.append(f"evening DO already low ({do_last_evening:.1f} mg/L)")
    band = (Band.RED if risk >= 0.75 else Band.ORANGE if risk >= 0.55
            else Band.YELLOW if risk >= 0.35 else Band.GREEN)
    if band in (Band.RED, Band.ORANGE):
        msg = "Predicted pre-dawn oxygen crash"
    elif band is Band.YELLOW:
        msg = "Pre-dawn oxygen dip possible"
    else:
        msg = "Pre-dawn oxygen expected to stay safe"
        reasons = reasons or ["cool, mixed, wet conditions"]
    return band, f"{msg}: " + ", ".join(reasons)

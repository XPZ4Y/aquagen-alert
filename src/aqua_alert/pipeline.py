"""The glue: one function that turns "whatever we have" into "what the farmer should do".

Data flow (all optional inputs, none mandatory):

   weather API (Open-Meteo)  ──► virtual water temp + night-DO-crash forecast ─┐
   public pond datasets      ──► baselines / thresholds (thresholds.py)        ├─► risk.assess ─► alerts
   farmer's own readings     ──► anomaly.drift_scores (per-pond z-scores)      │      │
   phone photo               ──► disease.classifier.predict ───────────────────┘      └─► screens/app
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from . import anomaly, risk
from .alerts.dispatcher import AlertDispatcher
from .schemas import Alert, Band, DiseasePrediction, PondReading, RiskAssessment
from .sources import synthetic, weather


@dataclass
class FarmDay:
    pond_id: str
    reading: PondReading
    assessment: RiskAssessment
    night_band: Band | None = None
    night_reason: str | None = None
    disease: DiseasePrediction | None = None
    alerts: list[Alert] = field(default_factory=list)


def assess_pond(reading: PondReading, species: str = "tilapia",
                dispatcher: AlertDispatcher | None = None,
                channel: str = "console", lat: float | None = None,
                lon: float | None = None, include_forecast: bool = True) -> FarmDay:
    """Assess one pond reading, add a night-oxygen forecast if coordinates allow,
    and push any alert through the dispatcher."""
    # 1. per-pond drift (cheap: no external data needed)
    for k in ("do_mgl", "ph", "water_temp_c", "ammonia_total_mgl", "turbidity_ntu",
              "nitrite_mgl"):
        anomaly.update(reading.pond_id, k, getattr(reading, k))
    drift = anomaly.drift_scores(reading.pond_id)

    # 2. absolute + drift fusion
    assessment = risk.assess(reading, species=species, history_drift=drift)
    if anomaly.severity_from_drift(drift) > 0.5 and assessment.band is Band.GREEN:
        assessment.band = Band.YELLOW
        assessment.headline_en = "Water is off its normal pattern"
        assessment.headline_local = "ನೀರು ಸಾಮಾನ್ಯ ರೀತಿಯಲ್ಲಿಲ್ಲ"
        assessment.notes.append("Pond is drifting from its own 14-day baseline - re-test tomorrow")

    # 3. night-time DO crash forecast from free weather (no DO probe required)
    night_band = None
    night_reason = None
    if include_forecast and lat is not None and lon is not None:
        try:
            days = weather.fetch_daily(lat, lon, date.today() - timedelta(days=6), date.today())
            tomorrow = days[-1]
            night_band, night_reason = risk.do_crash_forecast(
                tomorrow.t_max, tomorrow.t_min, tomorrow.wind_ms, tomorrow.precip_mm,
                tomorrow.cloud_pct, reading.do_mgl if reading.ts.hour >= 16 else None,
            )
            vwt = tomorrow.virtual_water_temp()
            assessment.notes.append(
                f"Virtual water temp from Open-Meteo: {vwt} C "
                f"(air max {tomorrow.t_max} C, min {tomorrow.t_min} C)"
            )
            if night_band in (Band.ORANGE, Band.RED) and assessment.band.value < night_band.value:
                assessment.band = night_band
                assessment.actions.insert(0,
                    "Run the aerator from 22:00 tonight until 07:00 "
                    "(ರಾತ್ರಿ 10 ರಿಂದ ಬೆಳಿಗ್ಗೆ 7 ರವರೆಗೆ ಏರೇಟರ್ ಚಾಲು ಮಾಡಿ)")
        except Exception as exc:  # offline / API down: never block the farmer's verdict
            assessment.notes.append(f"Weather forecast unavailable ({type(exc).__name__})")

    farm_day = FarmDay(reading.pond_id, reading, assessment, night_band, night_reason)

    if dispatcher is not None:
        farm_day.alerts.append(dispatcher.dispatch(assessment, channel=channel))
    return farm_day


def demo_day(pond_id: str = "KA-BLR-001", species: str = "tilapia",
             seed: int = 7, dispatcher: AlertDispatcher | None = None,
             channel: str = "console") -> FarmDay:
    """Runs the final *dawn* reading of the bundled 14-day sample - fully offline.

    Dawn (06:00) is the reading that matters: it is when dissolved oxygen is lowest
    and when a farmer is standing at the pond [FAO-DO].
    """
    rows = synthetic.load_csv()
    rows = [r for r in rows if r["pond_id"] == pond_id] or rows
    final_day = rows[-1]["ts"][:10]
    dawn = [r for r in rows if r["ts"][:10] == final_day and r["ts"][11:13] == "06"]
    last = dawn[0] if dawn else rows[-1]
    reading = PondReading(
        pond_id=last["pond_id"], ts=__import__("datetime").datetime.fromisoformat(last["ts"]),
        water_temp_c=float(last["water_temp_c"]), do_mgl=float(last["do_mgl"]),
        ph=float(last["ph"]), turbidity_ntu=float(last["turbidity_ntu"]),
        ammonia_total_mgl=float(last["ammonia_total_mgl"]),
        nitrite_mgl=float(last["nitrite_mgl"]),
        source="bundled sample (public-dataset-calibrated stand-in)",
    )
    return assess_pond(reading, species=species, dispatcher=dispatcher, channel=channel,
                       include_forecast=False)

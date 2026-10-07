"""FastAPI service: the thing a co-operative actually runs.

Endpoints
---------
GET  /health                     liveness
GET  /catalog                    the public-data catalogue (deliverable #2), JSON
GET  /pond/{pond_id}/status      last verdict for a pond (the app's polling endpoint)
POST /pond/{pond_id}/reading     farmer's phone/manual reading -> verdict + alerts
POST /pond/{pond_id}/photo       multipart JPEG -> disease triage (used by WhatsApp bot)
GET  /pond/{pond_id}/forecast    next 3 nights of DO-crash risk from Open-Meteo
POST /simulate/crash-night       demo: play the RED scenario end-to-end

Run:  uv run aqua-alert serve   (or: uv run uvicorn aqua_alert.api.server:app --host 0.0.0.0)
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .. import anomaly, pipeline, risk
from ..alerts.dispatcher import AlertDispatcher, escalation_matrix
from ..disease import classifier
from ..schemas import PondReading
from ..sources import registry, weather

app = FastAPI(
    title="AquaAlert - early disease & water-quality alerts for small fish farms",
    version="0.1.0",
    description="Sensorless-first alerting built on public datasets (see /catalog).",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# In-memory store: a real deployment swaps this for SQLite/Postgres. Small co-ops
# with 20-200 ponds genuinely do not need more than SQLite.
STATE: dict[str, dict] = {}
DISPATCHER = AlertDispatcher()

SPECIES_BY_POND = {
    "KA-BLR-001": "rohu",
    "KA-BLR-002": "tilapia",
    "AP-WGL-007": "shrimp",
}
FARM_COORDS = (12.9716, 77.5946)   # Bengaluru - override per pond in production


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "aqua-alert", "version": app.version}


@app.get("/catalog")
def catalog() -> JSONResponse:
    return JSONResponse({"count": len(registry.CATALOG), "entries": registry.to_json()})


@app.get("/catalog/{entry_id}")
def catalog_entry(entry_id: str) -> JSONResponse:
    e = registry.by_id(entry_id)
    if not e:
        raise HTTPException(404, f"no catalogue entry {entry_id}")
    return JSONResponse(e.model_dump(mode="json"))


@app.post("/pond/{pond_id}/reading")
def post_reading(pond_id: str, reading: PondReading, channel: str = "console") -> dict:
    reading.pond_id = pond_id
    species = SPECIES_BY_POND.get(pond_id, "tilapia")
    day = pipeline.assess_pond(reading, species=species, dispatcher=DISPATCHER,
                               channel=channel, lat=FARM_COORDS[0], lon=FARM_COORDS[1])
    STATE[pond_id] = {
        "assessment": day.assessment.model_dump(mode="json"),
        "night_band": day.night_band.value if day.night_band else None,
        "night_reason": day.night_reason,
        "alerts": [a.model_dump(mode="json") for a in day.alerts],
    }
    return STATE[pond_id]


@app.get("/pond/{pond_id}/status")
def status(pond_id: str) -> dict:
    if pond_id not in STATE:
        # cold start: derive from the bundled public-dataset-calibrated sample
        day = pipeline.demo_day(pond_id=pond_id,
                                species=SPECIES_BY_POND.get(pond_id, "tilapia"))
        STATE[pond_id] = {"assessment": day.assessment.model_dump(mode="json"),
                          "night_band": None, "night_reason": None, "alerts": []}
    return STATE[pond_id]


@app.post("/pond/{pond_id}/photo")
async def photo(pond_id: str, file: UploadFile = File(...)) -> dict:
    import tempfile
    suffix = "." + (file.filename or "photo.jpg").split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(await file.read())
        path = fh.name
    pred = classifier.predict(path, species=SPECIES_BY_POND.get(pond_id, "tilapia"))
    return {"pond_id": pond_id, "prediction": pred.model_dump(mode="json")}


@app.get("/pond/{pond_id}/forecast")
def forecast(pond_id: str, lat: float | None = None, lon: float | None = None) -> dict:
    lat = lat or FARM_COORDS[0]
    lon = lon or FARM_COORDS[1]
    days = weather.fetch_daily(lat, lon, date.today(), date.today() + timedelta(days=3))
    out = []
    for d in days:
        band, reason = risk.do_crash_forecast(d.t_max, d.t_min, d.wind_ms, d.precip_mm,
                                              d.cloud_pct, None)
        out.append({"date": d.day.isoformat(), "band": band.value, "reason": reason,
                    "virtual_water_temp_c": d.virtual_water_temp(),
                    "rain_mm": d.precip_mm, "t_max": d.t_max, "t_min": d.t_min,
                    "wind_ms": round(d.wind_ms, 1)})
    return {"pond_id": pond_id, "coords": {"lat": lat, "lon": lon}, "nights": out,
            "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"}


@app.get("/escalation-matrix")
def matrix() -> dict:
    return escalation_matrix()


@app.post("/simulate/crash-night")
def simulate_crash() -> dict:
    """Injects a classic pre-monsoon crash night: hot, still, 05:30 DO = 1.6 mg/L."""
    reading = PondReading(
        pond_id="KA-BLR-001", water_temp_c=33.4, do_mgl=1.6, ph=8.9,
        turbidity_ntu=180, ammonia_total_mgl=1.9, nitrite_mgl=0.8,
        source="simulated crash night",
    )
    day = pipeline.assess_pond(reading, species="rohu", dispatcher=DISPATCHER,
                               channel="console", include_forecast=False)
    return {"assessment": day.assessment.model_dump(mode="json"),
            "alerts": [a.model_dump(mode="json") for a in day.alerts]}


@app.get("/", response_class=HTMLResponse)
def farmer_screen() -> str:
    """Serve the farmer-facing alert screen (deliverable #3) from screens/."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[3] / "screens" / "farmer_app.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "<h1>AquaAlert</h1><p>screens/farmer_app.html not found</p>"


def reset_state() -> None:
    STATE.clear()
    anomaly.reset()

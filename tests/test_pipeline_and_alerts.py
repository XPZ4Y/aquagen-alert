"""End-to-end: catalogue integrity, offline demo, dedupe, drift, API contract."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aqua_alert import anomaly, pipeline
from aqua_alert.alerts.dispatcher import AlertDispatcher, escalation_matrix
from aqua_alert.schemas import Band, PondReading
from aqua_alert.sources import registry, synthetic


# --------------------------------------------------------------- catalogue -----
def test_catalogue_is_complete_and_cited():
    assert len(registry.CATALOG) >= 12
    ids = [e.id for e in registry.CATALOG]
    assert len(ids) == len(set(ids)), "catalogue ids must be unique"
    for e in registry.CATALOG:
        assert e.url.startswith("http")
        assert e.license, f"{e.id} is missing a license"
        assert e.replaces_sensor, f"{e.id} does not say which sensor it replaces"
        assert e.role_in_system


def test_catalogue_covers_every_deliverable_kind():
    kinds = {e.kind for e in registry.CATALOG}
    assert {"timeseries", "images", "weather", "remote_sensing", "thresholds"} <= kinds


# ------------------------------------------------------------------- data -------
def test_bundled_sample_is_pond_like(tmp_path):
    rows = synthetic.generate(days=14, seed=3)
    assert len(rows) == 56
    do = [r["do_mgl"] for r in rows]
    # a real pond: oxygen peaks in the afternoon and bottoms out near dawn
    dawn = [r["do_mgl"] for r in rows if r["ts"].endswith("T06:00")]
    noon = [r["do_mgl"] for r in rows if r["ts"].endswith("T12:00")]
    assert min(do) > 0.5
    assert sum(noon) / len(noon) > sum(dawn) / len(dawn)
    # and the overfeeding story produces a rising nitrogen signal
    assert rows[-1]["nitrite_mgl"] > rows[0]["nitrite_mgl"]


def test_sample_written_and_readable(tmp_path):
    p = synthetic.write_csv(tmp_path / "log.csv", days=3)
    assert len(synthetic.load_csv(p)) == 12


# ------------------------------------------------------------------ pipeline ----
def test_offline_demo_produces_a_verdict_and_an_alert():
    disp = AlertDispatcher()
    day = pipeline.demo_day(dispatcher=disp, channel="console")
    assert day.assessment.band in (Band.YELLOW, Band.ORANGE, Band.RED)
    assert day.assessment.actions
    assert disp.sent, "a non-green verdict must reach the farmer"


def test_green_pond_sends_nothing():
    disp = AlertDispatcher()
    r = PondReading(pond_id="G1", do_mgl=7.5, ph=7.5, water_temp_c=28.0)
    day = pipeline.assess_pond(r, dispatcher=disp, include_forecast=False)
    assert day.assessment.band is Band.GREEN
    assert not day.alerts[0].delivered
    assert "suppressed" in (day.alerts[0].error or "")


def test_alert_dedupe_stops_sms_fatigue():
    disp = AlertDispatcher()
    r = PondReading(pond_id="D1", do_mgl=2.0, ph=7.6, water_temp_c=29.0)
    first = disp.dispatch(pipeline.assess_pond(r, include_forecast=False).assessment)
    second = disp.dispatch(pipeline.assess_pond(r, include_forecast=False).assessment)
    assert first.delivered
    assert not second.delivered
    assert "anti-fatigue" in (second.error or "")


def test_sms_fits_one_gsm_part():
    disp = AlertDispatcher()
    day = pipeline.demo_day()
    msg = disp.compose(day.assessment, "sms")
    assert len(msg) <= 160
    assert "AQUA" in msg


def test_drift_flags_a_pond_that_is_fine_but_not_itself():
    anomaly.reset()
    for _ in range(12):   # this pond's DO normally sits around 7.8
        anomaly.update("DRIFT", "do_mgl", 7.8)
    z = anomaly.drift_scores("DRIFT")
    assert z == {} or abs(z.get("do_mgl", 0)) < 0.1
    anomaly.update("DRIFT", "do_mgl", 4.6)
    assert abs(anomaly.drift_scores("DRIFT")["do_mgl"]) >= 3.0
    assert anomaly.severity_from_drift(anomaly.drift_scores("DRIFT")) > 0.0


def test_escalation_matrix_covers_every_band():
    m = escalation_matrix()
    assert set(m) == {"GREEN", "YELLOW", "ORANGE", "RED"}
    assert "IVR" in m["ORANGE"] or "voice" in m["ORANGE"].lower()


# ---------------------------------------------------------------------- API -----
def test_api_smoke(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from aqua_alert.api import server

    server.reset_state()
    c = fastapi_testclient.TestClient(server.app)
    assert c.get("/health").json()["ok"] is True
    assert c.get("/catalog").json()["count"] >= 12

    r = c.post("/pond/KA-BLR-001/reading", json={
        "do_mgl": 1.7, "ph": 8.9, "water_temp_c": 33.0,
        "ammonia_total_mgl": 1.8, "nitrite_mgl": 0.7,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["assessment"]["band"] == "RED"
    assert body["assessment"]["headline_local"]        # farmer-facing localisation exists

    s = c.post("/simulate/crash-night")
    assert s.status_code == 200
    assert s.json()["assessment"]["band"] == "RED"
    assert c.get("/escalation-matrix").status_code == 200
    server.reset_state()


def test_api_coerces_json_datetime_and_keeps_old_readings(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from aqua_alert.api import server

    server.reset_state()
    c = fastapi_testclient.TestClient(server.app)
    ts = (datetime(2026, 10, 1, 5, 30) - timedelta(0)).isoformat()
    r = c.post("/pond/AP-WGL-007/reading", json={"ts": ts, "do_mgl": 3.1, "ph": 8.1,
                                                 "water_temp_c": 30.0})
    assert r.status_code == 200
    assert c.get("/pond/AP-WGL-007/status").json()["assessment"]["species"] \
        == "pacific_white_shrimp"
    server.reset_state()

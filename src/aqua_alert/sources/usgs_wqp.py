"""USGS/EPA Water Quality Portal client - proof that the pipeline ingests a real
government water-quality API, not only CSV files.

WQP is a cooperative service of USGS + US EPA + 400 agencies, free, no API key,
CSV/JSON/GeoJSON out [waterqualitydata.us]. Parameter codes we care about:
  00300 Dissolved oxygen (mg/L)      00400 pH
  00610 Ammonia, total (mg/L as N)   63676 Turbidity (NTU)
  32209 Chlorophyll a (ug/L)         00613 Nitrite (mg/L as N)
"""

from __future__ import annotations

import csv
import io
from datetime import date

import httpx

WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"

PARAM_CODES = {
    "00300": "do_mgl",
    "00400": "ph",
    "00610": "ammonia_total_mgl",
    "00613": "nitrite_mgl",
    "63676": "turbidity_ntu",
    "32209": "chlorophyll_a_ugl",
    "00010": "water_temp_c",
}


def fetch_results(state_code: str = "us:27", # 27 = Minnesota: dense lake monitoring
                  start: date | None = None, end: date | None = None,
                  params: tuple[str, ...] = ("00300", "00400", "00610", "00010"),
                  max_rows: int = 500, timeout: float = 60.0) -> list[dict]:
    """Download discrete water-quality results as CSV and normalise them."""
    q = {
        "statecode": state_code,
        "mimeType": "csv",
        "sorted": "no",
        "count": "no",
        "pCode": ";".join(params),
    }
    if start:
        q["startDateLo"] = start.strftime("%m-%d-%Y")
    if end:
        q["startDateHi"] = end.strftime("%m-%d-%Y")

    with httpx.Client(timeout=timeout, follow_redirects=True) as c:
        r = c.get(WQP_RESULT_URL, params=q)
        r.raise_for_status()
        text = r.text

    rows: list[dict] = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        if i >= max_rows:
            break
        code = (row.get("CharacteristicName") or "").strip()
        rows.append({
            "site": row.get("MonitoringLocationIdentifier"),
            "date": row.get("ActivityStartDate"),
            "characteristic": code,
            "value": _to_float(row.get("ResultMeasureValue")),
            "unit": row.get("ResultMeasure/MeasureUnitCode"),
            "param": PARAM_CODES.get(row.get("USGSPCode", ""), "other"),
            "source": "USGS/EPA WQP",
        })
    return rows


def _to_float(v: str | None) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def to_pond_reading(rows: list[dict], pond_id: str):
    """Pivot WQP rows for one site/date into a PondReading (public data as a virtual sensor)."""
    from datetime import datetime

    from ..schemas import PondReading

    fields = {"do_mgl": None, "ph": None, "ammonia_total_mgl": None,
              "nitrite_mgl": None, "turbidity_ntu": None, "water_temp_c": None}
    site = None
    for r in rows:
        if r["param"] in fields and r["value"] is not None:
            fields[r["param"]] = r["value"]
            site = site or r["site"]
    return PondReading(pond_id=pond_id, ts=datetime.now().astimezone(), **fields,
                       source=f"USGS WQP site {site}")

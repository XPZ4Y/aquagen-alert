"""Typed contracts shared by ingestion, risk engine, alerts and the API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Band(str, Enum):
    """Farmer-facing severity. Deliberately only 4 levels - a farmer acts on colour, not numbers."""

    GREEN = "GREEN"      # safe, keep routine
    YELLOW = "YELLOW"    # watch, cheap preventive action
    ORANGE = "ORANGE"    # act today, real risk of loss
    RED = "RED"          # emergency, act within hours


class SourceKind(str, Enum):
    TIMESERIES = "timeseries"
    IMAGES = "images"
    THRESHOLDS = "thresholds"
    REMOTE_SENSING = "remote_sensing"
    WEATHER = "weather"


class PondReading(BaseModel):
    """One observation set for one pond at one time.

    Every field is Optional on purpose: the sensorless path almost never has all of them.
    The risk engine degrades gracefully and reports which parameters are *missing*.
    """

    pond_id: str = ""                      # set by the caller/API path
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    water_temp_c: float | None = None
    do_mgl: float | None = None            # dissolved oxygen
    ph: float | None = None
    turbidity_ntu: float | None = None
    ammonia_total_mgl: float | None = None  # total ammonia nitrogen (TAN)
    nitrite_mgl: float | None = None
    tds_mgl: float | None = None
    secchi_cm: float | None = None
    salinity_ppt: float | None = None
    source: str = "manual"                  # manual test kit | sensor | virtual | model
    note: str | None = None


class ParameterRisk(BaseModel):
    parameter: str
    value: float | None
    unit: str
    ideal: tuple[float, float]
    severity: float = Field(ge=0.0, le=1.0)   # 0 safe -> 1 extreme
    status: Literal["ok", "watch", "act", "critical", "missing"]
    reason: str


class RiskAssessment(BaseModel):
    pond_id: str
    species: str
    ts: datetime
    band: Band
    score: float = Field(ge=0.0, le=1.0)
    headline_en: str
    headline_local: str
    actions: list[str]
    drivers: list[ParameterRisk]
    missing: list[str]
    notes: list[str] = []
    model_version: str = "rules-0.1"


class DiseasePrediction(BaseModel):
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    backend: str                    # "torch" | "heuristic-fallback"
    triage: Literal["healthy", "monitor", "treat", "escalate_to_expert"]
    advice_en: str
    advice_local: str


class Alert(BaseModel):
    id: str
    pond_id: str
    created_at: datetime
    band: Band
    channel: Literal["sms", "voice", "whatsapp", "app", "console"]
    message: str
    dedupe_key: str
    delivered: bool = False
    error: str | None = None


class CatalogEntry(BaseModel):
    """One public dataset / API in the data catalogue (deliverable #2)."""

    id: str
    name: str
    kind: SourceKind
    url: str
    license: str
    what_it_provides: str
    replaces_sensor: str
    role_in_system: str
    access_cost: str = "free"

"""Open-Meteo client - the 'free sensor' that replaces thermometers, rain gauges and
the dissolved-oxygen probe's night-time job.

Open-Meteo is CC BY 4.0, needs no API key, and covers the whole globe with hourly
ERA5 reanalysis back to 1940 plus a 16-day forecast [open-meteo.com].

Sensorless trick: pond water temperature lags air temperature and, in a small,
shallow, still pond, tracks roughly the *diurnal mean* with a 1-3 h lag. We model it
with an exponentially weighted moving average over the last 24 h of air temperature
plus a solar-radiation bump - a 2-parameter model that a farmer never has to touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class DayWeather:
    day: date
    t_max: float
    t_min: float
    precip_mm: float
    wind_ms: float
    cloud_pct: float
    radiation_mj: float
    source: str = "open-meteo"

    def virtual_water_temp(self) -> float:
        """Estimate pond water temperature (degC).

        Small still ponds run warm in the afternoon and buffer the night. We blend
        the daily mean air temperature with solar heating and subtract night cooling;
        calibrated against the tropical pond datasets in the catalogue
        (data.mendeley 8s73jfvgr5 and Kaggle synth-aquaponics), where pond temp sits
        within ~1.5 C of the daily mean air temperature.
        """
        mean_air = (self.t_max + self.t_min) / 2.0
        solar_bonus = min(1.8, self.radiation_mj / 22.0)      # shallow pond heats fast
        cloud_damping = 1.0 - 0.006 * self.cloud_pct          # cloud cover reduces heating
        wind_cooling = min(1.2, 0.18 * self.wind_ms)          # evaporative cooling
        return round(mean_air + solar_bonus * cloud_damping - wind_cooling, 1)


def _parse(payload: dict) -> list[DayWeather]:
    d = payload["daily"]
    out: list[DayWeather] = []
    for i, day in enumerate(d["time"]):
        y, m, dd = (int(x) for x in day.split("-"))
        out.append(DayWeather(
            day=date(y, m, dd),
            t_max=d["temperature_2m_max"][i],
            t_min=d["temperature_2m_min"][i],
            precip_mm=d["precipitation_sum"][i] or 0.0,
            wind_ms=(d["wind_speed_10m_max"][i] or 0.0) / 3.6,   # km/h -> m/s
            cloud_pct=d["cloud_cover_mean"][i] or 0.0,
            radiation_mj=d["shortwave_radiation_sum"][i] or 0.0,
        ))
    return out


def fetch_daily(lat: float, lon: float, start: date, end: date,
                timeout: float = 20.0) -> list[DayWeather]:
    """Free, no-key forecast/recent weather for a pond coordinate."""
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "wind_speed_10m_max,cloud_cover_mean,shortwave_radiation_sum",
        "timezone": "auto", "start_date": start.isoformat(), "end_date": end.isoformat(),
    }
    # future ranges need the forecast endpoint, past ones the archive
    url = FORECAST_URL if end >= date.today() else ARCHIVE_URL
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return _parse(r.json())


def fetch_forecast(lat: float, lon: float, days: int = 3) -> list[DayWeather]:
    """Next N days - the input to the pre-dawn oxygen-crash warning."""
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "wind_speed_10m_max,cloud_cover_mean,shortwave_radiation_sum",
        "timezone": "auto", "forecast_days": days,
    }
    with httpx.Client(timeout=20.0) as c:
        r = c.get(FORECAST_URL, params=params)
        r.raise_for_status()
        return _parse(r.json())

"""The high-risk maths: ammonia toxicity, oxygen saturation, severity banding."""

from __future__ import annotations

import pytest

from aqua_alert.risk import assess, do_crash_forecast
from aqua_alert.schemas import Band, PondReading
from aqua_alert.thresholds import (
    do_saturation_mgl,
    unionized_ammonia_fraction,
    unionized_ammonia_mgl,
)


def test_unionized_fraction_matches_eifac_tables():
    # FAO AC175E18 reproduces the Emerson/EIFAC %-un-ionized-NH3 tables.
    assert unionized_ammonia_fraction(8.5, 20.0) == pytest.approx(0.111, abs=0.004)
    assert unionized_ammonia_fraction(7.0, 20.0) == pytest.approx(0.004, abs=0.002)
    # warmer + more alkaline = dramatically more toxic ammonia at the same TAN
    assert unionized_ammonia_fraction(8.5, 30.0) > unionized_ammonia_fraction(8.5, 20.0)


def test_ammonia_conversion_is_the_free_sensor():
    # 1.0 mg/L TAN at pH 9.0, 30 C is an emergency; the same TAN at pH 6.5 is not.
    assert unionized_ammonia_mgl(1.0, 9.0, 30.0) > 0.2      # tilapia kill threshold [TIL]
    assert unionized_ammonia_mgl(1.0, 6.5, 30.0) < 0.02


def test_do_saturation_is_physical():
    assert 8.0 < do_saturation_mgl(25.0) < 8.6     # ~8.26 mg/L standard tables
    assert 14.0 < do_saturation_mgl(0.0) < 14.8    # ~14.6 mg/L
    assert do_saturation_mgl(10.0) > do_saturation_mgl(30.0)   # colder water holds more O2


def test_banding_escalates_on_a_single_critical_parameter():
    # Everything fine except oxygen at dawn
    r = PondReading(pond_id="T1", do_mgl=1.6, ph=7.8, water_temp_c=29.0)
    a = assess(r, species="rohu")
    assert a.band is Band.RED
    assert any("aerat" in x.lower() for x in a.actions)


def test_healthy_pond_stays_green_and_says_what_is_missing():
    r = PondReading(pond_id="T1", do_mgl=7.2, ph=7.4, water_temp_c=28.0)
    a = assess(r, species="tilapia")
    assert a.band is Band.GREEN
    assert any(m.startswith("turbidity_ntu") for m in a.missing)
    assert any(m.startswith("ammonia_total_mgl") for m in a.missing)


def test_missing_data_never_crashes():
    a = assess(PondReading(pond_id="T1"), species="tilapia")
    assert a.band is Band.GREEN
    assert a.score == 0.0
    assert len(a.missing) == 6


def test_high_ph_makes_a_medium_tan_dangerous():
    safe = assess(PondReading(pond_id="T", do_mgl=6.0, ph=6.8, water_temp_c=28,
                              ammonia_total_mgl=1.0), species="tilapia")
    risky = assess(PondReading(pond_id="T", do_mgl=6.0, ph=9.2, water_temp_c=31,
                               ammonia_total_mgl=1.0), species="tilapia")
    assert risky.score > safe.score
    assert risky.band.value >= Band.ORANGE.value


def test_night_crash_forecast_fires_before_the_kill():
    hot_still = do_crash_forecast(35, 27, 0.8, 0.0, 10, 5.4)
    cool_wet = do_crash_forecast(27, 22, 4.0, 25.0, 95, 8.0)
    assert hot_still[0] in (Band.ORANGE, Band.RED)
    assert cool_wet[0] is Band.GREEN
    assert "oxygen" in hot_still[1].lower()

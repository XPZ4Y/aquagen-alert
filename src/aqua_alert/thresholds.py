"""Species envelopes + the water-chemistry maths that turns cheap inputs into
parameters small farmers cannot afford to measure.

Why this module matters
-----------------------
A small farm cannot buy a dissolved-oxygen probe (INR 8k-25k), an ammonia
photometer (INR 15k+) and a turbidity meter. But it *can* buy, or already owns:
  * a 30-rupee thermometer,
  * a 150-rupee pH strip,
  * a rain gauge / the sky,
  * a mobile phone with a camera.
Everything else in this module is derived, not measured.

References
----------
[FAO-DO]   FAO Training Series, "Improving pond water quality", x6709e ch.2.5
           -> DO minima: trout 5, common carp 3, tilapia 2 mg/L.
[FAO-NH3]  FAO, "Site selection for aquaculture: chemical features of water",
           AC175E18 ch.18 (Emerson et al. 1975 / EIFAC 1986 pKa tables).
[EXT]      eXtension/SRAC "Water Quality in Aquaculture" species table.
[TIL]      El-Sherif & El-Feky 2008; Nehemia et al. 2012, reviewed in
           PMC9640449 "Effect of environmental factors on growth performance
           of Nile tilapia".
[APHA]     Standard Methods / Benson-Krause DO saturation equation.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Derived-parameter chemistry
# --------------------------------------------------------------------------- #


def unionized_ammonia_fraction(ph: float, temp_c: float) -> float:
    """Fraction of total ammonia (TAN) present as toxic NH3.

    Emerson et al. (1975) pKa model, as tabulated by EIFAC 1986 and reproduced by
    FAO [FAO-NH3]:  pKa = 0.09018 + 2729.92 / (T + 273.15);  f = 1 / (1 + 10^(pKa - pH)).

    >>> round(unionized_ammonia_fraction(8.5, 30.0) * 100, 1)   # warm, alkaline pond
    20.2
    >>> round(unionized_ammonia_fraction(8.5, 20.0) * 100, 1)   # same pH, cooler water
    11.1
    >>> round(unionized_ammonia_fraction(7.0, 20.0) * 100, 1)   # same TAN, far less toxic
    0.4
    """
    pka = 0.09018 + 2729.92 / (temp_c + 273.15)
    return 1.0 / (1.0 + 10 ** (pka - ph))


def unionized_ammonia_mgl(tan_mgl: float, ph: float, temp_c: float) -> float:
    """Toxic NH3-N in mg/L from a cheap total-ammonia test strip + pH strip + thermometer."""
    return tan_mgl * unionized_ammonia_fraction(ph, temp_c)


def do_saturation_mgl(temp_c: float, salinity_ppt: float = 0.0) -> float:
    """DO at 100% air saturation (mg/L). Benson-Krause / APHA [APHA].

    Used to convert "the water feels low" into a defensible % saturation, and to
    translate Open-Meteo water-temperature *estimates* into an oxygen capacity.
    """
    tk = temp_c + 273.15
    ln_c = (
        -139.34411
        + 1.575701e5 / tk
        - 6.642308e7 / tk**2
        + 1.243800e10 / tk**3
        - 8.621949e11 / tk**4
    )
    c = pow(2.718281828459045, ln_c)
    if salinity_ppt <= 0:                      # freshwater pond: no correction
        return c
    # Chen & Millero salinity correction, linearised (coastal shrimp ponds only)
    return c - 0.0841 * salinity_ppt + 0.000255 * salinity_ppt * temp_c


# --------------------------------------------------------------------------- #
# Species envelopes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Envelope:
    """Safe operating window for one parameter.

    ideal      : no action, no cost.
    act        : sub-optimal, cheap preventive action + a YELLOW warning.
    crit_low/high: physiology breaking down -> ORANGE/RED, must act today.
    weight     : contribution to the fused risk score (weights sum to 1.0).
    """

    ideal: tuple[float, float]
    act: tuple[float, float]
    crit_low: float
    crit_high: float
    weight: float
    unit: str
    note: str = ""


@dataclass(frozen=True)
class SpeciesProfile:
    name: str
    local_names: dict[str, str]
    envelopes: dict[str, Envelope]
    source: str
    feed_stop_below_do: float   # below this, advise feeding stop
    aeration_below_do: float    # below this, advise aerator / agitation


_TILAPIA = SpeciesProfile(
    name="tilapia",
    local_names={"en": "Tilapia", "kn": "ತಿಲಾಪಿಯಾ", "hi": "तिलापिया"},
    envelopes={
        # FAO: tilapia hard minimum 2 mg/L; growth <3.5; optimum 5-8 [FAO-DO][TIL]
        "do_mgl": Envelope((5.0, 12.0), (3.0, 14.0), 2.0, 18.0, 0.35, "mg/L",
                           "Feeding stops <3 mg/L; mortality risk <1 mg/L [TIL]"),
        # Toxic NH3-N (not TAN). >0.2 mg/L prolonged exposure starts mortality [TIL]
        "nh3_mgl": Envelope((0.0, 0.02), (0.02, 0.05), 0.0, 0.20, 0.20, "mg/L NH3-N",
                            "NH3-N >0.2 mg/L causes kills; appetite loss at 0.08 [TIL]"),
        "ph": Envelope((6.5, 8.5), (6.0, 9.0), 5.0, 9.5, 0.15, "pH"),
        "water_temp_c": Envelope((26.0, 32.0), (22.0, 35.0), 15.0, 38.0, 0.15, "degC"),
        "turbidity_ntu": Envelope((0.0, 80.0), (0.0, 150.0), 0.0, 300.0, 0.10, "NTU"),
        "nitrite_mgl": Envelope((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N"),
    },
    source="[FAO-DO], [TIL], [EXT]",
    feed_stop_below_do=3.0,
    aeration_below_do=4.0,
)

_ROHU = SpeciesProfile(
    name="indian_major_carp",   # rohu / catla / mrigal polyculture - the dominant Indian pond system
    local_names={"en": "Rohu-Catla (IMC)", "kn": "ರೋಹು-ಕಟ್ಲಾ", "hi": "रोहू-कतला"},
    envelopes={
        # FAO: common carp minimum 3 mg/L, preferred >=5 [FAO-DO]
        "do_mgl": Envelope((5.0, 12.0), (3.5, 14.0), 3.0, 18.0, 0.35, "mg/L"),
        "nh3_mgl": Envelope((0.0, 0.02), (0.02, 0.05), 0.0, 0.10, 0.20, "mg/L NH3-N",
                            "Carp tolerate total ammonia 3-4 mg/L at pH<8 [FAO-NH3]"),
        "ph": Envelope((6.5, 8.5), (6.0, 9.0), 5.5, 9.5, 0.15, "pH"),
        "water_temp_c": Envelope((24.0, 32.0), (20.0, 35.0), 12.0, 38.0, 0.15, "degC"),
        "turbidity_ntu": Envelope((0.0, 100.0), (0.0, 200.0), 0.0, 350.0, 0.10, "NTU"),
        "nitrite_mgl": Envelope((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N"),
    },
    source="[FAO-DO], [FAO-NH3], [EXT]",
    feed_stop_below_do=3.5,
    aeration_below_do=4.5,
)

_PANGASIUS = SpeciesProfile(
    name="pangasius_catfish",
    local_names={"en": "Pangasius / Catfish", "kn": "ಪಂಗಾಸಿಯಸ್", "hi": "पंगासियस"},
    envelopes={
        "do_mgl": Envelope((4.0, 12.0), (2.5, 16.0), 1.0, 20.0, 0.35, "mg/L",
                           "Air-breathers: survive very low DO but grow poorly <2.5 mg/L"),
        "nh3_mgl": Envelope((0.0, 0.02), (0.02, 0.06), 0.0, 0.15, 0.20, "mg/L NH3-N"),
        "ph": Envelope((6.5, 8.5), (6.0, 9.0), 5.0, 9.5, 0.15, "pH"),
        "water_temp_c": Envelope((26.0, 32.0), (23.0, 36.0), 18.0, 39.0, 0.15, "degC"),
        "turbidity_ntu": Envelope((0.0, 100.0), (0.0, 200.0), 0.0, 350.0, 0.10, "NTU"),
        "nitrite_mgl": Envelope((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N"),
    },
    source="[FAO-DO], [FAO-NH3]",
    feed_stop_below_do=2.5,
    aeration_below_do=3.0,
)

_PRAWN = SpeciesProfile(
    name="pacific_white_shrimp",
    local_names={"en": "Whiteleg shrimp", "kn": "ಸೀಗಡಿ", "hi": "झींगा"},
    envelopes={
        "do_mgl": Envelope((5.0, 12.0), (3.5, 14.0), 2.5, 16.0, 0.35, "mg/L"),
        "nh3_mgl": Envelope((0.0, 0.01), (0.01, 0.03), 0.0, 0.10, 0.20, "mg/L NH3-N",
                            "Shrimp are the most ammonia-sensitive animal in the pond"),
        "ph": Envelope((7.2, 8.5), (7.0, 9.0), 6.5, 9.5, 0.15, "pH"),
        "water_temp_c": Envelope((26.0, 31.0), (23.0, 34.0), 18.0, 36.0, 0.15, "degC"),
        "turbidity_ntu": Envelope((0.0, 60.0), (0.0, 120.0), 0.0, 250.0, 0.10, "NTU"),
        "nitrite_mgl": Envelope((0.0, 0.3), (0.0, 0.8), 0.0, 3.0, 0.05, "mg/L NO2-N"),
    },
    source="[FAO-DO], [EXT]",
    feed_stop_below_do=3.5,
    aeration_below_do=4.5,
)

PROFILES: dict[str, SpeciesProfile] = {
    "tilapia": _TILAPIA,
    "rohu": _ROHU,
    "carp": _ROHU,
    "pangasius": _PANGASIUS,
    "shrimp": _PRAWN,
}


def get_profile(species: str) -> SpeciesProfile:
    return PROFILES.get(species.strip().lower(), _TILAPIA)


# --------------------------------------------------------------------------- #
# Feeding & aeration decision aid (this is what saves money day-to-day)
# --------------------------------------------------------------------------- #


@dataclass
class FeedAdvice:
    feed_fraction: float        # fraction of normal ration to give today
    text_en: str
    text_local: str
    reason: str


def feed_advice(profile: SpeciesProfile, do_mgl: float | None, nh3_mgl: float | None,
                temp_c: float | None, local_lang: str = "kn") -> FeedAdvice:
    """Feeding into low DO is the classic small-farm money burner: uneaten feed
    decays, ammonia rises, oxygen falls further [TIL]."""
    reasons: list[str] = []
    frac = 1.0
    if do_mgl is not None and do_mgl < profile.feed_stop_below_do:
        frac = 0.0
        reasons.append(f"DO {do_mgl:.1f} < {profile.feed_stop_below_do} mg/L")
    elif do_mgl is not None and do_mgl < profile.envelopes["do_mgl"].ideal[0]:
        frac = 0.5
        reasons.append(f"DO {do_mgl:.1f} mg/L below optimum")
    if nh3_mgl is not None and nh3_mgl > profile.envelopes["nh3_mgl"].act[1]:
        frac = min(frac, 0.5)
        reasons.append(f"NH3-N {nh3_mgl:.3f} mg/L high")
    if temp_c is not None and temp_c > profile.envelopes["water_temp_c"].ideal[1]:
        frac = min(frac, 0.7)
        reasons.append(f"water {temp_c:.0f} C hot")
    reason = "; ".join(reasons) if reasons else "all monitored parameters in range"
    texts = {
        0.0: ("Do NOT feed today", "ಇಂದು ಮೀನುಗಳಿಗೆ ಆಹಾರ ಕೊಡಬೇಡಿ"),
        0.5: ("Feed only half the normal ration", "ಅರ್ಧ ಪ್ರಮಾಣದಷ್ಟೇ ಆಹಾರ ಕೊಡಿ"),
        0.7: ("Feed about 70% of the normal ration", "ಸಾಮಾನ್ಯದ 70% ಆಹಾರ ಕೊಡಿ"),
        1.0: ("Feed the normal ration", "ಸಾಮಾನ್ಯ ಪ್ರಮಾಣದಲ್ಲಿ ಆಹಾರ ಕೊಡಿ"),
    }
    en, local = texts[round(frac, 1)]
    return FeedAdvice(frac, en, local, reason)

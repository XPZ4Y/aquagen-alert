"""THE DATA CATALOGUE (deliverable #2).

Every entry below is a *public, existing* dataset or API that substitutes for a
physical sensor the small farm cannot buy. `replaces_sensor` says which sensor it
stands in for; `role_in_system` says what our code does with it.

Verified 2026-10-07. Access notes are in docs/DATA_SOURCES.md (with DOI/URLs).
"""

from __future__ import annotations

from ..schemas import CatalogEntry, SourceKind

CATALOG: list[CatalogEntry] = [
    # ---------------- A. Water-quality time series (replaces pH/DO/ammonia probes) ----
    CatalogEntry(
        id="kaggle-aquaponics-nigeria",
        name="Sensor Based Aquaponics Fish Pond Datasets (U. Nigeria Nsukka)",
        kind=SourceKind.TIMESERIES,
        url="https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets",
        license="CC BY 4.0 (see Kaggle page)",
        what_it_provides="5-second-interval temperature, turbidity, DO, pH, ammonia, nitrate "
                         "from 12 catfish ponds (Jun-Oct 2021), with fish population/length/weight",
        replaces_sensor="pH probe, DO probe, ammonia sensor, turbidity sensor",
        role_in_system="Primary training + calibration set for the absolute thresholds and for "
                       "the DO-crash forecaster; gives labelled 'normal pond' baselines.",
    ),
    CatalogEntry(
        id="kaggle-pondsdata-iot",
        name="Pondsdata - water quality data for aquaculture (IoT, 74,759 records)",
        kind=SourceKind.TIMESERIES,
        url="https://www.kaggle.com/datasets/apgopi/pondsdata",
        license="See Kaggle page",
        what_it_provides="pH, DO, temperature, turbidity, ammonia, nitrate, manganese, "
                         "Feb 2022-Jan 2023",
        replaces_sensor="Full sensor suite",
        role_in_system="Seasonal drift baselines (monsoon vs summer response of the same pond).",
    ),
    CatalogEntry(
        id="mendeley-aquaculture-monteria",
        name="Environmental Parameters in Aquaculture (Monteria, Colombia)",
        kind=SourceKind.TIMESERIES,
        url="https://data.mendeley.com/datasets/8s73jfvgr5/1",
        license="CC BY 4.0",
        what_it_provides="Temperature, DO, pH, turbidity with date/hour from tropical ponds, 2024",
        replaces_sensor="pH/DO/turbidity probes",
        role_in_system="DOI-citable calibration set; tropical pond diurnal curves for the "
                       "night-oxygen model.",
    ),
    CatalogEntry(
        id="usgs-wqp",
        name="Water Quality Portal (USGS + US EPA + 400 agencies)",
        kind=SourceKind.TIMESERIES,
        url="https://www.waterqualitydata.us/data/Result/search",
        license="Public domain (US federal data)",
        what_it_provides="REST/CSV discrete water-chemistry results (parameter codes 00300 DO, "
                         "00400 pH, 00610 ammonia, 63676 turbidity, 32209 chlorophyll-a)",
        replaces_sensor="Lab bench for ammonia/nitrite/nitrate/chl-a",
        role_in_system="Live ingestion of a real government API (code path exercised in "
                       "src/aqua_alert/sources/usgs_wqp.py); also the source of the "
                       "'this is what your pond should look like' reference distributions.",
    ),
    CatalogEntry(
        id="cpcb-nwmp-opencity",
        name="CPCB National Water Monitoring Programme - lakes/ponds/tanks/wetlands "
             "(2017-2022) via OpenCity",
        kind=SourceKind.TIMESERIES,
        url="https://data.opencity.in/dataset/water-quality-data-of-lakes-and-ponds-in-india",
        license="Public Domain",
        what_it_provides="Indian inland water body data: pH, DO, BOD, nitrate, faecal coliform, "
                         "temperature, per year CSVs/PDFs sourced from CPCB",
        replaces_sensor="Regional baseline for Indian surface water (not just US/EU)",
        role_in_system="Locality prior: what 'normal' looks like in Karnataka standing water, "
                       "used to sanity-check an alert before it reaches the farmer.",
    ),
    CatalogEntry(
        id="data-gov-in-aqi-cpcb",
        name="data.gov.in Open Government Data (CPCB) API",
        kind=SourceKind.TIMESERIES,
        url="https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
        license="Government Open Data License - India",
        what_it_provides="Free-tier JSON API (needs a free api-key) streaming CPCB environmental "
                         "station feeds, incl. water-quality resources",
        replaces_sensor="Not a sensor replacement - a free national data channel",
        role_in_system="Optional Indian deployment channel; keeps the stack on Indian public "
                       "infrastructure for a government-sponsored rollout.",
    ),

    # ---------------- B. Remote sensing (replaces chlorophyll/turbidity probes) --------
    CatalogEntry(
        id="sentinel2-ndci",
        name="Copernicus Sentinel-2 MSI (Level-2A) + NDCI algorithm",
        kind=SourceKind.REMOTE_SENSING,
        url="https://dataspace.copernicus.eu/ (Copernicus Data Space Ecosystem)",
        license="Copernicus open data (CC BY-SA 3.0 IGO / free, open)",
        what_it_provides="10 m x 5-day multi-spectral reflectance; NDCI = (B5-B4)/(B5+B4) is a "
                         "validated chlorophyll-a proxy for small water bodies "
                         "(adj. R^2 up to 0.84 in small reservoirs)",
        replaces_sensor="Chlorophyll-a / algal-bloom probe (INR 40k+), and spot turbidity checks",
        role_in_system="Weekly pond-scale 'greenness' score per farm polygon -> bloom-risk input; "
                       "validated against WQP chlorophyll-a (R^2 > 0.5 nationwide).",
    ),
    CatalogEntry(
        id="nasa-pace-sst",
        name="NASA PACE / MODIS-Aqua chlorophyll-a and SST products",
        kind=SourceKind.REMOTE_SENSING,
        url="https://oceancolor.gsfc.nasa.gov/",
        license="Public domain (NASA)",
        what_it_provides="Global chlorophyll-a and sea-surface temperature (MODIS Jul 2002-Feb 2022 "
                         "as used in peer-reviewed aquaculture-impact studies)",
        replaces_sensor="Regional water-temperature logger + bloom monitor",
        role_in_system="Fallback for farms whose pond is below Sentinel-2's useful size, and "
                       "long-run climate context for risk scoring.",
    ),

    # ---------------- C. Weather / hydrology (replaces the oxygen & rain sensors) ------
    CatalogEntry(
        id="open-meteo",
        name="Open-Meteo Historical Weather Archive + Forecast API (ERA5)",
        kind=SourceKind.WEATHER,
        url="https://api.open-meteo.com/v1/forecast",
        license="CC BY 4.0; free for non-commercial use, no API key, ~10,000 calls/day",
        what_it_provides="Hourly/daily air temperature, humidity, wind, rainfall, cloud cover, "
                         "solar radiation, soil moisture from 1940 to 16-day forecast, global",
        replaces_sensor="Rain gauge, air thermometer, anemometer, solar radiometer",
        role_in_system="Drives the pre-dawn DO-crash forecaster and the virtual water-temperature "
                       "model (risk.do_crash_forecast, sources/weather.py). This is the single "
                       "highest-leverage sensorless component.",
    ),
    CatalogEntry(
        id="noaa-gfs-ecmwf",
        name="NOAA GFS / ECMWF IFS forecasts (via Open-Meteo multi-model layer)",
        kind=SourceKind.WEATHER,
        url="https://open-meteo.com/en/docs",
        license="US Government public domain / ECMWF open data",
        what_it_provides="16-day ensemble-ish forecast horizons from 30+ national weather models",
        replaces_sensor="Nothing physical - extends lead time",
        role_in_system="Lets the app warn 'Thursday night is dangerous' 2-4 days ahead, which is "
                       "when a farmer can still hire an aerator or pump.",
    ),

    # ---------------- D. Disease / image data (replaces the microscope + vet) ----------
    CatalogEntry(
        id="kaggle-freshwater-south-asia",
        name="Freshwater Fish Disease Aquaculture in South Asia (7 classes x 250 imgs)",
        kind=SourceKind.IMAGES,
        url="https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
        license="CC0: Public Domain",
        what_it_provides="Bacterial (Aeromoniasis, Gill disease, Red disease), Fungal "
                         "(Saprolegniasis), Parasitic, Viral (White Tail Disease), Healthy - "
                         "collected partly from a farm in Odisha, India; 2,444 images in v4",
        replaces_sensor="Field vet visit / wet-lab examination",
        role_in_system="Primary classifier training set for the photo triage screen "
                       "(disease/classifier.py); CC0 license means it can be redistributed to "
                       "cooperatives.",
    ),
    CatalogEntry(
        id="kaggle-nile-tilapia",
        name="Enhancing Disease Detection in Nile Tilapia (NTD-1)",
        kind=SourceKind.IMAGES,
        url="https://www.kaggle.com/datasets/engineeringubu/enhancing-disease-detection-in-nile-tilapia",
        license="See Kaggle page (open access)",
        what_it_provides="1,043 annotated tilapia images: Streptococcosis (327), Columnaris "
                         "(362), Tilapia Lake Virus (354); bounding boxes + lesion masks "
                         "(LabelMe)",
        replaces_sensor="Pathology lab for the three diseases that cause sudden tilapia kills",
        role_in_system="Second training/specialisation set; lesion masks let the model highlight "
                       "*where* the sore is and how big it is, so the farmer can see progress "
                       "after treatment.",
    ),
    CatalogEntry(
        id="hf-panda992-vit",
        name="Pretrained ViT-base fish-disease classifier (panda992/fish_disease_datasets)",
        kind=SourceKind.IMAGES,
        url="https://huggingface.co/panda992/fish_disease_datasets",
        license="Apache-2.0",
        what_it_provides="google/vit-base-patch16-224-in21k fine-tuned on the 7-class "
                         "South-Asia fish-disease set (Bacterial Red disease, Aeromoniasis, "
                         "Gill disease, Saprolegniasis, Parasitic, White Tail Disease, Healthy)",
        replaces_sensor="Immediate disease triage with no training cost; the vet's first look",
        role_in_system="Default checkpoint of the photo triage screen (disease/classifier.py). "
                       "Verified on our demo photos: lesion -> 'Bacterial Red disease' 0.97; "
                       "healthy fish -> 'Healthy Fish' 0.98.",
    ),
    CatalogEntry(
        id="hf-allfahad-eus",
        name="Secondary checkpoint: EUS / gill-rot / lice classifier "
             "(allfahad/fish-disease-v1-original-20260608)",
        kind=SourceKind.IMAGES,
        url="https://huggingface.co/allfahad/fish-disease-v1-original-20260608",
        license="Not stated on the model card - verify before commercial use",
        what_it_provides="Classes incl. Epizootic Ulcerative Syndrome, Bacterial gill rot, "
                         "Lice, Healthy Fish (Bangladesh pond conditions)",
        replaces_sensor="Second opinion on the EUS-type ulcers common in South Asian ponds",
        role_in_system="Fallback/secondary checkpoint in disease/classifier.py; useful where "
                       "EUS rather than 'red disease' is the local vocabulary.",
    ),
    CatalogEntry(
        id="roboflow-fish-disease",
        name="Roboflow 'Fish diseases detection and classification using YOLOv8' pond dataset",
        kind=SourceKind.IMAGES,
        url="https://universe.roboflow.com/ (search: fish disease YOLOv8)",
        license="See workspace license (CC BY 4.0 typical)",
        what_it_provides="Object-detection annotations for ulcer/EUS lesions in real pond photos",
        replaces_sensor="Field observation by an extension officer",
        role_in_system="Detection (not just classification) head: counts and localises lesions. "
                       "Used to fine-tune a small YOLO-nano model for INR ~0 phone inference.",
    ),
    CatalogEntry(
        id="f4k",
        name="Fish4Knowledge (F4K) underwater video/image corpus",
        kind=SourceKind.IMAGES,
        url="https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/",
        license="Academic, open for research use",
        what_it_provides="27,370 verified fish images from 5,824 video clips, 23 species, 9 "
                         "fixed cameras at 3 angles, pre-split 16,430 train / 10,940 test",
        replaces_sensor="Manual count + behaviour observation (feeding response, crowding, "
                        "surface gasping)",
        role_in_system="Behaviour model: fish crowding at the surface / listlessness is a disease "
                       "and hypoxia signal that appears *before* lesions. Also the species-ID "
                       "stage so the risk engine picks the right thresholds.",
    ),

    # ---------------- E. Threshold / advisory knowledge (the 'expert in a box') --------
    CatalogEntry(
        id="fao-x6709e",
        name="FAO Training Series - Improving Pond Water Quality (x6709e)",
        kind=SourceKind.THRESHOLDS,
        url="https://www.fao.org/fishery/docs/CDrom/FAO_Training/FAO_Training/General/x6709e/x6709e02.htm",
        license="FAO open access",
        what_it_provides="Species DO minima and % saturation rules: trout 5, common carp 3, "
                         "tilapia 2 mg/L; saturation-vs-temperature tables",
        replaces_sensor="Not data - the decision rules that turn a reading into a verdict",
        role_in_system="Encoded in thresholds.py as the ORANGE/RED floors.",
    ),
    CatalogEntry(
        id="fao-ac175e18",
        name="FAO - Site Selection for Aquaculture: chemical features of water (AC175E18)",
        kind=SourceKind.THRESHOLDS,
        url="https://www.fao.org/4/ac175e/AC175E18.htm",
        license="FAO open access",
        what_it_provides="Ammonia chemistry: NH3 is 300-400x more toxic than NH4+, with "
                         "Emerson (1975)/EIFAC (1986) % un-ionized NH3 tables by pH and "
                         "temperature",
        replaces_sensor="Ammonia photometer - we convert a cheap TAN strip reading into "
                        "toxic-NH3 using pH + temperature",
        role_in_system="thresholds.unionized_ammonia_mgl(); the single biggest 'free sensor' win "
                       "in this design.",
    ),
    CatalogEntry(
        id="extension-srac-table",
        name="eXtension / SRAC 'Water Quality in Aquaculture' species table",
        kind=SourceKind.THRESHOLDS,
        url="https://freshwater-aquaculture.extension.org/water-quality-in-aquaculture/",
        license="US Cooperative Extension (public)",
        what_it_provides="Catfish/carp and tilapia windows: temp 65-80 F, DO 3-10 mg/L, pH 6-8, "
                         "alkalinity 50-250, ammonia % 0-0.03, nitrite 0-0.6 mg/L",
        replaces_sensor="Extension officer's memory",
        role_in_system="Cross-check on the envelope values (kept in the same units for tests).",
    ),
    CatalogEntry(
        id="tilapia-review-pmc9640449",
        name="Effect of environmental factors on Nile tilapia growth (review, PMC9640449)",
        kind=SourceKind.THRESHOLDS,
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9640449/",
        license="Open access (CC BY)",
        what_it_provides="DO 5-7 mg/L optimum, growth penalty <3.5, mortality <1; prolonged NH3 "
                         "0.2 mg/L starts deaths, appetite loss at 0.08 mg/L",
        replaces_sensor="Aquaculture specialist consult",
        role_in_system="Tilapia profile + the stop-feed rule.",
    ),
]


def by_id(entry_id: str) -> CatalogEntry | None:
    return next((c for c in CATALOG if c.id == entry_id), None)


def to_json() -> list[dict]:
    return [c.model_dump(mode="json") for c in CATALOG]

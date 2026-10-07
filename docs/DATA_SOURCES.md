# DATA SOURCES — the public data that replaces physical sensors

Deliverable #2 of the brief: *"Write a list of the public data you used. This data
replaces physical sensors."*

Status legend: **LIVE** = our code calls it today · **SAMPLE** = a bundled,
public-data-calibrated stand-in ships in the repo so the system runs offline ·
**PLANNED** = wired in the design and documented, adapter is the next engineering step.

Machine-readable version of this list: `GET /catalog` (or `uv run aqua-alert catalog`),
implemented in `src/aqua_alert/sources/registry.py`.

---

## A. Water-quality time series (replace pH / DO / ammonia / turbidity probes)

| # | Source | License | Replaces | Status |
|---|---|---|---|---|
| A1 | **Sensor Based Aquaponics Fish Pond Datasets** — HiPIC Research Group, University of Nigeria Nsukka. 12 catfish ponds, 5-second interval, temperature, turbidity, DO, pH, ammonia, nitrate + fish population/length/weight, Jun–Oct 2021. <https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets> · companion paper: *An internet of things labelled dataset for aquaponics fish pond water quality monitoring system* | CC BY 4.0 (per Kaggle listing) | pH probe, DO probe, ammonia sensor, turbidity sensor | PLANNED (download via `scripts/fetch_public_data.py --kaggle`) |
| A2 | **Pondsdata** — 74,759 IoT records, pH/DO/temperature/turbidity/ammonia/nitrate/manganese, Feb 2022–Jan 2023. <https://www.kaggle.com/datasets/apgopi/pondsdata> | see Kaggle listing | full sensor suite | PLANNED |
| A3 | **Environmental Parameters in Aquaculture** — Montería, Colombia, 2024: temperature, DO, pH, turbidity + timestamps, tropical ponds. Mendeley Data, DOI **10.17632/8s73jfvgr5.1**. <https://data.mendeley.com/datasets/8s73jfvgr5/1> | CC BY 4.0 | pH/DO/turbidity probes | SAMPLE (calibrates `sources/synthetic.py` diurnal curve) |
| A4 | **Water Quality Portal (WQP)** — USGS + US EPA + 400 state/tribal/local agencies; REST + CSV, no key. Parameter codes 00300 DO, 00400 pH, 00610 ammonia, 00613 nitrite, 63676 turbidity, 32209 chlorophyll-a. <https://www.waterqualitydata.us/data/Result/search> · cite as National Water Quality Monitoring Council, doi:**10.5066/P9QRKUVJ** | US public domain | lab bench for ammonia/nitrite/chl-a; cross-region reference distributions | **LIVE** (`sources/usgs_wqp.py`) |
| A5 | **USGS Water Data APIs** (modernised NWIS): continuous values, daily values, time-series metadata, OGC water-quality services. <https://api.waterdata.usgs.gov/> | US public domain | continuous logger at a reference site near the farm | PLANNED |
| A6 | **CPCB National Water Monitoring Programme (NWMP)** — lakes/ponds/tanks/wetlands, 2017–2022, pH/DO/BOD/nitrate/coliform/temperature; CSVs mirrored on OpenCity from CPCB source. <https://data.opencity.in/dataset/water-quality-data-of-lakes-and-ponds-in-india> | Public Domain | India-specific "what normal looks like" prior for standing water | PLANNED |
| A7 | **data.gov.in (OGD) API** — free API key; CPCB environmental resources incl. water quality. <https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69> | Government Open Data License – India | national data channel for a government rollout | PLANNED |

## B. Weather & hydrology (replaces oxygen, rain, temperature and wind sensors)

| # | Source | License | Replaces | Status |
|---|---|---|---|---|
| B1 | **Open-Meteo** forecast + archive API — hourly & daily air temperature, humidity, wind, precipitation, cloud cover, shortwave radiation, soil moisture; ERA5 archive from 1940, 16-day forecast, global, **no API key**. <https://open-meteo.com/en/docs> | CC BY 4.0 (attribution required) | rain gauge, anemometer, air thermometer, radiometer — and by proxy the **night-time DO probe** | **LIVE** (`sources/weather.py`, `risk.do_crash_forecast`, `GET /pond/{id}/forecast`) |
| B2 | **NOAA GFS / ECMWF IFS / 30+ national models** (exposed through Open-Meteo's multi-model layer) | US Gov public domain / ECMWF open data | extends lead time to 2–16 days | LIVE (same endpoint) |

## C. Satellite remote sensing (replaces chlorophyll / bloom / turbidity sensors)

| # | Source | License | Replaces | Status |
|---|---|---|---|---|
| C1 | **Copernicus Sentinel-2 MSI L2A** + **NDCI** = (B5−B4)/(B5+B4) — 10 m, ~5-day revisit; NDCI was the best-performing index for chlorophyll-a in *small* reservoirs (reported adj. R² up to 0.84, with reported RMSE 4 µg/L in one of the two systems studied; R² > 0.5 nationwide in the US assessment). <https://dataspace.copernicus.eu/> | Copernicus open data (free, open) | chlorophyll-a probe (₹40k+), bloom spot-checks | PLANNED (needs a per-farm polygon job; algorithm and validation cited) |
| C2 | **NASA OceanColor — MODIS-Aqua / PACE** chlorophyll-a & SST | NASA public domain | regional bloom + temperature baseline, long-run climate context | PLANNED |

## D. Fish disease & behaviour images (replace the vet, the microscope, the night watchman)

| # | Source | License | Replaces | Status |
|---|---|---|---|---|
| D1 | **Freshwater Fish Disease Aquaculture in South Asia** — 7 classes × 250 images (Bacterial Red disease, Aeromoniasis, Bacterial gill disease, Fungal Saprolegniasis, Parasitic, Viral White Tail Disease, Healthy), partly collected at an Odisha farm; 2,444 images in v4. <https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia> — cite Biswas et al., *IEEE Access* (2024) | **CC0: Public Domain** | field vet visit / wet-lab exam | PLANNED (primary training set; notebook provided) |
| D2 | **Enhancing Disease Detection in Nile Tilapia (NTD-1)** — 1,043 images: Streptococcosis 327, Columnaris 362, Tilapia Lake Virus 354, with bounding boxes + lesion masks (LabelMe). <https://www.kaggle.com/datasets/engineeringubu/enhancing-disease-detection-in-nile-tilapia> | open access (see Kaggle) | pathology lab for the 3 big tilapia killers | PLANNED (specialisation set; 93.5% classification accuracy reported in the source study) |
| D3 | **Pretrained checkpoint** `panda992/fish_disease_datasets` — ViT-base-patch16-224-in21k fine-tuned on the 7-class South-Asia set. <https://huggingface.co/panda992/fish_disease_datasets> | **Apache-2.0** | immediate photo triage with zero training cost | **LIVE** (`disease/classifier.py`; verified on the repo's demo photos: lesion → "Bacterial Red disease" 0.97, healthy → "Healthy Fish" 0.98) |
| D3b | **Pretrained checkpoint** `allfahad/fish-disease-v1-original-20260608` — EUS / bacterial gill rot / lice / healthy classes (Bangladesh pond conditions). <https://huggingface.co/allfahad/fish-disease-v1-original-20260608> | not stated on the model card → verify before commercial use | second opinion on EUS-type ulcers | **LIVE** (secondary checkpoint, same module) |
| D3c | `Saon110/fish-shrimp-disease-classifier` — ViT, 85.8M params, 11 fish+shrimp classes (CC BY-NC-SA 4.0, non-commercial). <https://huggingface.co/Saon110/fish-shrimp-disease-classifier> | CC BY-NC-SA 4.0 | wider class coverage incl. shrimp | available, but **gated** (needs an HF account + terms acceptance), so it is not the default |
| D4 | **Roboflow "Fish Disease" detection set** — 457 images, 7 lesion classes (BDA/BGD/BRD/FDS/HF/PD/WTD), CC BY 4.0; also a 454-image classification version. <https://universe.roboflow.com/yolo-eq1f5/fish-disease-qvxvl> | CC BY 4.0 | localises *where* the lesion is (count + area over time) | PLANNED |
| D5 | **Fish4Knowledge (F4K)** — 27,370 verified fish images from 5,824 videos, 23 species, 9 cameras × 3 angles, pre-split 16,430/10,940. <https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/> | academic/research use | manual count + behaviour watch (surface gasping, crowding, listlessness appear *before* lesions) | PLANNED |

## E. Thresholds & advisory knowledge (the "expert" that turns numbers into a decision)

| # | Source | License | Encoded in |
|---|---|---|---|
| E1 | **FAO Training Series, Improving Pond Water Quality** (x6709e ch. 2.5) — DO minima: trout 5, common carp 3, tilapia 2 mg/L; saturation-vs-temperature method. <https://www.fao.org/fishery/docs/CDrom/FAO_Training/FAO_Training/General/x6709e/x6709e02.htm> | FAO open access | `thresholds.PROFILES`, `do_saturation_mgl` |
| E2 | **FAO, Site Selection for Aquaculture — chemical features of water** (AC175E18 ch. 18) — NH₃ is 300–400× more toxic than NH₄⁺; Emerson (1975) / EIFAC (1986) %-un-ionized-NH₃ tables by pH & temperature. <https://www.fao.org/4/ac175e/AC175E18.htm> | FAO open access | `thresholds.unionized_ammonia_mgl` |
| E3 | **eXtension / SRAC "Water Quality in Aquaculture"** species table — catfish/carp & tilapia: 65–80 °F, DO 3–10 mg/L, pH 6–8, alkalinity 50–250, ammonia 0–0.03 %, nitrite 0–0.6 mg/L. <https://freshwater-aquaculture.extension.org/water-quality-in-aquaculture/> | US Cooperative Extension (public) | cross-check in `tests/` |
| E4 | **Nile tilapia environmental-factors review** (PMC9640449) — DO optimum 5–7, growth penalty <3.5, mortality <1 mg/L; prolonged NH₃ 0.2 mg/L starts deaths, appetite loss at 0.08 mg/L (El-Sherif & El-Feky 2008; Nehemia et al. 2012). <https://pmc.ncbi.nlm.nih.gov/articles/PMC9640449/> | CC BY (open access) | tilapia profile + stop-feed rule |
| E5 | **Sentinel-2 NDCI validation studies** (small reservoirs, Sci. Total Environ. 2022; nationwide US assessment, *Sustainability* 16(5):2090, 2024) | open access | NDCI thresholds for the bloom band (C1) |

---

## How the substitution actually works (one example, end to end)

The farmer cannot measure toxic ammonia — it needs a spectrophotometer. But the toxic
fraction of ammonia depends only on pH and temperature [E2]:

```
NH3-N (toxic) = TAN × 1 / (1 + 10^(pKa − pH)),      pKa = 0.09018 + 2729.92 / (T + 273.15)
```

So: ₹15 ammonia strip (TAN) + ₹5 pH strip + ₹30 thermometer ⇒ a defensible NH₃-N number.
At TAN = 1.8 mg/L, pH 8.9, 33 °C, **43.9 %** of the ammonia is un-ionized ⇒ 0.79 mg/L
NH₃-N — roughly 4× the tilapia kill threshold [E4]. At pH 7.4 and 28 °C the same TAN is
only 1.9 % un-ionized (0.04 mg/L) — harmless. That difference is invisible to the farmer
but decides whether the crop lives. `tests/test_chemistry.py` pins this behaviour against
the FAO/EIFAC tables.

## Access recipes

```bash
uv run aqua-alert catalog                          # the list above, from the code
uv run python scripts/fetch_public_data.py --weather        # Open-Meteo → data/public/
uv run python scripts/fetch_public_data.py --wqp            # USGS/EPA WQP results
uv sync --extra ml && uv run python scripts/fetch_public_data.py --kaggle --hf-model
```

Kaggle downloads need `~/.kaggle/kaggle.json`. Sentinel-2 needs a Copernicus account or a
Google Earth Engine project. **None** are needed for the demo: `data/samples/pond_log_14d.csv`
ships in the repo.

## Attribution we must display (license hygiene)

* Open-Meteo: "Weather data by Open-Meteo.com (CC BY 4.0)" — already printed by the CLI,
  the API response and the farmer screen.
* Sentinel-2 / Copernicus: attribution + "modified Copernicus Sentinel data [year]".
* CPCB/NWMP: "Source: CPCB NWMP, via OpenCity (public domain)".
* WQP: cite the DOI 10.5066/P9QRKUVJ.
* HF checkpoint D3 is **non-commercial**; the shipped production model must be trained on
  D1 (CC0) and D4 (CC BY 4.0) before any commercial deployment.

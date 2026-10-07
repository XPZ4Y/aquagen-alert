# AquaAlert — Python-assisted implementation plan

**Problem.** Small aquaculture farms (0.2–2 ha, 1–4 ponds, ₹1–5 lakh seasonal revenue)
lose whole crops to two things that are cheap to detect and expensive to miss:
(1) night-time dissolved-oxygen crashes and ammonia spikes, and (2) infectious disease
outbreaks. Large producers solve this with instruments (₹8k–25k per DO probe, ₹15k+
ammonia photometer, lab agreements) and staff. Small farms have a phone, a bucket, and
a test kit they use rarely.

**Objective.** Deliver a ≤₹50/farm/month early-warning service that, using existing
public data + a phone, gives (a) a safe/dangerous water verdict and (b) disease triage,
with an alert that reaches a feature phone in the farmer's language — and that is
*honest about its own uncertainty*.

**Non-goals.** Not a diagnosis service; not a replacement for the extension officer;
not a dosing calculator for antibiotics or chemicals.

---

## 1. Architecture (implemented in this repo)

```
┌────────────────── data plane (free) ──────────────────────────────────────────┐
│ Open-Meteo (CC BY 4.0, no key)      Sentinel-2 NDCI      Kaggle/Uni datasets  │
│  air T, wind, rain, cloud, solar     (10 m, 5-day)         pond IoT logs, fish │
│         │                                    │              images, thresholds │
└─────────┼────────────────────────────────────┼────────────────────▼───────────┘
          ▼                                    ▼            ┌──────────────────┐
   sources/weather.py                  ndci job (planned)   │ sources/registry │
   virtual water temperature           bloom-risk score     │  (catalogue)     │
          │                                    │            └──────────────────┘
          ▼                                    │                     │
  risk.do_crash_forecast  ─┐                   │                     │
   (pre-dawn DO crash)     │                   │                     │
                           ▼                   ▼                     ▼
   farmer reading  ──►  thresholds.py (species envelopes, ammonia & DO chemistry)
   (T, pH, TAN…)         anomaly.py (per-pond MAD drift)
                           │
                           ▼
                    risk.assess()  ──►  GREEN / YELLOW / ORANGE / RED + actions (+ Kannada)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   alerts/dispatcher.py          api/server.py (FastAPI)
   SMS ≤160 ch | IVR script      /pond/{id}/reading  /photo  /forecast  /catalog
   WhatsApp | app                screens/farmer_app.html  (deliverable #3)
              ▲
   disease/classifier.py  ◄── phone JPEG (deliverable #1, photo path)
```

Everything is one `uv sync` away; there are no paid dependencies in the core path.
The optional `ml` extra adds torch/transformers for the photo model.

## 2. Work plan (6 weeks, 2 engineers; every phase has an acceptance test)

### Phase 0 — Bootstrap (0.5 day) ✅ done in this repo
* `uv init`, `pyproject.toml`, `.python-version`, `uv.lock`; extras `ml`, `dev`.
* `uv run pytest` green; `uv run aqua-alert demo` prints a verdict without internet.
* **Acceptance:** a fresh machine goes from `git clone` to working demo in <5 minutes.

### Phase 1 — Decision engine (week 1) ✅ core done
* Species envelopes (rohu/catla, tilapia, pangasius, shrimp) from FAO [1][2], extension
  tables [3] and the tilapia review [4].
* Ammonia chemistry [2] and DO saturation [5]; severity → band fusion; feed/no-feed and
  aeration decisions; per-pond drift with robust z-scores (MAD).
* **Acceptance (met):** `tests/test_chemistry.py` reproduces published %-un-ionized-NH₃
  values (11.1 % at pH 8.5/20 °C; 20.2 % at pH 8.5/30 °C); a single critical parameter
  escalates the pond to RED; a fully-missing reading degrades to GREEN with a "missing"
  list instead of crashing.
* **Next:** calibrate weightings against ~30 labelled farm-days collected in Phase 5.

### Phase 2 — Free data ingestion (week 1–2) 🟡 weather live
* Open-Meteo client + virtual water temperature + pre-dawn crash forecaster ✅ live.
* USGS/EPA WQP client ✅ live (reference distributions), CPCB/NWMP loader, Kaggle loaders
  (`scripts/fetch_public_data.py`), Sentinel-2 NDCI job per farm polygon (planned).
* **Acceptance:** ≥3 of the catalogue's `timeseries`-kind sources ingest into a common
  schema with <1 % parse failures; a 30-day weather pull at the farm's coordinates
  reproduces the crash nights that the farmer remembers.

### Phase 3 — Alert tool (week 2–3) ✅ screen + dispatcher done, adapter next
* 4-band escalation matrix; SMS ≤160 chars (GSM-7); Kannada via IVR voice (UCS-2 SMS
  costs 3× and many farmers cannot read — the voice route is the inclusive one);
  WhatsApp for photos; anti-fatigue dedupe (6 h windows, hourly for RED); GREEN silenced.
* Provider: MSG91/Gupshup/Twilio (India) behind `AQUA_SMS_API_KEY`; commercial SMS needs
  a TRAI **DLT** sender ID + template registration → start that paperwork in week 1,
  it takes 1–2 weeks of lead time and blocks a pilot if left late.
* **Acceptance:** a RED verdict from `/simulate/crash-night` lands on a real phone within
  30 s; no more than 1 SMS/day/pond under normal conditions.

### Phase 4 — Disease model (week 3–5) 🟡 public checkpoint already wired in
* **Day-1 capability (done):** the app already triages photos with public, ungated
  checkpoints — `panda992/fish_disease_datasets` (Apache-2.0 ViT-base, 7 South-Asia
  classes) and `allfahad/fish-disease-v1-original-20260608` (EUS/gill-rot/lice). Verified
  on the repo's demo photos: lesion → "Bacterial Red disease" 0.97, healthy → "Healthy
  Fish" 0.98. Below 0.60 confidence the service **abstains** and refers.
* Train our own on D1 (CC0, 7 classes) + D2 (NTD-1, 1,043 annotated tilapia) + D4
  (Roboflow CC BY 4.0); Fish4Knowledge [9] later for behaviour. Notebook:
  `notebooks/01_train_fish_disease_cnn.ipynb`.
* Backbone: EfficientNet-B0 (or ViT-B/16) transfer learning, 224 px, aggressive
  augmentation (rotation, flip, colour jitter, blur — pond photos are badly lit and
  blurry), **subject-level** stratified 5-fold CV (photos of the same fish never straddle
  the split), temperature-scaling for calibrated confidence, abstain-and-refer when
  p < 0.60.
* Deploy: ONNX → TFLite, quantised, on-device inference (no photo leaves the phone
  without consent); server path stays as fallback.
* **Acceptance:** on a held-out, *field-collected* set of ≥300 photos: macro-F1 ≥ 0.80
  with ≥ 0.90 sensitivity for the "escalate to expert" classes (viral/bacterial
  septicaemia) — sensitivity, not accuracy, is the metric that matters: a missed
  outbreak costs the crop.
* Reported reference points from the source datasets (upper bounds, not promises):
  ~99 % F1 on the South-Asia set [10]; 93.5 % accuracy on NTD-1 [6].

### Phase 5 — Field pilot & validation (week 5–6, then ongoing)
* 10–20 ponds across 3 clusters (e.g. Karnataka inland, Andhra coastal shrimp, Odisha
  IMC), 6 weeks covering a monsoon transition.
* Protocol: dawn (05:30–06:30) + afternoon (15:00) reference measurement with a
  calibrated commercial meter (borrow/rent, do not buy) + daily photo of 3 fish + the
  farmer's own judgement, logged in the same app.
* Metrics: (a) RED-alert precision/recall against "true crash" (measured DO < 2 mg/L);
  target recall ≥ 0.90, ≥ 1 false RED per pond per month is acceptable, > 3 is not;
  (b) time-to-alert before the event (target ≥ 8 h); (c) farmer action compliance from
  voice-call follow-ups; (d) ammonia model error vs lab (target ±0.05 mg/L NH₃-N);
  (e) equity checks: feature-phone-only farms must receive equal-quality warnings.
* Human-in-the-loop labels: "Did the fish die? Did the treatment work?" replies are
  captured and become training labels (with consent) — this is how the model improves
  beyond the public datasets.

### Phase 6 — Scale & optional hardware (post-pilot)
* ₹700–1,500 upgrade kit (ESP32 + DS18B20 + pH module) posting to `/pond/{id}/reading`
  — the API is already shaped for it; the farmer's screen does not change.
* Co-operative dashboard (multi-pond), SQLite → Postgres, offline-first mobile client
  (queue readings, sync when signal returns).
* Handoff: fisheries-department/co-operative ownership; model retraining every season
  with fresh field labels; data-sharing agreement back to the public datasets used.

## 3. Cost model (per farm, per month)

| Item | Cost | Notes |
|---|---|---|
| Public data (weather, satellite, datasets) | ₹0 | CC BY 4.0 / public domain / CC0 |
| Verdict computation | ~₹0 | runs on a co-operative server or the farmer's phone |
| SMS (yellow/orange) | ₹0.15 each ≈ ₹5–15 | 1-part GSM, GREEN never sent |
| IVR voice call (Kannada, 20 s) | ₹0.30–0.60 each ≈ ₹5 | only for orange/red |
| Ammonia strip | ₹15/test × 4–8 tests/month | optional; improves NH₃ accuracy |
| **Total** | **₹10–50 / farm / month** | vs ₹8k–25k for one DO probe |

## 4. Risks and mitigations

| Risk | Impact | Mitigation (in the design) |
|---|---|---|
| False RED → farmer stops trusting alerts | Fatal to adoption | worst-case physics (not ML) drives RED; dedupe; GREEN silence; "why" panel shows the exact parameter; post-event review in the app |
| False GREEN on a dying pond | Crop loss | drift detector catches "fine in absolute terms, wrong for this pond"; nitrogen canary; re-test prompts |
| Weather API down / no internet | No forecast | absolute thresholds + farmer history still work offline; 3-day local cache; SMS fallback path independent of the app |
| Photo model fails on dark/blurry pond photos | Bad triage | abstain-and-refer below 0.60 confidence; heuristic never claims a diagnosis; field-set evaluation in Phase 5; on-device path preferred |
| Dataset bias (lab-clean South-Asia/NTD images) | Overstated accuracy | report field numbers only, never dataset numbers; subject-level splits; per-site evaluation |
| Antibiotic misuse amplified by the app | Public-health harm | the app never prescribes drugs; "escalate to expert" is a hard class; content reviewed by a fisheries pathologist |
| Language/literacy exclusion | Inequity | IVR voice in Kannada/Hindi/Tamil/Bengali; icon+colour screen; SMS in transliteration |
| Privacy (photos, pond GPS) | Trust | on-device inference by default, explicit consent for upload, no resale of farm data |
| Sentinel-2 pond too small / cloudy | Missing bloom signal | pond polygon + 3-day composite; fall back to farm-self-reported visual (secchi/turbidity) |

## 5. KPIs for the pilot (what "it works" means)

1. ≥ 90 % recall for crash-night RED alerts, ≥ 8 h median lead time.
2. ≥ 60 % of alerted farmers perform the recommended action within 6 h (measured by
   follow-up call/IVR keypress).
3. Disease triage macro-F1 ≥ 0.80 on field photos; ≥ 0.90 sensitivity on escalate classes.
4. ≤ 3 alerts/farm/week (any band) — alert fatigue ceiling.
5. Cost ≤ ₹50/farm/month at 200 farms per cluster.
6. 100 % of farms with feature phones receive alerts (equity check).

## 6. Team & effort

| Role | Weeks | Focus |
|---|---|---|
| Backend/data engineer | 6 | ingestion, API, dispatch, deploy (`uv`, FastAPI, SQLite) |
| ML engineer | 4 (from wk 3) | disease CNN, calibration, TFLite export, behaviour model |
| Aquaculture advisor (part-time) | 2 | thresholds review, advice content, field protocol |
| Field coordinator + 2 enumerators | 4 | 10–20 ponds, reference meters, farmer training, label collection |

## 7. Why this is defensible (the three sentences a judge should remember)

1. We never buy the sensor — we compute the dangerous quantity from data that is already
   free: pre-dawn oxygen crash from a weather API, toxic ammonia from a ₹15 strip plus FAO
   chemistry tables, blooms from Sentinel-2, and disease from public, licensed image sets.
2. We optimise the thing that matters in the field: **one action, at the right hour, in the
   farmer's language, on the cheapest phone in the village** — with GREEN deliberately
   silent so red still means red.
3. We are honest about the boundary of the science: physics and thresholds for water,
   a calibrated-but-abstaining ML model for disease, extension officer always in the loop.

## References

1. FAO, *Improving Pond Water Quality*, FAO Training Series x6709e ch. 2.5 — species DO minima.
2. FAO, *Site Selection for Aquaculture: chemical features of water*, AC175E18 ch. 18 — % un-ionized NH₃ (Emerson 1975 / EIFAC 1986).
3. eXtension/SRAC, *Water Quality in Aquaculture* — species window table.
4. El-Sherif & El-Feky (2008); Nehemia et al. (2012), reviewed in *Effect of environmental factors on growth performance of Nile tilapia*, PMC9640449.
5. Benson & Krause (1984) DO saturation / APHA Standard Methods.
6. *Robust Tilapia Disease Diagnosis…*, Appl. Sci. 16(13):6359 — NTD-1 dataset (1,043 images), 93.5 % classification accuracy.
7. Open-Meteo — free no-key weather archive/forecast (CC BY 4.0), <https://open-meteo.com>.
8. National Water Quality Monitoring Council, *Water Quality Portal*, doi:10.5066/P9QRKUVJ; USGS Water Data APIs, <https://api.waterdata.usgs.gov/>.
9. Fish4Knowledge — 27,370 images / 23 species, <https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/>.
10. Biswas et al., *IEEE Access* (2024) + Kaggle *Freshwater Fish Disease Aquaculture in South Asia* (CC0, 2,444 images).
11. *Mapping algal bloom dynamics in small reservoirs using Sentinel-2 imagery in Google Earth Engine*, Ecol. Indic. (2022) — NDCI adj. R² up to 0.84; *Sustainability* 16(5):2090 (2024) — nationwide 2BDA/NDCI assessment.
12. USP/CPCB NWMP lakes & ponds data via OpenCity (public domain); data.gov.in OGD API.

Full URLs and licenses for every dataset: **`docs/DATA_SOURCES.md`**.

# Project — coupon_RS working notes

_Living document. One source of truth for status, plan, and open decisions._

**Deliverable:** 10-page scientific report + code + contribution statement +
AI-tool-use honesty + 1-page learning reflection. **Due 28 September 2026.**
**Pitch:** 4 min + 1 min Q&A.

---

## 1 · Where we are

Progress is tracked by **phase**, not calendar days — some phases run in
parallel, some slip. Move a phase to ✅ only when its Definition of Done is met.

| # | Phase | Status | Definition of Done |
|---|-------|--------|--------------------|
| 1 | Data audit & EDA | ✅ Done | 6 audit questions answered, 5 data-quality checks passed, executive summary + limitations captured in the EDA notebook |
| 2 | Baselines + frozen eval harness | 🚧 In progress | `scripts/run_baselines.py` runs end-to-end; `src/eval_harness.py` and `src/config.py` frozen and signed; SOP filled in |
| 3 | Improved method (implicit ALS) | 🔜 | ALS model trained on train weeks, hyperparameters tuned on validation, scored through the frozen harness alongside baselines |
| 4 | Beyond-accuracy analysis | 🔜 | Redemption-uplift, coverage, and stratified metrics (by activity tier / RFM quintile) computed with bootstrap CIs |
| 5 | Pitch (4 min) | 🔜 | Deck finalised, notes rehearsed twice, group aligned on message |
| 6 | Report (10 pages) | 🔜 | All 10 sections drafted per the outline in § 5, contribution statement + AI-use note + learning reflection appended |

### Findings from Phase 1 (locked in as facts)

| # | Finding | Implication |
|---|---------|-------------|
| 1 | Density 37.18 % at commodity (27.89 % at department) | Not sparse — ALS will converge fast, low latent dim is fine |
| 2 | 3.9 % cold (< 10 baskets), 1.4 % heavy (> 500) | Stratified reporting will strengthen the story |
| 3 | Retailer targeting = **2.55× median lift**, every campaign > 1 | Baseline has real signal — beating it is a margin game |
| 4 | Redemption rate = **0.27 %** (adjusted) | Money-metric effect sizes will be small; report bootstrap CIs |
| 5 | Test window z = +0.51, 11 anomalous weeks in panel | Split is representative; anomalies are absorbed into training |
| 6 | Demographics cover 32 % of households, non-random (spend CI [3247, 3802]) | Do NOT use demographics as main-model features; keep for post-hoc |

---

## 2 · Phase 2 — Baselines + frozen eval harness

**Goal:** by end of this phase, any recommender can be scored through one
function, and four baselines already have results in the table.

### Concrete steps

1. **Pull the working tree.** `src/`, `scripts/`, `docs/`, `PROJECT.md`.
2. **Sanity-check `src/data_loader.py`** in a notebook cell — confirm dtypes
   load cleanly and row counts match the EDA notebook.
3. **Run the harness.** `python scripts/run_baselines.py` from repo root.
   Expect one printed block per baseline and a CSV at
   `artifacts/baseline_results.csv`.
4. **Freeze `src/eval_harness.py` and `src/config.py`.** Commit both with
   the message `freeze: eval harness and temporal split`. Fill in Owner and
   Frozen-on-date at the top of both files and in
   `docs/EVAL_HARNESS_SOP.md`.
5. **Announce in the group channel.** After this, every model routes
   through `src.eval_harness.evaluate(...)`.

### What we expect to see

- **Random baseline:** Recall@3 ≈ 3 / N_active × 3 ≈ 20–30 %. Sanity floor.
- **Popularity baseline:** ~30–40 % on Recall@3. Beats random by a lot.
- **Segment-popularity baseline:** ~35–45 % on Recall@3. The one to beat.
- **Last-category baseline:** ~35–45 %. May or may not beat segment-pop —
  either way is a real result.
- **Coverage:** popularity ≈ 3 / N_active (very low). Segment-pop meaningfully
  higher because different segments get different lists.
- **Redemption uplift ratio:** noisy on 2,500 households, likely wide CIs.

### Definition of Done

- `scripts/run_baselines.py` runs end-to-end without errors.
- `artifacts/baseline_results.csv` exists with one row per baseline.
- `src/eval_harness.py` and `src/config.py` are committed as frozen.
- `docs/EVAL_HARNESS_SOP.md` shows Owner and Frozen date at the top.
- Team channel message: _"Harness frozen. Improved-method phase can start."_

---

## 3 · Phase 3 preview — improved method

**Primary:** weighted implicit ALS via the `implicit` library.
`household × commodity` log-spend matrix, dim=64, alpha=40, reg=0.01 (defaults
in `src/config.py`; treat as starting point, tune on validation weeks 92–96).

Rank campaigns by average predicted score over covered commodities (same
`campaign → coupon → product → commodity` join used in the last-category
baseline).

**Stretch:** LightFM hybrid using the product hierarchy
(department + commodity) as side features. Only if ALS is running clean.

**All models route through the frozen harness.**

---

## 4 · Phase 4 preview — beyond-accuracy analysis

- **Redemption-uplift matched-vs-unmatched** analysis, with bootstrap 95 % CIs.
  Base rate is 0.27 % — expect wide bands. Report either way.
- **Stratified metrics** by household activity tier (cold / mid / heavy) and
  by RFM quintile — 3× more numbers, same eval function.
- **Category coverage & intra-list diversity** as the beyond-accuracy story.
- Optional: **difficulty-adjusted metric** — weight each recommendation error
  by 1 / campaign_lift so beating the retailer on a hard campaign counts more.

---

## 5 · Report outline (10 pages, single column)

Target section budget:

| Section | Pages | Notes |
|---------|-------|-------|
| 1. Introduction & problem framing | 0.75 | Target user (CM), business why (~70 % waste), scope |
| 2. Dataset & feedback signals | 1.00 | Panel description, tables, Phase 1 findings 1–6 |
| 3. Related work (light) | 0.50 | RFM history, implicit ALS, LightFM/FM lineage |
| 4. Method | 2.00 | Baselines, ALS, hybrid (if any), architecture diagram |
| 5. Evaluation protocol | 1.00 | Split, candidate set, ground truth, metrics, harness freeze |
| 6. Results & analysis | 2.00 | Main table (CIs), stratified table, coverage/diversity plots |
| 7. Beyond-accuracy discussion | 1.00 | Money-metric proxy, uplift interpretation, difficulty-adjusted |
| 8. Limitations & responsible use | 1.00 | Small panel, proxy-not-causal, fairness caveats |
| 9. Discussion & future work | 0.50 | A/B test as the honest next step, LightFM if deferred |
| 10. Conclusion | 0.25 | Two-line landing |

Plus (outside the 10 pages): contribution statement, honest AI-tool-use
report, 1-page learning reflection.

---

## 6 · Still missing / open decisions

### Blocking (decide before freezing the harness)
- [ ] **Owner of the frozen harness** — the person who signs the SOP.
- [ ] **Baseline of record** — RFM segment popularity is the default; confirm.
- [ ] **Author sections** — who writes what in the 10-page report.

### Phase 3 (improved-method) decisions
- [ ] Build the LightFM hybrid or defer to future work?
- [ ] Validation-set metric driving ALS hyperparameter choice — Recall@3 or NDCG@3?

### Phase 4 (analysis) decisions
- [ ] Time-series 3-fold CV or single split? (Recommended: single split for
      block week, add CV in the report's robustness section.)
- [ ] Implement the difficulty-adjusted metric or leave as future work?

### Not-yet-scheduled
- [ ] **Contribution statement template** — one paragraph per team member.
- [ ] **AI-tool-use honesty note** — list of tools used, what for.
- [ ] **1-page learning reflection** — each member's own.
- [ ] **Report bibliography** — decide citation style, aim for ~15 references.

---

## 7 · File map

```
coupon_RS/
├── PROJECT.md                          # <-- you are here
├── README.md                           # public repo readme
├── .gitignore
├── Data/                               # raw CSVs, gitignored
├── artifacts/                          # generated outputs, gitignored
│   └── baseline_results.csv            # after run_baselines.py
├── docs/
│   ├── Reco_Systems_Pitch.pdf
│   └── EVAL_HARNESS_SOP.md             # freeze rules
├── eda/
│   └── 01_EDA_Sparsity_Targeting_Signal_Data_Quality.ipynb
├── scripts/
│   └── run_baselines.py                # baseline runner
└── src/
    ├── __init__.py
    ├── config.py                       # FROZEN — paths, splits, K, seeds
    ├── data_loader.py                  # CSV loaders + day_range helper
    ├── eval_harness.py                 # FROZEN — evaluate() lives here
    └── baselines.py                    # random, popularity, RFM-seg, last-category
```

---

## 8 · Key architectural decisions (locked in)

- **Candidate set = active campaigns in test window.** ~30 items, closed
  universe, no negative-sampling debate.
- **Ground truth = retailer's own campaign assignment** (`campaign_table.csv`).
  Proxy for "correct target," acknowledged.
- **Temporal split, never random split** — weeks 1–91 / 92–96 / 97–102.
- **`household_key` is lowercase; every other ID is UPPERCASE.** Reflects the
  raw CSV schema; do not rename.
- **Do not modify `src/eval_harness.py` or `src/config.py` after freeze** —
  escalate via `docs/EVAL_HARNESS_SOP.md` instead.
- **Bootstrap CIs on Recall and NDCG** are the default; wide bands on money
  metric are expected and reported.
- **`hh_demographic` is NOT used as a model feature** in the main study
  (non-random 32 % coverage). Kept for post-hoc subgroup analysis only.

---

## 9 · Handy commands

```bash
# From repo root
python scripts/run_baselines.py                  # run baselines end-to-end
python -c "from src.data_loader import load_core; d = load_core(); print({k: len(v) for k, v in d.items()})"  # sanity check row counts
```

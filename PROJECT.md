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

## 6b · Study 2 — personalised COUPON_UPC recommendation

**Task.** For every household, predict the top-K individual coupons
(`COUPON_UPC`, ~1,135 in total) they will *redeem* in the test window.

**Why this study.** Study 1 recommends **campaigns** scored against retailer
**assignments** — a 12-item candidate set dominated by blanket TypeB/C
campaigns and only 1 TypeA in the test window. It's a well-scoped task, but
the ground truth is *"who did the retailer send this to,"* not *"who wanted
it."* Study 2 flips both axes: ~350-item candidate set (coupons whose parent
campaign overlaps the test window) and ground truth from
`coupon_redempt.csv` — actual redemption behaviour. This is the natural
setting for latent-factor methods, and it lets us reconstruct the private
TypeA logic Dunnhumby's PDF explicitly says is "outside the scope of this
database."

**Files added (no modification to Study 1 code):**
```
src/redemption_task.py         # task builder + coupon_values + evaluate_coupons()
src/coupon_baselines.py        # random, popularity, repeat_buy, last_category, item_knn
src/coupon_als.py              # implicit-ALS on hh x coupon binary matrix
scripts/run_coupon_recommender.py
```
Metric formulas (`recall_at_k`, `ndcg_at_k`, `bootstrap_mean_ci`) are imported
from the frozen `src/eval_harness.py` — no metric divergence between studies.
Study 2 adds one metric on top: `expected_revenue_at_k`, the hit-weighted
sum of per-coupon mean line-item SALES_VALUE (see definition below).

**Run:**
```bash
python scripts/run_coupon_recommender.py    # writes artifacts/coupon_results.csv
```

**Task shape (locked in on first run):**

| Field | Value |
|---|---|
| n candidate coupons | 354 |
| n active parent campaigns | 12 |
| n train redemptions | 1,305 |
| n test redemptions (in candidate set) | 946 |
| n eval households (≥1 test redemption) | 255 |
| n households with train history | 325 |
| median train-history length | 3 coupons |
| median test-truth length | 2 coupons |
| train DAY range | 1..551 |
| test DAY range | 587..711 |

**Results (K=3, bootstrap 95 % CIs, N=1000):**

| Model | Recall@3 | NDCG@3 | E[revenue]@3 | Coupon coverage |
|---|---|---|---|---|
| Random (sanity floor) | 0.007 [0.002, 0.017] | 0.010 | $0.10 [$0.03, $0.18] | 0.870 |
| Popularity | 0.041 [0.024, 0.059] | 0.061 | $0.54 [$0.39, $0.69] | 0.008 |
| Repeat-buy | 0.046 [0.030, 0.066] | 0.078 | $0.73 [$0.49, $1.06] | 0.096 |
| Item-kNN cosine | 0.023 [0.011, 0.037] | 0.034 | $0.29 [$0.17, $0.41] | 0.107 |
| ALS coupon (allow seen) | 0.044 [0.027, 0.064] | 0.075 | $0.67 [$0.45, $0.96] | 0.105 |
| ALS coupon (exclude seen) | 0.023 [0.010, 0.037] | 0.030 | $0.29 [$0.15, $0.45] | 0.082 |
| **Last-category** 🏆 | **0.200** [0.167, 0.238] | **0.247** | **$2.37** [$2.08, $2.68] | 0.040 |

Same ordering at K=10 (last_category 0.293, repeat_buy 0.097, ALS 0.091).
CIs of last_category are disjoint from every other model on all three metrics
— this is a robust, not marginal, win.

**E[revenue]@K definition.** For each household, sum the mean line-item
SALES_VALUE of every recommended coupon that appears in ground truth. Per
COUPON_UPC value = mean transaction-line SALES_VALUE across the products
that coupon covers, computed on train weeks only. Fallback = global median
coupon value ($3.20). This is a hit-weighted revenue proxy, not a causal
uplift — the same honest framing as Study 1's redemption_uplift.

**Honest findings from Study 2:**

1. **Random collapses to ~0.7 % Recall@3**, versus 24.5 % in Study 1. This is
   the direct effect of moving from a 12-item to a 354-item candidate set —
   the difficulty jump is real, and it's what makes Study 2 the more
   informative task.
2. **Last-category is the winner by a 4× margin** on Recall@3 and a 3.2×
   margin on expected revenue, with disjoint CIs. Ranking each candidate
   coupon by the household's train-window spend in the commodities that
   coupon covers turns out to be the strongest single signal — much stronger
   than raw redemption history alone. The category-spend link (coupon →
   product → COMMODITY_DESC → SALES_VALUE) is what latent-factor methods
   fail to reconstruct implicitly from binary redemption events.
3. **Repeat-buy still beats random and popularity**, but is now the second
   tier: personal redemption history matters, but it's dominated by "which
   aisle does this household actually shop in." A household who redeems a
   yogurt coupon this month is somewhat likely to redeem it again; a
   household who spends heavily in the yogurt aisle is *very* likely to
   redeem *any* yogurt coupon.
4. **ALS on binary redemptions matches repeat-buy** (0.044 vs 0.046, CIs
   overlap) and loses to last_category by 4.5×. The `als_coupon_noseen`
   ablation confirms ALS's performance *is* the repeat-purchase signal it
   learned. Latent-factor ALS on this feedback matrix cannot recover the
   category-spend signal that a direct coupon → commodity join delivers for
   free.
5. **Item-kNN underperforms** — median train history is only 3 coupons, too
   few neighbours for stable similarity scores. Genuine sparsity finding at
   the coupon level (contrast with 37 % density at the commodity level from
   EDA finding #1 — density in Study 2 lives in *commodities*, not
   *coupons*).
6. **Cross-study message.** In Study 1 the winner (segment_demographic)
   beat every latent-factor model on a candidate set dominated by blanket
   assignments. In Study 2 the winner (last_category) beats every
   latent-factor model on a candidate set of individual coupons with real
   redemption signal. Both times, the story is: **domain-informed feature
   engineering beats generic collaborative filtering on this panel**, and
   ALS's job is to *match* the right baseline — not to beat it — while
   giving us the stress test that confirms the baseline isn't a coincidence.

**How Study 2 slots into the 10-page report.** Method (§4), Evaluation (§5),
Results (§6) each get one extra paragraph or sub-table contrasting the two
formulations. Beyond-accuracy (§7) gains the "grocery-habit vs
retailer-blanket" comparison. Limitations (§8) picks up the sparsity note.
The two studies together are the sophisticated framing this repo now
supports — few capstones make the campaign-vs-redemption distinction
explicit.

**Open decisions specific to Study 2:**
- [ ] Confirm `last_category` as Study 2 winner of record (Recall@3 = 0.200,
      disjoint CIs vs every other model). `repeat_buy` is the honest
      strong-baseline label.
- [ ] Report `als_coupon` with `exclude_seen=False` (matches repeat-buy
      story) or add both variants? Current runner reports both.
- [ ] `expected_revenue_at_k` uses per-coupon MEAN line-item value on train
      as the weight. Alternatives: median (more robust to product-mix outliers),
      or coupon face value inferred from COUPON_DISC in transactions.

---

## 7 · File map

```
coupon_RS/
├── PROJECT.md                          # <-- you are here
├── README.md                           # public repo readme
├── .gitignore
├── Data/                               # raw CSVs, gitignored
├── artifacts/                          # generated outputs, gitignored
│   ├── model_results.csv               # after scripts/run_models.py
│   ├── fm_results.csv                  # after scripts/run_fm.py
│   └── coupon_results.csv              # after scripts/run_coupon_recommender.py
├── docs/
│   ├── Reco_Systems_Pitch.pdf
│   └── EVAL_HARNESS_SOP.md             # freeze rules
├── eda/
│   └── 01_EDA_Sparsity_Targeting_Signal_Data_Quality.ipynb
├── requirements.txt                    # numpy, pandas, scipy, sklearn, implicit, torch
├── scripts/
│   ├── run_models.py                   # Study 1 baselines + ALS runner
│   ├── run_fm.py                       # Study 1 FM runner (needs torch)
│   └── run_coupon_recommender.py       # Study 2 runner
└── src/                                # 7 modules (was 12 before consolidation)
    ├── __init__.py
    ├── config.py                       # FROZEN — paths, splits, K, seeds
    ├── data_loader.py                  # CSV loaders + day_range helper
    ├── eval_harness.py                 # FROZEN — Study 1 evaluate() lives here
    ├── baselines.py                    # Study 1 non-FM: random/pop/RFM/demographic/last-category/ALS
    ├── fm_model.py                     # Study 1 FM + campaign-level causal features
    └── study2.py                       # Study 2: task, evaluator, baselines, coupon-ALS
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
# From repo root — Study 1 (campaign recommendation)
python scripts/run_baselines.py                  # baselines only
python scripts/run_models.py                     # baselines + ALS + segment_demographic
python scripts/run_fm.py                         # PyTorch FM (needs local `pip install torch`)

# From repo root — Study 2 (personalised COUPON_UPC redemption)
python scripts/run_coupon_recommender.py        # baselines + ALS on coupons

# Sanity
python -c "from src.data_loader import load_core; d = load_core(); print({k: len(v) for k, v in d.items()})"
```

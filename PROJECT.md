# Project — coupon_RS working notes

_Living document. One source of truth for status, plan, and open decisions._

**Deliverable:** 10-page scientific report + code + contribution statement +
AI-tool-use honesty + 1-page learning reflection. **Due 28 September 2026.**
**Pitch:** 4 min + 1 min Q&A.

---

## 0 · Current direction (September 2026)

The project recommends, for each household, a short list of grocery **products**
to offer a coupon on, and checks those picks against what the household actually
**bought** in a held-back later period.

Everything for this version lives in:

- `notebooks/01_eda.ipynb` — data exploration
- `src/product_reco.py` — task builder, evaluator, and every baseline (incl. item-kNN)
- `src/als_model.py` — weighted implicit ALS
- `scripts/run_baselines.py`, `scripts/run_als_test.py` (+ their `*_validation.py`
  counterparts) — run and score every model, writing the CSVs the report and
  `scripts/run_responsible_use_summary.py` read from
- `report/report.md` — the graded write-up (see § 7 for the full file map)

An earlier version of the project asked a different question — which *coupon
campaigns* or *individual coupons* to send, checked against the retailer's own
targeting and against redemptions. That code has been moved into
`legacy/src/` (`baselines.py`, `fm_model.py`, `study2.py`) and `legacy/scripts/`
(their runners) — kept for reference, still runnable (imports point at the
frozen `src/config.py`, `src/data_loader.py` and `src/eval_harness.py`), but
**not part of the current pipeline**.

---

## 1 · Where we are

The modelling work and scientific-report draft are complete. The remaining
work is final reproducibility and submission quality assurance.

| # | Phase | Status | Definition of Done |
|---|-------|--------|--------------------|
| 1 | Data audit & EDA | ✅ Done | 6 audit questions answered, 5 data-quality checks passed, executive summary + limitations captured in the EDA notebook |
| 2 | Baselines + frozen evaluation protocol | ✅ Done | Eight baselines run through the shared product-level evaluator; protocol documented in `docs/EVAL_HARNESS_SOP.md` |
| 3 | Improved method (implicit ALS) | ✅ Done | ALS tuned only on validation weeks 80–84 and evaluated on final test weeks 85–102 |
| 4 | Beyond-accuracy and diagnostic analysis | ✅ Done | Coverage, activity-tier, warm/cold and include-/exclude-seen results reported with bootstrap CIs where applicable |
| 5 | Pitch (4 min) | ✅ Done | Pitch deck prepared and presented |
| 6 | Scientific report | 🚧 Final QA | Full draft, references, contribution statement, AI-use note and learning reflection exist; final PDF and 10-page check remain |

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

## 2 · Completed baseline and evaluation pipeline

Every current model returns `household_key` plus `rank_1` through `rank_5` and
is scored by `src.product_reco.evaluate(...)`. Metric formulas for Recall,
NDCG and bootstrap confidence intervals come from `src.eval_harness.py`.

The baseline ladder is random, popularity, RFM segment-popularity, trending,
Wilson uncertainty-aware popularity, repeat-buy, last-category and item-kNN.
The main comparison uses the include-seen replenishment condition; exclude-seen
is a separate discovery diagnostic.

---

## 3 · Completed improved method — weighted implicit ALS

Weighted implicit ALS uses the household × coupon-eligible-product interaction
matrix. Hyperparameters were selected on validation weeks 80–84 using NDCG@5:
4 factors, alpha 5.0, regularisation 0.1 and 10 iterations. The frozen
configuration was then evaluated once on test weeks 85–102.

Repeat-buy outperformed ALS on the final test. This negative result is retained
as the honest scientific conclusion; LightFM, BPR and EASE are future work, not
requirements for the final submission.

---

## 4 · Completed analysis

- Final relevance comparison: NDCG@5, Recall@5 and Hit Rate@5.
- Beyond-accuracy comparison: catalogue coverage.
- Diagnostics: light/mid/heavy activity tiers, warm/cold households and
  include-seen versus exclude-seen.
- Responsible-use conclusion: predicting purchases is not evidence that a
  coupon causes incremental purchases or revenue; deployment requires an A/B
  test and operational, privacy and fairness controls.

---

## 5 · Report outline (10 pages, single column)

The report follows the grading criteria directly. Final pagination will be
confirmed when the PDF is generated.

| Section | Pages | Notes |
|---------|-------|-------|
| 1. Problem | 0.75 | User, business motivation, task, research question and scope |
| 2. Dataset | 1.50 | Feedback signal, cleaning, sparsity, design implications and reproducibility |
| 3. Baseline | 1.25 | Baseline ladder, results and strongest bar to clear |
| 4. Method | 1.50 | Product representation, weighted implicit ALS, validation and implementation |
| 5. Evaluation | 2.25 | Frozen protocol, metrics, final results and diagnostics |
| 6. Discussion, limitations and responsible use | 2.25 | Interpretation, limitations, deployment and future work |
| 7. Conclusion | 0.50 | Final result and practical recommendation |

Plus (outside the 10 pages): contribution statement, honest AI-tool-use
report, 1-page learning reflection.

---

## 6 · Remaining work before submission

- [x] Evaluation protocol updated for the final product-level Top-5 task.
- [x] Baseline of record confirmed: repeat-buy.
- [x] ALS model selected on validation and scored on final test.
- [x] Author contributions written for all four members.
- [x] Honest AI-tool-use statement added.
- [x] Learning reflection drafted.
- [x] Bibliography added.
- [ ] Re-run the four current-pipeline commands from a clean environment and
      verify every reported number against the regenerated artifacts.
- [ ] Produce the final single-column PDF and verify that the scientific report
      does not exceed 10 pages.
- [ ] Confirm whether the learning reflection should be collective or individual.
- [ ] Complete the final editorial and visual consistency review.

---

## 6b · Legacy exploratory Study 2 — personalised COUPON_UPC recommendation

> Historical context only. This study is retained under `legacy/` and is not
> part of the final report, final results or current evaluation protocol.

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

**Files (moved into `legacy/` — see § 0):**
```
legacy/src/study2.py                    # task builder, coupon baselines, coupon-ALS, evaluator
legacy/scripts/run_coupon_recommender.py
```
Metric formulas (`recall_at_k`, `ndcg_at_k`, `bootstrap_mean_ci`) are imported
from the frozen `src/eval_harness.py` — no metric divergence between studies.
Study 2 adds one metric on top: `expected_revenue_at_k`, the hit-weighted
sum of per-coupon mean line-item SALES_VALUE (see definition below).

**Run:**
```bash
python legacy/scripts/run_coupon_recommender.py    # writes artifacts/coupon_results.csv
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

**Why it is retained.** It records an earlier task formulation and the evidence
that motivated the final product-level framing. No open decision in this legacy
study blocks the final submission.

---

## 7 · File map

```
coupon_RS/
├── PROJECT.md                          # <-- you are here
├── README.md                           # public repo readme
├── .gitignore
├── Data/                               # raw CSVs, gitignored
├── artifacts/                          # generated outputs, gitignored
│   ├── baseline_results.csv            # after scripts/run_baselines.py
│   ├── als_validation_results.csv      # after scripts/run_als_validation.py
│   ├── als_test_results.csv            # after scripts/run_als_test.py
│   ├── responsible_use_summary.md      # after scripts/run_responsible_use_summary.py
│   └── coupon_results*.csv             # legacy Study 2, after legacy/scripts/run_coupon_recommender.py
├── docs/
│   ├── Reco_Systems_Pitch.pdf
│   ├── dunnhumby - The Complete Journey User Guide.pdf
│   └── EVAL_HARNESS_SOP.md             # freeze rules
├── eda/
│   └── 01_EDA_Sparsity_Targeting_Signal_Data_Quality.ipynb
├── notebooks/
│   └── 01_eda.ipynb                    # current pipeline's exploratory notebook (§2.5 of the report)
├── report/
│   ├── report.md                       # the graded scientific report
│   └── figures/figure1_test_ndcg.png
├── requirements.txt                    # numpy, pandas, scipy, scikit-learn, implicit, torch, matplotlib
├── scripts/                            # current pipeline — product-level coupon recommendation
│   ├── run_baselines_validation.py     # baselines on train/val weeks
│   ├── run_baselines.py                # baselines on train/test weeks -> artifacts/baseline_results.csv
│   ├── run_als_validation.py           # ALS hyperparameter sweep on val weeks
│   ├── run_als_test.py                 # frozen ALS config on test weeks -> artifacts/als_test_results.csv
│   └── run_responsible_use_summary.py  # rebuilds §6 evidence from the two CSVs above, no manual numbers
├── src/                                # current pipeline
│   ├── __init__.py
│   ├── config.py                       # FROZEN — paths, splits, K, seeds
│   ├── data_loader.py                  # CSV loaders
│   ├── eval_harness.py                 # FROZEN — recall_at_k / ndcg_at_k / bootstrap_mean_ci
│   ├── product_reco.py                 # task builder, evaluator, all baselines incl. item-kNN
│   └── als_model.py                    # weighted implicit ALS (households x coupon-eligible products)
└── legacy/                             # earlier campaign-/coupon-level formulations, kept for reference
    ├── src/
    │   ├── baselines.py                 # Study 1 non-FM: random/pop/RFM/demographic/last-category/ALS
    │   ├── fm_model.py                  # Study 1 FM + campaign-level causal features
    │   └── study2.py                    # Study 2: coupon-redemption task, baselines, coupon-ALS
    └── scripts/
        ├── run_models.py                # Study 1 baselines + ALS runner
        ├── run_fm.py                    # Study 1 FM runner (needs torch)
        └── run_coupon_recommender.py    # Study 2 runner
```

---

## 8 · Key architectural decisions (locked in)

- **Task = Top-5 product recommendation.** Recommend coupon-eligible products
  that a household is likely to purchase in the later test window.
- **Candidate set = coupon-eligible products purchased during training.**
  The final test task contains 39,132 candidates.
- **Ground truth = later purchases.** Each household is evaluated against the
  candidate products it actually purchased in the evaluation window.
- **Temporal split, never random split** — train weeks 1–79, validation 80–84
  and final test 85–102.
- **`household_key` is lowercase; every other ID is UPPERCASE.** Reflects the
  raw CSV schema; do not rename.
- **Do not modify the frozen protocol** in `src/config.py`, the metric helpers
  in `src/eval_harness.py`, or the task/evaluation definitions in
  `src/product_reco.py` without the process in `docs/EVAL_HARNESS_SOP.md`.
- **NDCG@5 is the headline metric.** Recall@5, Hit Rate@5 and coverage provide
  complementary evidence; Recall and NDCG use household-level bootstrap CIs.
- **Include-seen is the main replenishment condition.** Exclude-seen is a
  separate discovery diagnostic.
- **`hh_demographic` is NOT used as a model feature** in the main study
  (non-random 32 % coverage). Kept for post-hoc subgroup analysis only.

---

## 9 · Handy commands

```bash
# From repo root — current pipeline (product-level coupon recommendation, the report's study)
python scripts/run_baselines.py                  # 8 baselines incl. item-kNN -> artifacts/baseline_results.csv
python scripts/run_als_validation.py             # ALS hyperparameter sweep on validation weeks
python scripts/run_als_test.py                   # frozen ALS config on test weeks -> artifacts/als_test_results.csv
python scripts/run_responsible_use_summary.py    # regenerates artifacts/responsible_use_summary.md

# From repo root — legacy (campaign- and coupon-level formulations, not part of the report)
python legacy/scripts/run_models.py              # Study 1 baselines + ALS + segment_demographic
python legacy/scripts/run_fm.py                  # PyTorch FM (needs local `pip install torch`)
python legacy/scripts/run_coupon_recommender.py  # Study 2 baselines + ALS on coupons

# Sanity
python -c "from src.data_loader import load_core; d = load_core(); print({k: len(v) for k, v in d.items()})"
```

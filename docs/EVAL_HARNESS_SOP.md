# SOP — Evaluation Protocol Freeze

**Owner:** Coupon RS team (Ramiro, Ana, Mayra and Fatima)

**Frozen on:** 8 September 2026

**Locations:** `src/config.py`, metric helpers in `src/eval_harness.py`, and the
product-task builder and evaluator in `src/product_reco.py`

This document describes the protocol used for the final product-level study.
Every baseline and ALS variant must be evaluated under this same protocol so
that the comparison remains controlled and reproducible.

## What is frozen

1. **Recommendation task** — for each household, rank the Top-5
   coupon-eligible products it is most likely to purchase in the test window.
2. **Temporal split** — weeks 1–79 for training, 80–84 for validation and
   85–102 for the final test, as defined in `src/config.py`.
3. **Candidate set** — products that appear in `coupon.csv` and were purchased
   by at least one household during the relevant training window. The final
   test candidate set contains 39,132 products.
4. **Ground truth** — for each household, the distinct candidate products it
   actually purchased during the evaluation window.
5. **Household filtering** — evaluation includes households with at least one
   ground-truth product in the evaluation window. A recommender may omit a row
   only when it cannot produce a valid recommendation; omitted rows are
   reported as dropped.
6. **Recommendation-list length** — `K = 5`.
7. **Primary metrics** — mean per-household NDCG@5 and Recall@5. NDCG@5 is the
   headline ranking metric. Percentile bootstrap 95% confidence intervals use
   1,000 household-level resamples and seed 42.
8. **Additional metrics** — Hit Rate@5, catalogue coverage, Recall@5 by
   household activity tier (light, mid and heavy), and warm/cold Recall@5.
9. **Seen-item policy** — include-seen is the primary replenishment condition.
   Exclude-seen is reported separately as a discovery diagnostic and must not
   be mixed into the main comparison.
10. **Model output contract** — every model returns one row per household with
    `household_key` and ranked product identifiers in `rank_1` through
    `rank_5`. Every current model is scored by `src.product_reco.evaluate(...)`,
    which imports the frozen Recall, NDCG and bootstrap formulas from
    `src.eval_harness`.
11. **Model selection** — ALS hyperparameters are selected only on validation
    weeks 80–84. The frozen configuration is then evaluated once on test weeks
    85–102; test results must not be used for tuning.

## What is allowed to change

- Modelling code and additional candidate methods.
- Analysis, plotting and reporting scripts.
- Documentation and report prose.
- Data-loading schema or typing improvements that do not change the semantic
  content of the returned data.

Any newly added model must use the frozen task definition, split, candidate
set, metrics and evaluator above.

## Escalation path if a real bug is found

1. Report the issue to the team with a minimal reproducible example.
2. Record the team decision: fix, defer or accept.
3. If fixed, re-run and re-score every previously reported model.
4. Record the change and affected models in the change log below.
5. Update every table, figure and claim that depends on the changed result.

## Change log

| Date | Change | Reason | Re-scored models |
|---|---|---|---|
| 8 Sep 2026 | Document aligned with the final product-level Top-5 protocol | Earlier text still described the legacy campaign-level K=3 study | All final results are produced by the product-level runners |

## Why the protocol is frozen

Freezing the protocol prevents post-hoc metric selection, inconsistent
comparisons and accidental test leakage. All algorithms are compared under the
same task, split, candidate set, filtering rules and metrics. This is especially
important because evaluation rigor represents 25% of the project grade.

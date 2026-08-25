# SOP — Evaluation Harness Freeze

**Owner:** _<team member — fill in at freeze>_
**Frozen on:** _<date — fill in at freeze>_
**Location:** `src/eval_harness.py`, `src/config.py`

The evaluation harness is the single source of truth for how every model in
this project is scored. Once frozen — at the start of modelling work, before
any candidate model is compared — no one modifies it without a team decision
recorded in the change log below.

## What is frozen

1. **Temporal split** — `TRAIN_WEEKS`, `VAL_WEEKS`, `TEST_WEEKS` in
   `src/config.py`. Weeks 1–91 train, 92–96 validation, 97–102 test.
2. **Candidate set** — campaigns whose `[START_DAY, END_DAY]` overlaps the
   test window's day range (computed from `transactions.WEEK_NO` at eval time).
3. **Ground truth** — `campaign_table.csv` filtered to active-in-test-window
   campaigns; a hit is `top_k ∩ assigned ≠ ∅`.
4. **Filtering** — households with zero assigned campaigns in the test window
   are excluded from evaluation (with a `dropped` count reported).
5. **Metrics** — Recall@K, NDCG@K, coverage, redemption-value uplift
   (matched vs unmatched households), with 95 % bootstrap CIs on Recall and
   NDCG.
6. **K** — 3.
7. **The `evaluate(...)` function signature and return schema.**

## What is allowed to change

- Modelling code — new baselines, ALS variants, LightFM, hybrids.
- Analysis notebooks that consume `evaluate(...)` output.
- Plotting and reporting scripts.
- The `docs/` folder.
- `src/data_loader.py` — schema/typing improvements are fine, but the returned
  data must remain semantically identical.

## Escalation path if a real bug is found

1. Open an issue in the team channel with a minimum reproducing example.
2. Team decides in ≤ 30 minutes: fix, defer, or accept.
3. If we fix: **re-score every prior model** with the new harness. Publish a
   before/after table in this file's Change log.
4. Log the change below.

## Change log

| Date | Change | Reason | Re-scored models |
|------|--------|--------|------------------|
|      |        |        |                  |

## Why we freeze — the one-paragraph rationale

Freezing the harness kills three failure modes at once: (a) post-hoc metric
shopping ("what if I change K to 5?" — you'll subconsciously find the tweak
that makes your model look better); (b) apples-to-oranges comparisons across
models scored under drifting protocols; and (c) irreproducibility in the
10-page report, which must describe exactly one protocol. Day 1 was explicit:
_"keep the protocol constant when comparing algorithms — a controlled
experiment."_ Evaluation rigour is the largest single grading criterion (25 %).

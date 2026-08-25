"""
Baseline runner — end-to-end evaluation of every non-model baseline.

Loads data, builds RFM on the training weeks, runs four baselines through
the frozen eval harness, prints per-baseline metrics, and writes a summary
CSV to artifacts/baseline_results.csv.

Baselines included:
    - random               : uniform sanity floor
    - popularity           : most-assigned campaigns (same top-K for everyone)
    - segment_popularity   : per-RFM-segment most-assigned campaigns  <-- baseline of record
    - last_category        : rank campaigns by household spend in covered commodities

Run from repo root:
    python scripts/run_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ARTIFACTS_DIR, K, TRAIN_WEEKS, TEST_WEEKS, SEED
from src.data_loader import load_core, day_range_for_weeks
from src.eval_harness import (
    active_campaigns_in_test,
    build_ground_truth,
    evaluate,
    format_results,
)
from src.baselines import (
    compute_rfm,
    random_baseline,
    popularity_baseline,
    segment_popularity_baseline,
    last_category_baseline,
)


def main():
    print("Loading data...")
    d = load_core()

    print("Determining test-window day range and active candidate campaigns...")
    day_start, day_end = day_range_for_weeks(d["transactions"], TEST_WEEKS)
    active = active_campaigns_in_test(d["campaign_desc"], day_start, day_end)
    print(f"  Test window: DAY {day_start}..{day_end}, {len(active)} active campaigns")

    print("Building ground truth for evaluable households...")
    truth = build_ground_truth(d["campaign_table"], active)
    hh_eval = sorted(truth.keys())
    print(f"  {len(hh_eval)} households have >=1 active-campaign assignment in the test window")

    print("Computing RFM on the training weeks...")
    rfm_df = compute_rfm(d["transactions"], TRAIN_WEEKS)
    print(f"  RFM covers {len(rfm_df)} households, {rfm_df['segment'].nunique()} segments")

    candidate_campaigns = active

    print("\n--- Running baselines ---")
    baselines = {
        "random":              random_baseline(hh_eval, list(candidate_campaigns), k=K, seed=SEED),
        "popularity":          popularity_baseline(d["campaign_table"], hh_eval, candidate_campaigns, k=K),
        "segment_popularity":  segment_popularity_baseline(d["campaign_table"], rfm_df, hh_eval, candidate_campaigns, k=K),
        "last_category":       last_category_baseline(d["transactions"], d["product"], d["coupon"], hh_eval, candidate_campaigns, k=K),
    }

    rows = []
    for name, recs in baselines.items():
        res = evaluate(
            recs_df=recs,
            campaign_table=d["campaign_table"],
            campaign_desc=d["campaign_desc"],
            coupon_redempt=d["coupon_redempt"],
            transactions=d["transactions"],
            k=K,
        )
        print()
        print(format_results(name, res))
        rows.append(
            {
                "model":       name,
                "recall_at_k": res["recall_at_k"],
                "recall_lo":   res["recall_at_k_ci"][0],
                "recall_hi":   res["recall_at_k_ci"][1],
                "ndcg_at_k":   res["ndcg_at_k"],
                "ndcg_lo":     res["ndcg_at_k_ci"][0],
                "ndcg_hi":     res["ndcg_at_k_ci"][1],
                "coverage":    res["coverage"],
                "uplift":      res["redemption_uplift"]["uplift_ratio"],
                "n_matched":   res["redemption_uplift"]["n_matched"],
                "n_unmatched": res["redemption_uplift"]["n_unmatched"],
            }
        )

    out = ARTIFACTS_DIR / "baseline_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

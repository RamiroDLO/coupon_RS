"""
Study 2 runner — personalised coupon (COUPON_UPC) recommendation.

Builds the redemption task, fits every Study 2 model, scores each through
the coupon-level evaluator, and dumps a results table with bootstrap CIs
to artifacts/coupon_results.csv.

Contrast with Study 1's run_models.py:
    - Item universe is 354 coupons (vs 12 campaigns) — dense enough for
      latent-factor methods to differentiate themselves.
    - Ground truth is real redemption behaviour (vs retailer assignment),
      so the "money metric" is the primary metric, not a proxy.

Run from repo root:
    python scripts/run_coupon_recommender.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ARTIFACTS_DIR, K, SEED
from src.data_loader import load_core
from src.study2 import (
    build_redemption_task,
    evaluate_coupons,
    format_coupon_results,
    random_coupon_baseline,
    popularity_coupon_baseline,
    repeat_buy_baseline,
    item_knn_baseline,
    last_category_coupon_baseline,
    train_coupon_als,
    als_coupon_recommendations,
)


def _collect_row(name: str, res: dict) -> dict:
    return {
        "model":                    name,
        "recall_at_k":              res["recall_at_k"],
        "recall_lo":                res["recall_at_k_ci"][0],
        "recall_hi":                res["recall_at_k_ci"][1],
        "ndcg_at_k":                res["ndcg_at_k"],
        "ndcg_lo":                  res["ndcg_at_k_ci"][0],
        "ndcg_hi":                  res["ndcg_at_k_ci"][1],
        "expected_revenue_at_k":    res["expected_revenue_at_k"],
        "expected_revenue_lo":      res["expected_revenue_at_k_ci"][0],
        "expected_revenue_hi":      res["expected_revenue_at_k_ci"][1],
        "coupon_coverage":          res["coverage"],
        "camp_coverage":            res["campaign_coverage"],
        "warm_recall":              res["warm_recall_at_k"],
        "cold_recall":              res["cold_recall_at_k"],
        "n_warm":                   res["n_warm"],
        "n_cold":                   res["n_cold"],
        "n_hh":                     res["n_households_evaluated"],
        "n_candidates":             res["n_candidate_coupons"],
        "k":                        res["k"],
    }


def main():
    print("Loading data...")
    d = load_core()

    print("Building Study 2 task (hh x COUPON_UPC, real redemptions)...")
    task = build_redemption_task(
        coupon_redempt=d["coupon_redempt"],
        coupon=d["coupon"],
        campaign_desc=d["campaign_desc"],
        transactions=d["transactions"],
    )
    print("Task built:")
    for k, v in task.meta.items():
        print(f"  {k:26s}: {v}")
    print(f"  train DAY range           : {task.train_day_range}")
    print(f"  test  DAY range           : {task.test_day_range}")
    print(f"  active test-window camps  : {sorted(task.active_campaigns)}")

    if task.meta["n_hh_eval"] == 0:
        print("No evaluable households — aborting.")
        return

    # ---- Fit ALS once ---------------------------------------------------
    print("\nTraining ALS on hh x coupon binary matrix...")
    als_model, als_hh, als_cp, _ = train_coupon_als(task)
    print(f"  ALS shape: {len(als_hh)} hh x {len(als_cp)} coupons")

    # ---- Build recs for every model ------------------------------------
    print("\nBuilding recommendations...")
    models = {
        "random":            random_coupon_baseline(task, k=K, seed=SEED),
        "popularity":        popularity_coupon_baseline(task, k=K),
        "repeat_buy":        repeat_buy_baseline(task, k=K),
        "last_category":     last_category_coupon_baseline(task, d["transactions"], d["product"], d["coupon"], k=K),
        "item_knn":          item_knn_baseline(task, k=K),
        "als_coupon":        als_coupon_recommendations(als_model, als_hh, als_cp, task, k=K),
        "als_coupon_noseen": als_coupon_recommendations(als_model, als_hh, als_cp, task, k=K, exclude_seen=True),
    }

    # ---- Score every model, at K=3 and K=10 ----------------------------
    all_rows = []
    for name, recs in models.items():
        res = evaluate_coupons(recs, task, k=K)
        print()
        print(format_coupon_results(name, res))
        all_rows.append(_collect_row(name, res))

    # K=10 supplement — deeper look for a much larger candidate set
    print("\n--- Supplementary evaluation at K=10 ---")
    K10 = 10
    models_k10 = {
        "random":            random_coupon_baseline(task, k=K10, seed=SEED),
        "popularity":        popularity_coupon_baseline(task, k=K10),
        "repeat_buy":        repeat_buy_baseline(task, k=K10),
        "last_category":     last_category_coupon_baseline(task, d["transactions"], d["product"], d["coupon"], k=K10),
        "item_knn":          item_knn_baseline(task, k=K10),
        "als_coupon":        als_coupon_recommendations(als_model, als_hh, als_cp, task, k=K10),
        "als_coupon_noseen": als_coupon_recommendations(als_model, als_hh, als_cp, task, k=K10, exclude_seen=True),
    }
    for name, recs in models_k10.items():
        res = evaluate_coupons(recs, task, k=K10)
        print(
            f"[K=10] {name:20s}  Recall@10 {res['recall_at_k']:.3f}  "
            f"NDCG@10 {res['ndcg_at_k']:.3f}  E[rev] ${res['expected_revenue_at_k']:.2f}  "
            f"cov {res['coverage']:.3f}"
        )
        all_rows.append(_collect_row(f"{name}@K10", res))

    out = ARTIFACTS_DIR / "coupon_results.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

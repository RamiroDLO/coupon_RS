"""
Final evaluation of the selected implicit ALS configuration.

The ALS hyperparameters were selected using VALIDATION weeks 80-84.

Final evaluation:
    TRAIN: weeks 1-79
    TEST : weeks 85-102

The test set is evaluated only after hyperparameter selection.

Run from repo root:

    python scripts/run_als_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    TRAIN_WEEKS,
    TEST_WEEKS,
    K,
    ARTIFACTS_DIR,
)

from src.data_loader import (  # noqa: E402
    load_purchases,
    load_product,
    load_coupon,
)

from src.product_reco import (  # noqa: E402
    build_purchase_task,
    evaluate,
)

from src.als_model import (  # noqa: E402
    build_als_data,
    fit_als,
    recommend_als,
)


OUTPUT_FILE = ARTIFACTS_DIR / "als_test_results.csv"


# ============================================================
# SELECTED ON VALIDATION — DO NOT TUNE ON TEST
# ============================================================

BEST_FACTORS = 4
BEST_ALPHA = 5.0
BEST_REGULARIZATION = 0.1
BEST_ITERATIONS = 10


def main() -> None:

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("Loading data ...")

    purchases = load_purchases()
    product = load_product()
    coupon = load_coupon()

    # ========================================================
    # 2. BUILD FINAL TEST TASK
    # ========================================================

    print("Building FINAL TEST task ...")

    test_task = build_purchase_task(
        purchases=purchases,
        product=product,
        coupon=coupon,
        train_weeks=TRAIN_WEEKS,
        test_weeks=TEST_WEEKS,
    )

    print("Test task shape:")

    for key, value in test_task.meta.items():
        print(f"  {key:<24s}: {value}")

    # ========================================================
    # 3. BUILD TRAINING MATRIX
    # ========================================================

    print()
    print("Building ALS training matrix ...")

    als_data = build_als_data(
        purchases=purchases,
        coupon=coupon,
        train_weeks=TRAIN_WEEKS,
    )

    print(
        f"ALS matrix: "
        f"{als_data.user_items.shape[0]} households x "
        f"{als_data.user_items.shape[1]} products"
    )

    print(f"Non-zero interactions: {als_data.user_items.nnz}")

    # ========================================================
    # 4. TRAIN SELECTED ALS
    # ========================================================

    print()
    print("Training selected ALS configuration ...")

    print(f"  factors        : {BEST_FACTORS}")
    print(f"  alpha          : {BEST_ALPHA}")
    print(f"  regularization : {BEST_REGULARIZATION}")
    print(f"  iterations     : {BEST_ITERATIONS}")

    artifacts = fit_als(
        data=als_data,
        factors=BEST_FACTORS,
        alpha=BEST_ALPHA,
        regularization=BEST_REGULARIZATION,
        iterations=BEST_ITERATIONS,
        show_progress=False,
    )

    # ========================================================
    # 5. GENERATE TEST RECOMMENDATIONS
    # ========================================================

    print()
    print("Generating TEST recommendations ...")

    recommendations = recommend_als(
        task=test_task,
        artifacts=artifacts,
        k=K,
        exclude_seen=False,
    )

    # Second condition: the same trained model, but forbidden from recommending
    # products the household already bought in training. This is the discovery
    # diagnostic — it puts ALS on the same footing as the exclude-seen baselines.
    recommendations_excl = recommend_als(
        task=test_task,
        artifacts=artifacts,
        k=K,
        exclude_seen=True,
    )

    # ========================================================
    # 6. FINAL TEST EVALUATION
    # ========================================================

    print()
    print("Evaluating on FINAL TEST ...")

    metrics = evaluate(
        recs_df=recommendations,
        task=test_task,
        k=K,
    )

    metrics_excl = evaluate(
        recs_df=recommendations_excl,
        task=test_task,
        k=K,
    )

    # ========================================================
    # 7. PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL ALS TEST RESULTS")
    print("=" * 70)

    print(
        f"Recall@{K}:   {metrics['recall_at_k']:.4f}  "
        f"CI [{metrics['recall_at_k_ci'][0]:.4f}, {metrics['recall_at_k_ci'][1]:.4f}]"
    )
    print(
        f"NDCG@{K}:     {metrics['ndcg_at_k']:.4f}  "
        f"CI [{metrics['ndcg_at_k_ci'][0]:.4f}, {metrics['ndcg_at_k_ci'][1]:.4f}]"
    )
    print(f"HitRate@{K}:  {metrics['hit_rate_at_k']:.4f}")
    print(f"Coverage:    {metrics['coverage']:.4f}")

    print()
    print(
        "Recall by tier: "
        f"light={metrics['recall_light']:.4f}, "
        f"mid={metrics['recall_mid']:.4f}, "
        f"heavy={metrics['recall_heavy']:.4f}"
    )

    print(
        "Warm / cold recall: "
        f"{metrics['warm_recall']:.4f} / "
        f"{metrics['cold_recall']:.4f}"
    )

    print()
    print("-" * 70)
    print("EXCLUDE-SEEN (discovery diagnostic)")
    print("-" * 70)
    print(f"Recall@{K}:   {metrics_excl['recall_at_k']:.4f}")
    print(f"NDCG@{K}:     {metrics_excl['ndcg_at_k']:.4f}")
    print(f"HitRate@{K}:  {metrics_excl['hit_rate_at_k']:.4f}")
    print(f"Coverage:    {metrics_excl['coverage']:.4f}")

    # ========================================================
    # 8. SAVE FINAL RESULT
    # ========================================================

    def _row(m: dict, exclude_seen: bool) -> dict:
        return {
            "model": "ALS",
            "split": "test",
            "k": K,
            "train_weeks": f"{TRAIN_WEEKS[0]}-{TRAIN_WEEKS[-1]}",
            "test_weeks": f"{TEST_WEEKS[0]}-{TEST_WEEKS[-1]}",
            "factors": BEST_FACTORS,
            "alpha": BEST_ALPHA,
            "regularization": BEST_REGULARIZATION,
            "iterations": BEST_ITERATIONS,
            "exclude_seen": exclude_seen,
            "recall_at_5": m["recall_at_k"],
            "recall_lo": m["recall_at_k_ci"][0],
            "recall_hi": m["recall_at_k_ci"][1],
            "ndcg_at_5": m["ndcg_at_k"],
            "ndcg_lo": m["ndcg_at_k_ci"][0],
            "ndcg_hi": m["ndcg_at_k_ci"][1],
            "hitrate_at_5": m["hit_rate_at_k"],
            "coverage": m["coverage"],
            "recall_light": m["recall_light"],
            "recall_mid": m["recall_mid"],
            "recall_heavy": m["recall_heavy"],
            "warm_recall": m["warm_recall"],
            "cold_recall": m["cold_recall"],
            "n_eval": m["n_households_evaluated"],
        }

    results = pd.DataFrame(
        [
            _row(metrics, exclude_seen=False),
            _row(metrics_excl, exclude_seen=True),
        ]
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Final ALS test results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
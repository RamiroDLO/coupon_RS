"""
Evaluate product-level baselines on the VALIDATION split.

Uses:
    TRAIN weeks 1-79
    VALIDATION weeks 80-84

This allows a fair comparison with ALS hyperparameter tuning.

Run from repo root:

    python scripts/run_baselines_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    TRAIN_WEEKS,
    VAL_WEEKS,
    K,
    ARTIFACTS_DIR,
)

from src.data_loader import (  # noqa: E402
    load_purchases,
    load_product,
    load_coupon,
)

from src.product_reco import (  # noqa: E402
    BASELINES,
    build_purchase_task,
    evaluate,
)


OUTPUT_FILE = ARTIFACTS_DIR / "baseline_validation_results.csv"


def main() -> None:

    print("Loading data ...")

    purchases = load_purchases()
    product = load_product()
    coupon = load_coupon()

    print("Building VALIDATION task ...")

    task = build_purchase_task(
        purchases=purchases,
        product=product,
        coupon=coupon,
        train_weeks=TRAIN_WEEKS,
        test_weeks=VAL_WEEKS,
    )

    print("Validation task shape:")

    for key, value in task.meta.items():
        print(f"  {key:<24s}: {value}")

    rows = []

    print()
    print("=" * 70)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 70)

    for name, baseline_function in BASELINES.items():

        recommendations = baseline_function(
            task,
            k=K,
            exclude_seen=False,
        )

        metrics = evaluate(
            recs_df=recommendations,
            task=task,
            k=K,
        )

        print()
        print(f"=== {name} ===")
        print(f"Recall@{K}:   {metrics['recall_at_k']:.4f}")
        print(f"NDCG@{K}:     {metrics['ndcg_at_k']:.4f}")
        print(f"HitRate@{K}:  {metrics['hit_rate_at_k']:.4f}")
        print(f"Coverage:    {metrics['coverage']:.4f}")

        rows.append(
            {
                "model": name,
                "split": "validation",
                "k": K,
                "train_weeks": f"{TRAIN_WEEKS[0]}-{TRAIN_WEEKS[-1]}",
                "validation_weeks": f"{VAL_WEEKS[0]}-{VAL_WEEKS[-1]}",
                "recall_at_5": metrics["recall_at_k"],
                "ndcg_at_5": metrics["ndcg_at_k"],
                "hitrate_at_5": metrics["hit_rate_at_k"],
                "coverage": metrics["coverage"],
                "recall_light": metrics["recall_light"],
                "recall_mid": metrics["recall_mid"],
                "recall_heavy": metrics["recall_heavy"],
                "warm_recall": metrics["warm_recall"],
                "cold_recall": metrics["cold_recall"],
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        by="ndcg_at_5",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print("=" * 70)
    print("BASELINES SORTED BY NDCG@5")
    print("=" * 70)

    print(
        results[
            [
                "model",
                "recall_at_5",
                "ndcg_at_5",
                "hitrate_at_5",
                "coverage",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Validation baseline results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
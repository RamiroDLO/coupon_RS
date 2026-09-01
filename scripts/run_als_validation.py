"""
Run a small hyperparameter search for implicit ALS on the
product-level coupon recommendation task.

All hyperparameters are evaluated on VALIDATION only.
The TEST period remains untouched until the final model is selected.

Run from repo root:

    python scripts/run_als_validation.py
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
    build_purchase_task,
    evaluate,
)

from src.als_model import (  # noqa: E402
    build_als_data,
    fit_als,
    recommend_als,
)


OUTPUT_FILE = ARTIFACTS_DIR / "als_validation_results.csv"


# ============================================================
# SMALL VALIDATION GRID
# ============================================================

ALS_CONFIGS = [
    {
        "factors": 4,
        "alpha": 5.0,
        "regularization": 0.1,
        "iterations": 10,
    },
    {
        "factors": 4,
        "alpha": 10.0,
        "regularization": 0.1,
        "iterations": 10,
    },
    {
        "factors": 8,
        "alpha": 5.0,
        "regularization": 0.1,
        "iterations": 10,
    },
    {
        "factors": 8,
        "alpha": 10.0,
        "regularization": 0.05,
        "iterations": 10,
    },
    {
        "factors": 8,
        "alpha": 10.0,
        "regularization": 0.1,
        "iterations": 10,
    },
    {
        "factors": 8,
        "alpha": 10.0,
        "regularization": 0.2,
        "iterations": 10,
    },
    {
        "factors": 8,
        "alpha": 10.0,
        "regularization": 0.1,
        "iterations": 15,
    },
    {
        "factors": 12,
        "alpha": 10.0,
        "regularization": 0.1,
        "iterations": 10,
    },
]

def main() -> None:

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("Loading data ...")

    purchases = load_purchases()
    product = load_product()
    coupon = load_coupon()

    # ========================================================
    # 2. BUILD VALIDATION TASK
    # ========================================================

    print("Building validation task ...")

    validation_task = build_purchase_task(
        purchases=purchases,
        product=product,
        coupon=coupon,
        train_weeks=TRAIN_WEEKS,
        test_weeks=VAL_WEEKS,
    )

    print("Validation task shape:")

    for key, value in validation_task.meta.items():
        print(f"  {key:<24s}: {value}")

    # ========================================================
    # 3. BUILD ALS MATRIX ONCE
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
    # 4. VALIDATION SEARCH
    # ========================================================

    rows = []

    for config_number, config in enumerate(ALS_CONFIGS, start=1):

        print()
        print("=" * 70)
        print(
            f"Configuration {config_number}/{len(ALS_CONFIGS)}: "
            f"factors={config['factors']}, "
            f"alpha={config['alpha']}, "
            f"reg={config['regularization']}, "
            f"iterations={config['iterations']}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        artifacts = fit_als(
            data=als_data,
            factors=config["factors"],
            alpha=config["alpha"],
            regularization=config["regularization"],
            iterations=config["iterations"],
            show_progress=False,
        )

        # ----------------------------------------------------
        # RECOMMEND
        # ----------------------------------------------------

        recommendations = recommend_als(
            task=validation_task,
            artifacts=artifacts,
            k=K,
            exclude_seen=False,
        )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        metrics = evaluate(
            recs_df=recommendations,
            task=validation_task,
            k=K,
        )

        print(
            f"Recall@{K}:   {metrics['recall_at_k']:.4f}"
        )

        print(
            f"NDCG@{K}:     {metrics['ndcg_at_k']:.4f}"
        )

        print(
            f"HitRate@{K}:  {metrics['hit_rate_at_k']:.4f}"
        )

        print(
            f"Coverage:    {metrics['coverage']:.4f}"
        )

        rows.append(
            {
                "model": "ALS",
                "split": "validation",
                "k": K,
                "train_weeks": (
                    f"{TRAIN_WEEKS[0]}-{TRAIN_WEEKS[-1]}"
                ),
                "validation_weeks": (
                    f"{VAL_WEEKS[0]}-{VAL_WEEKS[-1]}"
                ),
                "factors": config["factors"],
                "alpha": config["alpha"],
                "regularization": config["regularization"],
                "iterations": config["iterations"],
                "exclude_seen": False,
                "recall_at_5": metrics["recall_at_k"],
                "ndcg_at_5": metrics["ndcg_at_k"],
                "hitrate_at_5": metrics["hit_rate_at_k"],
                "coverage": metrics["coverage"],
                "recall_light": metrics["recall_light"],
                "recall_mid": metrics["recall_mid"],
                "recall_heavy": metrics["recall_heavy"],
                "warm_recall": metrics["warm_recall"],
                "cold_recall": metrics["cold_recall"],
                "n_eval": metrics["n_households_evaluated"],
            }
        )

    # ========================================================
    # 5. RESULTS TABLE
    # ========================================================

    results = pd.DataFrame(rows)

    results = results.sort_values(
        by=[
            "ndcg_at_5",
            "recall_at_5",
            "coverage",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ========================================================
    # 6. PRINT BEST CONFIGURATIONS
    # ========================================================

    print()
    print("=" * 70)
    print("ALS VALIDATION RESULTS")
    print("=" * 70)

    display_columns = [
        "factors",
        "alpha",
        "regularization",
        "iterations",
        "recall_at_5",
        "ndcg_at_5",
        "hitrate_at_5",
        "coverage",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # 7. BEST CONFIGURATION
    # ========================================================

    best = results.iloc[0]

    print()
    print("BEST VALIDATION CONFIGURATION")
    print("-----------------------------")

    print(f"factors        : {int(best['factors'])}")
    print(f"alpha          : {best['alpha']}")
    print(f"regularization : {best['regularization']}")
    print(f"iterations     : {int(best['iterations'])}")

    print()
    print(f"Recall@{K}     : {best['recall_at_5']:.4f}")
    print(f"NDCG@{K}       : {best['ndcg_at_5']:.4f}")
    print(f"HitRate@{K}    : {best['hitrate_at_5']:.4f}")
    print(f"Coverage       : {best['coverage']:.4f}")

    # ========================================================
    # 8. SAVE
    # ========================================================

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"ALS validation results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
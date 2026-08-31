"""
Implicit ALS model for personalized coupon-product recommendations.

Training uses only purchase information from the training period.
Validation must be used to select hyperparameters.
The test period must remain untouched until the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from .config import ALS_ALPHA, ALS_DIM, K, SEED, TRAIN_WEEKS
from .data_loader import load_coupon, load_purchases
from .product_reco import PurchaseTask


@dataclass
class ALSData:
    """Sparse matrix and mappings required by ALS."""

    user_items: csr_matrix
    households: list[int]
    products: list[int]
    household_to_index: dict[int, int]
    product_to_index: dict[int, int]


@dataclass
class ALSArtifacts:
    """Trained ALS model together with its matrix and mappings."""

    model: AlternatingLeastSquares
    data: ALSData
    alpha: float
    factors: int
    regularization: float
    iterations: int


def build_als_data(
    purchases: pd.DataFrame | None = None,
    coupon: pd.DataFrame | None = None,
    train_weeks: list[int] | None = None,
) -> ALSData:
    """
    Build a household-product matrix using coupon-eligible TRAIN purchases.

    Rows:
        households

    Columns:
        coupon-eligible products purchased at least once in training

    Values:
        purchase-line frequency per household-product pair
    """

    purchases = load_purchases() if purchases is None else purchases
    coupon = load_coupon() if coupon is None else coupon
    train_weeks = TRAIN_WEEKS if train_weeks is None else train_weeks

    eligible_products = set(
        int(product_id)
        for product_id in coupon["PRODUCT_ID"].unique()
    )

    train = purchases[
        purchases["WEEK_NO"].isin(train_weeks)
        & purchases["PRODUCT_ID"].isin(eligible_products)
    ].copy()

    interaction_counts = (
        train.groupby(
            ["household_key", "PRODUCT_ID"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "purchase_count"})
    )

    households = sorted(
        int(household)
        for household in interaction_counts["household_key"].unique()
    )

    products = sorted(
        int(product)
        for product in interaction_counts["PRODUCT_ID"].unique()
    )

    household_to_index = {
        household: index
        for index, household in enumerate(households)
    }

    product_to_index = {
        product: index
        for index, product in enumerate(products)
    }

    rows = (
        interaction_counts["household_key"]
        .map(household_to_index)
        .to_numpy(dtype=np.int32)
    )

    columns = (
        interaction_counts["PRODUCT_ID"]
        .map(product_to_index)
        .to_numpy(dtype=np.int32)
    )

    values = (
        interaction_counts["purchase_count"]
        .astype(np.float32)
        .to_numpy()
    )

    user_items = csr_matrix(
        (values, (rows, columns)),
        shape=(len(households), len(products)),
        dtype=np.float32,
    )

    return ALSData(
        user_items=user_items,
        households=households,
        products=products,
        household_to_index=household_to_index,
        product_to_index=product_to_index,
    )


def fit_als(
    data: ALSData,
    factors: int = ALS_DIM,
    alpha: float = ALS_ALPHA,
    regularization: float = 0.01,
    iterations: int = 20,
    seed: int = SEED,
    show_progress: bool = True,
) -> ALSArtifacts:
    """
    Train weighted implicit ALS.

    Purchase counts are converted into confidence by multiplying them
    by alpha.

    Hyperparameters must be selected using VALIDATION only.
    """

    confidence = data.user_items * np.float32(alpha)

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )

    model.fit(
        confidence,
        show_progress=show_progress,
    )

    return ALSArtifacts(
        model=model,
        data=data,
        alpha=alpha,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )


def _complete_recommendations(
    recommendations: list[int],
    fallback: list[int],
    k: int,
    excluded: set[int] | None = None,
) -> list[int]:
    """
    Complete a recommendation list with popularity fallback.

    Ensures:
        - exactly K products
        - no duplicates
        - excluded products remain excluded
    """

    excluded = set() if excluded is None else set(excluded)

    result: list[int] = []
    used = set(excluded)

    for product in recommendations + fallback:
        product = int(product)

        if product in used:
            continue

        result.append(product)
        used.add(product)

        if len(result) == k:
            break

    if len(result) < k:
        raise ValueError(
            f"Could not produce {k} unique recommendations."
        )

    return result


def recommend_als(
    task: PurchaseTask,
    artifacts: ALSArtifacts,
    k: int = K,
    exclude_seen: bool = False,
) -> pd.DataFrame:
    """
    Generate Top-K ALS recommendations for every evaluation household.

    Warm households:
        personalized ALS ranking

    Cold-start households:
        global popularity fallback

    When exclude_seen=True, previously purchased products are removed.
    """

    model = artifacts.model
    data = artifacts.data

    recommendations_by_household: dict[int, list[int]] = {}

    for household in task.hh_eval:

        household = int(household)

        excluded_products = (
            set(task.hh_train_products.get(household, []))
            if exclude_seen
            else set()
        )

        # ----------------------------------------------------
        # Cold start
        # ----------------------------------------------------

        if household not in data.household_to_index:

            recommendations_by_household[household] = (
                _complete_recommendations(
                    recommendations=[],
                    fallback=task.popularity,
                    k=k,
                    excluded=excluded_products,
                )
            )

            continue

        # ----------------------------------------------------
        # ALS recommendations
        # ----------------------------------------------------

        user_index = data.household_to_index[household]

        user_row = data.user_items[user_index]

        product_indices, _ = model.recommend(
            userid=user_index,
            user_items=user_row,
            N=k,
            filter_already_liked_items=exclude_seen,
        )

        als_products = [
            data.products[int(product_index)]
            for product_index in product_indices
        ]

        recommendations_by_household[household] = (
            _complete_recommendations(
                recommendations=als_products,
                fallback=task.popularity,
                k=k,
                excluded=excluded_products,
            )
        )

    # --------------------------------------------------------
    # Convert to standard recommender output
    # --------------------------------------------------------

    rows = []

    for household in task.hh_eval:

        household = int(household)

        recommendations = recommendations_by_household[household]

        row = {
            "household_key": household,
        }

        for position, product in enumerate(
            recommendations,
            start=1,
        ):
            row[f"rank_{position}"] = int(product)

        rows.append(row)

    return pd.DataFrame(rows)
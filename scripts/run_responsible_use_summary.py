"""Generate an evidence summary for limitations and responsible use.

This script reads the final model-result artifacts and rebuilds the final
purchase task. It avoids copying numerical results manually into the report.

Run from the repository root:
    python scripts/run_responsible_use_summary.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


# Make imports work when this file is executed from the scripts directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product_reco import build_purchase_task  # noqa: E402


ARTIFACTS_DIR = ROOT / "artifacts"
BASELINE_RESULTS = ARTIFACTS_DIR / "baseline_results.csv"
ALS_TEST_RESULTS = ARTIFACTS_DIR / "als_test_results.csv"
OUTPUT_FILE = ARTIFACTS_DIR / "responsible_use_summary.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact and fail clearly if it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {path}\n"
            "Run the corresponding evaluation script first."
        )

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    source: Path,
) -> None:
    """Check that an upstream artifact still has the expected schema."""
    if not rows:
        raise ValueError(f"No rows found in {source}")

    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"{source.name} is missing expected columns: "
            f"{', '.join(sorted(missing))}"
        )


def as_float(row: dict[str, str], column: str) -> float:
    return float(row[column])


def fmt(value: float) -> str:
    return f"{value:.4f}"


def select_baseline(
    rows: list[dict[str, str]],
    model: str,
    exclude_seen: bool = False,
) -> dict[str, str]:
    """Select one baseline under the requested seen-item policy."""
    expected = str(exclude_seen)

    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["exclude_seen"].strip().lower() == expected.lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for model={model!r}, "
            f"exclude_seen={exclude_seen}; found {len(matches)}."
        )

    return matches[0]


def calculate_repeat_share(task) -> tuple[int, int, float, float]:
    """Measure repeat versus new purchases in the filtered final ground truth."""
    repeat_pairs = 0
    new_pairs = 0

    for household, products in task.ground_truth.items():
        seen = set(task.hh_train_products.get(household, []))

        for product in products:
            if product in seen:
                repeat_pairs += 1
            else:
                new_pairs += 1

    total = repeat_pairs + new_pairs
    repeat_share = repeat_pairs / total if total else 0.0
    new_share = new_pairs / total if total else 0.0

    return repeat_pairs, new_pairs, repeat_share, new_share


def main() -> None:
    print("Reading final evaluation artifacts ...")

    baseline_rows = read_csv(BASELINE_RESULTS)
    als_rows = read_csv(ALS_TEST_RESULTS)

    require_columns(
        baseline_rows,
        {
            "model",
            "exclude_seen",
            "recall",
            "recall_lo",
            "recall_hi",
            "ndcg",
            "ndcg_lo",
            "ndcg_hi",
            "hit_rate",
            "coverage",
            "recall_light",
            "recall_mid",
            "recall_heavy",
            "warm_recall",
            "cold_recall",
            "n_eval",
        },
        BASELINE_RESULTS,
    )

    require_columns(
        als_rows,
        {
            "model",
            "split",
            "k",
            "train_weeks",
            "test_weeks",
            "factors",
            "alpha",
            "regularization",
            "iterations",
            "exclude_seen",
            "recall_at_5",
            "ndcg_at_5",
            "hitrate_at_5",
            "coverage",
            "recall_light",
            "recall_mid",
            "recall_heavy",
            "warm_recall",
            "cold_recall",
            "n_eval",
        },
        ALS_TEST_RESULTS,
    )

    repeat_buy = select_baseline(baseline_rows, "repeat_buy")
    popularity = select_baseline(baseline_rows, "popularity")

    als_test_rows = [
        row for row in als_rows
        if row["model"].upper() == "ALS" and row["split"].lower() == "test"
    ]
    if len(als_test_rows) != 1:
        raise ValueError(
            "Expected exactly one final ALS test row in "
            f"{ALS_TEST_RESULTS.name}; found {len(als_test_rows)}."
        )
    als = als_test_rows[0]

    print("Rebuilding final purchase task ...")
    task = build_purchase_task()
    meta = task.meta

    repeat_pairs, new_pairs, repeat_share, new_share = (
        calculate_repeat_share(task)
    )

    n_cold_eval = sum(
        1
        for household in task.hh_eval
        if household not in task.hh_train_products
    )

    repeat_recall = as_float(repeat_buy, "recall")
    repeat_ndcg = as_float(repeat_buy, "ndcg")
    als_recall = as_float(als, "recall_at_5")
    als_ndcg = as_float(als, "ndcg_at_5")

    recall_gap = (
        (repeat_recall - als_recall) / repeat_recall
        if repeat_recall
        else 0.0
    )
    ndcg_gap = (
        (repeat_ndcg - als_ndcg) / repeat_ndcg
        if repeat_ndcg
        else 0.0
    )

    tier_counts = meta["eval_tier_counts"]

    markdown = f"""# Responsible-use evidence summary

This file is generated automatically by
`scripts/run_responsible_use_summary.py`. Do not edit its numerical results
manually.

## Final evaluation task

- Training weeks: {als["train_weeks"]}
- Test weeks: {als["test_weeks"]}
- Recommendation-list length: K={als["k"]}
- Candidate products: {meta["n_candidate_products"]:,}
- Evaluated households: {meta["n_eval_households"]:,}
- Households with training history: {meta["n_hh_with_train_history"]:,}
- Median distinct training history: {meta["median_train_history"]}
- Median ground-truth size: {meta["median_ground_truth"]}
- Activity tiers: light={tier_counts["light"]}, mid={tier_counts["mid"]},
  heavy={tier_counts["heavy"]}

## Final test results

| Model | Recall@5 | 95% CI | NDCG@5 | 95% CI | Hit Rate@5 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Repeat-buy | {fmt(repeat_recall)} | [{fmt(as_float(repeat_buy, "recall_lo"))}, {fmt(as_float(repeat_buy, "recall_hi"))}] | {fmt(repeat_ndcg)} | [{fmt(as_float(repeat_buy, "ndcg_lo"))}, {fmt(as_float(repeat_buy, "ndcg_hi"))}] | {fmt(as_float(repeat_buy, "hit_rate"))} | {fmt(as_float(repeat_buy, "coverage"))} |
| Popularity | {fmt(as_float(popularity, "recall"))} | [{fmt(as_float(popularity, "recall_lo"))}, {fmt(as_float(popularity, "recall_hi"))}] | {fmt(as_float(popularity, "ndcg"))} | [{fmt(as_float(popularity, "ndcg_lo"))}, {fmt(as_float(popularity, "ndcg_hi"))}] | {fmt(as_float(popularity, "hit_rate"))} | {fmt(as_float(popularity, "coverage"))} |
| ALS | {fmt(als_recall)} | Not saved | {fmt(als_ndcg)} | Not saved | {fmt(as_float(als, "hitrate_at_5"))} | {fmt(as_float(als, "coverage"))} |

ALS configuration: factors={als["factors"]}, alpha={als["alpha"]},
regularization={als["regularization"]}, iterations={als["iterations"]}.

Relative to repeat-buy, ALS is {recall_gap:.1%} lower in Recall@5 and
{ndcg_gap:.1%} lower in NDCG@5.

## Stratified Recall@5

| Model | Light | Mid | Heavy | Warm | Cold |
|---|---:|---:|---:|---:|---:|
| Repeat-buy | {fmt(as_float(repeat_buy, "recall_light"))} | {fmt(as_float(repeat_buy, "recall_mid"))} | {fmt(as_float(repeat_buy, "recall_heavy"))} | {fmt(as_float(repeat_buy, "warm_recall"))} | {fmt(as_float(repeat_buy, "cold_recall"))} |
| Popularity | {fmt(as_float(popularity, "recall_light"))} | {fmt(as_float(popularity, "recall_mid"))} | {fmt(as_float(popularity, "recall_heavy"))} | {fmt(as_float(popularity, "warm_recall"))} | {fmt(as_float(popularity, "cold_recall"))} |
| ALS | {fmt(as_float(als, "recall_light"))} | {fmt(as_float(als, "recall_mid"))} | {fmt(as_float(als, "recall_heavy"))} | {fmt(as_float(als, "warm_recall"))} | {fmt(as_float(als, "cold_recall"))} |

Cold-start results must be interpreted cautiously because the final evaluation
contains only {n_cold_eval:,} evaluated households without training history.

## Repeat versus new ground-truth products

Using the same candidate-set filtering as the final evaluator:

- Repeat household-product pairs: {repeat_pairs:,} ({repeat_share:.1%})
- New household-product pairs: {new_pairs:,} ({new_share:.1%})

“New” means that the household did not buy the product during the training
window. It does not mean that the product itself is new to the catalogue.

## Limitations supported by the pipeline

1. The evaluation uses a single retailer panel and only
   {meta["n_hh_with_train_history"]:,} households with observed training
   history.
2. Purchases are implicit, MNAR feedback. A missing purchase is not evidence
   that a household disliked or rejected a product.
3. The candidate set is closed: only coupon-eligible products purchased at
   least once during training can be recommended.
4. Results come from one temporal test window and may not generalise to other
   periods, retailers or populations.
5. Cold-start evidence is especially weak because the cold group is extremely
   small.
6. ALS confidence intervals are not currently stored in
   `als_test_results.csv`, so uncertainty cannot yet be reported consistently
   across all final models.

## Responsible-use interpretation

The offline task predicts whether a household later purchases a product. It
does not estimate whether receiving a coupon causes an incremental purchase.
A high-ranking product may be something the household would have bought
without a discount. Therefore, these results must not be presented as causal
lift, incremental revenue or proof that the recommender outperforms the
retailer's campaign targeting.

Any deployment should include margin, stock, contact-frequency and privacy
controls, subgroup monitoring, and a randomised A/B test measuring incremental
outcomes rather than purchases alone.

## Future-work links

- Test BPR or EASE to model personalised ranking and repeat-purchase structure.
- Test a LightFM hybrid with household, product and contextual features.
- Use rolling temporal evaluation rather than a single validation/test split.
- Add diversity, novelty and benefit-distribution measures.
- Evaluate incremental margin net of coupon cost through a randomised A/B test.
"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(markdown, encoding="utf-8")

    print(f"Responsible-use summary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

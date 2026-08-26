"""
Feature-aware Factorization Machine runner.

Adaptation of the course's Day 3 Block 4 tutorial to our Dunnhumby coupon
recommender. Uses the same PyTorch FM structure taught there (FeatureEncoder
+ bi_interaction + FMModel), with fields adapted to our schema:
    Household : household_id, age_group, income, kids
    Campaign  : campaign_id, campaign_type, department_avg,
                display_bucket, mailer_bucket

Scores every model through the frozen eval harness, appends to a new CSV
(artifacts/fm_results.csv). Does NOT touch scripts/run_models.py or its output.

External deps: torch (pip install torch), pandas, numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from src.fm_model import compute_campaign_causal_features
from src.fm_model import (
    build_field_index_maps,
    build_training_examples,
    train_fm,
    fm_recommendations,
)


def _popularity_ranking(campaign_table, candidate_campaigns):
    counts = (
        campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)]
        .groupby("CAMPAIGN").size().sort_values(ascending=False)
    )
    return counts.index.tolist()


def main():
    print("Loading data...")
    d = load_core()

    day_start, day_end = day_range_for_weeks(d["transactions"], TEST_WEEKS)
    active = active_campaigns_in_test(d["campaign_desc"], day_start, day_end)
    print(f"Test window: DAY {day_start}..{day_end}, {len(active)} active campaigns")

    truth = build_ground_truth(d["campaign_table"], active)
    hh_eval = sorted(truth.keys())
    print(f"{len(hh_eval)} evaluable households")

    print("Computing causal (display + mailer) features per campaign...")
    causal_feats = compute_campaign_causal_features(d["coupon"], d["campaign_desc"])
    print(causal_feats.describe().round(3))

    all_households = sorted(d["campaign_table"]["household_key"].unique().tolist())
    all_campaigns = sorted(d["campaign_desc"]["CAMPAIGN"].unique().tolist())

    print("Building field index maps...")
    bundle = build_field_index_maps(
        campaign_desc=d["campaign_desc"],
        hh_demographic=d["hh_demographic"],
        coupon=d["coupon"],
        product=d["product"],
        causal_feats=causal_feats,
        all_households=all_households,
        all_campaigns=all_campaigns,
    )
    print(f"  Fields: n_hh={bundle['n_households']}  n_camp={bundle['n_campaigns']}  "
          f"n_depts={bundle['n_departments']}  n_age={bundle['n_age']}  "
          f"n_income={bundle['n_income']}  n_kids={bundle['n_kids']}")

    # Train-window campaigns: those that started before train_day_max
    train_day_max = d["transactions"][
        d["transactions"]["WEEK_NO"].isin(TRAIN_WEEKS)
    ]["DAY"].max()
    train_camps = set(
        d["campaign_desc"][d["campaign_desc"]["START_DAY"] <= int(train_day_max)]["CAMPAIGN"]
    )
    print(f"  Train-window campaigns: {sorted(train_camps)}")

    print("Building positive + negative training pairs...")
    train_users, train_items, train_labels = build_training_examples(
        campaign_table=d["campaign_table"],
        train_campaigns=train_camps,
        hh_to_row=bundle["hh_to_row"],
        camp_to_row=bundle["camp_to_row"],
        negatives_per_positive=2,
        seed=SEED,
    )
    print(f"  {len(train_labels)} training examples, positive fraction "
          f"{train_labels.mean():.3f}")

    print("Training FM...")
    encoder_kwargs = dict(
        n_households=bundle["n_households"],
        n_campaigns=bundle["n_campaigns"],
        n_age_groups=bundle["n_age"],
        n_incomes=bundle["n_income"],
        n_kids=bundle["n_kids"],
        n_campaign_types=bundle["n_campaign_types"],
        n_departments=bundle["n_departments"],
        n_display_buckets=bundle["n_display"],
        n_mailer_buckets=bundle["n_mailer"],
    )
    model, history = train_fm(
        encoder_kwargs=encoder_kwargs,
        train_users=train_users,
        train_items=train_items,
        train_labels=train_labels,
        field_bundle=bundle,
        embedding_dim=12,
        epochs=4,
        batch_size=1024,
    )

    print("Generating recommendations from FM...")
    fallback = _popularity_ranking(d["campaign_table"], active)
    fm_recs = fm_recommendations(
        model, bundle, hh_eval, sorted(active), fallback, k=K,
    )

    print("Scoring through the frozen eval harness...")
    res = evaluate(
        recs_df=fm_recs,
        campaign_table=d["campaign_table"],
        campaign_desc=d["campaign_desc"],
        coupon_redempt=d["coupon_redempt"],
        transactions=d["transactions"],
        k=K,
    )
    print()
    print(format_results("fm_with_causal", res))

    out = ARTIFACTS_DIR / "fm_results.csv"
    row = {
        "model":       "fm_with_causal",
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
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

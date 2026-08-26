"""
End-to-end runner — every model in the project, scored through the frozen harness.

Baselines:
    random, popularity, segment_popularity, segment_demographic, last_category
Improved:
    als_commodity_max, als_campaign

Reports two evaluations:
    1. FULL — all 12 active campaigns in test window (mixed TypeA/B/C)
    2. TYPE_A — restricted to the 1 TypeA campaign (real targeting)

Saves a summary CSV to artifacts/model_results.csv.

Run from repo root:
    python scripts/run_models.py
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
from src.baselines import (
    compute_rfm,
    random_baseline,
    popularity_baseline,
    segment_popularity_baseline,
    segment_demographic_baseline,
    last_category_baseline,
    build_hh_commodity_matrix,
    build_hh_campaign_matrix,
    train_als_model,
    als_commodity_recommendations,
    als_campaign_recommendations,
)


def _popularity_ranking(campaign_table, candidate_campaigns):
    counts = (
        campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)]
        .groupby("CAMPAIGN")
        .size()
        .sort_values(ascending=False)
    )
    return counts.index.tolist()


def _evaluate_and_collect(models, d, active, k, tag):
    """Score every model and return a list of result dicts."""
    rows = []
    for name, recs in models.items():
        res = evaluate(
            recs_df=recs,
            campaign_table=d["campaign_table"],
            campaign_desc=d["campaign_desc"][d["campaign_desc"]["CAMPAIGN"].isin(active)],
            coupon_redempt=d["coupon_redempt"],
            transactions=d["transactions"],
            k=k,
        )
        print()
        print(format_results(f"[{tag}] {name}", res))
        rows.append(
            {
                "eval_scope":  tag,
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
                "n_hh":        res["n_households_evaluated"],
                "n_active":    res["n_active_campaigns"],
            }
        )
    return rows


def main():
    print("Loading data...")
    d = load_core()

    day_start, day_end = day_range_for_weeks(d["transactions"], TEST_WEEKS)
    active_all = active_campaigns_in_test(d["campaign_desc"], day_start, day_end)
    print(f"Test window: DAY {day_start}..{day_end}, {len(active_all)} active campaigns")

    # ---- Campaign type breakdown ----
    active_desc = d["campaign_desc"][d["campaign_desc"]["CAMPAIGN"].isin(active_all)]
    type_a = set(active_desc[active_desc["DESCRIPTION"] == "TypeA"]["CAMPAIGN"])
    type_bc = set(active_desc[active_desc["DESCRIPTION"].isin(["TypeB", "TypeC"])]["CAMPAIGN"])
    print(f"  TypeA (targeting): {sorted(type_a)}")
    print(f"  TypeB+C (blanket): {sorted(type_bc)}")

    truth = build_ground_truth(d["campaign_table"], active_all)
    hh_eval = sorted(truth.keys())
    print(f"{len(hh_eval)} households have >=1 active-campaign assignment")

    hh_demo = d["hh_demographic"]
    covered = len(set(hh_eval) & set(hh_demo["household_key"]))
    print(f"  Demographic coverage of eval subset: {covered}/{len(hh_eval)} ({100*covered/len(hh_eval):.1f}%)")

    rfm_df = compute_rfm(d["transactions"], TRAIN_WEEKS)
    print(f"RFM covers {len(rfm_df)} households, {rfm_df['segment'].nunique()} segments")

    fallback = _popularity_ranking(d["campaign_table"], active_all)

    print("Building hh x commodity matrix + training ALS (commodity)...")
    mat_c, hh_c, com_c = build_hh_commodity_matrix(d["transactions"], d["product"], TRAIN_WEEKS)
    print(f"  {mat_c.shape[0]} hh x {mat_c.shape[1]} commodities, nnz={mat_c.nnz}")
    als_c = train_als_model(mat_c)

    print("Building hh x campaign matrix + training ALS (campaign)...")
    train_day_max = d["transactions"][d["transactions"]["WEEK_NO"].isin(TRAIN_WEEKS)]["DAY"].max()
    mat_p, hh_p, camp_p = build_hh_campaign_matrix(d["campaign_table"], d["campaign_desc"], int(train_day_max))
    print(f"  {mat_p.shape[0]} hh x {mat_p.shape[1]} campaigns, nnz={mat_p.nnz}")
    als_p = train_als_model(mat_p)

    print("\n--- Building recs (full candidate set) ---")
    models_full = {
        "random":               random_baseline(hh_eval, list(active_all), k=K, seed=SEED),
        "popularity":           popularity_baseline(d["campaign_table"], hh_eval, active_all, k=K),
        "segment_popularity":   segment_popularity_baseline(d["campaign_table"], rfm_df, hh_eval, active_all, k=K),
        "segment_demographic":  segment_demographic_baseline(d["campaign_table"], rfm_df, d["hh_demographic"], hh_eval, active_all, k=K),
        "last_category":        last_category_baseline(d["transactions"], d["product"], d["coupon"], hh_eval, active_all, k=K),
        "als_commodity_max":    als_commodity_recommendations(als_c, hh_c, com_c, d["coupon"], d["product"], active_all, hh_eval, fallback, k=K, aggregation="max"),
        "als_campaign":         als_campaign_recommendations(als_p, hh_p, camp_p, active_all, hh_eval, fallback, k=K),
    }

    all_rows = []
    all_rows += _evaluate_and_collect(models_full, d, active_all, k=K, tag="FULL")

    # ---- TypeA-only evaluation ----
    if len(type_a) >= 1:
        print(f"\n--- Building recs (TypeA-only candidate set, {len(type_a)} campaign(s)) ---")
        truth_a = build_ground_truth(d["campaign_table"], type_a)
        hh_eval_a = sorted(truth_a.keys())
        print(f"  {len(hh_eval_a)} households have >=1 TypeA assignment in test window")

        # For TypeA-only, K=1 makes more sense since there's only 1 candidate
        k_a = min(K, len(type_a))
        fallback_a = _popularity_ranking(d["campaign_table"], type_a)
        models_a = {
            "random":               random_baseline(hh_eval_a, list(type_a), k=k_a, seed=SEED),
            "popularity":           popularity_baseline(d["campaign_table"], hh_eval_a, type_a, k=k_a),
            "segment_popularity":   segment_popularity_baseline(d["campaign_table"], rfm_df, hh_eval_a, type_a, k=k_a),
            "segment_demographic":  segment_demographic_baseline(d["campaign_table"], rfm_df, d["hh_demographic"], hh_eval_a, type_a, k=k_a),
            "last_category":        last_category_baseline(d["transactions"], d["product"], d["coupon"], hh_eval_a, type_a, k=k_a),
            "als_commodity_max":    als_commodity_recommendations(als_c, hh_c, com_c, d["coupon"], d["product"], type_a, hh_eval_a, fallback_a, k=k_a, aggregation="max"),
        }
        all_rows += _evaluate_and_collect(models_a, d, type_a, k=k_a, tag="TYPE_A")

    out = ARTIFACTS_DIR / "model_results.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

"""
Run every product-framing baseline, both include-seen and exclude-seen, score
each through the shared evaluator, and write artifacts/baseline_results.csv.

    python scripts/run_baselines.py        # from the repo root
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.config import K, ARTIFACTS_DIR  # noqa: E402
from src.product_reco import (  # noqa: E402
    BASELINES,
    build_purchase_task,
    evaluate,
    format_result,
)


def main() -> None:
    print("Building task ...")
    task = build_purchase_task()
    print("Task shape:")
    for key, val in task.meta.items():
        print(f"  {key:<24s}: {val}")
    print()

    rows = []
    for name, fn in BASELINES.items():
        for exclude_seen in (False, True):
            recs = fn(task, k=K, exclude_seen=exclude_seen)
            res = evaluate(recs, task, k=K)
            print(format_result(name, exclude_seen, res))
            print()
            rows.append(
                {
                    "model": name,
                    "exclude_seen": exclude_seen,
                    "recall": res["recall_at_k"],
                    "recall_lo": res["recall_at_k_ci"][0],
                    "recall_hi": res["recall_at_k_ci"][1],
                    "ndcg": res["ndcg_at_k"],
                    "ndcg_lo": res["ndcg_at_k_ci"][0],
                    "ndcg_hi": res["ndcg_at_k_ci"][1],
                    "hit_rate": res["hit_rate_at_k"],
                    "coverage": res["coverage"],
                    "recall_light": res["recall_light"],
                    "recall_mid": res["recall_mid"],
                    "recall_heavy": res["recall_heavy"],
                    "warm_recall": res["warm_recall"],
                    "cold_recall": res["cold_recall"],
                    "n_eval": res["n_households_evaluated"],
                }
            )

    out_path = ARTIFACTS_DIR / "baseline_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

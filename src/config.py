"""
Single source of truth for paths, temporal split, seeds and top-K.

Import from here in every notebook and script — never hard-code these values elsewhere.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal split — FROZEN, do not modify without team decision.
# Weeks are inclusive on both ends; matches transactions.WEEK_NO values.
# ---------------------------------------------------------------------------
TRAIN_WEEKS = list(range(1, 92))    # weeks  1..91  (91 weeks)
VAL_WEEKS   = list(range(92, 97))   # weeks 92..96  ( 5 weeks)
TEST_WEEKS  = list(range(97, 103))  # weeks 97..102 ( 6 weeks)

# Sanity: WEEK_NO values in the panel cover 1..102 (102 weeks total).

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Recommender knobs
# ---------------------------------------------------------------------------
K = 3                    # top-K recommendation size
RFM_QUANTILES = 5        # 5x5x5 = 125 possible segments
ALS_DIM = 64             # latent dim for weighted implicit ALS (Wed)
ALS_ALPHA = 40.0         # Hu-Koren-Volinsky confidence scaling (Wed)
ALS_REG = 0.01           # L2 regularisation (Wed)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 1000       # resamples for CI computation
BOOTSTRAP_ALPHA = 0.05   # 95% CIs

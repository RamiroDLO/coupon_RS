"""
Factorization Machine adapted from the course's Day 3 Block 4 tutorial.

Follows the same PyTorch pattern the course teaches:
    FeatureEncoder builds one embedding per "field", plus per-field linear
    terms and a global bias.
    bi_interaction() computes the pairwise-interaction sum via the
    O(n_fields) formula (sum-of-squares vs square-of-sum).
    FMModel returns linear + bi_interaction.sum(-1).

Fields in our adaptation (Dunnhumby coupon recommendation):
    Household side : household_id, age_group, income, kids
    Campaign side  : campaign_id, campaign_type, department_avg,
                     display_bucket, mailer_bucket

`department_avg` mirrors the course's "genre" field — a multi-hot vector
that is combined into a single field via a weighted embedding sum
(weights = normalised multi-hot), so a campaign covering many departments
still contributes just one field to the FM.

External dependency: torch (pip install torch).

Also contains: campaign-level causal features (display + mailer rates)
derived from causal_data.csv — see compute_campaign_causal_features().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED, K, TRAIN_WEEKS


# ---------------------------------------------------------------------------
# Feature encoders — mirrors course cell 13, adapted to our 9 fields
# ---------------------------------------------------------------------------
class FeatureEncoder(nn.Module):
    def __init__(
        self,
        n_households: int,
        n_campaigns: int,
        n_age_groups: int,
        n_incomes: int,
        n_kids: int,
        n_campaign_types: int,
        n_departments: int,
        n_display_buckets: int,
        n_mailer_buckets: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.hh_embedding = nn.Embedding(n_households, embedding_dim)
        self.camp_embedding = nn.Embedding(n_campaigns, embedding_dim)
        self.age_embedding = nn.Embedding(n_age_groups, embedding_dim)
        self.income_embedding = nn.Embedding(n_incomes, embedding_dim)
        self.kids_embedding = nn.Embedding(n_kids, embedding_dim)
        self.ctype_embedding = nn.Embedding(n_campaign_types, embedding_dim)
        self.dept_embedding = nn.Embedding(n_departments, embedding_dim)
        self.display_embedding = nn.Embedding(n_display_buckets, embedding_dim)
        self.mailer_embedding = nn.Embedding(n_mailer_buckets, embedding_dim)

        # Per-field linear terms (bias contribution)
        self.hh_linear = nn.Embedding(n_households, 1)
        self.camp_linear = nn.Embedding(n_campaigns, 1)
        self.age_linear = nn.Embedding(n_age_groups, 1)
        self.income_linear = nn.Embedding(n_incomes, 1)
        self.kids_linear = nn.Embedding(n_kids, 1)
        self.ctype_linear = nn.Embedding(n_campaign_types, 1)
        self.dept_linear = nn.Embedding(n_departments, 1)
        self.display_linear = nn.Embedding(n_display_buckets, 1)
        self.mailer_linear = nn.Embedding(n_mailer_buckets, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in [
            self.hh_embedding, self.camp_embedding,
            self.age_embedding, self.income_embedding, self.kids_embedding,
            self.ctype_embedding, self.dept_embedding,
            self.display_embedding, self.mailer_embedding,
        ]:
            nn.init.normal_(emb.weight, std=0.05)
        for lin in [
            self.hh_linear, self.camp_linear,
            self.age_linear, self.income_linear, self.kids_linear,
            self.ctype_linear, self.dept_linear,
            self.display_linear, self.mailer_linear,
        ]:
            nn.init.zeros_(lin.weight)

    def forward(
        self,
        hh_idx, camp_idx,
        age_idx, income_idx, kids_idx,
        ctype_idx, dept_weights,
        display_idx, mailer_idx,
    ):
        # department multi-hot -> weighted average embedding (course pattern)
        dept_vector = dept_weights @ self.dept_embedding.weight
        dept_linear = dept_weights @ self.dept_linear.weight

        field_embeddings = torch.stack([
            self.hh_embedding(hh_idx),
            self.camp_embedding(camp_idx),
            self.age_embedding(age_idx),
            self.income_embedding(income_idx),
            self.kids_embedding(kids_idx),
            self.ctype_embedding(ctype_idx),
            dept_vector,
            self.display_embedding(display_idx),
            self.mailer_embedding(mailer_idx),
        ], dim=1)

        linear = (
            self.hh_linear(hh_idx).squeeze(-1)
            + self.camp_linear(camp_idx).squeeze(-1)
            + self.age_linear(age_idx).squeeze(-1)
            + self.income_linear(income_idx).squeeze(-1)
            + self.kids_linear(kids_idx).squeeze(-1)
            + self.ctype_linear(ctype_idx).squeeze(-1)
            + dept_linear.squeeze(-1)
            + self.display_linear(display_idx).squeeze(-1)
            + self.mailer_linear(mailer_idx).squeeze(-1)
            + self.global_bias
        )
        return field_embeddings, linear


# ---------------------------------------------------------------------------
# FM interaction — course cell 15, verbatim
# ---------------------------------------------------------------------------
def bi_interaction(field_embeddings: torch.Tensor) -> torch.Tensor:
    summed = field_embeddings.sum(dim=1)
    summed_square = summed ** 2
    square_summed = (field_embeddings ** 2).sum(dim=1)
    return 0.5 * (summed_square - square_summed)


class FMModel(nn.Module):
    def __init__(self, feature_encoder: FeatureEncoder):
        super().__init__()
        self.features = feature_encoder

    def forward(self, *inputs):
        field_embeddings, linear = self.features(*inputs)
        interaction_vector = bi_interaction(field_embeddings)
        fm_score = interaction_vector.sum(dim=1)
        return linear + fm_score


# ---------------------------------------------------------------------------
# Feature preparation from raw dataframes
# ---------------------------------------------------------------------------
def _bucketize(values: pd.Series, quantiles: int = 5) -> tuple[np.ndarray, int]:
    """Quantile-bucket a numeric column into 1..q integer codes."""
    codes = pd.qcut(values, q=quantiles, labels=False, duplicates="drop")
    codes = codes.fillna(0).astype(int).values
    return codes, int(codes.max()) + 1


def build_field_index_maps(
    campaign_desc: pd.DataFrame,
    hh_demographic: pd.DataFrame,
    coupon: pd.DataFrame,
    product: pd.DataFrame,
    causal_feats: pd.DataFrame | None,
    all_households: list[int],
    all_campaigns: list[int],
) -> dict:
    """
    Build a bundle of arrays + lookup dicts to feed the FM.
    Every household in `all_households` and every campaign in `all_campaigns`
    gets a row in the corresponding feature table.
    """
    # Household -> demographic bucket ints
    demo = hh_demographic.set_index("household_key")[
        ["AGE_DESC", "INCOME_DESC", "KID_CATEGORY_DESC"]
    ].reindex(all_households).fillna("MISSING")
    age_map = {v: i for i, v in enumerate(sorted(demo["AGE_DESC"].unique()))}
    income_map = {v: i for i, v in enumerate(sorted(demo["INCOME_DESC"].unique()))}
    kids_map = {v: i for i, v in enumerate(sorted(demo["KID_CATEGORY_DESC"].unique()))}
    hh_age = np.array([age_map[v] for v in demo["AGE_DESC"]], dtype=np.int64)
    hh_income = np.array([income_map[v] for v in demo["INCOME_DESC"]], dtype=np.int64)
    hh_kids = np.array([kids_map[v] for v in demo["KID_CATEGORY_DESC"]], dtype=np.int64)

    # Household id -> row index in the embedding table
    hh_to_row = {h: i for i, h in enumerate(all_households)}

    # Campaign -> type
    ctype_map = {"TypeA": 0, "TypeB": 1, "TypeC": 2}
    cdesc = campaign_desc.set_index("CAMPAIGN").reindex(all_campaigns)
    camp_type = np.array(
        [ctype_map.get(t, 1) for t in cdesc["DESCRIPTION"].fillna("TypeB")],
        dtype=np.int64,
    )

    # Campaign -> normalised department multi-hot (mirrors course "genre")
    dep_map = {d: i for i, d in enumerate(sorted(product["DEPARTMENT"].dropna().unique()))}
    n_depts = len(dep_map)
    dept_matrix = np.zeros((len(all_campaigns), n_depts), dtype=np.float32)
    cp = coupon.merge(product[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="inner")
    for camp, grp in cp.groupby("CAMPAIGN"):
        if camp not in all_campaigns:
            continue
        row = all_campaigns.index(camp)
        for d in grp["DEPARTMENT"].dropna().unique():
            dept_matrix[row, dep_map[d]] = 1.0
    # Normalise rows to sum to 1 (weighted average embedding, per course pattern)
    row_sums = dept_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    dept_matrix = dept_matrix / row_sums

    # Campaign -> causal buckets
    n_display_b, n_mailer_b = 5, 5
    if causal_feats is not None:
        cf = causal_feats.reindex(all_campaigns).fillna(0.0)
        display_buckets, n_display_b = _bucketize(cf["display_rate"], quantiles=5)
        mailer_buckets, n_mailer_b = _bucketize(cf["mailer_rate"], quantiles=5)
    else:
        display_buckets = np.zeros(len(all_campaigns), dtype=np.int64)
        mailer_buckets = np.zeros(len(all_campaigns), dtype=np.int64)

    camp_to_row = {c: i for i, c in enumerate(all_campaigns)}

    return {
        "n_households":    len(all_households),
        "n_campaigns":     len(all_campaigns),
        "n_age":           len(age_map),
        "n_income":        len(income_map),
        "n_kids":          len(kids_map),
        "n_campaign_types": 3,
        "n_departments":   n_depts,
        "n_display":       n_display_b,
        "n_mailer":        n_mailer_b,
        "hh_to_row":       hh_to_row,
        "camp_to_row":     camp_to_row,
        "hh_age":          hh_age,
        "hh_income":       hh_income,
        "hh_kids":         hh_kids,
        "camp_type":       camp_type,
        "dept_matrix":     dept_matrix,
        "display_bucket":  display_buckets,
        "mailer_bucket":   mailer_buckets,
    }


# ---------------------------------------------------------------------------
# Training pair construction — mirrors course cell 11
# ---------------------------------------------------------------------------
def build_training_examples(
    campaign_table: pd.DataFrame,
    train_campaigns: set[int],
    hh_to_row: dict[int, int],
    camp_to_row: dict[int, int],
    negatives_per_positive: int = 2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ct = campaign_table[campaign_table["CAMPAIGN"].isin(train_campaigns)]
    train_camp_list = list(train_campaigns)

    hh_pos = ct.groupby("household_key")["CAMPAIGN"].apply(set).to_dict()
    users, items, labels = [], [], []
    for hh, pos_camps in hh_pos.items():
        if hh not in hh_to_row:
            continue
        u = hh_to_row[hh]
        pool = [c for c in train_camp_list if c not in pos_camps and c in camp_to_row]
        for pc in pos_camps:
            if pc not in camp_to_row:
                continue
            users.append(u)
            items.append(camp_to_row[pc])
            labels.append(1.0)
            if pool:
                n = min(negatives_per_positive, len(pool))
                for nc in rng.choice(pool, size=n, replace=False):
                    users.append(u)
                    items.append(camp_to_row[int(nc)])
                    labels.append(0.0)
    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(items, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------
def train_fm(
    encoder_kwargs: dict,
    train_users: np.ndarray,
    train_items: np.ndarray,
    train_labels: np.ndarray,
    field_bundle: dict,
    embedding_dim: int = 12,
    epochs: int = 4,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = SEED,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    encoder = FeatureEncoder(embedding_dim=embedding_dim, **encoder_kwargs)
    model = FMModel(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Attach household + campaign side features for each training row
    ex_age = field_bundle["hh_age"][train_users]
    ex_income = field_bundle["hh_income"][train_users]
    ex_kids = field_bundle["hh_kids"][train_users]
    ex_ctype = field_bundle["camp_type"][train_items]
    ex_dept = field_bundle["dept_matrix"][train_items]
    ex_display = field_bundle["display_bucket"][train_items]
    ex_mailer = field_bundle["mailer_bucket"][train_items]

    ds = TensorDataset(
        torch.from_numpy(train_users),
        torch.from_numpy(train_items),
        torch.from_numpy(ex_age),
        torch.from_numpy(ex_income),
        torch.from_numpy(ex_kids),
        torch.from_numpy(ex_ctype),
        torch.from_numpy(ex_dept),
        torch.from_numpy(ex_display),
        torch.from_numpy(ex_mailer),
        torch.from_numpy(train_labels),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            *inputs, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            logits = model(*inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        avg = total / max(count, 1)
        history.append(avg)
        print(f"    Epoch {epoch:02d}  BCE={avg:.4f}")
    return model, history


@torch.no_grad()
def fm_recommendations(
    model,
    field_bundle: dict,
    eval_households: list[int],
    candidate_campaigns: list[int],
    fallback_ranking: list[int],
    k: int = K,
) -> pd.DataFrame:
    device = torch.device("cpu")
    model.eval()

    hh_to_row = field_bundle["hh_to_row"]
    camp_to_row = field_bundle["camp_to_row"]

    cand_rows = [camp_to_row[c] for c in candidate_campaigns if c in camp_to_row]
    cand_camps = [c for c in candidate_campaigns if c in camp_to_row]
    unknown_cands = [c for c in candidate_campaigns if c not in camp_to_row]
    if not cand_rows:
        # No known candidates — fallback for everyone
        rows = []
        for hh in eval_households:
            top = list(fallback_ranking[:k])
            while len(top) < k:
                top.append(top[-1] if top else candidate_campaigns[0])
            rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
        return pd.DataFrame(rows)

    camp_type_t = torch.from_numpy(field_bundle["camp_type"][cand_rows]).long().to(device)
    dept_t = torch.from_numpy(field_bundle["dept_matrix"][cand_rows]).float().to(device)
    display_t = torch.from_numpy(field_bundle["display_bucket"][cand_rows]).long().to(device)
    mailer_t = torch.from_numpy(field_bundle["mailer_bucket"][cand_rows]).long().to(device)
    camp_idx_t = torch.tensor(cand_rows, dtype=torch.long, device=device)

    rows = []
    for hh in eval_households:
        if hh not in hh_to_row:
            top = list(fallback_ranking[:k])
            while len(top) < k:
                top.append(top[-1] if top else candidate_campaigns[0])
            rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
            continue

        u = hh_to_row[hh]
        n = len(cand_rows)
        hh_idx_t = torch.full((n,), u, dtype=torch.long, device=device)
        age_t = torch.full((n,), int(field_bundle["hh_age"][u]), dtype=torch.long, device=device)
        income_t = torch.full((n,), int(field_bundle["hh_income"][u]), dtype=torch.long, device=device)
        kids_t = torch.full((n,), int(field_bundle["hh_kids"][u]), dtype=torch.long, device=device)

        logits = model(hh_idx_t, camp_idx_t, age_t, income_t, kids_t,
                       camp_type_t, dept_t, display_t, mailer_t)
        scores = torch.sigmoid(logits).cpu().numpy()
        order = np.argsort(-scores)
        top = [cand_camps[i] for i in order[:k]]
        # Pad with unknown candidates then fallback
        for c in unknown_cands:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        for c in fallback_ranking:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        while len(top) < k:
            top.append(top[-1] if top else candidate_campaigns[0])
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# Campaign-level causal features (display + mailer rates from causal_data.csv)
# ===========================================================================
from .config import DATA_DIR  # for load_causal_data_slim

# 1 week ≈ 7 days; used to convert campaign day ranges to week ranges
DAYS_PER_WEEK = 7


def _day_to_week(day: int) -> int:
    return max(1, (day - 1) // DAYS_PER_WEEK + 1)


def load_causal_data_slim(product_ids_of_interest: set[int]) -> pd.DataFrame:
    """
    Load causal_data.csv, filtered to products of interest and aggregated
    to (product, week) with binary display/mailer flags (ANY store).
    """
    print(f"    Loading causal_data.csv (subset to {len(product_ids_of_interest)} products)...")
    cd = pd.read_csv(
        DATA_DIR / "causal_data.csv",
        dtype={"PRODUCT_ID": "int32", "STORE_ID": "int32", "WEEK_NO": "int16",
               "display": "string", "mailer": "string"},
    )
    cd = cd[cd["PRODUCT_ID"].isin(product_ids_of_interest)]
    cd["on_display"] = (cd["display"] != "0").astype("int8")
    cd["on_mailer"] = (cd["mailer"] != "0").astype("int8")
    # Collapse across stores: any-display per (product, week)
    agg = (
        cd.groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False)
        .agg(display_any=("on_display", "max"), mailer_any=("on_mailer", "max"))
    )
    return agg


def compute_campaign_causal_features(
    coupon: pd.DataFrame,
    campaign_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return DataFrame indexed by CAMPAIGN with columns:
        display_rate, mailer_rate, n_covered_products
    """
    product_ids = set(coupon["PRODUCT_ID"].unique().tolist())
    causal_slim = load_causal_data_slim(product_ids)

    # Coupon -> (campaign, product_ids)
    campaign_products = (
        coupon.groupby("CAMPAIGN")["PRODUCT_ID"].apply(set).rename("products").reset_index()
    )
    campaign_products["n_covered_products"] = campaign_products["products"].apply(len)

    rows = []
    for _, camp in campaign_desc.iterrows():
        camp_id = camp["CAMPAIGN"]
        start_w = _day_to_week(int(camp["START_DAY"]))
        end_w = _day_to_week(int(camp["END_DAY"]))
        prods_row = campaign_products[campaign_products["CAMPAIGN"] == camp_id]
        if prods_row.empty:
            rows.append({"CAMPAIGN": camp_id, "display_rate": 0.0, "mailer_rate": 0.0,
                         "n_covered_products": 0})
            continue
        prods = prods_row["products"].iloc[0]
        n_covered = len(prods)
        sub = causal_slim[
            (causal_slim["PRODUCT_ID"].isin(prods))
            & (causal_slim["WEEK_NO"] >= start_w)
            & (causal_slim["WEEK_NO"] <= end_w)
        ]
        if len(sub) == 0:
            display_rate, mailer_rate = 0.0, 0.0
        else:
            display_rate = float(sub["display_any"].mean())
            mailer_rate = float(sub["mailer_any"].mean())
        rows.append({"CAMPAIGN": camp_id, "display_rate": display_rate,
                     "mailer_rate": mailer_rate, "n_covered_products": n_covered})

    return pd.DataFrame(rows).set_index("CAMPAIGN")

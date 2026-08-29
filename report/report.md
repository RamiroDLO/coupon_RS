<!--
  WORKING DRAFT — 10-page scientific report (single-column).
  Course: Recommender Systems Block Week, MSc Applied Data Science & AI, HSLU — Dr. Guang Lu
  Deadline: 28 September 2026. Email report + notebook to guang.lu@hslu.ch; share repo with `lugulugu`.

  HOW TO USE THIS FILE
  - Each `##` heading carries a page budget and, in [brackets], the grading criterion it earns.
  - `<!-- TODO: ... -->`  = content to write.
  - `**DECISION:** ...`    = a team choice that must be settled before the section is final.
  - Keep every number generated from code, not hand-typed (Inputs.docx point 15).
  - Grading weights: Evaluation rigor 25% · Method 20% · Data understanding & reproducibility 15%
    · Meaningful baseline 15% · Problem framing 10% · Analysis/limitations/responsible use 10%
    · Communication 5%.
-->

# Personalized Grocery Coupon Recommendation on the Dunnhumby Complete Journey Panel

**Authors:** Ana Mas Urquijo · Ramiro D.L.O. · <member 3> · <member 4>
**Course:** Recommender Systems Block Week — MSc Applied Data Science & AI, HSLU · Dr. Guang Lu
**Date:** September 2026

---

## Abstract  <!-- ~0.25 pg · write LAST -->

<!-- TODO: 5–7 sentences: problem · dataset · task · baseline vs improved method · headline
     result (state it honestly, incl. "domain baseline wins / ALS matches not beats") · one
     trade-off (e.g. coverage collapse). No causal claim. -->

---

## 1. Introduction & Problem Framing  <!-- ~1.0 pg · [Problem framing 10%] -->

<!-- TODO:
  - Target user: category manager, US grocery retailer. Business cost: ~70% of untargeted
    coupon impressions wasted (docs/Reco_Systems_Pitch.pdf).
  - One line each on the four Day-1 objectives: user value / business value / system
    constraints / responsible use.
  - Frame as Top-K ranking under biased implicit feedback (Day 1 Block 1). Not search, not
    rating prediction.
  - Paragraph: "Why not evaluate against the retailer's own targeting?" — retailer assignment
    is `observed feedback is biased` (Day 1 Block 1), so redemption behaviour is the ground
    truth instead. (This is where Study 1 lives now — see DECISION below.)
  - Scope sentence: what the study claims and does NOT claim.
  - Contributions: 3 bullets.
-->

**Research question** (Day 1 Block 6 template — fill P/X/Y/Z/W to match §5):

> Under evaluation protocol **P** *(temporal split, closed candidate set of N active coupons,
> households with ≥1 redemption)*, does method **X** *(weighted implicit ALS)* improve
> metric **Y** *(Recall@K, NDCG@K)* over baseline **Z** *(<baseline of record>)* on the
> Dunnhumby panel, and what trade-off does it create for metric **W** *(catalogue coverage)*?

**DECISION:** one study or two?
- *Recommended:* one study — Study 2, coupon-redemption ground truth. Demote the
  campaign-assignment framing (Study 1) to the paragraph above.
- *Alternative:* keep both; §4/§5/§6 each gain a short "Study 1 vs Study 2" contrast block
  and lose depth elsewhere.

---

## 2. Dataset & Feedback Signals  <!-- ~1.25 pg · [Data understanding & reproducibility 15%] -->

<!-- TODO:
  - Dunnhumby "The Complete Journey": 2,500 households, ~2.6M transactions, ~92k products,
    ~102 weeks, 8 tables + campaign / coupon / redemption ledger + causal in-store promo flags.
  - The household × item matrix: which granularity the MODEL sees (commodity / product /
    coupon) and why. Report density AT THAT granularity — do NOT mix the commodity-level
    (~37%) and product-level (~0.61%) numbers; state which one the model uses.
  - Feedback type: implicit, MNAR. "Missing ≠ disliked" (Day 1 Block 1; Day 2 Block 5).
  - Table of EDA findings that drive modelling decisions: activity distribution · cold vs
    heavy households · redemption base rate (~0.27%) · demographic coverage (~32%, non-random)
    · test-window representativeness.
  - Reproducibility: repo link, seed, `pip install -r requirements.txt`, the ONE command
    that regenerates the results table and every figure.
-->

**DECISION:** interaction signal for the matrix — net spend (`SALES_VALUE`) vs purchase
frequency / basket count / recency / binary / RFM combo (Inputs.docx point 4). State which,
and why.

---

## 3. Related Work  <!-- ~0.75 pg -->

<!-- TODO (~12–15 refs):
  - RFM segmentation (industry practice).
  - Implicit ALS — Hu, Koren, Volinsky 2008.
  - BPR — Rendle et al. 2009.  FM — Rendle 2010.  EASE — Steck 2019.
  - Item-based CF — Sarwar et al. 2001.  Evaluation — Herlocker et al. 2004.
  - 1–2 sentences placing deep / sequential / LLM RS (Day 3–4) and why they are out of scope
    here: no session sequence, small panel, interpretability need.
-->

---

## 4. Method  <!-- ~2.0 pg · [Method design & implementation 20%] + [Meaningful baseline 15%] -->

### 4.1 Task definition
<!-- TODO: users · items · candidate set (closed, size N) · output = ranked Top-K · K = ? -->
**DECISION:** K = 3 or 5 (Inputs.docx: "cambiemos el ndcg al 3 o al 5"). Fix once, everywhere.

### 4.2 Baseline ladder (Day 1 Block 4)
<!-- TODO: give each with its exact scoring formula.
  1. Random — sanity floor.
  2. Popularity — most-redeemed coupon on train.
  3. Segment / RFM popularity — top of the household's RFM segment.
  Domain-informed:
  - last-category — score a coupon by household train-window spend in the commodities it
    covers (content-based recipe, Day 2 Block 1).
  - repeat-buy — rank coupons the household already redeemed.
-->
**DECISION:** baseline of record — the single baseline the improved method must beat. Name it.

### 4.3 Improved method — weighted implicit ALS (Day 2 Block 5)
<!-- TODO: matrix definition · confidence c = 1 + alpha * r · latent dim · reg · iterations ·
     alpha · Hu-Koren-Volinsky citation · how per-household scores map back to a ranked
     candidate list · train-only fitting. -->

### 4.4 (optional) Factorization Machine / hybrid
<!-- TODO: only if kept — fields used, why FM generalises MF (Day 2 Block 6). Else delete. -->

### 4.5 What we deliberately did not run
<!-- TODO: BPR, EASE, LightGCN, sequential/transformer — one sentence each on why not
     (→ §9 Future work). Shows Day 2–4 awareness. -->

<!-- FIG 1: architecture diagram (data → matrix → ALS → candidate scoring → Top-K). -->

---

## 5. Evaluation Protocol  <!-- ~1.5 pg · [Evaluation rigor 25% — part 1] -->

Protocol = **split + relevance threshold + candidate set + filtering rule** (Day 1 Block 2).

- **Split.** Temporal, weeks <a>/<b>/<c> (train / val / test). Justify temporal over random
  ("predicting future interactions from past is the deployed setting", Day 1 Block 2).
  **DECISION:** split boundaries. Justify why the test window has enough active campaigns /
  targeted households / coupons / redemptions to be non-degenerate (Inputs.docx point 12).
- **Relevance threshold.** Binary — a hit is <redeemed coupon in the test window>.
- **Candidate set.** Closed, N = <active coupons whose parent campaign overlaps the test
  window>. No sampled negatives — justify (small closed universe).
- **Filtering rule.** Evaluate households with ≥1 ground-truth item; report the dropped count.
- **Leakage controls (state each one explicitly — this is where rigor marks are won):**
  - popularity counts, RFM segments, user profiles, ALS hyperparameters — all fit on TRAIN only;
  - the ground-truth / assignment table is used ONLY for scoring, never to rank a baseline
    (Inputs.docx baseline-leakage note);
  - sparsity recomputed on the TRAIN matrix only (Inputs.docx point 5);
  - campaigns handled by Type A / B / C where the exposure assumption differs
    (Inputs.docx points 2, 9) — or excluded, with reason.
- **Metrics.** Recall@K, NDCG@K (≥2 relevance) + **coverage** (beyond-accuracy, required by
  Day 1 Block 2) + **E[revenue]@K** (business-value proxy — framed as NOT causal). Exact
  formulas for all four.
- **Uncertainty.** Bootstrap 95% CIs on Recall & NDCG (N = 1000). One frozen protocol —
  no post-hoc metric shopping (docs/EVAL_HARNESS_SOP.md).

---

## 6. Results & Analysis  <!-- ~1.75 pg · [Evaluation rigor 25% — part 2] -->

<!-- TODO:
  - TABLE 1: every model × {Recall@K, NDCG@K, coverage, E[revenue]@K} with 95% CIs.
  - FIG 2: Recall@K bar chart with CI whiskers.
  - FIG 3: coverage-vs-recall scatter (the accuracy/coverage trade-off).
  - TABLE 2: metrics stratified by household activity tier (cold / mid / heavy).
  - Answer the RQ in one sentence: did X beat Z, by how much, are the CIs disjoint?
  - Ablation: ALS allow-seen vs exclude-seen — what it reveals about the signal ALS learned.
-->

---

## 7. Beyond-Accuracy Discussion  <!-- ~1.0 pg · [Analysis 10% — part 1] -->

<!-- TODO:
  - Why the simple domain baseline wins: "complexity must earn its keep" (Day 1 Block 4;
    Day 2 Block 5 EASE argument). Frame as a RESULT, not a failure.
  - Popularity bias & coverage collapse of the winning baseline (numbers).
  - Business reading of E[revenue]@K and its limits (hit-weighted proxy, not uplift).
-->

---

## 8. Limitations & Responsible Use  <!-- ~0.75 pg · [Analysis / responsible use 10% — part 2] -->

<!-- TODO:
  - Small panel (2,500 hh); offline-only evaluation; proxy-not-causal.
  - Exposure / selection bias in the feedback (Day 1 Block 1).
  - Non-random demographic coverage → not a model feature; post-hoc subgroup analysis only.
  - Fairness / popularity-bias note; privacy (loyalty-card data).
  - One concrete failure mode of the deployed recommender.
-->

---

## 9. Conclusion & Future Work  <!-- ~0.5 pg -->

<!-- TODO:
  - Two sentences landing on the RQ answer.
  - Future: BPR / EASE / LightFM hybrid; A/B test as the honest next step
    ("offline eval can mislead", Day 1 Block 2).
-->

---

## References

<!-- TODO: ~12–15 entries. Decide citation style (team). -->

---

<!-- ================= OUTSIDE THE 10-PAGE LIMIT ================= -->

## Appendix A — Contribution Statement
<!-- TODO: one paragraph per team member. -->

## Appendix B — Responsible Use of AI (Coding) Tools
<!-- TODO: tools used · what for · what was human-verified. -->

## Appendix C — Learning Reflection
<!-- TODO: max 1 page. Each member writes their own (may be submitted separately). -->

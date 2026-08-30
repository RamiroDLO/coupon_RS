# Personalized Grocery Coupon Recommendation on the Dunnhumby Complete Journey Panel

<!-- Grading: % after each heading is that criterion's weight (Day 1, Block 6).
     Communication quality (5%) applies to the whole document. Sections 1–6 = 95%; +5% = 100%. -->

## Abstract

## 1. Problem — Problem framing and relevance (10%)

### 1.1 The business problem

A grocery retailer sends coupon offers to its loyalty-card households. It already aims
campaigns at broadly relevant shoppers — households chosen for a campaign had spent
about 2.5 times the average in that campaign's product categories — but it does not
tailor the specific products offered to each household. Because households shop very
differently, many offers still reach people who would not have bought the product
anyway, wasting both the customer's attention and the retailer's limited campaign
capacity. In the two years of data we use, only about **12% of the household–campaign
pairs the retailer targeted led to any redemption**; the rest did not. The user we
design for is a **category or CRM manager** who must decide, before each campaign,
which products to offer each household a coupon on.

### 1.2 What the recommender does

For each household, the system produces a short ranked list of products to offer a
coupon on — the products that household is most likely to buy in the near future. It
can only choose from **coupon-eligible products**: the roughly 44,000 items (out of a
~92,000-product catalogue) that a coupon exists for. This is a ranking task — order
the candidates, return the top few — not a yes/no prediction, and not a search where
the customer types in what they want.

### 1.3 What the model learns from

The only signal available is what households bought, never what they thought of a
product. This is *implicit feedback*: a purchase is a positive signal, but the absence
of a purchase is not a negative one — the household may simply never have been offered
the product or noticed it. The model treats "not bought" as *unknown*, not as
*disliked*. We deliberately do not score the recommender against the retailer's own
past targeting choices: those record *who the retailer decided to contact*, not *who
wanted the offer*, and matching them would just teach the model to copy the retailer,
bias included. Instead we score every method against what households **actually
bought** in a later period.

### 1.4 Research question and scope

Following the course template — *under protocol P, does method X improve metric Y over
baseline Z, and at what cost to metric W?*:

> Under a temporal train/validation/test split of the Dunnhumby "Complete Journey"
> panel, with a closed candidate set of coupon-eligible products and households scored
> on their test-period purchases, does **weighted implicit ALS** improve **Recall@5
> and NDCG@5** over a **most-popular-products baseline**, and what does it cost in
> **catalogue coverage**?

In plain terms: can personalising the offer list beat "send everyone the bestsellers"
at predicting what each household buys next, without shrinking the range of products
we ever recommend?

**Scope.** If the personalised model wins, we can say it predicts future purchases
more accurately than a simple rule. We cannot claim that sending a coupon *causes* a
purchase — this is an offline study of prediction, not a controlled experiment.

**Contributions.** (1) A clean, reproducible setup for the task: a frozen past/future
split, a defined product menu, and a clear definition of a correct recommendation.
(2) A like-for-like comparison of a simple baseline against a personalised model,
scored on real purchase behaviour. (3) An honest account of the trade-off between
relevance and coverage, and of the study's limitations.

## 2. Dataset — Data understanding and reproducibility (15%)

### 2.1 The dataset

We use the Dunnhumby "The Complete Journey" panel: every purchase made by 2,500
loyalty-card households at a US grocery retailer over about two years (102 shopping
weeks), together with the retailer's coupon-campaign records. It contains roughly
**2.6 million purchase lines** across a **~92,000-product catalogue**, plus **1,135
distinct coupons** and a record of which campaign each household received and which
coupons it redeemed. Every product also carries a category label: it rolls up into
one of ~300 broad **commodities** (e.g. "yogurt") and ~40 **departments**
(e.g. "dairy"). A further file marks whether a product was in the weekly mailer or on
an in-store display; we do not use it, because this study predicts purchases rather
than measuring the causal effect of promotions.

### 2.2 Data quality and cleaning

The data is unusually clean. There are **no missing values** in the purchase, product
or coupon tables, **no duplicated purchase lines**, and every purchase and every
coupon links to a real catalogue product. We also confirmed each raw file matches the
published dataset size, so nothing was truncated or re-exported.

The only issue is a small set of odd lines: about **0.6% of purchase lines have a
quantity of zero or below**, and about **0.7% have a sales value of exactly zero**
(returns, giveaways or scanning glitches). We exclude these — a purchase, for us, is a
line where money changed hands and at least one unit was bought — which removes
roughly **1% of the data** and leaves the rest untouched. No other cleaning is
applied.

One caveat about the money field: the recorded "sales value" is what the **retailer
received** after loyalty and coupon discounts, not the price the shopper paid. This
does not affect the model, which only uses *whether* a product was bought, but it
would matter if the model were later weighted by spend.

### 2.3 What we recommend, and how thin the evidence is

The recommender can only offer a coupon for a **coupon-eligible product** — one that
appears in the coupon records. There are **44,133** of these, about **48% of the
catalogue**; **39,132** were bought by at least one household during training, and
those form the pool the model chooses from. Coupons map to products very unevenly: a
typical coupon covers about a dozen products, but a handful of broad coupons ("any
private-label frozen vegetable") cover thousands.

The evidence the model has to work with is very sparse. Laying out a grid of every
household against every coupon-eligible product, only about **0.7% of the cells** are
filled — the rest are combinations that simply never occurred. Sparsity at this level
is normal for grocery; it means the model cannot rely on rich detail about each
household and must learn from patterns across similar households. At the coarser
category level the same shopping is much denser (a typical household has bought in
about a third of the ~300 commodities), which is a fallback signal when a household's
per-product history runs thin.

### 2.4 What the data tells us about the method

Several findings from the exploratory analysis shape the modelling choices:

- **Demand is spread out.** The 100 best-selling products account for only about 15%
  of all purchases, so a plain "recommend the bestsellers" rule can only go so far and
  personalisation has genuine room to add value.
- **The task is mainstream.** Coupon-eligible products make up about **60% of all
  shopping** (by lines and by money spent), so this is a central lever for the
  retailer, not a niche one.
- **Histories are long.** A typical household made about 80 shopping trips and bought
  roughly 200 different coupon-eligible products in training — plenty to personalise
  from. Only ~4% of households barely shopped; ~1% shopped so heavily they would
  dominate an unadjusted "most-bought" ranking.
- **Almost every household can be scored, and cold start is not a concern.** Of the
  2,500 households, **2,364** bought at least one coupon-eligible product in the test
  period and can be scored, and **99.8%** of them have prior history.
- **The test period is a demanding target.** In the test window a household buys about
  **70** different coupon-eligible products, and about **60% of those are products it
  had never bought before** — so a recommender that only replays past favourites would
  miss most of what happens. Because the target per household is large, absolute
  Recall figures will be small; the comparison *between* methods is what matters
  (see §5).
- **Purchases are a clean preference signal.** Only about **1.6%** of the
  coupon-eligible purchases in the test window involved a manufacturer coupon, so
  "the household bought it" reflects genuine preference, not a reaction to a promotion.
- **Demographics are not used as a model input.** The household demographic file (age,
  income, family) covers only about **32%** of households, and those households spend
  markedly more than the rest — the coverage is not random. Feeding it to the model
  would bias it toward that third of the panel; it is kept only for a possible
  subgroup check.

For context, the retailer's current targeting sets a low but non-zero bar: only about
**12%** of the household–campaign pairs it targeted led to any redemption, though the
households it chose had already spent about **2.5×** the average in the relevant
categories.

### 2.5 Reproducibility

All figures above come from a single exploratory notebook (`notebooks/01_eda.ipynb`)
that loads the raw files, applies the cleaning rule, and prints every number used
here. The past/future split — training weeks 1–79, validation 80–84, test 85–102 — is
fixed in a shared configuration file and imported everywhere, so it cannot drift
between the exploration, the baselines and the final model. The dataset is publicly
available on Kaggle; the repository holds the code and the split definition, and one
command regenerates the results.

## 3. Baseline — Meaningful baseline (15%)

## 4. Method — Method design and implementation (20%)

## 5. Evaluation — Evaluation rigor (25%)

## 6. Reflection — Analysis, limitations and responsible use (10%)

## 7. Conclusion

## Author Contributions

## Responsible Use of AI (Coding) Tools

## References

## Appendix A — Learning Reflection (max 1 page, not graded)

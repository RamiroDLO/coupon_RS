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

## 3. Baseline — Meaningful baseline (15%)

## 4. Method — Method design and implementation (20%)

## 5. Evaluation — Evaluation rigor (25%)

## 6. Reflection — Analysis, limitations and responsible use (10%)

## 7. Conclusion

## Author Contributions

## Responsible Use of AI (Coding) Tools

## References

## Appendix A — Learning Reflection (max 1 page, not graded)

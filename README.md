# coupon_RS

Capstone project for the Recommender Systems course (Master Applied Data Science and AI) — a coupon/targeting
recommender built on the Dunnhumby **"The Complete Journey"** household-panel grocery dataset.

## Repository structure

```
coupon_RS/
├── eda/
│   └── 01_EDA_Sparsity_Targeting_Signal_Data_Quality.ipynb
├── docs/
│   └── Reco_Systems_Pitch.pdf
├── Data/                 # raw CSVs — not tracked in git, see below
└── .gitignore
```

## Data

The `Data/` folder is intentionally **not** tracked in this repository (see `.gitignore`) — the raw CSVs
are large (the transaction and causal-data files alone are several hundred MB) and are excluded to keep the
repo lightweight.

To reproduce the notebooks, download the Dunnhumby "The Complete Journey" dataset and place the CSVs in a
local `Data/` folder at the repo root:

```
Data/
├── campaign_desc.csv
├── campaign_table.csv
├── causal_data.csv
├── coupon.csv
├── coupon_redempt.csv
├── hh_demographic.csv
├── product.csv
└── transaction_data.csv
```

## Notebooks

### `eda/01_EDA_Sparsity_Targeting_Signal_Data_Quality.ipynb`

Exploratory data analysis that scopes how the capstone recommender should be built and evaluated. Runs five
data-quality checks, then answers six questions:

1. How sparse is the household x commodity matrix?
2. What does the household activity distribution look like?
3. Does the retailer's own targeting have signal?
4. What's the base-rate coupon redemption rate?
5. Is the test window (weeks 97-102) representative?
6. Does the demographic table cover households at random?

The notebook closes with an executive-summary table and a limitations list to carry into the write-up.

## Docs

`docs/Reco_Systems_Pitch.pdf` — the project pitch deck.

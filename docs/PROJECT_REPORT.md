# Sasol Customer Value Challenge: Project Report

Author: Amare
Competition: Zindi, Sasol Customer Value Recruitment Challenge (23 Sep to 19 Oct 2026)
Code: Kedro project `sasol_clv` in this folder

This report explains the project in plain language first, then covers the technical detail Zindi asks for
in its code review (architecture, ETL, modelling, inference, run time, metrics, error handling, maintenance).
If you only have five minutes, read sections 1 to 4 and section 12.

---

## 1. The problem in one paragraph

Sasol runs a loyalty programme at its fuel stations. Every time a card holder buys fuel or something from the
shop, the transaction is recorded. Sasol wants to look at what a customer did in the past and answer three
questions about the **next three months**:

1. How many **litres of fuel** will this customer buy?
2. How many **rand will they spend in the shop** (everything that is not fuel)?
3. What is the **most likely change in their behaviour**? Will they stop coming (Inactivity), start buying a
   category they have never bought (New category adoption), buy noticeably more of something they already buy
   (Existing-category growth), or just carry on as before (Stable)?

Why Sasol cares: if you know who is about to drift away you can send them an offer before they go. If you
know who is about to start buying coffee you can put a coffee voucher in front of them. This is called
predicting **Customer Lifetime Value (CLV)** and finding **cross-sell** opportunities.

## 2. An everyday analogy

Think of a spaza shop owner who knows her regulars. She notices that Thabo used to come in every Friday but
has not been in for six weeks, so he might be gone. She notices Lerato buys bread every day and has recently
started buying cold drinks, so drinks might become a bigger part of her spend. Our model does the same thing,
but for thousands of customers at once, using numbers instead of memory.

## 3. The data

| File | What it holds | Size |
|---|---|---|
| `train.csv` | Every line item bought by loyalty customers from 19 Mar 2024 to 30 Nov 2025 | 467 880 rows, 6 204 customers, 390 sites |
| `test.csv` | The 5 488 customers we must predict for | one column: ID |
| `SampleSubmission.csv` | The format Zindi expects | ID, CLV_fuel, CLV_nonfuel, Opportunity |
| `label_config.json`, `LABEL_RULES.md`, `label_rules.py` | The exact rules Zindi uses to create the answers | small |
| `DATA_DICTIONARY.md` | What each column means | small |

Each row is one line on a till slip. Important columns:

- `DateTimeZA`: when it happened
- `TransactionId`: the till slip (basket). One basket can have many rows
- `ID`: the anonymised customer
- `ItemCategoryLevel2`: category, for example Fuel, Beverages, Snacking
- `Quantity`: litres for fuel, units for shop items. Negative means a return
- `TotalExclAmount`: rand value of the line

Things we learned in the data audit (`notebooks/01_data_audit.ipynb`):

- No missing values. The category **"NA" is a real category name**, not a blank, so we must read it as text.
- There are a few hundred returns (negative rows) and 317 discount lines (for example "TOTAL/ROUNDING DISCO")
  that the rules say to ignore.
- Every test customer appears in the training history with at least 3 baskets.
- Fuel is about 73% of all rows. About 30% of customers never buy anything in the shop.

## 4. What exactly we have to hand in

A CSV with one row per test customer:

| Column | Meaning | How it is scored |
|---|---|---|
| `CLV_fuel` | predicted fuel litres, **transformed** | RMSE (average size of the error) |
| `CLV_nonfuel` | predicted shop rand, **transformed** | RMSE |
| `Opportunity` | one of 17 labels | Weighted F1 (how often we pick the right label, weighted by how common each label is) |

**The transform.** Customers range from 0 litres to thousands of litres. To stop the huge customers from
dominating the score, Zindi squashes the numbers with a log: `z = ln(1 + y) / s`. The constant `s` is given
(2.3528 for fuel, 2.4201 for non-fuel). A customer with no fuel gets 0. We submit `z`, not litres.

**The final score** combines the three parts with weights 30%, 30% and 40%. On the leaderboard higher is
better. The Opportunity label is the single biggest part of the score.

**The 17 Opportunity labels** follow fixed rules (`LABEL_RULES.md`):

- **Inactivity**: no real purchase in the next 3 months.
- **New category adoption: X**: buys a category never bought before, with positive spend.
- **Existing-category growth: X**: spends at least R20 more **and** more than 25% more in a category than in
  the previous 3 months.
- **Stable**: still active, but none of the above.
- If more than one applies, the biggest rand increase wins. Small categories are grouped as "Other".

In practice four labels cover about 85% of customers: Stable, Inactivity, Existing-category growth: Fuel and
Existing-category growth: Other.

## 5. The key idea: making our own practice exams

The training file has transactions but **no answers**. Zindi keeps the real answers (Dec 2025 to Feb 2026)
hidden. So how do we train a model?

We turn back the clock. We pick a date in the past, for example 1 June 2025, and pretend that is "today".

- Everything **before** 1 June is the customer's history. We turn it into features (section 7).
- The three months **after** 1 June (June to August) already happened, so we can work out the correct answers
  using Zindi's own rules.

We do this for every month start from June 2024 to September 2025 (16 "practice exams"), stack them together
and train on all of them. The final model then looks at history up to 1 Dec 2025 and predicts Dec to Feb.

**How we know our answers are right.** Zindi published its label code (`label_rules.py`). Our own label
builder gives exactly the same labels for 100% of customers, and it reproduces Zindi's scaling constants to
the last decimal. So our practice exams are marked exactly like the real one.

## 6. Architecture

```
                  data/01_raw
   train.csv   test.csv   SampleSubmission.csv   label_config.json
       |           |               |                    |
       v           |               |                    |
 [1 data_processing]               |                    |
   clean_transactions  <-----------+--------------------+
       |  adds flags: fuel litres row, shop row, adjustment, qualifying purchase
       v
   transactions.parquet (02_intermediate)
       |
       v
 [2 feature_engineering]
   build_train_table: for each of 16 past cutoffs
        eligible customers -> features (history) + labels (next 3 months)
   build_test_table: features at 1 Dec 2025 for the 5 488 test customers
       |
       v
   train_table.parquet, test_table.parquet (05_model_input)
       |
       v
 [3 modelling]
   train_and_validate: train on older cutoffs, score on the Sep 2025 quarter
        -> validation_metrics.json (08_reporting)
   train_final: retrain on all 16 cutoffs -> final_models.pkl (06_models)
       |
       v
 [4 submission]
   make_submission: predict, check format -> submission.csv (07_model_output)
```

Kedro runs these four pipelines in order (`kedro run`). Every file location is set in
`conf/base/catalog.yml` and every setting in `conf/base/parameters.yml`, so nothing is hard coded.

## 7. ETL (extract, transform, load)

**Extract.** The four raw files are read from `data/01_raw`. IDs and product codes are read as text, and the
literal "NA" category is kept as text. About 72 MB, loads in a few seconds.

**Transform.**

1. Parse dates and add simple true/false flags that the label rules need: is this a fuel litres row, a shop
   row, a discount line, an excluded category (Value added services, Print media, Carrier Bags), a qualifying
   purchase (positive quantity and amount).
2. For each cutoff, keep only customers with at least 3 baskets before it (Zindi's eligibility rule).
3. Build about 160 features per customer (98 used by the regressors, all 164 by the classifier) from history only. The main groups:

| Group | Examples | Why it helps |
|---|---|---|
| Activity | baskets, days since last visit, days as a customer, gaps between visits | recency is the strongest sign of Inactivity |
| Recent windows | fuel litres and shop rand in the last 30, 90 and 180 days | the recent past is the best guide to the near future |
| Trend | last quarter versus the quarter before, 6 month slope | speeding up or slowing down |
| Category history | ever bought, spend last quarter, baskets per category | separates adoption from growth |
| Growth gap | rand needed next quarter to count as growth versus the customer's usual quarter | directly mirrors the growth rule |
| Habits | weekend share, litres per fill, diesel share, returns | spending style |

4. Build the labels with the official rules (fuel litres, shop rand and the Opportunity label).

**Load.** Tables are saved as Parquet files (fast and compact), models as pickle files, the submission as CSV.

## 8. Modelling

**Algorithm.** `HistGradientBoosting` from scikit-learn (free and open source). It builds many small decision
trees one after the other, each fixing the mistakes of the previous ones. It handles missing values and skewed
numbers well and is fast.

- Two **regressors** predict `CLV_fuel` and `CLV_nonfuel` directly on the transformed scale, so they are
  trained on exactly what is scored. Each is the average of 3 models with different seeds for stability.
- One **classifier** predicts the Opportunity label out of 17 classes.

**Settings** (`parameters.yml`): learning rate 0.02 (regression) and 0.05 (classifier), trees with at most 15
leaves, at least 300 customers per leaf, strong L2 regularisation (10), each tree sees half the features, no
early stopping. Seeds are fixed (42, 7, 2026) so every run gives the same result.

**Validation.** We copy the real test: train only on cutoffs whose 3 month future ends before 1 Sep 2025, then
score on the Sep to Nov 2025 quarter. Nothing from the scored quarter leaks into training. We also checked
on the Jun 2025 quarter and on the Dec 2024 quarter (the closest match to the festive test season).

## 9. What we tried and what worked

| Change | Result | Kept? |
|---|---|---|
| Version 1: gradient boosting with default early stopping | public score 0.3016812, rank 9 | baseline |
| **Turning off early stopping and adding strong regularisation** | fuel and shop errors down on all 3 test quarters, F1 up | yes, biggest win |
| Growth gap and rhythm features for the classifier | F1 up about 0.003 on both quarters | yes (classifier only) |
| Nudging the rare labels (x2 other growth, x3 adoption) before picking the top label | F1 up about 0.006 on both quarters | yes |
| Classifier skips the Jun 2024 cutoff (its previous quarter is cut short because the data starts on 19 Mar, so growth labels there are too easy) | F1 up about 0.002 to 0.003 on two quarters | yes (v3) |
| Removing the cutoff month feature | small gain, and one less way to overfit | yes |
| Averaging 3 seeds for the regressors | steadier predictions | yes |
| Extra Trees and Random Forest blends | small gain on their own, no gain once boosting was fixed | no |
| Two-step "zero or not, then how much" model | no gain | no |
| Extra mid-month cutoffs (more training rows) | slightly worse | no |
| Weighting recent cutoffs more | no gain | no |

**Why the early stopping change mattered (a good interview story).** Early stopping holds back a random 10%
of rows to decide when to stop training. But our rows come from overlapping practice exams, so the same
customer at 1 June and 1 July looks almost identical. The held back rows were not really new, the model
thought it was still improving, and it trained too long (overfitting). Turning early stopping off and making
each tree simpler fixed it. Lesson: **your validation must respect how the data was built.**

**Validation results, version 1 versus version 2**

| Quarter scored | Fuel RMSE v1 to v2 | Shop RMSE v1 to v2 | Weighted F1 v1 to v2 |
|---|---|---|---|
| Dec 2024 to Feb 2025 | 0.634 to 0.622 | 0.827 to 0.817 | 0.436 to 0.445 |
| Jun to Aug 2025 | 0.615 to 0.594 | 0.747 to 0.738 | 0.492 to 0.510 |
| Sep to Nov 2025 | 0.607 to 0.599 | 0.753 to 0.747 | 0.508 to 0.510 |

Every number moved the right way on every quarter, which is why we trust the change.

Simple baselines on the Sep 2025 quarter, to show the model adds value: "next quarter equals last quarter"
scores fuel 0.689, shop 0.928, F1 0.113. Predicting all zeros scores fuel 1.834, shop 1.224.

## 10. Inference (making the predictions)

`kedro run` retrains on all 16 cutoffs, builds features for the test customers at 1 Dec 2025, predicts,
applies the rare-label nudge, and writes `data/07_model_output/submission.csv`. Before saving, the code checks
5 488 rows, same IDs as the sample, no blanks and the right column order.

In a real Sasol setting the same pipeline would run each month: drop in the latest transactions, set
`test_cutoff` to the new month, run `kedro run`, and send the predictions to the marketing system.

## 11. Run time and environment

- Machine: Windows laptop, Python virtual environment, Kedro 1.6.0, kedro-datasets 9.6.0, scikit-learn,
  pandas (see `requirements.txt`). (Add your laptop CPU and RAM here.)
- Full `kedro run`: about 3 to 6 minutes.
  - cleaning: about 5 s
  - training table (16 cutoffs): about 40 s
  - test table: about 5 s
  - validation models: about 2 minutes
  - final models: about 2 to 3 minutes
  - submission: under 5 s
- Notebooks: the audit takes about 1 minute, the EDA about 1 minute.

## 12. How to explain this in an interview (the 60 second version)

"Sasol gave us two years of loyalty transactions and asked us to predict, for each customer, their fuel
litres and shop spend over the next quarter, and what kind of change in behaviour to expect. There were no
labels, so I rebuilt the problem at 16 past dates: history before each date became features and the three
months after it became the answers, using Sasol's own label rules. I proved my labels matched theirs 100%.
I engineered about 160 customer features on recency, frequency, spend, trends and category habits, trained
gradient boosted trees, and validated on the most recent unseen quarter so the test was realistic. The biggest
improvement came from spotting that early stopping was leaking because my training rows overlap in time. I
built it as a Kedro pipeline so it is reproducible with one command."

Likely follow up questions and short answers:

- **Why log transform?** Spend is very skewed. Without it, a few big fleet customers would dominate the error.
- **How do you avoid leakage?** Features only use data before the cutoff. Validation quarters never overlap
  training outcomes.
- **Why weighted F1 and not accuracy?** Classes are unbalanced. Weighted F1 balances precision and recall per
  class, weighted by how common each class is.
- **What drives Inactivity?** Days since last visit. Over half of customers not seen for 60 days are inactive
  the next quarter, versus about 3% of those seen in the last week.
- **Business use?** Win-back offers for likely inactive customers, category vouchers for likely adopters,
  and forecasting fuel demand per customer segment.
- **What would you do with more time?** Separate model for Inactivity, per-category models for growth,
  more careful handling of the festive season, and monitoring drift in the label mix.

## 13. Error handling and logging

- Kedro logs every step with timestamps, and our nodes log row counts and validation scores.
- The label builder raises an error if a category is missing from the config.
- The submission node refuses to save if rows, IDs, columns or blanks are wrong.
- The submission CSV is written with plain line endings. An earlier file uploaded from Windows got an extra
  blank line after every row, and Zindi rejected it with "Missing entries for IDs". Never open and re-save
  the file in Excel before uploading, because Excel can also turn long IDs into numbers like 4.001E+12.

## 14. Maintenance and monitoring

- Retrain monthly with the newest transactions. Add the new cutoffs to `train_cutoffs`.
- Watch the label mix over time. Inactivity grew from 7% to 27% between mid 2024 and late 2025, so a model
  trained on old months can under-predict churn.
- Track the validation metrics JSON for every run, and keep a submission log (date, change, local scores,
  public score).
- Scaling: the pipeline handles about 470 000 rows in minutes on a laptop. For millions of rows, the same
  Kedro nodes can move to Spark or Dask datasets without changing the pipeline shape.

## 15. Rules we followed

Only the provided data, only open source tools, no AutoML, seeds fixed, one account, data never shared
(`data/` is in `.gitignore`).

## 16. Scores

| Version | Change | Local F1 (Sep 25) | Public score | Private score |
|---|---|---|---|---|
| v1 | first model | 0.508 | 0.3016812 | at close |
| v2 | regularised models, new features, label nudge | 0.510 | 0.3040199 | at close |
| v3 | classifier skips the Jun 2024 cutoff | 0.513 | fill in after upload | at close |

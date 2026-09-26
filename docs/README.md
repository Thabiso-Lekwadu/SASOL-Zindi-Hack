# Sasol Customer Value Challenge: what to do and how to win it

A short guide for this project. Read it once before touching the code.

## 1. The challenge in plain words

Sasol gives us loyalty card transactions (fuel and shop purchases) from 19 Mar 2024 to 30 Nov 2025.
For 5 488 customers we must predict what happens in the next three months (1 Dec 2025 to 28 Feb 2026):

| Column | What it is | How it is scored |
|---|---|---|
| `CLV_fuel` | total fuel litres in the 3 months | RMSE on the transformed value |
| `CLV_nonfuel` | total shop spend in rand (excl VAT) | RMSE on the transformed value |
| `Opportunity` | one label out of 17 (see below) | weighted F1 |

**Important: we do not submit litres or rand.** We submit the transformed value

```
z = ln(1 + y) / s
s_fuel    = 2.352825386185113
s_nonfuel = 2.4200904988927476
```

Zero stays zero. Our pipeline trains straight on z, so the submission file already has the right scale.

Final score as written on Zindi: `0.3 x RMSE_fuel + 0.3 x RMSE_nonfuel + 0.4 x weighted F1`.
Mixing an error (lower is better) with F1 (higher is better) is odd, so watch the leaderboard direction
after your first upload. In practice: push both RMSEs down and F1 up. Our code logs both the Zindi formula
and an `error_score` that uses `1 - F1` so lower is always better.

## 2. The Opportunity label

Built with fixed rules (see `data/01_raw/LABEL_RULES.md`). In short, per customer:

1. **Inactivity**: no real purchase in the next 3 months.
2. **New category adoption: X**: buys a category they have never bought before.
3. **Existing-category growth: X**: spends at least R20 more and at least 25% more in a category than in the last 3 months.
4. **Stable**: active, but none of the above.

If a customer qualifies for several, the biggest rand increase wins. Small categories are grouped into "Other".
The big four classes are Stable, Existing-category growth: Fuel, Inactivity and Existing-category growth: Other.
Weighted F1 means these big classes count the most.

## 3. The trick that makes this work

The train file has no labels. We make our own by pretending "today" is an earlier date (a cutoff),
building features from everything before it, and building labels from the 3 months after it, using the
official rules. We do this for every month start from Jun 2024 to Sep 2025 and stack the results.

We proved our label code is correct: rebuilding the official scaling constants over the five calibration
quarters gives exactly the numbers in `label_config.json` (see the audit notebook, section 8).

Validation copies the leaderboard: train on cutoffs whose future ends before 1 Sep 2025, test on the
1 Sep 2025 cutoff (one unseen quarter). Never let a training cutoff's future window overlap the validation quarter.

## 4. Project layout (Kedro)

```
conf/base/catalog.yml       # where every dataset lives
conf/base/parameters.yml    # cutoffs, windows, model settings, seed
data/01_raw/                # put the Zindi files here
notebooks/01_data_audit.ipynb
notebooks/02_eda.ipynb
src/sasol_clv/pipelines/
    data_processing/        # clean rows, add flags from the label rules
    feature_engineering/    # labels.py (official rules), features.py (customer features)
    modelling/              # validation run, final fit on all cutoffs
    submission/             # predict test and write the csv
```

## 5. How to run it (Windows)

```
cd SASOL-Zindi-Hack
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

kedro run
```

Outputs:

- `data/08_reporting/validation_metrics.json`: local scores, compare with the baselines inside
- `data/07_model_output/submission.csv`: the file you upload

Useful commands:

```
kedro run --pipelines features     # only rebuild features
kedro run --pipelines modelling    # retrain using saved features
kedro run --pipelines submission   # re-predict with saved models
kedro viz                         # picture of the pipeline (pip install kedro-viz)
```

## 6. Making a submission on Zindi

1. Open the challenge page and click **Submit**.
2. Upload `data/07_model_output/submission.csv`.
3. Add a short note, for example "v1 hist gbm baseline".
4. You get 10 uploads a day and 200 in total. Before the end, **pick 2 submissions** for the private leaderboard.
   Pick your best local validation score, not only the best public score (public is only about 30% of test).

Checks the code does before saving: 5 488 rows, same IDs as the sample, no blanks, same column order.

## 7. How to push the score up (in this order)

1. **Track every run.** Keep a small table: date, change, local scores, public score.
2. **Opportunity first (40% of the score).**
   - Class weights or a tuned decision rule on top of the probabilities.
   - Features for each category: last quarter spend versus the quarter before, days since last purchase in the category.
   - Try a two step model: first Inactive or not, then which growth or adoption class.
3. **Regression targets.**
   - Split into two parts: chance of zero times expected value when not zero.
   - Features on visit rhythm: gaps between visits, trend in the last 4 to 8 weeks.
   - Seasonality: Dec to Feb. The Dec 2024 cutoff is our closest example, check how the model does on it.
4. **Validate on more than one quarter** (for example Jun 2025 and Sep 2025) so you do not chase noise.
5. **Blend** a few models with different seeds or settings. Only open source libraries, no AutoML.

## 8. Rules that can get you disqualified

- Only the provided data. No outside data.
- No AutoML tools.
- Always set the seed (ours is in `parameters.yml`). Top 10 must hand in code that reproduces the score within 48 hours.
- Do not share code or data outside the platform. The data has no licence to share, so never push `data/` to a public repo (the `.gitignore` already blocks it).
- Team size 1, one account.

## 9. Key dates

- Close: 18 Oct 2026 (Zindi shows 19 Oct)
- Private leaderboard: shortly after close
- Top 10 get a code review request and a job interview invite (Data Scientist or Senior Data Scientist).

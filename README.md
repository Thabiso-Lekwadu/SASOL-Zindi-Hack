# sasol_clv

Kedro project for the Zindi Sasol Customer Value Recruitment Challenge.

Predicts, per loyalty customer, next quarter fuel litres, next quarter non-fuel spend and an Opportunity label.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
kedro run
```

Put `train.csv`, `test.csv`, `SampleSubmission.csv` and `label_config.json` in `data/01_raw/` first.

The submission ends up in `data/07_model_output/submission.csv` and local scores in
`data/08_reporting/validation_metrics.json`.

See `docs/README.md` for the full guide: what the challenge wants, how the labels work and how to improve the score.

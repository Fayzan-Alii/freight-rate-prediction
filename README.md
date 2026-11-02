# Freight Rate Prediction

Predicts the price of a truck load from its route, trailer type, weight and date. The full write-up is in [report.pdf](report.pdf).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the assessment CSV files in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Run

```bash
python pipeline.py      # clean, test, train, write validation_predictions.csv and december_predictions.csv
python experiments.py   # tests behind each feature choice
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

## Files

- `eda.ipynb`: data exploration
- `pipeline.py`: the full pipeline
- `experiments.py`: feature tests
- `validation_predictions.csv`: the submission
- `report.pdf`: the report

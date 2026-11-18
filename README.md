# Telecom Churn Business Analytics Dashboard

A Python analysis of telecom customer churn. It reports the overall churn and retention rates, compares churn across partners, states, age groups, and salary bands, and generates ten charts.

The project started as a [Google Colab notebook](https://colab.research.google.com/drive/1g0vEJizD_HRBQrGDesET5o_xu0cSx3H0). The original notebook duplicated the analysis cells and expected a CSV at the fixed path `/content/telecom_churn.csv`. The cleaned notebook in `notebooks/` now uses the reusable analysis module in this repository, and the CSV location is supplied explicitly.

## Project files

- `telecom_churn_analysis.py` — CSV validation, preparation, summary tables, charts, and command-line entry point.
- `notebooks/telecom_churn_dashboard.ipynb` — a concise notebook for interactive use.
- `requirements.txt` — Python dependencies.
- `data/` — place your private or licensed dataset here. The dataset is not included in this repository.
- `reports/figures/` — generated chart images (ignored by Git).

## Requirements

- Python 3.9 or newer
- A telecom churn CSV with the columns listed below

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the analysis

Put the CSV at `data/telecom_churn.csv`, or provide its path directly:

```bash
python telecom_churn_analysis.py data/telecom_churn.csv --no-show
```

The command prints summary metrics and a partner benchmark, then saves ten PNG charts to `reports/figures/`. By default, charts are also displayed in interactive environments. Use `--output-dir` to choose another chart folder and `--as-of YYYY-MM-DD` to set a fixed date for tenure calculations.

```bash
python telecom_churn_analysis.py /path/to/telecom_churn.csv \
  --as-of 2026-01-01 \
  --output-dir reports/figures \
  --no-show
```

## Dataset columns

| Column | Expected content |
| --- | --- |
| `customer_id` | Customer identifier |
| `telecom_partner` | Telecom partner or provider name |
| `state` | Customer state or region |
| `date_of_registration` | Registration date parseable by pandas |
| `age` | Numeric age in years |
| `calls_made` | Numeric call count or call-use measure |
| `sms_sent` | Numeric SMS count or use measure |
| `data_used` | Numeric data-use measure |
| `estimated_salary` | Numeric estimated salary |
| `churn` | `0`/`1`, `false`/`true`, or `no`/`yes` |

Extra columns are retained. Missing or invalid dates are reported and omitted from tenure summaries. Invalid numeric values and unrecognized or missing churn labels stop the run with a clear error. The program does not impute customer data.

## Use the notebook

Open `notebooks/telecom_churn_dashboard.ipynb` in Jupyter or Google Colab with `telecom_churn_analysis.py` available on the Python path. Upload the analysis module and your CSV to the Colab session, or open the notebook from a checked-out copy of this repository. Set `DATA_PATH` in the notebook to the CSV location before running the analysis cell.

## Interpretation

The charts and rates describe the provided dataset; they do not establish why a customer churned or predict future churn. Recommendations are generated from observed group rates, and should be checked against sample sizes, data quality, and operational context before acting on them.

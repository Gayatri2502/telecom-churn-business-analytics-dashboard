#!/usr/bin/env python3
"""Analyze a telecom customer churn CSV and generate summary charts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger("telecom_churn")
REQUIRED_COLUMNS = {
    "customer_id",
    "telecom_partner",
    "state",
    "date_of_registration",
    "age",
    "calls_made",
    "sms_sent",
    "data_used",
    "estimated_salary",
    "churn",
}
NUMERIC_COLUMNS = (
    "age",
    "calls_made",
    "sms_sent",
    "data_used",
    "estimated_salary",
)
CHURN_VALUES = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "no": 0,
    "yes": 1,
    "not churned": 0,
    "churned": 1,
}
AGE_BINS = [-np.inf, 17, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["Under 18", "18–25", "26–35", "36–45", "46–55", "56–65", "66+"]
SALARY_LABELS = ["Low", "Medium", "High", "Very high"]


class DataValidationError(ValueError):
    """Raised when the input CSV cannot support the requested analysis."""


def _normalize_churn(values: pd.Series) -> pd.Series:
    """Convert common churn encodings to integers 0 and 1."""
    normalized = values.astype("string").str.strip().str.casefold()
    converted = normalized.map(CHURN_VALUES)
    invalid = values.notna() & converted.isna()
    if invalid.any():
        examples = values[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise DataValidationError(
            "Column 'churn' must contain 0/1, true/false, or yes/no values; "
            f"unrecognized values include {examples}."
        )
    if values.isna().any():
        raise DataValidationError(
            f"Column 'churn' contains {int(values.isna().sum())} missing value(s). "
            "Fill or remove them before analysis."
        )
    return converted.astype("int8")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Read and validate the CSV expected by this analysis."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Pass the CSV path as the first argument."
        )
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse CSV {path}: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise DataValidationError(f"CSV has duplicate column names: {duplicates}.")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DataValidationError(
            "Dataset is missing required column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise DataValidationError("Dataset contains a header but no customer rows.")

    frame["churn"] = _normalize_churn(frame["churn"])
    for column in NUMERIC_COLUMNS:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise DataValidationError(
                f"Column '{column}' contains non-numeric values such as {examples}."
            )
        frame[column] = converted

    frame["date_of_registration"] = pd.to_datetime(
        frame["date_of_registration"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    return frame


def prepare_data(
    frame: pd.DataFrame, as_of: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    """Add tenure and robust age/salary bands without mutating the input."""
    prepared = frame.copy()
    reference_date = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC").date())
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_convert(None)
    reference_date = reference_date.normalize()

    missing_dates = prepared["date_of_registration"].isna()
    if missing_dates.any():
        LOGGER.warning(
            "%s row(s) have invalid registration dates; tenure will be left blank.",
            int(missing_dates.sum()),
        )
    prepared["tenure_days"] = (reference_date - prepared["date_of_registration"]).dt.days
    future_dates = prepared["tenure_days"] < 0
    if future_dates.any():
        LOGGER.warning(
            "%s registration date(s) are after the analysis date; tenure will be left blank.",
            int(future_dates.sum()),
        )
        prepared.loc[future_dates, "tenure_days"] = np.nan

    prepared["age_group"] = pd.cut(
        prepared["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    salary = prepared["estimated_salary"]
    prepared["salary_group"] = pd.Series(pd.NA, index=prepared.index, dtype="object")
    valid_salary = salary.dropna()
    if valid_salary.nunique() == 1:
        prepared.loc[valid_salary.index, "salary_group"] = "All salaries"
    elif valid_salary.nunique() > 1:
        bands = pd.qcut(valid_salary, q=4, duplicates="drop")
        count = len(bands.cat.categories)
        labels = SALARY_LABELS[:count]
        label_map = dict(zip(bands.cat.categories, labels))
        prepared.loc[valid_salary.index, "salary_group"] = bands.map(label_map).astype("object")
    return prepared


def _churn_rate(series: pd.Series) -> float:
    """Return a churn percentage, or NaN when a group has no usable records."""
    clean = series.dropna()
    return float(clean.mean() * 100) if not clean.empty else np.nan


def _group_churn(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame.dropna(subset=[column])
        .groupby(column, observed=True)["churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Build business summaries and benchmark tables."""
    churn_rate = _churn_rate(frame["churn"])
    partner_report = (
        frame.dropna(subset=["telecom_partner"])
        .groupby("telecom_partner", observed=True)
        .agg(
            customers=("customer_id", "size"),
            avg_calls=("calls_made", "mean"),
            avg_sms=("sms_sent", "mean"),
            avg_data_used=("data_used", "mean"),
            avg_estimated_salary=("estimated_salary", "mean"),
            churn_rate=("churn", "mean"),
        )
    )
    if not partner_report.empty:
        partner_report["churn_rate"] *= 100
        partner_report = partner_report.sort_values("churn_rate", ascending=False)

    median_usage = frame["data_used"].median()
    usage_rates: dict[str, float] = {"High usage": np.nan, "Low usage": np.nan}
    if pd.notna(median_usage):
        high_usage = frame["data_used"] > median_usage
        low_usage = frame["data_used"].notna() & ~high_usage
        usage_rates["High usage"] = _churn_rate(frame.loc[high_usage, "churn"])
        usage_rates["Low usage"] = _churn_rate(frame.loc[low_usage, "churn"])

    return {
        "customer_count": len(frame),
        "churn_rate": churn_rate,
        "retention_rate": 100 - churn_rate,
        "partner_report": partner_report,
        "partner_churn": _group_churn(frame, "telecom_partner").sort_values(),
        "state_churn": _group_churn(frame, "state"),
        "age_churn": _group_churn(frame, "age_group"),
        "salary_churn": _group_churn(frame, "salary_group"),
        "usage_churn": usage_rates,
    }


def print_report(summary: dict[str, object]) -> None:
    """Print a concise executive summary and partner benchmark."""
    print("\n" + "=" * 60)
    print("TELECOM CHURN BUSINESS OVERVIEW")
    print("=" * 60)
    print(f"Total customers: {summary['customer_count']:,}")
    print(f"Churn rate:      {summary['churn_rate']:.2f}%")
    print(f"Retention rate:  {summary['retention_rate']:.2f}%")

    partner_report = summary["partner_report"]
    print("\nPartner benchmark (sorted by churn rate)")
    if partner_report.empty:
        print("No partner values are available.")
    else:
        display = partner_report.copy()
        display["churn_rate"] = display["churn_rate"].map(lambda value: f"{value:.2f}%")
        print(display.to_string())

    print("\nUsage segment churn")
    for segment, rate in summary["usage_churn"].items():
        print(f"{segment}: {rate:.2f}%" if pd.notna(rate) else f"{segment}: unavailable")

    recommendations: list[str] = []
    partner_churn = summary["partner_churn"]
    state_churn = summary["state_churn"]
    age_churn = summary["age_churn"]
    if not partner_churn.empty:
        recommendations.append(
            f"Review retention performance for {partner_churn.idxmax()}, "
            f"the partner with the highest observed churn rate ({partner_churn.max():.2f}%)."
        )
    if not state_churn.empty:
        recommendations.append(
            f"Investigate customer experience in {state_churn.index[0]}, "
            f"the state with the highest observed churn rate ({state_churn.iloc[0]:.2f}%)."
        )
    if not age_churn.empty:
        recommendations.append(
            f"Review the {age_churn.index[0]} segment, which has the highest observed age-group churn."
        )
    print("\nData-led follow-up")
    if recommendations:
        for number, recommendation in enumerate(recommendations, 1):
            print(f"{number}. {recommendation}")
    else:
        print("No segment recommendations can be calculated from the available data.")
    print("These group-level comparisons are descriptive and do not establish causation.")


def _save_or_show(fig: plt.Figure, filename: str, output_dir: Path | None, show: bool) -> None:
    fig.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def create_charts(
    frame: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Generate the dashboard charts; optionally save them to a directory."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=frame, x="churn", ax=ax)
    ax.set(title="Customer Churn Distribution", xlabel="Churn (0 = retained, 1 = churned)", ylabel="Customers")
    _save_or_show(fig, "01_churn_distribution.png", output_dir, show)

    partner_churn = summary["partner_churn"]
    fig, ax = plt.subplots(figsize=(10, 6))
    if partner_churn.empty:
        ax.text(0.5, 0.5, "No partner data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        partner_churn.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn Rate by Telecom Partner")
    _save_or_show(fig, "02_partner_churn.png", output_dir, show)

    state_churn = summary["state_churn"].head(15).sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    if state_churn.empty:
        ax.text(0.5, 0.5, "No state data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        state_churn.plot(kind="barh", ax=ax)
        ax.set_xlabel("Churn rate (%)")
    ax.set_title("15 States with Highest Churn")
    _save_or_show(fig, "03_state_churn.png", output_dir, show)

    fig, ax = plt.subplots(figsize=(9, 5))
    age_churn = summary["age_churn"]
    if age_churn.empty:
        ax.text(0.5, 0.5, "No age data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        age_churn.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn by Age Group")
    _save_or_show(fig, "04_age_churn.png", output_dir, show)

    for filename, column, title, ylabel in (
        ("05_data_usage_vs_churn.png", "data_used", "Data Usage vs Churn", "Data used"),
        ("06_calls_vs_churn.png", "calls_made", "Calls Made vs Churn", "Calls made"),
        ("07_sms_vs_churn.png", "sms_sent", "SMS Usage vs Churn", "SMS sent"),
        ("08_tenure_vs_churn.png", "tenure_days", "Customer Tenure vs Churn", "Tenure (days)"),
    ):
        chart_data = frame[["churn", column]].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        if chart_data.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()
        else:
            sns.boxplot(data=chart_data, x="churn", y=column, ax=ax)
            ax.set_xlabel("Churn (0 = retained, 1 = churned)")
            ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save_or_show(fig, filename, output_dir, show)

    salary_churn = summary["salary_churn"]
    fig, ax = plt.subplots(figsize=(8, 5))
    if salary_churn.empty:
        ax.text(0.5, 0.5, "No salary data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        salary_churn.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn by Estimated Salary Group")
    _save_or_show(fig, "09_salary_churn.png", output_dir, show)

    numeric = frame.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(12, 8))
    if numeric.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix")
    _save_or_show(fig, "10_correlation_matrix.png", output_dir, show)


def run_analysis(
    frame: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    output_dir: Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Prepare, summarize, and chart a validated customer dataframe."""
    prepared = prepare_data(frame, as_of=as_of)
    summary = summarize(prepared)
    print_report(summary)
    create_charts(prepared, summary, output_dir=output_dir, show=show)
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize telecom churn data and create dashboard charts."
    )
    parser.add_argument("csv_path", help="Path to the telecom churn CSV file")
    parser.add_argument(
        "--as-of",
        help="Reference date for tenure calculations (YYYY-MM-DD; defaults to today)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for generated PNG charts (default: reports/figures)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save charts without opening interactive windows",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        frame = load_dataset(args.csv_path)
        run_analysis(
            frame,
            as_of=args.as_of,
            output_dir=args.output_dir,
            show=not args.no_show,
        )
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

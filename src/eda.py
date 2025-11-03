"""Exploratory data analysis helpers."""

from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Collect high-level metadata about the dataset."""

    summary = {
        "shape": df.shape,
        "head": df.head(),
    }
    if "Category Name" in df.columns:
        summary["category_count"] = df["Category Name"].nunique()
    return summary


def top_missing_columns(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return the top-n columns with the most missing values."""

    return df.isna().sum().sort_values(ascending=False).head(top_n)


def missing_counts(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Count missing values for the provided column subset."""

    selected = [col for col in columns if col in df.columns]
    if not selected:
        return pd.Series(dtype="int64")
    return df[selected].isna().sum().sort_values(ascending=False)


def calories_statistics(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for the Calories feature."""

    stats = df["Calories"].describe()[["mean", "50%", "std"]]
    stats.index = ["mean", "median", "std"]
    return stats.to_dict()


def interpret_calorie_skew(mean_value: float, median_value: float) -> str:
    """Interpret skewness based on mean and median comparison."""

    if mean_value > median_value:
        return "right-skewed"
    if mean_value < median_value:
        return "left-skewed"
    return "approximately symmetric"


def top_protein_categories(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """Identify category names with the highest average protein content."""

    if "Category Name" not in df.columns:
        return []
    protein_means = (
        df.groupby("Category Name")["Protein"].mean().sort_values(ascending=False)
    )
    return protein_means.head(top_n).index.tolist()


def macro_correlation(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    """Compute correlation matrix and the strongest Calories correlate."""

    matrix = df[list(columns)].corr()
    strongest = (
        matrix["Calories"].drop("Calories").abs().sort_values(ascending=False).index[0]
    )
    return matrix, strongest

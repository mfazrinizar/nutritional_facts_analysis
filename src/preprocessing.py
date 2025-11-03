"""Data preparation helpers."""

from typing import Iterable, Tuple

import pandas as pd


NON_NUMERIC_COLUMNS = {"Food Name", "Category Name"}
DEFAULT_IMPUTE_COLUMNS = ("Vitamin D", "Trans Fat")


def coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Convert eligible features to numeric dtype in-place."""

	numeric_candidates = df.columns.difference(list(NON_NUMERIC_COLUMNS))
	df[numeric_candidates] = df[numeric_candidates].apply(pd.to_numeric, errors="coerce")
	return df


def impute_with_median(
	df: pd.DataFrame,
	columns: Iterable[str] = DEFAULT_IMPUTE_COLUMNS,
) -> Tuple[pd.DataFrame, dict]:
	"""Fill missing values using the median for selected columns."""

	imputation_summary = {}
	for column in columns:
		if column in df.columns:
			median_value = df[column].median()
			df[column].fillna(median_value, inplace=True)
			imputation_summary[column] = median_value
	return df, imputation_summary


"""Utilities for retrieving datasets and managing project paths."""

from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter


DATASET_ID = "beridzeg45/food-nutritional-facts"
DATASET_FILENAME = "foodstruct_nutritional_facts.csv"


def load_food_dataset(file_path: str = DATASET_FILENAME):
	"""Fetch the food nutrition dataset using KaggleHub."""

	return kagglehub.load_dataset(
		KaggleDatasetAdapter.PANDAS,
		DATASET_ID,
		file_path,
	)


def ensure_output_directory(directory: str = "outputs") -> Path:
	"""Create the directory for generated artefacts if needed."""

	path = Path(directory)
	path.mkdir(parents=True, exist_ok=True)
	return path


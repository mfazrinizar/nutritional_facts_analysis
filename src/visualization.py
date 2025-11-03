"""Plotting utilities for the nutrition analysis."""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_protein_boxplot(
    df: pd.DataFrame,
    categories: Iterable[str],
    output_dir: Path,
    filename: str = "protein_boxplot.png",
) -> Optional[Path]:
    """Generate and display the boxplot for top protein categories."""

    categories = list(categories)
    if not categories or "Category Name" not in df.columns:
        return None

    subset = df[df["Category Name"].isin(categories)].copy()
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=subset,
        x="Category Name",
        y="Protein",
        order=categories,
        palette="viridis",
    )
    plt.title("Protein Distribution for Top Categories")
    plt.xlabel("Category")
    plt.ylabel("Protein (g)")
    plt.xticks(rotation=30, ha="right")
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
    return path


def plot_macro_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_dir: Path,
    filename: str = "macro_correlation_heatmap.png",
) -> Path:
    """Render and display the macronutrient correlation heatmap."""

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Macronutrient Correlation Heatmap")
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
    return path


def plot_cluster_scatter(
    scaled_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "kmeans_clusters.png",
) -> Path:
    """Display cluster assignments on the standardised feature space."""

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=scaled_df,
        x="Carbs_scaled",
        y="Protein_scaled",
        hue="Cluster",
        palette="tab10",
        alpha=0.7,
        edgecolor="white",
        s=60,
    )
    plt.title("K-Means Clusters on Standardised Features")
    plt.xlabel("Carbohydrates (scaled)")
    plt.ylabel("Protein (scaled)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
    return path

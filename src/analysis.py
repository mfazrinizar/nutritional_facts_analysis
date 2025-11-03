"""Analytical routines for simulation and clustering."""

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def simulate_sodium_intake(
    iterations: int = 10_000,
    mean: float = 2000.0,
    std: float = 300.0,
    threshold: float = 2300.0,
    seed: int = 42,
) -> float:
    """Estimate the probability that sodium intake exceeds the threshold."""

    rng = np.random.default_rng(seed=seed)
    samples = rng.normal(loc=mean, scale=std, size=iterations)
    return float(np.mean(samples > threshold))


def cluster_macronutrients(
    df: pd.DataFrame,
    features: Sequence[str] = ("Carbs", "Protein"),
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Run K-Means clustering on selected macronutrient features."""

    feature_columns = [feature for feature in features if feature in df.columns]
    if len(feature_columns) < len(features):
        raise ValueError("Not all requested features are present in the DataFrame.")

    feature_df = df[feature_columns].dropna().copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{feature}_scaled" for feature in feature_columns],
        index=feature_df.index,
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(scaled_df)

    clustered_df = feature_df.copy()
    clustered_df["Cluster"] = labels
    scaled_df["Cluster"] = labels

    cluster_profiles = clustered_df.groupby("Cluster")[feature_columns].mean()
    cluster_counts = clustered_df["Cluster"].value_counts().sort_index()

    # Define the "protein rich, lower carb" cluster assuming the last feature is Protein.
    sort_order = [feature_columns[-1], feature_columns[0]]
    target_cluster = (
        cluster_profiles.sort_values(sort_order, ascending=[False, True]).index[0]
    )

    return {
        "clustered": clustered_df,
        "scaled": scaled_df,
        "profiles": cluster_profiles,
        "counts": cluster_counts,
        "target_cluster": int(target_cluster),
    }

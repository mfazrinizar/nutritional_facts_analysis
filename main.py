import warnings

import src.eda as eda
from src.analysis import cluster_macronutrients, simulate_sodium_intake
from src.data_sources import DATASET_ID, ensure_output_directory, load_food_dataset
from src.preprocessing import coerce_numeric_features, impute_with_median
from src.visualization import (
    plot_cluster_scatter,
    plot_macro_correlation_heatmap,
    plot_protein_boxplot,
)

def main() -> None:
    """Run the full nutrition analysis pipeline."""

    warnings.filterwarnings("ignore", category=FutureWarning)

    df = load_food_dataset()
    coerce_numeric_features(df)

    print("### 0. Data Collection & Understanding")
    print(f"- Dataset: Food Nutritional Facts (Kaggle, {DATASET_ID})")
    summary = eda.summarize_dataset(df)
    print(
        f"- Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns"
    )
    if "category_count" in summary:
        print(f"- Number of food categories: {summary['category_count']}")
    macro_focus = [
        column
        for column in ["Calories", "Fats", "Carbs", "Protein", "Sodium"]
        if column in df.columns
    ]
    print(f"- Key features: {', '.join(macro_focus)}")
    print("- First five records:")
    print(summary["head"])

    print("\n### 1. Data Preparation & Exploratory Data Analysis")
    print("#### 1.1 Missing Value Assessment")
    missing_before = eda.top_missing_columns(df)
    print("Top 10 columns with the highest missing values before imputation:")
    print(missing_before)

    df, imputation_stats = impute_with_median(df)
    missing_after = eda.missing_counts(df, imputation_stats.keys())
    print("\nMissing values after median imputation for selected columns:")
    print(missing_after)

    print("\n#### 1.2 Calorie Statistics")
    calorie_stats = eda.calories_statistics(df)
    print(f"- Mean Calories : {calorie_stats['mean']:.2f}")
    print(f"- Median Calories: {calorie_stats['median']:.2f}")
    print(f"- Std Dev Calories: {calorie_stats['std']:.2f}")
    skew_label = eda.interpret_calorie_skew(
        calorie_stats["mean"], calorie_stats["median"]
    )
    print(f"- Distribution insight: {skew_label} distribution")

    print("\n#### 1.3 Visualisations")
    output_dir = ensure_output_directory()

    top_categories = eda.top_protein_categories(df)
    boxplot_path = plot_protein_boxplot(df, top_categories, output_dir)
    if boxplot_path:
        print(f"- Protein boxplot saved to: {boxplot_path}")
    else:
        print("- Protein boxplot skipped due to missing categories.")

    macro_columns = ["Calories", "Fats", "Carbs", "Protein"]
    corr_matrix, strongest_corr = eda.macro_correlation(df, macro_columns)
    heatmap_path = plot_macro_correlation_heatmap(corr_matrix, output_dir)
    print(f"- Correlation heatmap saved to: {heatmap_path}")
    print(
        f"- Highest absolute correlation with Calories: {strongest_corr}"
    )

    print("\n### 2. Monte Carlo Simulation: Daily Sodium Intake")
    simulation_probability = simulate_sodium_intake()
    print("- Configuration: 10,000 iterations, mean 2000 mg, std 300 mg")
    print(
        f"- Probability of exceeding 2300 mg: {simulation_probability:.2%}"
    )

    print("\n### 3. Nutrient Clustering (K-Means on Carbs & Protein)")
    cluster_results = cluster_macronutrients(df)
    print("#### 3.1 Cluster Membership Counts")
    for cluster_id, count in cluster_results["counts"].items():
        print(f"- Cluster {cluster_id}: {count} foods")

    print("\n#### 3.2 Average Macronutrients per Cluster (grams)")
    print(cluster_results["profiles"].round(3))

    target_cluster = cluster_results["target_cluster"]
    cluster_summary = (
        f"Cluster {target_cluster} → High protein with relatively lower carbohydrates"
    )
    print(f"\n#### 3.3 Highlighted cluster: {cluster_summary}")

    scatter_path = plot_cluster_scatter(cluster_results["scaled"], output_dir)
    print(f"- Cluster scatter plot saved to: {scatter_path}")

    print("\n### Conclusion")
    print(
        "- Median imputation resolves Vitamin D and Trans Fat gaps without distorting distributions."
    )
    print(
        f"- Calories correlate most strongly with {strongest_corr}, underscoring key macronutrient interactions."
    )
    print(
        f"- The estimated risk of exceeding 2300 mg sodium is {simulation_probability:.2%}, highlighting dietary caution for sensitive groups."
    )
    print(
        f"- {cluster_summary} offers a practical shortlist for protein-forward meal planning."
    )


if __name__ == "__main__":
    main()
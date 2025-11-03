# Nutritional Facts Analysis

This project explores the Food Nutritional Facts dataset from Kaggle using a notebook-style Python pipeline. The workflow covers data acquisition with KaggleHub, cleaning, exploratory analysis, Monte Carlo simulation, and K-Means clustering to highlight protein-rich meal clusters. Generated plots are stored in the `outputs/` directory and displayed during execution to mirror a Google Colab experience.

## Project Structure

- `main.py` – orchestrates the full analysis in a linear, notebook-like flow.
- `src/` – reusable modules for data loading, preprocessing, exploratory analysis, visualisation, and analytical routines.
- `outputs/` – generated figures such as the protein boxplot, macronutrient correlation heatmap, and cluster scatter plot.
- `requirements.txt` – Python dependencies needed to run the project.
- `run.sh` / `run.bat` – convenience scripts for Linux/macOS and Windows.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Authenticate Kaggle (if required)**
   KaggleHub may prompt for authentication the first time the dataset (`beridzeg45/food-nutritional-facts`) is downloaded. Follow the on-screen instructions to provide your Kaggle API token.

3. **Run the analysis**
   - Linux/macOS:
     ```bash
     ./run.sh
     ```
   - Windows:
     ```bat
     run.bat
     ```

The script prints notebook-style commentary and saves artefacts to `outputs/` while also calling `plt.show()` for immediate figure previews.

## Analytical Highlights

1. **Data Understanding** – dataset summary, feature list, and head preview.
2. **Data Preparation** – numeric coercion and median imputation for Vitamin D and Trans Fat.
3. **Exploratory Analysis** – missing value audit, calorie distribution statistics, and macronutrient correlations.
4. **Visualisations** – protein distribution across top categories, correlation heatmap, and cluster scatter plot.
5. **Monte Carlo Simulation** – sodium intake risk estimation across 10,000 scenarios.
6. **Clustering** – K-Means segmentation on carbohydrates and protein to flag protein-forward, lower-carbohydrate meal options.

## Notes

- The analysis is intentionally linear to emulate a Google Colab notebook while remaining modular under `src/` for maintainability.
- Re-running the pipeline refreshes the figures in `outputs/` and reprints all commentary, making it easy to trace updates.

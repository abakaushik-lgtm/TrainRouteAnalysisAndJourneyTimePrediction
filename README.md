# Train Route Analysis and Journey Time Prediction

## Objective
Building an end-to-end data pipeline to analyze train routes and predict journey durations using Linear Regression.

## Tech Stack
* **Language & Framework**: Python, FastAPI
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Modeling & Inference**: Scikit-learn, Joblib


## System Architecture

The project is structured modularly to avoid duplication, enforce clean software engineering principles, and expose reusable APIs for all 6 levels:

```
Train Route Analysis and Journey Time Prediction/
│
├── Dataset1.csv                   # Raw train route dataset
├── train_modeling_df.csv          # Cleaned, train-level aggregated dataset (Output of Level 2)
│
├── src/                           # Shared modules (Source Package)
│   ├── __init__.py
│   ├── data_loader.py             # Data loading, parsing, cleaning and feature engineering
│   ├── model.py                   # Model training, evaluation and persistence
│   └── pipeline.py                # Reusable end-to-end integration pipeline
│
├── level1_data_overview.py        # Ingests raw data & audits structural inconsistencies
├── level2_data_cleaning.py        # Cleans dataset, tracks day boundaries & aggregates features
├── level3_data_exploration.py     # Performs traffic and correlation analyses
├── level4_visualization.py        # Generates boxplots, histograms and regression plots
├── level5_model_development.py    # Trains LinearRegression, evaluates, & saves the model
├── level6_final_integration.py    # Runs end-to-end pipeline, plots accuracy & prints LinkedIn summaries
│
├── reports/                       # Visualizations and model storage
│   ├── duration_distribution.png  # Journey duration distribution plot (Level 3)
│   ├── correlation_heatmap.png    # Correlation heatmap plot (Level 3)
│   ├── duration_boxplot.png       # Boxplot of major route durations (Level 4)
│   ├── station_traffic_dist.png   # Distribution plot of station traffic (Level 4)
│   ├── distance_vs_duration.png   # Scatter plot with trendline (Level 4)
│   ├── journey_time_model.joblib  # Trained model serialized (Level 5)
│   └── actual_vs_predicted.png    # Actual vs. Predicted scatter plot (Level 6)
│
└── requirements.txt               # Project dependencies
```

---

## 🚀 How to Run the Project

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution of Levels
Each level is designed to run independently, or you can run them sequentially:

* **Level 1: Data Overview**
  ```bash
  python level1_data_overview.py
  ```
  *Prints shape, datatypes, train endpoints, and programmatically audits structural inconsistencies (gaps in sequence numbers, circular routes, truncated route endpoints).*

* **Level 2: Data Cleaning & Feature Engineering**
  ```bash
  python level2_data_cleaning.py
  ```
  *Removes duplicates, tracks day-boundary wrap-around events using time minute offset increments, aggregates metrics per train route, and outputs `train_modeling_df.csv`.*

* **Level 3: Data Exploration (EDA)**
  ```bash
  python level3_data_exploration.py
  ```
  *Performs statistical analysis, computes highest/lowest traffic stations, calculates the Pearson correlation matrix, and saves `reports/duration_distribution.png` and `reports/correlation_heatmap.png`.*

* **Level 4: Visualization & Pattern Analysis**
  ```bash
  python level4_visualization.py
  ```
  *Generates comparative boxplots of major routes, log-scaled station traffic histograms, and distance vs. duration regression scatter plots under `reports/`.*

* **Level 5: Prediction Model Development**
  ```bash
  python level5_model_development.py
  ```
  *Splits dataset 80/20, trains scikit-learn OLS Linear Regression, calculates test metrics (MAE, RMSE, R²), and serializes the model to `reports/journey_time_model.joblib`.*

* **Level 6: Final Integration & Prediction System**
  ```bash
  python level6_final_integration.py
  ```
  *Runs the consolidated pipeline, plots actual vs. predicted values alongside y = x, and prints a structured console log summary ready for a LinkedIn video walkthrough.*

---

## 📈 Key Insights & Results

### 1. Structural Inconsistencies Audited
* **Sequence Gaps**: 100 stations skipped in raw sequences (e.g. SN jumps from 18 to 21).
* **Truncated Routes**: Train `12978` (Route 1) is missing stops 1-25; it starts at `SN 26` and `Distance 2277`.
* **Circular Loops**: 30 repeating stations in routes (e.g. Train 290 at station `DSJ` at SN 1 and SN 14).

### 2. Time Engineering Solution
By parsing time objects to cumulative minutes and tracking the decreases in HH:MM:SS values sequentially, the system successfully resolved day-boundary crossings. No negative travel times were generated, ensuring a robust feature engineering layer.

### 3. Model Parameters & Interpretation
* **Intercept**: `-30.2819 minutes` (OLS baseline adjustment).
* **Total_Distance Coefficient**: `1.0370` (For every 1 km, travel duration increases by ~1.04 minutes, corresponding to an average train running speed of **57.8 km/h**).
* **Number_of_Stops Coefficient**: `6.0926` (Every intermediate stop adds **6.1 minutes** of dwell, deceleration, and acceleration latency).
* **Performance**:
  * **Mean Absolute Error (MAE)**: `59.11 minutes`
  * **Root Mean Squared Error (RMSE)**: `158.56 minutes`
  * **R² Score**: `0.9437` (The model explains **94.4%** of the variance in journey durations).

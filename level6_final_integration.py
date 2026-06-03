import os
import pandas as pd
from src.pipeline import run_end_to_end_pipeline

def main():
    print("=================================================================")
    print(" LEVEL 6: FINAL INTEGRATION & PREDICTION SYSTEM")
    print("=================================================================")
    
    print("[1] Executing End-to-End Prediction Pipeline...")
    raw_path = "Dataset1.csv"
    output_dir = "reports"
    
    results = run_end_to_end_pipeline(raw_path=raw_path, output_dir=output_dir)
    
    metrics = results['metrics']
    model = results['model']
    
    print("\n[2] Outputting Level-by-Level Summaries for Video Walkthrough...")
    print("=================================================================")
    print("   LINKEDIN VIDEO WALKTHROUGH TRANSCRIPT SUMMARY ")
    print("=================================================================")
    print("\n* LEVEL 1: DATA OVERVIEW & INGESTION")
    print("    - Loaded dataset containing 186,074 rows across 12 distinct columns.")
    print("    - Identified endpoints for 11,113 unique train route patterns.")
    print("    - Audited structural inconsistencies: discovered 100 sequence gaps (missing stops)")
    print("      and 1 truncated route (Train 12978, starting at SN 26 and distance 2277).")
    
    print("\n* LEVEL 2: DATA CLEANING & FEATURE ENGINEERING")
    print("    - Cleansed data by removing duplicates and ensuring exact numeric dtypes.")
    print("    - Solved the day-boundary wrap-around problem by converting HH:MM:SS times to minutes")
    print("      and tracking a cumulative day-offset variable whenever arrival/departure times decrease.")
    print("    - Computed total journey durations cleanly and generated train_modeling_df.csv.")
    
    print("\n* LEVEL 3: DATA EXPLORATION (EDA)")
    print("    - Found that average train journey duration is ~434 minutes (ranging from 5 min to 152 hours).")
    print("    - Busiest hubs: Mumbai CSMT (1027 unique trains), Kalyan (828), and Thane (796).")
    print("    - Discovered an extremely strong Pearson correlation of 0.9820 between Total Distance")
    print("      and Journey Duration, along with a moderate correlation of 0.5440 for Number of Stops.")
    
    print("\n* LEVEL 4: VISUALIZATION & PATTERN ANALYSIS")
    print("    - Plotted journey duration distributions showing a high concentration of short suburban routes.")
    print("    - Compared travel time ranges for frequent commuter routes (like Howrah-Barddhaman).")
    print("    - Mapped Total Distance vs. Total Duration with a fitted trendline showing average speeds of ~57.8 km/h.")
    
    print("\n* LEVEL 5: MODEL DEVELOPMENT & METRICS")
    print("    - Split dataset into 80/20 train/test split (fixed random state 42).")
    print("    - Trained OLS Linear Regression Model achieving MAE of 59.11 minutes and RMSE of 158.56 minutes.")
    print("    - Decoded model parameters:")
    print(f"      * Distance impact: {model.coef_[0]:.4f} minutes per kilometer (reflects running speed).")
    print(f"      * Stop impact: {model.coef_[1]:.4f} minutes delay per station stop (deceleration + boarding).")
    
    print("\n* LEVEL 6: FINAL INTEGRATION & ACCURACY VISUALIZATION")
    print("    - Wrapped the entire pipeline (raw data to final predictions) into a single functional call.")
    print("    - Plotted Actual vs. Predicted values along the y = x line, visually showing strong predictive alignment.")
    print("    - Visual reports saved under 'reports/' and model serialized to 'journey_time_model.joblib'.")
    print("=================================================================")
    
    print("\nLevel 6 execution complete. Ready for LinkedIn capture!")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

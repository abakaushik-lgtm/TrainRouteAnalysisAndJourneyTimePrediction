import pandas as pd
from src.data_loader import load_raw_dataset, clean_and_engineer_data, aggregate_to_train_level

def main():
    print("=================================================================")
    print(" LEVEL 2: DATA CLEANING & FEATURE ENGINEERING")
    print("=================================================================")
    
    print("[1] Loading Raw Dataset...")
    df_raw = load_raw_dataset("Dataset1.csv")
    print(f"    Raw records loaded: {len(df_raw)}")
    
    print("\n[2] Cleaning & Preprocessing (Handling Day Boundaries)...")
    # This resolves day wrap-around by converting times and tracking day offset
    df_processed = clean_and_engineer_data(df_raw)
    print(f"    Processed station-level records: {len(df_processed)}")
    print("    Sample preprocessed data (with absolute elapsed time):")
    print(df_processed[['Train_No', 'Route_Number', 'SN', 'Station_Code', 'Arrival_time', 
                        'Departure_Time', 'Abs_Arrival_Minutes', 'Abs_Departure_Minutes']].head(10).to_string(index=False))
    
    print("\n[3] Aggregating Train-Level Metrics & Filtering Truncated Routes...")
    # Computes total distance, stops, and duration for each train.
    # Excludes truncated routes (like Train 12978).
    df_modeling = aggregate_to_train_level(df_processed)
    print(f"    Created Modeling Dataframe with {len(df_modeling)} train routes.")
    
    print("\n[4] Saving Aggregated Modeling Dataset...")
    output_path = "train_modeling_df.csv"
    df_modeling.to_csv(output_path, index=False)
    print(f"    Successfully saved modeling dataset to: {output_path}")
    
    print("\nModeling DataFrame Info & Summary Statistics:")
    print(df_modeling.info())
    print("\nDescriptive statistics for engineered features:")
    print(df_modeling.describe())
    
    print("\nLevel 2 execution complete.")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

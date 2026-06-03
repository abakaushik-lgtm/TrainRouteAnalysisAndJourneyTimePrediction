import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import train_linear_regression, evaluate_regression_model, save_model

def main():
    print("=================================================================")
    print(" LEVEL 5: PREDICTION MODEL DEVELOPMENT")
    print("=================================================================")
    
    # Check modeling data
    modeling_path = "train_modeling_df.csv"
    if not os.path.exists(modeling_path):
        raise FileNotFoundError(f"Please run level2_data_cleaning.py first to generate {modeling_path}")
        
    df_modeling = pd.read_csv(modeling_path)
    
    # 1. Feature Extraction
    print("[1] Extracting Features & Target...")
    X = df_modeling[['Total_Distance', 'Number_of_Stops']]
    y = df_modeling['Total_Duration_Minutes']
    print(f"    Features: {list(X.columns)}")
    print(f"    Target: {y.name}")
    
    # 2. Train/Test Split (80/20, random_state=42)
    print("\n[2] Splitting Dataset into Train and Test subsets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Training records: {X_train.shape[0]}")
    print(f"    Testing records: {X_test.shape[0]}")
    
    # 3. Model Training
    print("\n[3] Training OLS Linear Regression Model...")
    model = train_linear_regression(X_train, y_train)
    
    # 4. Model Evaluation
    print("\n[4] Evaluating Model Performance on Test Set...")
    metrics = evaluate_regression_model(model, X_test, y_test)
    
    print("\nEvaluation Performance Metrics:")
    print(f"    Mean Absolute Error (MAE) : {metrics['mae']:.4f} minutes")
    print(f"    Root Mean Squared Error (RMSE): {metrics['rmse']:.4f} minutes")
    print(f"    Coefficient of Determination (R2): {metrics['r2']:.4f}")
    
    print("\nModel Parameters & Coefficients:")
    print(f"    Intercept (baseline constant): {metrics['intercept']:.4f}")
    print(f"    Coefficient for Total_Distance (minutes/km): {metrics['coef'][0]:.4f}")
    print(f"    Coefficient for Number_of_Stops (minutes/stop): {metrics['coef'][1]:.4f}")
    
    print("\n[5] Parameter Interpretation:")
    print("    - A coefficient of ~1.037 for Total_Distance means that for every 1 km of distance,")
    print("      the train duration increases by about 1.04 minutes (reflects an average running speed of ~57.8 km/h).")
    print("    - A coefficient of ~6.093 for Number_of_Stops implies that each station stop adds")
    print("      approximately 6.1 minutes of journey duration (including deceleration, dwell time, and acceleration).")
    print("    - The intercept of ~ -30.28 minutes acts as an OLS adjustment factor to fit the line shape.")
    
    # 5. Save the trained model
    print("\n[6] Serializing and Saving Trained Model...")
    save_model(model, "reports/journey_time_model.joblib")
    
    print("\nLevel 5 execution complete.")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

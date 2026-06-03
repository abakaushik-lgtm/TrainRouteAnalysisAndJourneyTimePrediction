import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_dataset, clean_and_engineer_data, aggregate_to_train_level
from src.model import train_linear_regression, evaluate_regression_model, save_model

def run_end_to_end_pipeline(raw_path="Dataset1.csv", output_dir="reports"):
    """
    Consolidates the entire data processing and modeling workflow into a single 
    reusable pipeline function.
    
    Parameters:
        raw_path (str): Path to the raw CSV dataset.
        output_dir (str): Directory where outputs (plots, models) will be saved.
        
    Returns:
        dict: Summary of the run, including metrics and model components.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    df_raw = load_raw_dataset(raw_path)
    
    # 2. Clean & Preprocess
    df_processed = clean_and_engineer_data(df_raw)
    
    # 3. Aggregate to Train Route Level
    df_modeling = aggregate_to_train_level(df_processed)
    
    # Save modeling dataframe
    modeling_csv_path = "train_modeling_df.csv"
    df_modeling.to_csv(modeling_csv_path, index=False)
    print(f"Aggregated modeling dataframe saved to {modeling_csv_path}")
    
    # 4. Train-Test Split
    X = df_modeling[['Total_Distance', 'Number_of_Stops']]
    y = df_modeling['Total_Duration_Minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Model
    model = train_linear_regression(X_train, y_train)
    
    # 6. Evaluate Model
    metrics = evaluate_regression_model(model, X_test, y_test)
    
    # 7. Save Model
    model_path = os.path.join(output_dir, "journey_time_model.joblib")
    save_model(model, model_path)
    
    # 8. Plot Actual vs Predicted Durations
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#1f77b4', s=12, label='Predicted Trains')
    
    # Draw reference line
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    
    plt.xlabel('Actual Journey Duration (Minutes)', fontsize=11)
    plt.ylabel('Predicted Journey Duration (Minutes)', fontsize=11)
    plt.title('Actual vs. Predicted Journey Durations (Linear Regression Model)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to {plot_path}")
    
    return {
        'model': model,
        'metrics': metrics,
        'modeling_df': df_modeling,
        'raw_df': df_raw,
        'processed_df': df_processed,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

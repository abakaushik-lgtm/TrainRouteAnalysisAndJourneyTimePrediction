import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_raw_dataset

def main():
    print("=================================================================")
    print(" LEVEL 3: DATA EXPLORATION (EDA)")
    print("=================================================================")
    
    os.makedirs("reports", exist_ok=True)
    
    # Load modeling data
    modeling_path = "train_modeling_df.csv"
    if not os.path.exists(modeling_path):
        raise FileNotFoundError(f"Please run level2_data_cleaning.py first to generate {modeling_path}")
        
    df_modeling = pd.read_csv(modeling_path)
    
    # 1. Compare and Aggregate Journey Durations
    print("[1] Analyzing Train Journey Durations...")
    print(df_modeling['Total_Duration_Minutes'].describe())
    
    # Save a distribution plot of durations
    plt.figure(figsize=(8, 5))
    sns.histplot(df_modeling['Total_Duration_Minutes'], kde=True, color='#2ca02c', bins=30)
    plt.xlabel('Total Journey Duration (Minutes)', fontsize=11)
    plt.ylabel('Count / Density', fontsize=11)
    plt.title('Distribution of Train Journey Durations', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    dist_plot_path = "reports/duration_distribution.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"    Journey duration distribution plot saved to: {dist_plot_path}")
    
    # 2. Station Traffic Analysis (Unique train visits per station)
    print("\n[2] Performing Station Traffic Analysis...")
    df_raw = load_raw_dataset("Dataset1.csv")
    station_traffic = df_raw.groupby(['Station_Code', 'Station_Name'])['Train_No'].nunique().reset_index()
    station_traffic.rename(columns={'Train_No': 'Unique_Train_Visits'}, inplace=True)
    
    top5 = station_traffic.sort_values(by='Unique_Train_Visits', ascending=False).head(5)
    bottom5 = station_traffic.sort_values(by='Unique_Train_Visits', ascending=True).head(5)
    
    print("\nTop 5 Highest-Traffic Stations (Busiest Junctions):")
    print(top5.to_string(index=False))
    
    print("\nBottom 5 Lowest-Traffic Stations (Quiet Stops):")
    print(bottom5.to_string(index=False))
    
    # 3. Correlation Analysis
    print("\n[3] Conducting Correlation Analysis...")
    corr_cols = ['Total_Distance', 'Number_of_Stops', 'Total_Duration_Minutes']
    corr_matrix = df_modeling[corr_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.to_string())
    
    # Generate Heatmap of Correlation Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", vmin=-1, vmax=1, square=True, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 11, "weight": "bold"})
    plt.title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    heatmap_path = "reports/correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"\n    Correlation matrix heatmap saved to: {heatmap_path}")
    
    print("\nLevel 3 execution complete.")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

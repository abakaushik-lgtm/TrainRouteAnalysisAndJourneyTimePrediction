import pandas as pd
from src.data_loader import load_raw_dataset

def main():
    print("=================================================================")
    print(" LEVEL 1: DATA OVERVIEW")
    print("=================================================================")
    
    # 1. Load raw dataset and display summary
    print("[1] Loading Dataset...")
    df = load_raw_dataset("Dataset1.csv")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nData Types and Missing Values:")
    print(df.dtypes)
    print("\nNull count per column:")
    print(df.isnull().sum())
    
    # 2. Train Route Start & End Station Summary Table
    print("\n[2] Identifying Train Route Endpoints (Start & Destination)...")
    # Group by Train_No and Route_Number to handle multiple route paths
    df_sorted = df.sort_values(by=['Train_No', 'Route_Number', 'SN'])
    
    # Find start station (earliest SN / min distance) and end station (latest SN / max distance)
    endpoints = []
    for (t_no, r_no), group in df_sorted.groupby(['Train_No', 'Route_Number']):
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        endpoints.append({
            'Train_No': t_no,
            'Route_Number': r_no,
            'Start_Station': first_row['Station_Name'],
            'Start_Code': first_row['Station_Code'],
            'End_Station': last_row['Station_Name'],
            'End_Code': last_row['Station_Code'],
            'Min_Distance': first_row['Distance'],
            'Max_Distance': last_row['Distance']
        })
    df_endpoints = pd.DataFrame(endpoints)
    print("\nSample Train Endpoints Summary Table:")
    print(df_endpoints.head(10).to_string(index=False))
    
    # 3. Descriptive Statistics for Distance and Stops count
    print("\n[3] Calculating Distance & Route Stops Statistics...")
    print("\nDescriptive Statistics for Cumulative Distance:")
    print(df['Distance'].describe())
    
    stops_per_train = df.groupby(['Train_No', 'Route_Number']).size().reset_index(name='Total_Stops')
    print("\nDescriptive Statistics for Number of Stops per Train Route:")
    print(stops_per_train['Total_Stops'].describe())
    
    # 4. Programmatic Structural Inconsistency Auditing
    print("\n[4] Programmatically Auditing Structural Inconsistencies...")
    
    # Check for gaps in sequence number (SN)
    df_sorted['expected_SN'] = df_sorted.groupby(['Train_No', 'Route_Number']).cumcount() + 1
    sn_mismatches = df_sorted[df_sorted['SN'] != df_sorted['expected_SN']]
    print(f" -> SN Sequence Gaps (missing intermediate stops): {len(sn_mismatches)} rows affected.")
    if len(sn_mismatches) > 0:
        print("    Sample mismatch train routes:")
        print(sn_mismatches[['Train_No', 'Route_Number', 'SN', 'expected_SN', 'Station_Code']].head(5).to_string(index=False))
        
    # Check for truncated routes (missing starting stations, first distance is not 0)
    train_min_dist = df.groupby(['Train_No', 'Route_Number'])['Distance'].min().reset_index()
    no_zero_start = train_min_dist[train_min_dist['Distance'] != 0]
    print(f" -> Truncated Route Gaps (first stop distance > 0): {len(no_zero_start)} train routes affected.")
    if len(no_zero_start) > 0:
        print("    Truncated routes details:")
        print(no_zero_start.to_string(index=False))
        
    # Check for duplicate stations within the same train route (circular/loop back routes)
    dup_stations = df.duplicated(subset=['Train_No', 'Route_Number', 'Station_Code']).sum()
    print(f" -> Circular Train Routes (repeating stations in same route): {dup_stations} occurrences.")
    
    print("\nLevel 1 execution complete.")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

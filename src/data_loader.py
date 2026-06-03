import os
import pandas as pd
import numpy as np

def load_raw_dataset(filepath="Dataset1.csv"):
    """
    Loads the raw train route dataset (Dataset1.csv).
    
    Parameters:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Raw DataFrame.
    """
    # Look for the file in multiple common search paths if not found directly
    if not os.path.exists(filepath):
        search_paths = [
            filepath,
            os.path.join("..", filepath),
            os.path.abspath(filepath),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath)
        ]
        for path in search_paths:
            if os.path.exists(path):
                filepath = path
                break
        else:
            raise FileNotFoundError(f"Could not locate dataset file: {filepath}")
            
    df = pd.read_csv(filepath)
    return df

def clean_and_engineer_data(df):
    """
    Cleans raw train route data, parses times, and tracks day boundaries to calculate 
    absolute elapsed times from the start of each train's journey.
    
    Parameters:
        df (pd.DataFrame): Raw train route data.
        
    Returns:
        pd.DataFrame: Cleaned station-level dataframe with absolute time columns.
    """
    # 1. Drop exact duplicate rows (if any)
    df_cleaned = df.drop_duplicates().copy()
    
    # 2. Convert column types for stability
    numeric_cols = ['SN', 'Train_No', '1A', '2A', '3A', 'SL', 'Route_Number', 'Distance']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
            
    # 3. Clean string columns
    str_cols = ['Station_Code', 'Station_Name', 'Arrival_time', 'Departure_Time']
    for col in str_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
    # 4. Sort route entries sequentially
    df_sorted = df_cleaned.sort_values(by=['Train_No', 'Route_Number', 'SN']).copy()
    
    # 5. Track day boundaries group-by-group (Train_No & Route_Number)
    results = []
    for (t_no, r_no), group in df_sorted.groupby(['Train_No', 'Route_Number']):
        group = group.copy()
        
        # Extract hours and minutes
        times = []
        for idx, row in group.iterrows():
            arr_str = row['Arrival_time']
            dep_str = row['Departure_Time']
            
            # Extract values from HH:MM:SS
            arr_h, arr_m, _ = map(int, arr_str.split(':'))
            dep_h, dep_m, _ = map(int, dep_str.split(':'))
            
            times.append({
                'arr_min': arr_h * 60 + arr_m,
                'dep_min': dep_h * 60 + dep_m
            })
            
        curr_offset_days = 0
        prev_min = None
        abs_arrs = []
        abs_deps = []
        
        for i, t in enumerate(times):
            arr = t['arr_min']
            dep = t['dep_min']
            
            if i == 0:
                # First stop: arrival is usually "00:00:00" and not applicable
                abs_arr = None
                abs_dep = dep  # Departure time relative to midnight of starting day
                prev_min = dep
            else:
                # Check for arrival day-boundary crossing
                if arr < prev_min:
                    curr_offset_days += 1
                abs_arr = curr_offset_days * 24 * 60 + arr
                prev_min = arr
                
                # Check for departure day-boundary crossing (except for terminating stop)
                if dep == 0 and i == len(times) - 1:
                    abs_dep = None
                else:
                    if dep < prev_min:
                        curr_offset_days += 1
                    abs_dep = curr_offset_days * 24 * 60 + dep
                    prev_min = dep
                    
            abs_arrs.append(abs_arr)
            abs_deps.append(abs_dep)
            
        group['Abs_Arrival_Minutes'] = abs_arrs
        group['Abs_Departure_Minutes'] = abs_deps
        results.append(group)
        
    return pd.concat(results)

def aggregate_to_train_level(df_processed):
    """
    Aggregates station-level data to train route-level metrics.
    Filters out truncated routes (where starting distance != 0).
    
    Parameters:
        df_processed (pd.DataFrame): Preprocessed data with absolute arrival/departure minutes.
        
    Returns:
        pd.DataFrame: Train route-level aggregated modeling dataframe.
    """
    records = []
    for (t_no, r_no), group in df_processed.groupby(['Train_No', 'Route_Number']):
        group = group.sort_values(by='SN')
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        
        # Calculate features
        total_distance = last_row['Distance'] - first_row['Distance']
        number_of_stops = len(group)
        
        # Check if route is truncated (missing starting stops)
        is_truncated = (first_row['Distance'] != 0)
        
        abs_dep_start = first_row['Abs_Departure_Minutes']
        abs_arr_end = last_row['Abs_Arrival_Minutes']
        
        if abs_dep_start is not None and abs_arr_end is not None:
            total_duration = abs_arr_end - abs_dep_start
        else:
            total_duration = np.nan
            
        records.append({
            'Train_No': t_no,
            'Route_Number': r_no,
            'Start_Station': first_row['Station_Code'],
            'End_Station': last_row['Station_Code'],
            'Total_Distance': total_distance,
            'Number_of_Stops': number_of_stops,
            'Total_Duration_Minutes': total_duration,
            'Is_Truncated': is_truncated
        })
        
    df_agg = pd.DataFrame(records)
    
    # Filter out truncated routes for modeling stability
    df_clean = df_agg[df_agg['Is_Truncated'] == False].copy()
    
    # Drop intermediate columns that are no longer needed
    df_clean.drop(columns=['Is_Truncated'], inplace=True)
    
    return df_clean

"""
Helper module to map new Excel column names to scoring engine expected names.
"""

# Column name mapping from new Excel format to expected metric names
COLUMN_MAPPING = {
    # Possession
    'Possession, %': 'Possession, %',
    
    # Passes
    'Passes %': 'Percentage of accurate passes',
    'Passes accurate': 'Accurate passes',
    'Passes Total': 'Total passes',
    
    # Back/Lateral/Forward passes
    'Back passes %': 'Percentage of accurate back passes',
    'Lateral passes %': 'Percentage of accurate lateral passes',
    'Forward passes %': 'Percentage of accurate forward passes',
    
    # Progressive passes  
    'Progressive passes %': 'Percentage of acccurate progressive passes',
    'Progressive passes Total': 'Total progressive passes',
    
    # Passes to final third
    'Passes to final third %': 'Percentage of accurate passes to final third',
    
    # Defensive duels
    'Defensive duels %': 'Percentage defensive duels  won',
    
    # Aerial duels
    'Aerial duels %': 'Percentage aerial duels  won',
    
    # Offensive duels
    'Offensive duels %': 'Percentage offensive duels  won',
    
    # Sliding tackles
    'Sliding tackles %': 'Percentage successful sliding tackles',
    
    # Shots
    'Shots %': 'Percentage of shots on target',
    'Shots on target': 'Shots on target',
    'Shots Total': 'Total shots',
    
    # Shots against
    'Shots against %': 'Percentage of shots against on target',
    'Shots against on target': 'Total shots against on target',
    'Shots against Total': 'Total shots against',
    
    # Crosses
    'Crosses %': 'Percentage of accurate crosses',
    
    # Other metrics that should remain unchanged
    'Goals': 'Goals',
    'xG': 'xG',
    'PPDA': 'PPDA',
    'Match tempo': 'Match tempo',
    'Conceded goals': 'Conceded goals',
    'Interceptions': 'Interceptions',
    'Clearances': 'Clearances',
    'Fouls': 'Fouls',
}


def standardize_columns(df):
    """
    Standardize DataFrame columns to match scoring engine expectations.
    """
    import pandas as pd
    
    # Create reverse mapping for columns that exist
    rename_dict = {}
    for new_col, old_col in COLUMN_MAPPING.items():
        if new_col in df.columns:
            rename_dict[new_col] = old_col
    
    # Apply renaming
    df_standardized = df.rename(columns=rename_dict)
    
    return df_standardized


def get_metric_value(data_row, metric_name):
    """
    Get metric value from data row, trying multiple possible column names.
    
    Args:
        data_row: pandas Series or dict
        metric_name: Expected metric name (e.g., 'Percentage of accurate passes')
        
    Returns:
        Metric value or 0 if not found
    """
    # Try exact match first
    if metric_name in data_row:
        return data_row[metric_name]
    
    # Try to find in reverse mapping
    for new_col, old_col in COLUMN_MAPPING.items():
        if old_col == metric_name and new_col in data_row:
            return data_row[new_col]
    
    # Not found
    return 0

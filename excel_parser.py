"""
Excel Parser Module for Team Intelligence Suite

Handles smart parsing of Team Stats Excel files with:
- Header row detection (hunts for "Date" or "Match")
- Multi-level header parsing (2-3 rows)
- Column name flattening
- Metric mapping (Main Metric → Sub columns)
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List, Optional


def find_header_row(filepath: str, max_search_rows: int = 20) -> Optional[int]:
    """
    Find the header row containing 'Date' or 'Match'.
    
    Args:
        filepath: Path to Excel file
        max_search_rows: Maximum rows to search for header
        
    Returns:
        Row index (0-based) of header row, or None if not found
    """
    # Read first N rows without headers to search
    df_raw = pd.read_excel(filepath, header=None, nrows=max_search_rows)
    
    for idx, row in df_raw.iterrows():
        # Convert row to strings and check each cell
        row_str = row.astype(str).str.lower()
        
        # Check if any cell contains "date" or "match"
        if any('date' in str(val) for val in row_str) or \
           any('match' in str(val) for val in row_str):
            print(f"✅ Header row found at index {idx}")
            return int(idx)
    
    print(f"⚠️  Warning: No header row found in first {max_search_rows} rows")
    return None


def parse_multi_level_headers(filepath: str, header_row_idx: int, num_header_rows: int = 3) -> pd.DataFrame:
    """
    Parse Excel file with multi-level headers starting from header_row_idx.
    
    Args:
        filepath: Path to Excel file
        header_row_idx: Starting row index for headers
        num_header_rows: Number of header rows (2 or 3)
        
    Returns:
        DataFrame with multi-index columns
    """
    header_indices = list(range(header_row_idx, header_row_idx + num_header_rows))
    
    try:
        df = pd.read_excel(filepath, header=header_indices)
        print(f"✅ Parsed {num_header_rows} header rows starting at index {header_row_idx}")
        return df
    except Exception as e:
        # If 3 rows fail, try 2 rows
        if num_header_rows == 3:
            print(f"⚠️  3-row headers failed, trying 2-row headers...")
            return parse_multi_level_headers(filepath, header_row_idx, num_header_rows=2)
        else:
            raise e


def flatten_columns(multi_index_cols) -> List[str]:
    """
    Flatten multi-level column index into meaningful single-level names.
    
    Pattern recognition:
    - "Passes / accurate" with 3 sub-columns → "Passes Total", "Passes Accurate", "Passes %"
    - "Recoveries / Low / Medium / High" with 4 sub-columns → "Recoveries Total", "Recoveries Low", "Recoveries Medium", "Recoveries High"
    
    Args:
        multi_index_cols: MultiIndex columns from DataFrame
        
    Returns:
        List of flattened column names
    """
    flattened = []
    i = 0
    cols = list(multi_index_cols)
    
    while i < len(cols):
        col = cols[i]
        
        # Get main metric name from first level
        main_metric = str(col[0]).strip()
        
        # Skip if it's an "Unnamed" column at top level
        if "Unnamed" in main_metric:
            # Try to extract from lower levels
            valid_parts = [str(c) for c in col if "Unnamed" not in str(c) and pd.notna(c)]
            if valid_parts:
                flattened.append(" ".join(valid_parts))
            else:
                flattened.append(f"Col_{i}")
            i += 1
            continue
        
        # Count how many consecutive columns share this main metric
        group = [col]
        j = i + 1
        while j < len(cols) and str(cols[j][0]).strip() == main_metric:
            group.append(cols[j])
            j += 1
        
        group_size = len(group)
        
        # Pattern matching based on group size
        if group_size == 3 and "/" in main_metric:
            # Pattern: "Passes / accurate" → Total, Accurate, %
            parts = [p.strip() for p in main_metric.split("/")]
            base_name = parts[0]
            detail_name = parts[-1] if len(parts) > 1 else "Count"
            
            flattened.append(f"{base_name} Total")
            flattened.append(f"{base_name} {detail_name}")
            flattened.append(f"{base_name} %")
            
        elif group_size == 4 and main_metric.count("/") >= 3:
            # Pattern: "Recoveries / Low / Medium / High" → Total, Low, Medium, High
            parts = [p.strip() for p in main_metric.split("/")]
            base_name = parts[0]
            
            flattened.append(f"{base_name} Total")
            for part in parts[1:]:
                flattened.append(f"{base_name} {part}")
                
        elif group_size == 2 and "/" in main_metric:
            # Pattern: 2 columns with "/"
            parts = [p.strip() for p in main_metric.split("/")]
            base_name = parts[0]
            detail_name = parts[-1] if len(parts) > 1 else "Count"
            
            flattened.append(f"{base_name} Total")
            flattened.append(f"{base_name} {detail_name}")
            
        elif group_size == 1:
            # Single column - keep main metric name
            flattened.append(main_metric)
            
        else:
            # Irregular pattern - use generic naming
            for k in range(group_size):
                flattened.append(f"{main_metric} {k+1}")
        
        i += group_size
    
    return flattened


def build_metric_mapping(columns: List[str]) -> Dict[str, List[str]]:
    """
    Build mapping of Main Metric → List of sub-columns.
    
    Example:
        "Passes" → ["Passes Total", "Passes Accurate", "Passes %"]
    
    Args:
        columns: List of column names
        
    Returns:
        Dictionary mapping main metric to its sub-columns
    """
    mapping = {}
    
    for col in columns:
        # Extract base metric name (before first space or special char)
        # Handle cases like "Passes Total", "Passes %", etc.
        parts = col.split()
        if len(parts) > 1:
            base = parts[0]
            if base not in mapping:
                mapping[base] = []
            mapping[base].append(col)
        else:
            # Single-word columns
            mapping[col] = [col]
    
    return mapping


def load_team_stats_smart(filepath: str, team_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load Team Stats Excel file with smart header detection.
    
    Args:
        filepath: Path to Excel file
        team_name: Optional team name to filter data
        
    Returns:
        Tuple of (DataFrame, metric_mapping)
    """
    print(f"\n{'='*80}")
    print(f"Loading: {filepath}")
    print(f"{'='*80}")
    
    # Step 1: Find header row
    header_row_idx = find_header_row(filepath)
    if header_row_idx is None:
        print("❌ Could not find header row. Using row 0 as fallback.")
        header_row_idx = 0
    
    # Step 2: Parse multi-level headers
    df = parse_multi_level_headers(filepath, header_row_idx, num_header_rows=3)
    
    # Step 3: Flatten columns
    new_columns = flatten_columns(df.columns)
    
    # Ensure column count matches
    if len(new_columns) != len(df.columns):
        print(f"⚠️  Column count mismatch: {len(new_columns)} flattened vs {len(df.columns)} original")
        # Fallback to simple join
        new_columns = [" | ".join([str(c) for c in col if "Unnamed" not in str(c) and pd.notna(c)]) 
                      for col in df.columns]
    
    df.columns = new_columns
    
    # Step 4: Standardize key columns
    col_map = {}
    for c in df.columns:
        c_lower = c.lower()
        if "date" in c_lower and "Date" not in col_map.values():
            col_map[c] = "Date"
        elif c == "Team" or (c_lower == "team" and "Team" not in col_map.values()):
            col_map[c] = "Team"
    
    df.rename(columns=col_map, inplace=True)
    
    # Step 5: Convert Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates (likely sub-header rows)
        df.dropna(subset=['Date'], inplace=True)
    
    # Step 6: Filter by team if specified
    if team_name and 'Team' in df.columns:
        df = df[df['Team'] == team_name].copy()
    
    # Step 7: Build metric mapping
    metric_mapping = build_metric_mapping(df.columns.tolist())
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"✅ Built metric mapping with {len(metric_mapping)} main metrics")
    
    return df, metric_mapping


if __name__ == "__main__":
    # Quick test
    import os
    test_file = os.path.join("database", "Team Stats", "Germany Frauen Bundesliga 25_26", 
                             "Team Stats Bayern München 25_26.xlsx")
    
    if os.path.exists(test_file):
        df, mapping = load_team_stats_smart(test_file)
        print(f"\nSample columns: {df.columns[:10].tolist()}")
        print(f"\nSample metric mapping:")
        for k, v in list(mapping.items())[:5]:
            print(f"  {k}: {v}")
    else:
        print(f"Test file not found: {test_file}")

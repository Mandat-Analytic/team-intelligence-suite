import pandas as pd
import os
import glob

# Find a team stats file recursively
team_dir = os.path.join("database", "Team Stats")
files = glob.glob(os.path.join(team_dir, "**", "*.xlsx"), recursive=True)

if files:
    target_file = files[0]
    print(f"Inspecting file: {target_file}")
    
    # Read with header=None to see raw grid
    df = pd.read_excel(target_file, header=None, nrows=10)
    print("\n--- RAW GRID DUMP (First 10 rows) ---")
    print(df.to_string())
else:
    print("No Team Stats files found.")

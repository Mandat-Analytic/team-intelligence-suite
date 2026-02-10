import data_loader
import os
import glob
import pandas as pd

# Find file
team_dir = os.path.join("database", "Team Stats")
files = glob.glob(os.path.join(team_dir, "**", "*.xlsx"), recursive=True)

if files:
    f = files[0]
    print(f"Testing loader on: {f}")
    df = data_loader.load_team_stats(f)
    if df is not None:
        print("Success!")
        print("Columns found:")
        print(df.columns.tolist()[:20]) # First 20
        print("...")
        print("Sample data:")
        print(df[['Date', 'Team', 'Goals']].head())
    else:
        print("Loader returned None")
else:
    print("No files found")

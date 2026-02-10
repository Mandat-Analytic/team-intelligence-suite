import data_loader
import pandas as pd
import os

# 1. Load Player Data
league = "Germany Frauen Bundesliga 25_26"
print(f"Loading player data for: {league}")
player_df = data_loader.load_player_data(league)

if player_df.empty:
    print("!!! Player DF is empty")
else:
    print(f"Loaded {len(player_df)} rows")
    
    # 2. Check Teams
    if 'Team' in player_df.columns:
        player_teams = sorted(player_df['Team'].unique().astype(str))
        print("\n--- Player Data Teams (First 10) ---")
        print(player_teams[:10])
    else:
        print("!!! 'Team' column missing in Player Data")
        print("Columns:", player_df.columns.tolist()[:10])

# 3. Check Team Stats Teams (what the selector uses)
print("\n--- Available Teams (from data_loader) ---")
available_teams = data_loader.get_teams_in_league(league)
print(sorted(available_teams)[:10])

# 4. Check Junction
if 'Team' in player_df.columns:
    matches = [t for t in available_teams if t in player_df['Team'].unique()]
    print(f"\nMatching Teams: {len(matches)} / {len(available_teams)}")
    if len(matches) < len(available_teams):
        print("Missing in Player Data:")
        for t in available_teams:
            if t not in matches:
                print(f" - '{t}'")

# 5. Check Metrics presence
print("\n--- Metric Check ---")
# Sample metrics from PHASE_INFO
sample_metrics = [
    "Passes per 90", "Accurate passes, %", "Back passes per 90", # Build Up
    "Goals per 90", "xG per 90", # Output
    "Defensive duels per 90", # Pressing
]
print("Checking sample metrics existence in Player DF:")
for m in sample_metrics:
    exists = m in player_df.columns
    print(f"'{m}': {exists}")

if not player_df.empty:
    print("\nAll Player DF Columns:")
    print(sorted(player_df.columns.tolist()))

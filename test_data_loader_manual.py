"""
Manual test script to validate data_loader.py functionality
"""
import os
import sys
import glob
import pandas as pd

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

import data_loader

def test_team_stats_loading():
    """Test loading Team Stats Excel files"""
    print("=" * 80)
    print("TESTING DATA LOADER")
    print("=" * 80)
    
    # Find all Team Stats files
    team_stats_dir = os.path.join("database", "Team Stats")
    pattern = os.path.join(team_stats_dir, "**", "*.xlsx")
    files = glob.glob(pattern, recursive=True)
    
    # Filter out temp files
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    print(f"\nFound {len(files)} Team Stats file(s):")
    for f in files:
        print(f"  - {f}")
    
    if not files:
        print("\n⚠️  No Team Stats files found!")
        return
    
    # Test loading each file
    for filepath in files:
        print("\n" + "-" * 80)
        print(f"Testing file: {os.path.basename(filepath)}")
        print("-" * 80)
        
        try:
            df = data_loader.load_team_stats(filepath)
            
            if df is None or df.empty:
                print("❌ FAILED: Returned None or empty DataFrame")
                continue
            
            print(f"✅ Successfully loaded!")
            print(f"   Shape: {df.shape} (rows, columns)")
            print(f"   Columns: {len(df.columns)}")
            print(f"\nFirst 5 column names:")
            for i, col in enumerate(df.columns[:5]):
                print(f"   {i+1}. {col}")
            
            # Check for key columns
            has_date = 'Date' in df.columns
            has_team = 'Team' in df.columns
            
            print(f"\nKey columns present:")
            print(f"   Date: {'✅' if has_date else '❌'}")
            print(f"   Team: {'✅' if has_team else '❌'}")
            
            if has_date:
                print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
            
            if has_team:
                teams = df['Team'].unique()
                print(f"\nTeams found ({len(teams)}):")
                for team in teams[:5]:
                    print(f"   - {team}")
                if len(teams) > 5:
                    print(f"   ... and {len(teams) - 5} more")
            
            # Show sample row
            print("\nSample row (first 10 columns):")
            print(df.iloc[0][:10])
            
        except Exception as e:
            print(f"❌ ERROR loading file: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_team_stats_loading()

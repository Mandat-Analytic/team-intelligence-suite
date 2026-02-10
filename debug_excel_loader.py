"""
Debug script to test Excel parsing logic
"""
import os
import sys
import glob

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

import excel_parser

def main():
    print("="*80)
    print("EXCEL PARSER DEBUG TOOL")
    print("="*80)
    
    # Find Team Stats files
    team_stats_dir = os.path.join("database", "Team Stats")
    pattern = os.path.join(team_stats_dir, "**", "*.xlsx")
    files = glob.glob(pattern, recursive=True)
    
    # Filter out temp files
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    if not files:
        print("\n❌ No Team Stats files found!")
        return
    
    # Test with first file
    test_file = files[0]
    print(f"\n📁 Testing with: {os.path.basename(test_file)}")
    print("-"*80)
    
    try:
        # Load with smart parser
        df, metric_mapping = excel_parser.load_team_stats_smart(test_file)
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        print(f"\n📋 First 20 Column Names:")
        for i, col in enumerate(df.columns[:20], 1):
            print(f"   {i:2d}. {col}")
        
        if len(df.columns) > 20:
            print(f"   ... and {len(df.columns) - 20} more columns")
        
        print(f"\n🗂️  Metric Mapping (first 10):")
        for i, (main_metric, sub_cols) in enumerate(list(metric_mapping.items())[:10], 1):
            print(f"   {i:2d}. {main_metric}")
            for sub in sub_cols:
                print(f"       → {sub}")
        
        # Check for expected columns
        print(f"\n✅ Key Columns Check:")
        expected = ['Date', 'Team', 'Passes Total', 'Passes Accurate', 'Passes %']
        for col in expected:
            status = "✅" if col in df.columns else "❌"
            print(f"   {status} {col}")
        
        # Show sample data
        if not df.empty:
            print(f"\n📄 Sample Row (first 10 columns):")
            sample = df.iloc[0][:10]
            for col, val in sample.items():
                print(f"   {col}: {val}")
        
        # Date range
        if 'Date' in df.columns:
            print(f"\n📅 Date Range:")
            print(f"   From: {df['Date'].min()}")
            print(f"   To:   {df['Date'].max()}")
        
        # Teams
        if 'Team' in df.columns:
            teams = df['Team'].unique()
            print(f"\n⚽ Teams ({len(teams)}):")
            for team in teams[:5]:
                print(f"   - {team}")
            if len(teams) > 5:
                print(f"   ... and {len(teams) - 5} more")
        
        print(f"\n{'='*80}")
        print("✅ PARSING SUCCESSFUL!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

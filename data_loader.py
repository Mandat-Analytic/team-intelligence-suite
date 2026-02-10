"""
Enhanced Data Loader for Team Intelligence Suite

Provides:
- League selection (scan Team Stats subfolders)
- Team selection within leagues
- League-wide data loading for comparisons
- Integration with smart Excel parser
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Optional
import excel_parser


def get_available_leagues() -> List[str]:
    """
    Scan Team Stats directory for league subfolders.
    
    Returns:
        List of league names (subfolder names)
    """
    team_stats_dir = os.path.join("database", "Team Stats")
    
    if not os.path.exists(team_stats_dir):
        return []
    
    # Get all subdirectories
    leagues = [d for d in os.listdir(team_stats_dir) 
               if os.path.isdir(os.path.join(team_stats_dir, d))]
    
    return sorted(leagues)


def get_teams_in_league(league_name: str) -> List[str]:
    """
    Get list of teams in a specific league.
    
    Args:
        league_name: Name of the league (subfolder name)
        
    Returns:
        List of team names (extracted from filenames)
    """
    league_dir = os.path.join("database", "Team Stats", league_name)
    
    if not os.path.exists(league_dir):
        return []
    
    # Find all Excel files
    pattern = os.path.join(league_dir, "Team Stats *.xlsx")
    files = glob.glob(pattern)
    
    # Extract team names from filenames
    teams = []
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith("Team Stats ") and basename.endswith(".xlsx"):
            # Remove "Team Stats " prefix and ".xlsx" suffix
            team_name = basename.replace("Team Stats ", "").replace(".xlsx", "")
            # Remove " 25_26" or similar season suffix
            team_name = team_name.rsplit(" ", 1)[0] if " " in team_name and team_name.split()[-1].count("_") > 0 else team_name
            teams.append(team_name)
    
    return sorted(teams)


def load_team_data(league_name: str, team_name: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load data for a specific team in a league.
    
    Args:
        league_name: Name of the league
        team_name: Name of the team
        
    Returns:
        Tuple of (DataFrame, metric_mapping)
    """
    # Construct filepath - try with and without season suffix
    league_dir = os.path.join("database", "Team Stats", league_name)
    
    # Try to find the file (might have season suffix)
    pattern = os.path.join(league_dir, f"Team Stats {team_name}*.xlsx")
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No data found for team '{team_name}' in league '{league_name}'")
    
    filepath = matches[0]
    
    # Use smart Excel parser
    df, metric_mapping = excel_parser.load_team_stats_smart(filepath, team_name=team_name)
    
    return df, metric_mapping


def load_league_data(league_name: str, cache: Dict = None) -> pd.DataFrame:
    """
    Load data for all teams in a league (for comparisons).
    
    Args:
        league_name: Name of the league
        cache: Optional cache dictionary to store/retrieve results
        
    Returns:
        Combined DataFrame with all teams' data
    """
    # Check cache first
    if cache is not None and league_name in cache:
        print(f"✅ Using cached data for league '{league_name}'")
        return cache[league_name]
    
    print(f"📊 Loading league data for '{league_name}'...")
    
    teams = get_teams_in_league(league_name)
    
    if not teams:
        print(f"⚠️  No teams found in league '{league_name}'")
        return pd.DataFrame()
    
    all_data = []
    
    for team in teams:
        try:
            df, _ = load_team_data(league_name, team)
            all_data.append(df)
        except Exception as e:
            print(f"⚠️  Failed to load {team}: {str(e)}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all team data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Store in cache
    if cache is not None:
        cache[league_name] = combined
    
    print(f"✅ Loaded {len(combined)} total matches from {len(all_data)} teams")
    
    return combined


def calculate_season_avg(team_df: pd.DataFrame) -> pd.Series:
    """
    Calculate season average (mean) for a specific team.
    
    Args:
        team_df: DataFrame containing team's match data
        
    Returns:
        Series with mean values for all numeric columns
    """
    numeric_cols = team_df.select_dtypes(include=[np.number]).columns
    return team_df[numeric_cols].mean()


def calculate_league_avg(league_df: pd.DataFrame) -> pd.Series:
    """
    Calculate league average (mean) across all teams.
    
    Args:
        league_df: DataFrame containing all league match data
        
    Returns:
        Series with mean values for all numeric columns
    """
    numeric_cols = league_df.select_dtypes(include=[np.number]).columns
    return league_df[numeric_cols].mean()


def calculate_top_25_pct(league_df: pd.DataFrame) -> pd.Series:
    """
    Calculate 75th percentile (top 25%) across league.
    
    Args:
        league_df: DataFrame containing all league match data
        
    Returns:
        Series with 75th percentile values for all numeric columns
    """
    numeric_cols = league_df.select_dtypes(include=[np.number]).columns
    return league_df[numeric_cols].quantile(0.75)


def load_player_data(league_name: str) -> pd.DataFrame:
    """
    Load Player Stats for a specific league from position-specific files.
    
    Looks in: database/Player Stats/[league]/
    Files: Goalkeeper, Defender, Midfielder, Forwards
    
    Args:
        league_name: Name of the league
        
    Returns:
        DataFrame with player statistics (all positions combined)
    """
    player_stats_dir = os.path.join("database", "Player Stats", league_name)
    
    if not os.path.exists(player_stats_dir):
        print(f"Warning: No player stats directory found for league '{league_name}'")
        return pd.DataFrame()
    
    position_files = {
        "Goalkeeper": f"Goalkeeper {league_name}.xlsx",
        "Defender": f"Defender {league_name}.xlsx",
        "Midfielder": f"Midfielder {league_name}.xlsx",
        "Forwards": f"Forwards {league_name}.xlsx"
    }
    
    all_players = []
    
    for position_type, filename in position_files.items():
        filepath = os.path.join(player_stats_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_excel(filepath)
                if not df.empty:
                    print(f"Loaded {len(df)} {position_type} players")
                    all_players.append(df)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {str(e)}")
        else:
            print(f"Info: {filename} not found")
    
    if all_players:
        combined_df = pd.concat(all_players, ignore_index=True)
        
        # Normalize Team column
        if 'Team within selected timeframe' in combined_df.columns:
            combined_df.rename(columns={'Team within selected timeframe': 'Team'}, inplace=True)
            
        print(f"Total loaded player stats: {len(combined_df)} players")
        return combined_df
    else:
        print(f"Warning: No player stats found for league '{league_name}'")
        return pd.DataFrame()


# Backward compatibility - keep old function names
def get_available_teams():
    """Legacy function - get teams from first league"""
    leagues = get_available_leagues()
    if leagues:
        return get_teams_in_league(leagues[0])
    return []


def load_data(team_name: str):
    """Legacy function - load team data from first league"""
    leagues = get_available_leagues()
    if not leagues:
        return None, None, pd.DataFrame(), pd.DataFrame()
    
    try:
        team_df, _ = load_team_data(leagues[0], team_name)
        
        # Split into team and opponent data
        if 'Team' in team_df.columns:
            team_rows = team_df[team_df['Team'] == team_name].copy()
            # For opponents, we'd need to parse match results - simplified for now
            opp_rows = pd.DataFrame()
        else:
            team_rows = team_df
            opp_rows = pd.DataFrame()
        
        # Load player data
        player_df = load_player_data(leagues[0])
        league_df = player_df.copy()
        
        if 'Team' in player_df.columns:
            player_df = player_df[player_df['Team'] == team_name].copy()
        
        return team_rows, opp_rows, player_df, league_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")
    
    leagues = get_available_leagues()
    print(f"\nAvailable leagues: {leagues}")
    
    if leagues:
        league = leagues[0]
        teams = get_teams_in_league(league)
        print(f"\nTeams in {league}: {teams}")
        
        if teams:
            team = teams[0]
            df, mapping = load_team_data(league, team)
            print(f"\nLoaded {team}: {len(df)} matches, {len(df.columns)} columns")

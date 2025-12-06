import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import PyPizza

# File paths for different leagues
# Base path for all database files
BASE_PATH = r"C:\Users\USER\Desktop\Rembatz Analisis\Selangor FC\Wyscout Database"

# Function to automatically scan and load all league files
def scan_league_files(base_path):
    import os
    import glob
    
    league_files = {}
    
    # Get all Excel files in the directory
    excel_files = glob.glob(os.path.join(base_path, "*.xlsx"))
    
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        
        # Detect position type from filename
        if filename.startswith("Defender"):
            position_code = "(D)"
            league_name = filename.replace("Defender ", "").replace(".xlsx", "")
        elif filename.startswith("Midfielder"):
            position_code = "(M)"
            league_name = filename.replace("Midfielder ", "").replace(".xlsx", "")
        elif filename.startswith("Forwards"):
            position_code = "(F)"
            league_name = filename.replace("Forwards ", "").replace(".xlsx", "")
        else:
            continue  # Skip files that don't match the pattern
        
        # Convert abbreviated season format to full format
        # e.g., "24_25" -> "2024/2025", "2023" -> "2023"
        import re
        season_match = re.search(r'(\d{2})_(\d{2})$', league_name)
        if season_match:
            year1, year2 = season_match.groups()
            league_name = league_name.replace(f"{year1}_{year2}", f"20{year1}/20{year2}")
        
        # Create the display name
        display_name = f"{league_name} {position_code}"
        
        # Add to dictionary
        league_files[display_name] = file_path
    
    return league_files

# Scan and load all league files
league_files = scan_league_files(BASE_PATH)

# Define metrics for different templates
defender_metrics = {
    "Non-penalty goals per 90": "Non-penalty goals\nper 90",
    "Successful dribbles, %": "Successful\ndribbles %",
    "Offensive duels won, %": "Offensive duels\nwon %",
    "xA per 90": "Expected Assists\n(xA) per 90",
    "Forward passes per 90": "Forward Passes\nper 90",
    "Accurate forward passes, %": "Accurate Forward\nPasses, %",
    "Long passes per 90": "Long Passes\nper 90",
    "Accurate long passes, %": "Accurate Long\nPasses, %",
    "Progressive passes per 90": "Progressive\nPasses per 90",
    "Accurate progressive passes, %":"Accurate Progressive\nPasses, %",
    "Progressive runs per 90": "Progressive\nRuns per 90",
    "Successful defensive actions per 90": "Defensive actions\nper 90",
    "Defensive duels won, %": "Defensive duels\nWon %",
    "PAdj Sliding tackles":"Posession-Adjusted\nSliding Tackles",
    "PAdj Interceptions":"Posession-Adjusted\nInterceptions",
    "Aerial duels won, %": "Aerial duels\nwon %",
    "Fouls per 90": "Fouls\nper 90"
}

midfielder_metrics = {
    "Non-penalty goals per 90": "Non-penalty goals\nper 90",
    "Shots per 90": "Shots\nper 90",
    "Successful dribbles, %": "Successful\ndribbles %",
    "Offensive duels won, %": "Offensive duels\nwon %",
    "xA per 90": "Expected Assists\n(xA) per 90",
    "Shot assists per 90": "Shot Assists\nper 90",
    "Smart passes per 90": "Smart Passes\nper 90",
    "Accurate passes to final third, %": "Final Third\nPasses %",
    "Accurate through passes, %": "Through\nPasses %",
    "Progressive passes per 90": "Progressive\nPasses per 90",
    "Progressive runs per 90": "Progressive\nRuns per 90",
    "Successful defensive actions per 90": "Defensive actions\nper 90",
    "Defensive duels won, %": "Defensive duels\nwon %",
    "Aerial duels won, %": "Aerial duels\nwon %",
    "Fouls per 90": "Fouls\nper 90"
}

forward_metrics = {
    "Non-penalty goals per 90": "Non-penalty goals\nper 90",
    "xG per 90":"xG per 90",
    "Goals/xG": "Goals/xG",
    "Successful dribbles, %": "Successful\ndribbles %",
    "Shots per 90":"Shots per 90",
    "Shots on target, %":"Shots On\nTarget, %",
    "Goal conversion, %":"Goal\nConversion, %",
    "Touches in box per 90":"Touches In\nBox per 90",
    "Offensive duels won, %": "Offensive duels\nwon %",
    "xA per 90": "Expected Assists\n(xA) per 90",
    "Shot assists per 90": "Shot Assists\nper 90",
    "Passes to penalty area per 90": "Passes To Penalty\nArea per 90",
    "Deep completions per 90": "Deep Completions\nper 90",
    "Smart passes per 90": "Smart Passes\nper 90",
    "Progressive passes per 90": "Progressive\nPasses per 90",
    "Progressive runs per 90": "Progressive\nRuns per 90",
    "Defensive duels won, %": "Defensive Duels\nWon %",
    "Aerial duels won, %":"Aerial Duels\nWon, %",
    "Aerial duels per 90":"Aerial Duels\nper 90"
}

# Define position filters for templates
template_positions = {
    "Defender": ['CB', 'RCB', 'LCB', 'RB', 'LB', 'LWB', 'RWB'],
    "Midfielder": ['DMF', 'CMF', 'AMF'],
    "Forward":['CF', 'RWF', 'LWF', 'AMF', 'RW', 'LW']
}

# Define slice colors for templates
defender_slice_colors = ["#1A78CF"] * 3 + ["#FF9300"] * 8 + ["#D70232"] * 6
midfielder_slice_colors = ["#1A78CF"] * 4 + ["#FF9300"] * 7 + ["#D70232"] * 4
forward_slice_colors = ["#1A78CF"] * 9 + ["#FF9300"] * 7 + ["#D70232"] * 3

# Streamlit UI with improved layout
st.set_page_config(page_title="Player Radar Chart", layout="wide")
st.title("Selangor FC: Scouting Dashboard\nPlayer Performance Radar Chart")

# Create columns for template selection and other controls
template_col, _ = st.columns([1, 3])

with template_col:
    # Template selection
    template = st.selectbox("Select Template", ["Defender", "Midfielder","Forward"])

# Get the appropriate metrics, positions, and colors based on template
if template == "Defender":
    metrics = defender_metrics
    positions = template_positions["Defender"]
    slice_colors = defender_slice_colors
    available_leagues = [league for league in league_files.keys() if "(D)" in league]
    
elif template == "Forward":
    metrics = forward_metrics
    positions = template_positions["Forward"]
    slice_colors = forward_slice_colors
    available_leagues = [league for league in league_files.keys() if "(F)" in league]

else:  # Midfielder template
    metrics = midfielder_metrics
    positions = template_positions["Midfielder"]
    slice_colors = midfielder_slice_colors
    available_leagues = [league for league in league_files.keys() if "(M)" in league]
    
# Create columns for control panel and chart
col1, col2 = st.columns([1, 3])

with col1:
    # Control panel
    st.subheader("Player 1 Controls")
    
    # League selection for player 1
    league_1 = st.selectbox("Select League for Player 1", available_leagues, key="league_1")
    
    # Load data based on selected league
    def load_data(file_path):
        return pd.read_excel(file_path)
    
    df_1 = load_data(league_files[league_1])
    
    # Filter players based on position and minutes played
    players_1 = df_1[(df_1['Position'].apply(lambda x: any(pos in str(x) for pos in positions))) & 
                     (df_1['Minutes played'] >= 250)]
    
    # Player 1 selection
    player_name_1 = st.selectbox("Select Player 1", players_1['Player'].unique(), key="player_1")
    
    # Compare checkbox
    compare_player = st.checkbox("Compare with another player?")
    
    # Player 2 controls - only show if compare_player is checked
    if compare_player:
        st.markdown("---")
        st.subheader("Player 2 Controls")
        
        # League selection for player 2
        league_2 = st.selectbox("Select League for Player 2", available_leagues, key="league_2")
        
        # Load data for player 2's league
        df_2 = load_data(league_files[league_2])
        
        # Filter players for player 2
        players_2 = df_2[(df_2['Position'].apply(lambda x: any(pos in str(x) for pos in positions))) & 
                         (df_2['Minutes played'] >= 300)]
        
        # Player 2 selection
        player_name_2 = st.selectbox("Select Player 2", players_2['Player'].unique(), key="player_2")
    else:
        player_name_2 = None
        league_2 = None
        players_2 = None
    
    # Add some information about the chart
    st.markdown("---")
    st.markdown("### Chart Legend")
    st.markdown("🔵 **Blue**: Attack metrics")
    st.markdown("🟠 **Orange**: Possession/Creation metrics")
    st.markdown("🔴 **Red**: Defense metrics")
    st.markdown("---")
    st.markdown(f"Values shown are percentile ranks compared to all {template.lower()}s in the selected league (minimum 300 minutes played).")

# Function to get percentiles for a player
def get_percentiles(player_name, players_df):
    player_stats = players_df[list(metrics.keys())]
    percentiles = player_stats.rank(pct=True) * 100
    player_data = players_df[players_df['Player'] == player_name]
    
    if not player_data.empty:
        return percentiles[players_df['Player'] == player_name].values.flatten(), player_data[list(metrics.keys())].values.flatten()
    return None, None

# Function to create radar chart
def create_radar_chart(player_percentiles, player_values, player_name, league_name, comparison=False):
    background = "#101010"
    text_color = "white"
    
    # Format player values for display
    formatted_values = []
    for i, value in enumerate(player_values):
        if "%" in list(metrics.keys())[i]:  # Check if it's a percentage metric
            formatted_values.append(f"({value:.2f}%)")
        else:  # For per 90 metrics
            formatted_values.append(f"({value:.2f})")
    
    # Create modified parameter labels with values in brackets underneath
    params_with_values = []
    for i, param in enumerate(metrics.values()):
        params_with_values.append(f"{param}\n{formatted_values[i]}")
    
    # Labels for the radar chart
    text_colors = ["white"] * len(metrics)
    
    # Create the radar chart
    baker = PyPizza(
        params=params_with_values,
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-."
    )
    
    # Make pizza for our data
    fig, ax = baker.make_pizza(
        player_percentiles,
        figsize=(11, 11) if comparison else (12, 12),
        color_blank_space="same",
        param_location=110,
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(
            facecolor="cornflowerblue", edgecolor="#000000",
            zorder=3, linewidth=1
        ),
        kwargs_params=dict(
            color=text_color, fontsize=11,
            va="center",
            linespacing=1.2
        ),
        kwargs_values=dict(
            color="#ffffff", fontsize=12, fontweight="bold", 
            bbox=dict(facecolor="cornflowerblue", edgecolor="#000000", boxstyle="round,pad=0.2", alpha=0.7),
            zorder=4
        )
    )
    
    # Get the text objects for percentile values and update them
    texts = baker.get_value_texts()
    for i, text in enumerate(texts):
        text.set_text(f"{int(player_percentiles[i])}")
        # Update the background color based on the slice color
        text.set_bbox(dict(facecolor=slice_colors[i], edgecolor="#000000", boxstyle="round,pad=0.2", alpha=0.7))
    
    # Extract the position type from the template
    position_type = template
    
    # Add title and subtitle
    fig.text(0.5, 1.05, f"{player_name} - {position_type} Performance", ha="center", fontsize=17, color="#ffffff", fontweight="bold")
    fig.text(0.5, 1.0, f"Compared to all {position_type.lower()}s in {league_name.split(' (')[0]}\nMinimum 300 minutes played", ha="center", fontsize=12, color="#ffffff")
    
    # Add color-coded category labels
    if template == "Defender":
        fig.text(0.16, 0.97, "Attack", size=12, color="#1A78CF", ha="center", fontweight="bold")
        fig.text(0.50, 0.97, "Possession/Creation", size=12, color="#FF9300", ha="center", fontweight="bold")
        fig.text(0.84, 0.97, "Defense", size=12, color="#D70232", ha="center", fontweight="bold")
    else:  # Midfielder
        fig.text(0.16, 0.97, "Attack", size=12, color="#1A78CF", ha="center", fontweight="bold")
        fig.text(0.50, 0.97, "Possession/Creation", size=12, color="#FF9300", ha="center", fontweight="bold")
        fig.text(0.84, 0.97, "Defense", size=12, color="#D70232", ha="center", fontweight="bold")
    
    # Set tight layout to maximize chart space
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.set_facecolor(background)
    return fig

# Chart area
with col2:
    if player_name_1:
        # Get data for player 1
        player_percentiles_1, player_values_1 = get_percentiles(player_name_1, players_1)
        
        if player_percentiles_1 is not None:
            if compare_player and player_name_2:
                # If comparing, create a 1x2 grid for charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Player 1 chart
                    st.markdown(f"### {player_name_1}")
                    fig1 = create_radar_chart(
                        player_percentiles_1, 
                        player_values_1, 
                        player_name_1, 
                        league_1,
                        comparison=True
                    )
                    st.pyplot(fig1)
                
                with chart_col2:
                    # Player 2 chart
                    player_percentiles_2, player_values_2 = get_percentiles(player_name_2, players_2)
                    
                    if player_percentiles_2 is not None:
                        st.markdown(f"### {player_name_2}")
                        fig2 = create_radar_chart(
                            player_percentiles_2, 
                            player_values_2, 
                            player_name_2, 
                            league_2,
                            comparison=True
                        )
                        st.pyplot(fig2)
                    else:
                        st.error(f"Player {player_name_2} not found in the dataset.")
                
                # Display comparison table below charts
                st.markdown("---")
                st.markdown("### Comparison Statistics")
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    f'{player_name_1} Value': player_values_1,
                    f'{player_name_1} Percentile': player_percentiles_1
                })
                
                if player_percentiles_2 is not None:
                    comparison_df[f'{player_name_2} Value'] = player_values_2
                    comparison_df[f'{player_name_2} Percentile'] = player_percentiles_2
                
                # Format the percentiles as integers with % symbol
                for col in comparison_df.columns:
                    if 'Percentile' in col:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{int(x)}%")
                
                # Display the stats table
                st.dataframe(comparison_df, hide_index=True)
                
            else:
                # Single player chart (full width)
                fig = create_radar_chart(player_percentiles_1, player_values_1, player_name_1, league_1)
                st.pyplot(fig)
                
                # Display player stats table
                st.markdown("### Player Statistics")
                # Create DataFrame for stats table - Fixed to handle numpy array
                stats_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': player_values_1,
                    'Percentile': [f"{int(p)}%" for p in player_percentiles_1]  # Fixed here - directly format numpy array
                })
                st.dataframe(stats_df, hide_index=True)
        else:
            st.error(f"Player {player_name_1} not found in the dataset.")
            
#Path: cd "C:/Users/USER/Desktop/Rembatz Analisis/Python Script"
# Execute: streamlit run "Streamlit Radar Chart.py"


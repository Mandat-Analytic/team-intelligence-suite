import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import pandas as pd
import numpy as np

# --- Configuration ---

DEFENDER_METRICS = {
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

MIDFIELDER_METRICS = {
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

FORWARD_METRICS = {
    "Non-penalty goals per 90": "Non-penalty goals\nper 90",
    "xG per 90":"xG per 90",
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

TEMPLATE_CONFIG = {
    "Defender": {
        "metrics": DEFENDER_METRICS,
        "positions": ['CB', 'RCB', 'LCB', 'LB', 'LWB', 'RB', 'RWB'],
        "colors": ["#1A78CF"] * 3 + ["#FF9300"] * 8 + ["#D70232"] * 6
    },
    "Midfielder": {
        "metrics": MIDFIELDER_METRICS,
        "positions": ['LAMF', 'RAMF', 'AMF', 'CM', 'LCM', 'RCM', 'DM', 'LDM', 'RDM'],
        "colors": ["#1A78CF"] * 4 + ["#FF9300"] * 7 + ["#D70232"] * 4
    },
    "Forward": {
        "metrics": FORWARD_METRICS,
        "positions": ['LWF', 'LW', 'LAMF', 'RW', 'RWF', 'RAMF', 'AMF', 'CF'],
        "colors": ["#1A78CF"] * 9 + ["#FF9300"] * 7 + ["#D70232"] * 2
    }
}

def get_player_percentiles(player_name, league_df, template_name):
    """Calculate percentiles for a player against their position group in the league."""
    config = TEMPLATE_CONFIG.get(template_name)
    if not config: return None, None, None
    
    metrics = config["metrics"]
    positions = config["positions"]
    
    # Filter league data for relevant positions
    # Use string matching for positions as they might be "CF, RW"
    position_mask = league_df['Position'].apply(lambda x: any(pos in str(x) for pos in positions))
    # Filter by minutes played (e.g., > 300) to avoid outliers
    minutes_mask = league_df['Minutes played'] >= 300
    
    cohort_df = league_df[position_mask & minutes_mask].copy()
    
    if player_name not in cohort_df['Player'].values:
        return None, None, None
        
    # Calculate percentiles
    player_stats = cohort_df[list(metrics.keys())]
    # Handle missing cols if any
    missing_cols = [c for c in metrics.keys() if c not in player_stats.columns]
    if missing_cols:
        # Fill missing with 0 or skip? Let's fill 0
        for c in missing_cols:
            player_stats[c] = 0
            
    percentiles = player_stats.rank(pct=True) * 100
    
    player_p_vals = percentiles[cohort_df['Player'] == player_name].values.flatten()
    player_raw_vals = cohort_df[cohort_df['Player'] == player_name][list(metrics.keys())].values.flatten()
    
    return player_p_vals, player_raw_vals, list(metrics.values())

def create_pizza_chart(player_name, percentiles, raw_values, param_labels, template_name):
    """Create the Pizza chart using mplsoccer."""
    config = TEMPLATE_CONFIG.get(template_name)
    slice_colors = config["colors"]
    
    # Format raw values
    formatted_values = []
    for i, val in enumerate(raw_values):
        # Heuristic: if label contains %, format as %, else as float
        if "%" in param_labels[i]:
            formatted_values.append(f"{val:.1f}%")
        else:
            formatted_values.append(f"{val:.2f}")
            
    # Combine label and value
    params_with_values = [f"{label}\n({val})" for label, val in zip(param_labels, formatted_values)]
    
    baker = PyPizza(
        params=params_with_values,
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-."
    )
    
    fig, ax = baker.make_pizza(
        percentiles,
        figsize=(7.5, 7.5),  # Reduced by 25%
        color_blank_space="same",
        param_location=110,
        slice_colors=slice_colors,
        value_colors=["white"] * len(percentiles),
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(facecolor="cornflowerblue", edgecolor="#000000", zorder=3, linewidth=1),
        kwargs_params=dict(color="white", fontsize=9, va="center"),
        kwargs_values=dict(
            color="#ffffff", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="cornflowerblue", edgecolor="#000000", boxstyle="round,pad=0.2", alpha=0.7),
            zorder=4
        )
    )
    
    # Update text values to be integers (percentiles)
    texts = baker.get_value_texts()
    for i, text in enumerate(texts):
        text.set_text(f"{int(percentiles[i])}")
        text.set_bbox(dict(facecolor=slice_colors[i], edgecolor="#000000", boxstyle="round,pad=0.2", alpha=0.7))
        
    # Title
    fig.text(0.5, 0.97, f"{player_name} - {template_name} Template", ha="center", fontsize=16, color="white", fontweight="bold")
    
    # Legend
    if template_name in ["Defender", "Midfielder"]:
        fig.text(0.16, 0.93, "Attack", size=10, color="#1A78CF", ha="center", fontweight="bold")
        fig.text(0.50, 0.93, "Possession", size=10, color="#FF9300", ha="center", fontweight="bold")
        fig.text(0.84, 0.93, "Defense", size=10, color="#D70232", ha="center", fontweight="bold")
    else: # Forward
        fig.text(0.16, 0.93, "Attack", size=10, color="#1A78CF", ha="center", fontweight="bold")
        fig.text(0.50, 0.93, "Possession", size=10, color="#FF9300", ha="center", fontweight="bold")
        fig.text(0.84, 0.93, "Defense", size=10, color="#D70232", ha="center", fontweight="bold")
        
    fig.set_facecolor("#0e1117") # Match Streamlit dark theme
    return fig

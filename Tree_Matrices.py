import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import textwrap

# Import custom modules
import scoring_engine
import forecasting_model
import radar_chart
import base64
import json
import data_loader
import column_mapping

# Page configuration
st.set_page_config(page_title="Team Intelligence Suite", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    h1, h2, h3, .main-header {
        font-family: 'Outfit', sans-serif !important;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        text-transform: uppercase;
        letter-spacing: 4px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants for Tagging ---
TAG_OPTIONS = {
    "Quality": [
        "Key player", "Regular starter", "Squad player", 
        "Squad player with starter potential", "Development player", 
        "Investment player", "Training player", "Versatile player"
    ],
    "Salary": ["Top earner", "High earner", "Medium earner", "Low earner"],
    "Contract": ["Licensed contract", "Contracted player", "Development contract"],
    "Player Type": [
        "Captain", "Winner", "Vocal leader", "Social leader", 
        "Neutral", "Socially negative"
    ],
    "Role": [
        "Comedian", "Motivator", "Enforcer", "Mentor", "Verbal leader", 
        "Non-verbal leader", "Team player", "Star player", 
        "Social convener", "Career distractor", "Complainer"
    ],
    "Contract Options": ["Contract extension", "Release clause"],
    "Registration": ["League", "Cup"]
}

# Color mappings for categories
TAG_COLORS = {
    "Quality": {
        "Key player": "#10b981", "Regular starter": "#10b981", # Green
        "Squad player": "#f59e0b", "Squad player with starter potential": "#f59e0b", "Versatile player": "#f59e0b", # Yellow
        "Development player": "#3b82f6", "Investment player": "#3b82f6", # Blue
        "Training player": "#ef4444" # Red
    },
    "Salary": {
        "Top earner": "#ef4444", "High earner": "#f97316", # Red/Orange
        "Medium earner": "#f59e0b", # Yellow
        "Low earner": "#10b981" # Green
    },
    "Contract": {
        "Licensed contract": "#10b981", 
        "Contracted player": "#f59e0b", 
        "Development contract": "#3b82f6"
    },
    "Role": {
        "Star player": "#8b5cf6", "Mentor": "#3b82f6", "Team player": "#10b981",
        "Comedian": "#f59e0b", "Motivator": "#f59e0b", "Enforcer": "#ef4444",
        "Complainer": "#9ca3af", "Career distractor": "#9ca3af"
    }
}

# --- Tagging Persistence ---

def load_player_tags():
    tag_file = os.path.join("database", "player_tags.json")
    if os.path.exists(tag_file):
        try:
            with open(tag_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_player_tags(tags):
    tag_file = os.path.join("database", "player_tags.json")
    os.makedirs("database", exist_ok=True)
    with open(tag_file, "w") as f:
        json.dump(tags, f, indent=4)

# --- Data Loading ---

@st.cache_data
def get_available_leagues():
    return data_loader.get_available_leagues()

@st.cache_data
def get_teams_in_league(league_name):
    return data_loader.get_teams_in_league(league_name)

@st.cache_data
def load_team_data_cached(league_name, team_name):
    df, metric_mapping = data_loader.load_team_data(league_name, team_name)
    # Standardize columns
    df = column_mapping.standardize_columns(df)
    return df, metric_mapping

@st.cache_data
def load_league_data_cached(league_name):
    # Use a simple dict as cache
    league_df = data_loader.load_league_data(league_name, cache={})
    # Standardize columns
    league_df = column_mapping.standardize_columns(league_df)
    return league_df

@st.cache_data
def load_player_data_cached(league_name):
    return data_loader.load_player_data(league_name)

# --- Tree Node Component ---

def render_full_tree(scores, kpi_thresholds, team_data_row):
    def get_style(score, kpi):
        # Color based on comparison with KPI (Benchmark)
        if score > kpi + 0.05: color = "#10b981" # Green (Increase)
        elif abs(score - kpi) <= 0.05: color = "#f59e0b" # Yellow (Same)
        else: color = "#ef4444" # Red (Decrease)
        
        if score >= kpi: arrow = "↑"; arrow_color = "#10b981"
        else: arrow = "↓"; arrow_color = "#ef4444"
        return color, arrow, arrow_color

    def render_node(name, key):
        c, a, ac = get_style(scores[name], kpi_thresholds[name])
        
        # Determine if this node is selected
        is_selected = st.session_state.selected_phase == name
        border_style = f"border-left: 6px solid {c};"
        if is_selected:
            border_style += " background: rgba(59, 130, 246, 0.2); border-color: #3b82f6;"

        # Use st.markdown for the aesthetic and a button for interaction
        st.markdown(f"""
            <style>
            .tree-node-container {{
                background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                display: flex;
                flex-direction: column;
                align-items: center;
                {border_style}
                margin-bottom: 10px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .node-header-tree {{
                font-weight: 700;
                font-size: 0.85rem;
                color: #e2e8f0;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 4px;
                text-align: center;
            }}
            .node-score-tree {{
                font-size: 1.6rem;
                font-weight: 800;
                margin: 4px 0;
            }}
            .node-target-tree {{
                font-size: 0.65rem;
                color: #94a3b8;
                margin-bottom: 8px;
            }}
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            # We use a transparent button overlay or just a button below
            # To maintain the design, we'll use a button with the text and style it
            if st.button(f"{name}", key=f"btn_{key}", use_container_width=True):
                st.session_state.selected_phase = name
                st.rerun()
            
            # Show the status info below the button
            st.markdown(f"""
                <div style="text-align: center; margin-top: -10px;">
                    <div class="node-score-tree" style="color: {c}">{scores[name]:.1f} <span style="color: {ac}">{a}</span></div>
                    <div class="node-target-tree">Target: {kpi_thresholds[name]:.1f}</div>
                </div>
            """, unsafe_allow_html=True)

    # Main Tree Layout using Streamlit columns
    st.markdown('<div class="tree-container-new">', unsafe_allow_html=True)
    
    # Level 0: Game Model
    col0_1, col0_2, col0_3 = st.columns([1, 1, 1])
    with col0_2:
        render_node('Game Model', 'gm')
    
    st.markdown("<hr style='border: 0.5px solid rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Level 1: In Possession / Out of Possession
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        render_node('In Possession', 'in_poss')
    with col1_2:
        render_node('Out of Possession', 'out_poss')
    
    st.markdown("<hr style='border: 0.5px solid rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Level 2: Sub-phases
    in_phases = ['Build Up', 'Progression', 'Transition (D to A)', 'Set Pieces (Off)', 'Output']
    out_phases = ['Pressing', 'Defensive Block', 'Transition (A to D)', 'Set Pieces (Def)', 'Output Against']
    
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        # In Possession Sub-phases
        sub_cols = st.columns(3)
        for i, p in enumerate(in_phases):
            with sub_cols[i % 3]:
                render_node(p, f"in_{i}")
                
    with col2_2:
        # Out of Possession Sub-phases
        sub_cols = st.columns(3)
        for i, p in enumerate(out_phases):
            with sub_cols[i % 3]:
                render_node(p, f"out_{i}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Squad Planner Helpers ---

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_squad_planner(league_data, selected_team):
    st.markdown("## Squad Planner")
    
    pitch_image_path = os.path.join("resource", "football-pitch-diagram.png")
    if not os.path.exists(pitch_image_path):
        st.warning("Pitch image not found in resource folder.")
        return

    bin_str = get_base64_of_bin_file(pitch_image_path)
    
    if league_data.empty:
        st.warning("No player data available.")
        return

    # Position mapping for filtering
    position_keywords = {
        'GK': ['GK'],
        'LB': ['LB', 'LWB'],
        'LCB': ['CB', 'LCB'],
        'RCB': ['CB', 'RCB'],
        'RB': ['RB', 'RWB'],
        'LDM': ['DM', 'LDM', 'CDM'],
        'RDM': ['DM', 'RDM', 'CDM'],
        'AMF': ['AMF', 'CAM', 'AM'],
        'LW': ['LW', 'LWF', 'LAMF'],
        'CF': ['CF', 'ST', 'SS'],
        'RW': ['RW', 'RWF', 'RAMF']
    }
    
    # Team column
    team_col = 'Team within selected timeframe' if 'Team within selected timeframe' in league_data.columns else 'Team'
    
    # Function to filter players by position
    def get_players_for_position(pos):
        keywords = position_keywords.get(pos, [])
        if not keywords:
            return league_data['Player'].unique().tolist()
        mask = league_data['Position'].apply(lambda x: any(kw in str(x) for kw in keywords))
        return league_data[mask]['Player'].unique().tolist()
    
    # Function to get player info
    def get_player_info(player_name):
        row = league_data[league_data['Player'] == player_name]
        if row.empty:
            return None, None, None
        row = row.iloc[0]
        age = int(row.get('Age', 0)) if pd.notna(row.get('Age', 0)) else 0
        team = row.get(team_col, 'Unknown')
        minutes = row.get('Minutes played', 0) if pd.notna(row.get('Minutes played', 0)) else 0
        return age, team, minutes
    
    # Function to get minutes percentile for a position
    def get_minutes_percentile(player_name, pos):
        keywords = position_keywords.get(pos, [])
        mask = league_data['Position'].apply(lambda x: any(kw in str(x) for kw in keywords))
        pos_players = league_data[mask]
        if pos_players.empty:
            return 50  # Default to middle
        
        player_row = pos_players[pos_players['Player'] == player_name]
        if player_row.empty:
            return 50
        
        player_minutes = player_row.iloc[0].get('Minutes played', 0)
        if pd.isna(player_minutes):
            player_minutes = 0
        
        all_minutes = pos_players['Minutes played'].fillna(0).values
        percentile = (np.sum(all_minutes <= player_minutes) / len(all_minutes)) * 100
        return percentile
    
    # Function to get color based on percentile
    def get_color_for_percentile(percentile):
        if percentile <= 33:
            return "#ef4444"  # Red
        elif percentile <= 66:
            return "#f59e0b"  # Yellow
        else:
            return "#10b981"  # Green
    
    # Initialize squad with team players (sorted by minutes)
    if 'squad_planner_data' not in st.session_state:
        st.session_state.squad_planner_data = {
            'GK': [], 'LB': [], 'LCB': [], 'RCB': [], 'RB': [],
            'LDM': [], 'RDM': [], 'AMF': [],
            'LW': [], 'CF': [], 'RW': []
        }
        # Prepopulate with team players from selected team, prioritized by Minutes played
        if team_col in league_data.columns:
            team_players = league_data[league_data[team_col] == selected_team].copy()
            for pos in st.session_state.squad_planner_data.keys():
                keywords = position_keywords.get(pos, [])
                mask = team_players['Position'].apply(lambda x: any(kw in str(x) for kw in keywords))
                pos_team_players = team_players[mask].sort_values('Minutes played', ascending=False)
                if not pos_team_players.empty:
                    st.session_state.squad_planner_data[pos] = [pos_team_players.iloc[0]['Player']]
    
    # Initialize player tags
    if 'player_tags' not in st.session_state:
        st.session_state.player_tags = load_player_tags()

    st.markdown("### Squad Selection")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        selected_position = st.selectbox("Select Position", list(st.session_state.squad_planner_data.keys()))
    
    # Get ALL players filtered by selected position (from entire league)
    filtered_players = get_players_for_position(selected_position)
    
    with col_input2:
        current_selection = st.session_state.squad_planner_data[selected_position]
        valid_current = [p for p in current_selection if p in filtered_players]
        new_selection = st.multiselect(
            f"Select Players for {selected_position}", 
            filtered_players, 
            default=valid_current,
            key=f"select_{selected_position}"
        )
        st.session_state.squad_planner_data[selected_position] = new_selection

    # --- Player Tagging UI ---
    st.markdown("---")
    with st.expander("Player Tagging", expanded=False):
        # Get all currently selected players on the pitch
        current_squad_players = []
        for players in st.session_state.squad_planner_data.values():
            current_squad_players.extend(players)
        
        if not current_squad_players:
            st.info("Add players to the squad to tag them.")
        else:
            tag_player = st.selectbox("Select Player to Tag", sorted(list(set(current_squad_players))))
            
            if tag_player not in st.session_state.player_tags:
                st.session_state.player_tags[tag_player] = {}
            
            # Tagging Inputs
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.session_state.player_tags[tag_player]["Quality"] = st.selectbox(
                    "Quality", [""] + TAG_OPTIONS["Quality"], 
                    index=TAG_OPTIONS["Quality"].index(st.session_state.player_tags[tag_player].get("Quality")) + 1 if st.session_state.player_tags[tag_player].get("Quality") in TAG_OPTIONS["Quality"] else 0,
                    key=f"qual_{tag_player}"
                )
                st.session_state.player_tags[tag_player]["Salary"] = st.selectbox(
                    "Salary", [""] + TAG_OPTIONS["Salary"],
                    index=TAG_OPTIONS["Salary"].index(st.session_state.player_tags[tag_player].get("Salary")) + 1 if st.session_state.player_tags[tag_player].get("Salary") in TAG_OPTIONS["Salary"] else 0,
                    key=f"sal_{tag_player}"
                )
            with col_t2:
                st.session_state.player_tags[tag_player]["Contract"] = st.selectbox(
                    "Contract", [""] + TAG_OPTIONS["Contract"],
                    index=TAG_OPTIONS["Contract"].index(st.session_state.player_tags[tag_player].get("Contract")) + 1 if st.session_state.player_tags[tag_player].get("Contract") in TAG_OPTIONS["Contract"] else 0,
                    key=f"cont_{tag_player}"
                )
                st.session_state.player_tags[tag_player]["Role"] = st.selectbox(
                    "Role", [""] + TAG_OPTIONS["Role"],
                    index=TAG_OPTIONS["Role"].index(st.session_state.player_tags[tag_player].get("Role")) + 1 if st.session_state.player_tags[tag_player].get("Role") in TAG_OPTIONS["Role"] else 0,
                    key=f"role_{tag_player}"
                )
            with col_t3:
                st.session_state.player_tags[tag_player]["Player Type"] = st.multiselect(
                    "Player Type", TAG_OPTIONS["Player Type"],
                    default=st.session_state.player_tags[tag_player].get("Player Type", []),
                    key=f"type_{tag_player}"
                )
                st.session_state.player_tags[tag_player]["Contract Options"] = st.multiselect(
                    "Contract Options", TAG_OPTIONS["Contract Options"],
                    default=st.session_state.player_tags[tag_player].get("Contract Options", []),
                    key=f"copt_{tag_player}"
                )
                st.session_state.player_tags[tag_player]["Registration"] = st.multiselect(
                    "Registration", TAG_OPTIONS["Registration"],
                    default=st.session_state.player_tags[tag_player].get("Registration", []),
                    key=f"reg_{tag_player}"
                )
            
            if st.button("Update Tagging"):
                save_player_tags(st.session_state.player_tags)
                st.success(f"Tags updated for {tag_player} and saved!")

    # --- Visualization Controls ---
    st.markdown("---")
    color_mode = st.selectbox("Color Code By", ["Minutes Played", "Quality", "Salary", "Contract", "Role"])
    
    # Generate Legend HTML
    legend_html = ""
    if color_mode == "Minutes Played":
        legend_items = [
            ("0-33%", "#ef4444"), ("34-66%", "#f59e0b"), ("67-100%", "#10b981")
        ]
    elif color_mode in TAG_COLORS:
        legend_items = [(k, v) for k, v in TAG_COLORS[color_mode].items()]
    else:
        legend_items = []
    
    if legend_items:
        legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;justify-content:center;">'
        for label, color in legend_items:
            legend_html += f'<div style="display:flex;align-items:center;font-size:0.8rem;"><span style="width:12px;height:12px;background-color:{color};border-radius:50%;display:inline-block;margin-right:4px;"></span>{label}</div>'
        legend_html += '</div>'

    pos_coords = {
        'GK': (90, 50), 'LB': (75, 15), 'LCB': (75, 38), 'RCB': (75, 62), 'RB': (75, 85),
        'LDM': (60, 38), 'RDM': (60, 62), 'AMF': (40, 50),
        'LW': (20, 15), 'CF': (15, 50), 'RW': (20, 85)
    }
    
    player_cards_html = ""
    for pos, players in st.session_state.squad_planner_data.items():
        if players:
            top, left = pos_coords[pos]
            card_content = ""
            for p in players:
                age, team, minutes = get_player_info(p)
                
                # Determine Color
                color = "#94a3b8" # Default gray
                if color_mode == "Minutes Played":
                    percentile = get_minutes_percentile(p, pos)
                    color = get_color_for_percentile(percentile)
                elif color_mode in TAG_COLORS:
                    # Get tag value for player
                    tags = st.session_state.player_tags.get(p, {})
                    val = tags.get(color_mode)
                    if val and val in TAG_COLORS[color_mode]:
                        color = TAG_COLORS[color_mode][val]
                
                # Profile card with color-coded border
                card_content += f'<div class="profile-card" style="border-left:4px solid {color};"><div class="player-name">{p}</div><div class="player-info">Age: {age} | {team}</div></div>'
            player_cards_html += f'<div class="player-card-container" style="top:{top}%;left:{left}%;"><div class="position-label">{pos}</div>{card_content}</div>'

    # Use components.html for reliable HTML rendering
    pitch_html = f'''<!DOCTYPE html><html><head><style>body{{margin:0;padding:0;background:transparent;}}.pitch-container{{position:relative;width:100%;max-width:600px;height:900px;margin:0 auto;background-image:url("data:image/png;base64,{bin_str}");background-size:contain;background-position:center;background-repeat:no-repeat;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.3);}}.player-card-container{{position:absolute;transform:translate(-50%,-50%);background-color:rgba(15,23,42,0.95);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:6px 10px;text-align:center;min-width:120px;box-shadow:0 2px 8px rgba(0,0,0,0.5);}}.player-card-container:hover{{z-index:10;transform:translate(-50%,-50%) scale(1.05);border-color:#3b82f6;}}.position-label{{font-size:0.65rem;color:#94a3b8;font-weight:bold;text-transform:uppercase;margin-bottom:4px;padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,0.1);}}.profile-card{{background:rgba(0,0,0,0.3);border-radius:4px;padding:4px 6px;margin:3px 0;text-align:left;}}.player-name{{font-size:0.8rem;color:#f8fafc;font-weight:600;white-space:nowrap;}}.player-info{{font-size:0.6rem;color:#94a3b8;white-space:nowrap;}}</style></head><body><div style="color:#e0e0e0;text-align:center;margin-bottom:5px;">{legend_html}</div><div class="pitch-container">{player_cards_html}</div></body></html>'''
    
    components.html(pitch_html, height=980)

    # --- Summary Table ---
    st.markdown("### Squad Summary")
    
    current_squad_players = []
    for pos, players in st.session_state.squad_planner_data.items():
        for p in players:
            current_squad_players.append((p, pos))
            
    if current_squad_players:
        table_data = []
        for p_name, pos in current_squad_players:
            age, team, minutes = get_player_info(p_name)
            tags = st.session_state.player_tags.get(p_name, {})
            
            row = {
                "Player": p_name,
                "Age": age,
                "Minutes Played": minutes,
                "Position": pos,
                "Quality": tags.get("Quality", ""),
                "Salary": tags.get("Salary", ""),
                "Contract": tags.get("Contract", ""),
                "Role": tags.get("Role", ""),
                "Player Type": ", ".join(tags.get("Player Type", [])),
                "Contract Options": ", ".join(tags.get("Contract Options", [])),
                "Registration": ", ".join(tags.get("Registration", []))
            }
            table_data.append(row)
            
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No players in squad to display.")

# --- Main App ---

def main():
    st.sidebar.header("CONFIGURATION")
    
    # Initialize session state for interactivity
    if 'selected_phase' not in st.session_state:
        st.session_state.selected_phase = None
    if 'show_top_players' not in st.session_state:
        st.session_state.show_top_players = False
        
    page = st.sidebar.radio("Navigation", ["Performance Tree", "Deep Analytics", "Trend Analytics"])
    
    # League Selection
    available_leagues = get_available_leagues()
    if not available_leagues:
        st.error("No leagues found in database/Team Stats/")
        return
    
    selected_league = st.sidebar.selectbox("League", available_leagues)
    
    # Team Selection
    available_teams = get_teams_in_league(selected_league)
    if not available_teams:
        st.error(f"No teams found in {selected_league}")
        return
    
    selected_team = st.sidebar.selectbox("Team", available_teams)
    
    # Load team data
    try:
        team_data, metric_mapping = load_team_data_cached(selected_league, selected_team)
    except Exception as e:
        st.error(f"Error loading team data: {str(e)}")
        return
    
    if team_data is None or team_data.empty:
        st.error("No data available for selected team")
        return
    
    # Standardize Match numbering: Earliest date should be Match 1
    if 'Date' in team_data.columns:
        team_data['Date'] = pd.to_datetime(team_data['Date'])
        team_data = team_data.sort_values('Date').reset_index(drop=True)
    
    # Match Selection
    match_options = [f"Match {i+1} ({row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else 'N/A'})" 
                     for i, row in team_data.iterrows()]
    selected_match_idx = st.sidebar.selectbox("Match", range(len(match_options)), format_func=lambda x: match_options[x])
    
    st.sidebar.markdown("---")
    
    # Benchmark Selection with League-aware comparisons
    kpi_mode = st.sidebar.radio("Benchmark", ["Season Avg", "League Avg", "Top 25%", "Custom"])
    custom_target = 80
    if kpi_mode == "Custom":
        custom_target = st.sidebar.slider("Target", 0, 100, 80)
    
    # Calculate scores for current match
    current_team_row = team_data.iloc[selected_match_idx]
    current_opp_row = pd.Series()  # Opponent data is in same row
    
    scores = scoring_engine.calculate_match_scores(current_team_row, current_opp_row)
    
    # Calculate all scores for team's season
    all_scores_list = []
    for i in range(len(team_data)):
        try:
            match_scores = scoring_engine.calculate_match_scores(team_data.iloc[i], pd.Series())
            all_scores_list.append(match_scores)
        except Exception as e:
            print(f"Error scoring match {i}: {str(e)}")
            continue
    
    all_scores_df = pd.DataFrame(all_scores_list) if all_scores_list else pd.DataFrame()
    
    # Calculate KPI thresholds based on selected mode
    if kpi_mode == "Season Avg" and not all_scores_df.empty:
        kpi_thresholds = all_scores_df.mean().to_dict()
        st.sidebar.info(f"📊 Comparing vs team's season average")
        
    elif kpi_mode == "League Avg":
        # Load and calculate league average
        with st.spinner(f"Loading league data for {selected_league}..."):
            league_df = load_league_data_cached(selected_league)
            
        if not league_df.empty:
            # Calculate scores for all league matches
            league_scores = []
            for i in range(min(len(league_df), 500)):  # Limit to avoid timeout
                try:
                    match_scores = scoring_engine.calculate_match_scores(league_df.iloc[i], pd.Series())
                    league_scores.append(match_scores)
                except:
                    continue
            
            league_scores_df = pd.DataFrame(league_scores) if league_scores else pd.DataFrame()
            if not league_scores_df.empty:
                kpi_thresholds = league_scores_df.mean().to_dict()
                st.sidebar.success(f"✅ Calculated from {len(league_scores_df)} league matches")
            else:
                kpi_thresholds = {col: 70 for col in scores.keys()}
                st.sidebar.warning("⚠️  Using default values")
        else:
            kpi_thresholds = {col: 70 for col in scores.keys()}
            st.sidebar.warning("⚠️  League data not available")
            
    elif kpi_mode == "Top 25%":
        # Load and calculate top 25% (75th percentile)
        with st.spinner(f"Loading league data for {selected_league}..."):
            league_df = load_league_data_cached(selected_league)
            
        if not league_df.empty:
            league_scores = []
            for i in range(min(len(league_df), 500)):
                try:
                    match_scores = scoring_engine.calculate_match_scores(league_df.iloc[i], pd.Series())
                    league_scores.append(match_scores)
                except:
                    continue
            
            league_scores_df = pd.DataFrame(league_scores) if league_scores else pd.DataFrame()
            if not league_scores_df.empty:
                kpi_thresholds = league_scores_df.quantile(0.75).to_dict()
                st.sidebar.success(f"✅ 75th percentile from {len(league_scores_df)} matches")
            else:
                kpi_thresholds = {col: 80 for col in scores.keys()}
                st.sidebar.warning("⚠️  Using default values")
        else:
            kpi_thresholds = {col: 80 for col in scores.keys()}
            st.sidebar.warning("⚠️  League data not available")
            
    else:  # Custom
        kpi_thresholds = {col: custom_target for col in scores.keys()}
    
    # Render appropriate page
    if page == "Performance Tree":
        st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
        
        # Display the tree
        render_full_tree(scores, kpi_thresholds, current_team_row)
        
        # Dynamic Content underneath the tree
        if st.session_state.selected_phase:
            phase = st.session_state.selected_phase
            st.markdown("---")
            
            # Action Buttons
            col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
            with col_act1:
                if st.button("Hide", use_container_width=True):
                    st.session_state.selected_phase = None
                    st.session_state.show_top_players = False
                    st.rerun()
            with col_act2:
                btn_label = "Hide Top Contributing Players" if st.session_state.show_top_players else "Show Top Contributing Players (Percentile Rank)"
                if st.button(btn_label, use_container_width=True):
                    st.session_state.show_top_players = not st.session_state.show_top_players
                    st.rerun()
            
            # Trend Chart
            st.markdown(f"### {phase} Performance Trend")
            if not all_scores_df.empty and phase in all_scores_df.columns:
                trend_data = all_scores_df[[phase]].copy()
                trend_data['Match Number'] = [f"Match {i+1}" for i in range(len(trend_data))]
                benchmark_val = kpi_thresholds.get(phase, 70)
                
                fig = go.Figure()
                # Benchmark line
                fig.add_shape(type="line", x0=-0.5, y0=benchmark_val, x1=len(trend_data)-0.5, y1=benchmark_val,
                             line=dict(color="rgba(148, 163, 184, 0.5)", width=2, dash="dash"))
                
                # Performance line
                fig.add_trace(go.Scatter(
                    x=trend_data['Match Number'],
                    y=trend_data[phase],
                    mode='lines+markers+text',
                    name=phase,
                    text=[f"{v:.1f}" for v in trend_data[phase]],
                    textposition="top center",
                    line=dict(color="#3b82f6", width=4, shape='spline'),
                    marker=dict(
                        size=10,
                        color=["#10b981" if v >= benchmark_val else "#ef4444" for v in trend_data[phase]]
                    )
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(15, 23, 42, 0.5)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 110]),
                    margin=dict(l=20, r=20, t=40, b=20), height=400, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top Contributing Players (if toggled)
            if st.session_state.show_top_players:
                st.markdown("---")
                st.markdown(f"### Top Contributing Players for {phase}")
                
                # We need to map phase names to analysis phases used in Deep Analytics
                phase_map = {
                    "Game Model": "Output", # Fallback or specific mapping
                    "In Possession": "Progression",
                    "Out of Possession": "Pressing",
                    "Build Up": "Build Up",
                    "Progression": "Progression",
                    "Transition (D to A)": "Transition D→A",
                    "Set Pieces (Off)": "Offensive Set Pieces",
                    "Output": "Output",
                    "Pressing": "Pressing",
                    "Defensive Block": "Defensive Block",
                    "Transition (A to D)": "Transition A→D",
                    "Set Pieces (Def)": "Defensive Set Pieces",
                    "Output Against": "Output Against"
                }
                
                analytics_phase = phase_map.get(phase, "Build Up")
                league_player_data = load_player_data_cached(selected_league)
                
                # Extract Top Players Logic from Deep Analytics (reusable)
                PHASE_INFO_LOCAL = {
                    "Build Up": {"player_metrics": ["Passes per 90", "Accurate passes, %", "Back passes per 90", "Lateral passes per 90"]},
                    "Progression": {"player_metrics": ["Forward passes per 90", "Progressive passes per 90", "Deep completions per 90", "Progressive runs per 90", "xA per 90"]},
                    "Transition D→A": {"player_metrics": ["Accelerations per 90", "Progressive runs per 90", "Successful defensive actions per 90"]},
                    "Offensive Set Pieces": {"player_metrics": ["Crosses per 90", "Accurate crosses, %", "Aerial duels won, %"]},
                    "Output": {"player_metrics": ["Goals per 90", "xG per 90", "Shots on target, %"]},
                    "Pressing": {"player_metrics": ["Defensive duels per 90", "Defensive duels won, %", "PAdj Interceptions"]},
                    "Defensive Block": {"player_metrics": ["Shots blocked per 90", "Aerial duels won, %", "Successful defensive actions per 90"]},
                    "Transition A→D": {"player_metrics": ["PAdj Sliding tackles", "Fouls per 90", "Red cards per 90"]},
                    "Defensive Set Pieces": {"player_metrics": ["Aerial duels won, %", "Aerial duels per 90", "Clearances per 90"]},
                    "Output Against": {"player_metrics": ["Shots blocked per 90", "Interceptions per 90", "Successful defensive actions per 90"]}
                }
                
                # (Remaining logic similar to Deep Analytics but condensed)
                if analytics_phase in PHASE_INFO_LOCAL and not league_player_data.empty:
                    team_players = league_player_data[league_player_data['Team'] == selected_team].copy().drop_duplicates(subset=['Player'])
                    metrics = PHASE_INFO_LOCAL[analytics_phase]["player_metrics"]
                    available_metrics = [m for m in metrics if m in team_players.columns]
                    
                    if available_metrics:
                        team_players['Phase Score'] = 0
                        for m in available_metrics:
                            max_v = team_players[m].max()
                            if max_v > 0: team_players['Phase Score'] += (team_players[m] / max_v)
                        
                        top_p = team_players.sort_values('Phase Score', ascending=False).head(10)
                        
                        for _, row in top_p.iterrows():
                            p_name, pos = row['Player'], row['Position'] if 'Position' in row else "N/A"
                            st.markdown(f"**{p_name}** ({pos})")
                            cols = st.columns(len(available_metrics))
                            for i, m in enumerate(available_metrics):
                                pct = radar_chart.calculate_player_percentile(row[m], m, "Midfielder", league_player_data) # Simplified pos_group
                                color = "green" if pct >= 67 else "orange" if pct >= 34 else "red"
                                with cols[i]:
                                    st.caption(m)
                                    st.progress(int(pct)/100)
                                    st.markdown(f":{color}[**{int(pct)}**]")
                            st.markdown("---")

    elif page == "Deep Analytics":
        st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
        st.subheader("DEEP ANALYTICS - Overall Team Strategy & Player Impact")
        
        # Load League Data for Percentiles
        league_player_data = load_player_data_cached(selected_league)
        
        # --- 1. Team Phase Analysis (Pizza Charts) ---
        st.markdown("### 1. Team Phase Performance (Season Average)")
        
        if team_data.empty:
            st.warning("No data available for this team.")
            return

        # Calculate Season Averages for Team
        numeric_cols = team_data.select_dtypes(include=[np.number]).columns
        season_avg = team_data[numeric_cols].mean()
        
        # Helper to get phase scores
        scorer = scoring_engine.GameModelScorer()
        
        in_possession_phases = {
            "Build Up": scorer.score_build_up_phase,
            "Progression": scorer.score_progression_phase,
            "Transition D→A": scorer.score_transition_d_to_a,
            "Offensive Set Pieces": scorer.score_offensive_set_pieces,
            "Output": scorer.score_output_phase
        }
        
        out_possession_phases = {
            "Pressing": scorer.score_pressing_phase,
            "Defensive Block": scorer.score_defensive_block,
            "Transition A→D": scorer.score_transition_a_to_d,
            "Defensive Set Pieces": scorer.score_defensive_set_pieces,
            "Output Against": scorer.score_output_against
        }
        
        # --- LEAGUE CONTEXT CALCULATION ---
        with st.spinner("Calculating league context..."):
            league_matches = load_league_data_cached(selected_league)
            
            if not league_matches.empty:
                # 1. Calculate Season Avg for every team in league
                league_teams_avg = league_matches.groupby('Team')[numeric_cols].mean()
                
                # 2. Calculate phase scores for every team
                league_team_phase_scores = []
                for t_name, t_avg in league_teams_avg.iterrows():
                    res = {}
                    for name, func in in_possession_phases.items():
                        res[name] = func(t_avg) * 10
                    for name, func in out_possession_phases.items():
                        res[name] = func(t_avg) * 10
                    league_team_phase_scores.append(res)
                
                league_scores_df = pd.DataFrame(league_team_phase_scores)
                
                # 3. Calculate current team absolute scores
                current_team_abs = {}
                for name, func in in_possession_phases.items():
                    current_team_abs[name] = func(season_avg) * 10
                for name, func in out_possession_phases.items():
                    current_team_abs[name] = func(season_avg) * 10
                
                current_team_abs_df = pd.DataFrame([current_team_abs])
                
                # 4. Use scoring_engine to normalize
                norm_results = scoring_engine.normalize_scores_to_league(current_team_abs_df, league_scores_df)
                
                # 5. Extract results for visualization
                in_poss_scores = {name: norm_results[name]['min_max'] for name in in_possession_phases}
                in_poss_percentiles = {name: norm_results[name]['percentile'] for name in in_possession_phases}
                
                out_poss_scores = {name: norm_results[name]['min_max'] for name in out_possession_phases}
                out_poss_percentiles = {name: norm_results[name]['percentile'] for name in out_possession_phases}
            else:
                # Fallback to absolute scores if league data missing
                in_poss_scores = {name: int(func(season_avg) * 10) for name, func in in_possession_phases.items()}
                in_poss_percentiles = in_poss_scores
                out_poss_scores = {name: int(func(season_avg) * 10) for name, func in out_possession_phases.items()}
                out_poss_percentiles = out_poss_scores

        # Display Pizza Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**In Possession Strategy**")
            fig_in = radar_chart.create_phase_pizza(in_poss_scores, in_poss_percentiles, "In Possession")
            st.pyplot(fig_in, use_container_width=True)
            
        with col2:
            st.markdown("**Out of Possession Strategy**")
            fig_out = radar_chart.create_phase_pizza(out_poss_scores, out_poss_percentiles, "Out of Possession")
            st.pyplot(fig_out, use_container_width=True)
            
        st.markdown("---")
        
        # --- 2. Player Contributions (Percentiles) ---
        st.markdown("### 2. Top Contributing Players (Percentile Rank)")
        
        # Phase Selector
        selected_phase = st.selectbox(
            "Select Analysis Phase",
            list(in_possession_phases.keys()) + list(out_possession_phases.keys())
        )
        
        # Define Phase Definitions (Metrics)
        PHASE_INFO = {
            "Build Up": {
                "player_metrics": ["Passes per 90", "Accurate passes, %", "Back passes per 90", "Lateral passes per 90"],
                "team_metrics": ["Back passes %", "Lateral passes %", "Possession, %", "Passes %"]
            },
            "Progression": {
                "player_metrics": ["Forward passes per 90", "Progressive passes per 90", "Deep completions per 90", "Progressive runs per 90", "xA per 90"],
                "team_metrics": ["Forward passes %", "Progressive passes %", "Passes to final third %"]
            },
            "Transition D→A": {
                "player_metrics": ["Accelerations per 90", "Progressive runs per 90", "Successful defensive actions per 90"],
                "team_metrics": ["Counterattacks Total", "Recoveries High", "PPDA"]
            },
            "Offensive Set Pieces": {
                "player_metrics": ["Crosses per 90", "Accurate crosses, %", "Aerial duels won, %"],
                "team_metrics": ["Crosses %", "Set pieces with shots", "Corners with shots"]
            },
            "Output": {
                "player_metrics": ["Goals per 90", "xG per 90", "Shots on target, %"],
                "team_metrics": ["Goals", "xG", "Shots %", "Shots on target"]
            },
            "Pressing": {
                "player_metrics": ["Defensive duels per 90", "Defensive duels won, %", "PAdj Interceptions"],
                "team_metrics": ["Defensive duels %", "Recoveries High", "PPDA"]
            },
            "Defensive Block": {
                "player_metrics": ["Shots blocked per 90", "Aerial duels won, %", "Successful defensive actions per 90"],
                "team_metrics": ["Defensive duels %", "Aerial duels %", "Sliding tackles %"]
            },
            "Transition A→D": {
                "player_metrics": ["PAdj Sliding tackles", "Fouls per 90", "Red cards per 90"],
                "team_metrics": ["Losses High", "Recoveries Medium", "Fouls"] 
            },
            "Defensive Set Pieces": {
                "player_metrics": ["Aerial duels won, %", "Aerial duels per 90", "Clearances per 90"],
                "team_metrics": ["Aerial duels %", "Clearances", "Shots against Total"]
            },
            "Output Against": {
                "player_metrics": ["Shots blocked per 90", "Interceptions per 90", "Successful defensive actions per 90"],
                "team_metrics": ["Shots against %", "xG against", "Conceded goals"]
            }
        }
        
        phase_data = PHASE_INFO.get(selected_phase)
        
        # --- PHASE METRICS RADAR CHART ---
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown(f"**{selected_phase} Metrics**")
            # Calculate Team Metrics and Percentiles
            team_metrics = phase_data.get("team_metrics", [])
            
            if team_metrics:
                # 1. Get Team Raw Values (Season Avg)
                team_raw = season_avg.get(team_metrics, pd.Series()).fillna(0).to_dict()
                
                # 2. Get Score/Percentile for visualization
                # For simplicty, we'll calculate percentile against LEAGUE TEAMS
                # We need league data for this. Since we don't have a "load_league_teams" handy,
                # we'll use a simpler approach: Map raw values to a logical 0-100 scale or just load league data if fast enough.
                # Actually, earlier we loaded league_player_data. We don't have league TEAM data loaded for comparison easily here.
                # Use a heuristic for now? Or load the league data properly.
                # Let's check if 'load_league_data_cached' (returns matches) is available.
                
                league_matches = load_league_data_cached(selected_league)
                metric_percentiles = {}
                
                if not league_matches.empty:
                    # Resolve column names (handle short vs long mapping)
                    from column_mapping import COLUMN_MAPPING
                    
                    actual_metrics = []
                    display_names = {} # Map actual col name -> display name (short)
                    
                    for m in team_metrics:
                        if m in league_matches.columns:
                            actual_metrics.append(m)
                            display_names[m] = m
                        elif m in COLUMN_MAPPING and COLUMN_MAPPING[m] in league_matches.columns:
                            long_name = COLUMN_MAPPING[m]
                            actual_metrics.append(long_name)
                            display_names[long_name] = m
                    
                    if actual_metrics:
                        # Group by team to get team averages
                        league_teams_avg = league_matches.groupby('Team')[actual_metrics].mean()
                        
                        metric_percentiles = {}
                        team_raw_resolved = {}
                        
                        for col in actual_metrics:
                            # Get raw value from team data
                            val = team_raw.get(col, season_avg.get(col, 0))
                            
                            short_name = display_names[col]
                            team_raw_resolved[short_name] = val
                            
                            # Percentile rank
                            if col in league_teams_avg.columns:
                                # Percentile of team value in league distribution
                                pct = (league_teams_avg[col] < val).mean() * 100
                                metric_percentiles[short_name] = pct
                            else:
                                metric_percentiles[short_name] = 50 # Default
                                
                        # Create Chart
                        fig_phase = radar_chart.create_metric_pizza(team_raw_resolved, metric_percentiles, f"{selected_phase} Profile")
                        st.pyplot(fig_phase, use_container_width=True)
                    else:
                        st.warning("Metrics not found in league data columns")
                else:
                    st.warning("League data unavailable for comparison")
            else:
                st.info("No metrics defined for this phase")

        # --- TOP PLAYERS ---
        with c2:
            st.markdown(f"**Top Contributors**")
            if not league_player_data.empty and 'Team' in league_player_data.columns:
                team_players = league_player_data[league_player_data['Team'] == selected_team].copy()
                
                # DEDUPLICATE PLAYERS
                team_players = team_players.drop_duplicates(subset=['Player'])
                
                if not team_players.empty:
                    # Calculate composite score for sorting
                    available_metrics = [m for m in phase_data["player_metrics"] if m in team_players.columns]
                    
                    if available_metrics:
                        team_players['Phase Score'] = 0
                        for metric in available_metrics:
                            max_val = team_players[metric].max()
                            if max_val > 0:
                                team_players['Phase Score'] += (team_players[metric] / max_val)
                        
                        # Sort (Show all players)
                        top_players = team_players.sort_values('Phase Score', ascending=False)
                        
                        # Display as formatted list
                        for idx, row in top_players.iterrows():
                            player_name = row['Player']
                            position = row['Position'] if 'Position' in row else "Unknown"
                            
                            # Determine simpler position group for comparison
                            pos_str = str(position)
                            if "GK" in pos_str: pos_group = "Goalkeeper"
                            elif any(x in pos_str for x in ["CB", "LB", "RB", "WB"]): pos_group = "Defender"
                            elif any(x in pos_str for x in ["CF", "RW", "LW", "FW"]): pos_group = "Forward"
                            else: pos_group = "Midfielder"
                            
                            st.markdown(f"**{player_name}** ({position})")
                            
                            cols = st.columns(len(available_metrics))
                            for i, metric in enumerate(available_metrics):
                                val = row[metric]
                                
                                # Calculate percentile
                                pct = radar_chart.calculate_player_percentile(
                                    val, metric, pos_group, league_player_data
                                )
                                
                                # Color logic
                                if pct >= 67: color = "green"
                                elif pct >= 34: color = "orange" # Streamlit yellow is orange
                                else: color = "red"
                                
                                with cols[i]:
                                    st.caption(f"{metric}")
                                    st.progress(int(pct) / 100)
                                    st.markdown(f":{color}[**{int(pct)}**] ({val:.2f})") # Removed 'th Percentile' text
                                    
                            st.markdown("---")
                    else:
                        st.warning("Relevant metrics not found in player data.")
            else:
                st.error("Player data not available.")

    elif page == "Trend Analytics":
        st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
        st.subheader("TREND ANALYTICS - Performance Flow Over Time")
        
        if all_scores_df.empty:
            st.warning("No performance data available to show trends.")
            return

        # 1. Selection of Phase
        phases_to_chart = [
            "Game Model", 
            "In Possession", "Out of Possession",
            "Build Up", "Progression", "Transition (D to A)", "Set Pieces (Off)", "Output",
            "Pressing", "Defensive Block", "Transition (A to D)", "Set Pieces (Def)", "Output Against"
        ]
        
        # Filter only existing columns in scores
        available_phases = [p for p in phases_to_chart if p in all_scores_df.columns]
        
        selected_trend_phase = st.selectbox("Select Phase to Analyze", available_phases)
        
        # 2. Prepare Data for Charting
        trend_df = all_scores_df[[selected_trend_phase]].copy()
        trend_df['Match Number'] = [f"Match {i+1}" for i in range(len(trend_df))]
        
        # Get dates for tooltip
        dates = []
        for i in range(len(trend_df)):
            if i < len(team_data):
                d = team_data.iloc[i]['Date']
                dates.append(d.strftime('%Y-%m-%d') if pd.notna(d) else f'M{i+1}')
            else:
                dates.append(f'M{i+1}')
        trend_df['Date'] = dates
        
        benchmark_val = kpi_thresholds.get(selected_trend_phase, 70)
        
        # 3. Create Plotly Line Chart
        fig = go.Figure()

        # Benchmark line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=benchmark_val,
            x1=len(trend_df)-0.5,
            y1=benchmark_val,
            line=dict(color="rgba(148, 163, 184, 0.5)", width=2, dash="dash"),
            name="Benchmark"
        )
        
        fig.add_annotation(
            x=len(trend_df)-1 if len(trend_df) > 0 else 0,
            y=benchmark_val,
            text=f"Benchmark: {benchmark_val:.1f}",
            showarrow=False,
            yshift=15,
            font=dict(color="#94a3b8", size=12),
            bgcolor="rgba(15, 23, 42, 0.8)"
        )

        # Performance line
        fig.add_trace(go.Scatter(
            x=trend_df['Match Number'],
            y=trend_df[selected_trend_phase],
            mode='lines+markers+text',
            name=selected_trend_phase,
            text=[f"{v:.1f}" for v in trend_df[selected_trend_phase]],
            textposition="top center",
            line=dict(color="#3b82f6", width=4, shape='spline'),
            marker=dict(
                size=12,
                color=["#10b981" if v >= benchmark_val else "#ef4444" for v in trend_df[selected_trend_phase]],
                line=dict(color="#1e293b", width=2)
            ),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<br>Date: %{customdata}<extra></extra>",
            customdata=trend_df['Date']
        ))

        # Update Layout
        fig.update_layout(
            title=dict(
                text=f"{selected_trend_phase} Performance Trend",
                x=0.5,
                font=dict(size=22, color="#f8fafc", family="Outfit, sans-serif")
            ),
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Matches",
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#94a3b8"),
                showgrid=True
            ),
            yaxis=dict(
                title="Performance Score",
                range=[0, min(110, max(trend_df[selected_trend_phase].max() + 20, 100))],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#94a3b8"),
                showgrid=True
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            height=500,
            showlegend=False,
            hoverlabel=dict(bgcolor="#1e293b", font_size=14, font_family="Inter")
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Insightful cards below
        st.markdown("### Key Observations")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        avg_score = trend_df[selected_trend_phase].mean()
        max_score = trend_df[selected_trend_phase].max()
        current_score = trend_df[selected_trend_phase].iloc[-1]
        
        with col_m1:
            st.metric("Average Score", f"{avg_score:.1f}", f"{avg_score - benchmark_val:+.1f} vs Target")
        with col_m2:
            st.metric("Peak Performance", f"{max_score:.1f}", f"{max_score - avg_score:+.1f} vs Avg")
        with col_m3:
            prev_score = trend_df[selected_trend_phase].iloc[-2] if len(trend_df) > 1 else current_score
            delta = current_score - prev_score
            st.metric("Latest Form", f"{current_score:.1f}", f"{delta:+.1f} vs Prev")
        with col_m4:
            consistency = 100 - trend_df[selected_trend_phase].std()
            st.metric("Consistency Index", f"{consistency:.1f}", help="Based on standard deviation across matches")

if __name__ == "__main__":
    main()
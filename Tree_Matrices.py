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

# Page configuration
st.set_page_config(page_title="Team Intelligence Suite", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255,255,255,0.05);
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
def get_available_teams():
    team_dir = os.path.join("database", "Team Stats")
    if not os.path.exists(team_dir): return []
    files = glob.glob(os.path.join(team_dir, "*.xlsx"))
    teams = []
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith("Team Stats ") and basename.endswith(".xlsx"):
            teams.append(basename.replace("Team Stats ", "").replace(".xlsx", ""))
    return teams

@st.cache_data
def load_data(team_name):
    team_file = os.path.join("database", "Team Stats", f"Team Stats {team_name}.xlsx")
    if not os.path.exists(team_file): return None, None, None, None
    
    df = pd.read_excel(team_file)
    if 'Team' in df.columns:
        team_data = df[df['Team'] == team_name].copy()
        matches = []
        for i in range(0, len(df) - 1, 2): 
            row1 = df.iloc[i]; row2 = df.iloc[i+1]
            if row1['Team'] == team_name: matches.append((row1, row2))
            elif row2['Team'] == team_name: matches.append((row2, row1))
        
        if matches:
            team_rows = pd.DataFrame([m[0] for m in matches])
            opp_rows = pd.DataFrame([m[1] for m in matches])
        else:
            team_rows, opp_rows = team_data, pd.DataFrame()
    else:
        team_rows, opp_rows = df, pd.DataFrame()

    league_file = os.path.join("database", "Player Stats", "Serbia Super Liga 25_26.xlsx")
    player_df = pd.DataFrame()
    league_df = pd.DataFrame()
    if os.path.exists(league_file):
        df_league = pd.read_excel(league_file)
        league_df = df_league.copy()
        if 'Team' in df_league.columns:
            player_df = df_league[df_league['Team'] == team_name].copy()
        
    return team_rows, opp_rows, player_df, league_df

# --- Tree Node Component ---

def render_full_tree(scores, kpi_thresholds, team_data_row):
    def get_style(score, kpi):
        if score >= 80: color = "#10b981"
        elif score >= 60: color = "#f59e0b"
        else: color = "#ef4444"
        if score >= kpi: arrow = "↑"; arrow_color = "#10b981"
        else: arrow = "↓"; arrow_color = "#ef4444"
        return color, arrow, arrow_color

    def get_metrics_html(metrics_dict):
        parts = ['<div class="metrics-container">']
        for m_name, m_val in metrics_dict.items():
            parts.append(f'<div class="metric-row"><span>{m_name}</span><span>{m_val:.1f}</span></div>')
        parts.append('</div>')
        return "".join(parts)

    nodes = {}
    c, a, ac = get_style(scores['Game Model'], kpi_thresholds['Game Model'])
    nodes['gm'] = {'score': scores['Game Model'], 'kpi': kpi_thresholds['Game Model'], 'color': c, 'arrow': a, 'arrow_color': ac}
    
    for name in ['In Possession', 'Out of Possession']:
        c, a, ac = get_style(scores[name], kpi_thresholds[name])
        nodes[name] = {'score': scores[name], 'kpi': kpi_thresholds[name], 'color': c, 'arrow': a, 'arrow_color': ac}

    in_phases = ['Build Up', 'Progression', 'Transition (D to A)', 'Set Pieces (Off)', 'Output']
    in_metrics_map = {
        'Build Up': {'Back Pass %': team_data_row.get('Percentage of accurate back passes', 0), 'Lat Pass %': team_data_row.get('Percentage of accurate lateral passes', 0), 'Poss %': team_data_row.get('Possession, %', 0)},
        'Progression': {'Fwd Pass %': team_data_row.get('Percentage of accurate forward passes', 0), 'Prog Pass %': team_data_row.get('Percentage of acccurate progressive passes', 0), 'Final 3rd %': team_data_row.get('Percentage of accurate passes to final third', 0)},
        'Transition (D to A)': {'CA Shots': team_data_row.get('Total counterattacks with shots', 0), 'Hi Recov': team_data_row.get('Recoveries (high)', 0), 'PPDA': team_data_row.get('PPDA', 0)},
        'Set Pieces (Off)': {'SP Shots': team_data_row.get('Total set pieces with shots', 0), 'Corn Shots': team_data_row.get('Total corners with shots', 0), 'Cross %': team_data_row.get('Percentage of accurate crosses', 0)},
        'Output': {'Goals': team_data_row.get('Goals', 0), 'xG': team_data_row.get('xG', 0), 'SoT %': team_data_row.get('Percentage of shots on target', 0)}
    }
    out_phases = ['Pressing', 'Defensive Block', 'Transition (A to D)', 'Set Pieces (Def)', 'Output Against']
    out_metrics_map = {
        'Pressing': {'PPDA': team_data_row.get('PPDA', 0), 'Hi Recov': team_data_row.get('Recoveries (high)', 0), 'Int': team_data_row.get('Interceptions', 0)},
        'Defensive Block': {'Def Duel %': team_data_row.get('Percentage defensive duels  won', 0), 'Aerial %': team_data_row.get('Percentage aerial duels  won', 0), 'Tackle %': team_data_row.get('Percentage successful sliding tackles', 0)},
        'Transition (A to D)': {'Loss High': team_data_row.get('Ball losses (high)', 0), 'Recov Med': team_data_row.get('Recoveries (medium)', 0), 'Fouls': team_data_row.get('Fouls', 0)},
        'Set Pieces (Def)': {'Shots Ag': team_data_row.get('Total shots against', 0), 'Aerial %': team_data_row.get('Percentage aerial duels  won', 0), 'Clear': team_data_row.get('Clearances', 0)},
        'Output Against': {'Conc': team_data_row.get('Conceded goals', 0), 'Shots Ag': team_data_row.get('Total shots against', 0), 'SoT Ag %': team_data_row.get('Percentage of shots against on target', 0)}
    }

    def generate_sub_nodes(phases, metrics_map):
        parts = []
        for p in phases:
            c, a, ac = get_style(scores[p], kpi_thresholds[p])
            m_html = get_metrics_html(metrics_map.get(p, {}))
            parts.append(f'<div class="tree-node sub-node"><div class="node-header">{p}</div><div class="node-score" style="color: {c}">{scores[p]:.1f} <span style="color: {ac}">{a}</span></div><div class="node-target">Target: {kpi_thresholds[p]:.1f}</div>{m_html}</div>')
        return "".join(parts)

    in_html = generate_sub_nodes(in_phases, in_metrics_map)
    out_html = generate_sub_nodes(out_phases, out_metrics_map)

    css = "<style>.tree-container { display: grid; grid-template-rows: auto auto 1fr; gap: 10px; padding: 15px; background: #0f172a; border-radius: 16px; height: 85vh; box-sizing: border-box; overflow-y: auto; }.level-0 { display: flex; justify-content: center; align-items: center; padding-bottom: 5px; }.level-1 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 10px; }.level-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }.sub-phase-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; align-content: start; }.tree-node { background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); display: flex; flex-direction: column; align-items: center; transition: transform 0.2s; }.tree-node:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.3); border-color: rgba(255,255,255,0.2); }.gm-node { width: 300px; border-left: 6px solid #3b82f6; }.main-node { border-left-width: 5px; }.sub-node { padding: 10px; min-height: 120px; }.node-header { font-weight: 700; font-size: 0.9rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; text-align: center; }.node-score { font-size: 1.8rem; font-weight: 800; margin: 4px 0; }.node-target { font-size: 0.7rem; color: #94a3b8; margin-bottom: 8px; }.metrics-container { width: 100%; background: rgba(0,0,0,0.2); border-radius: 6px; padding: 6px; margin-top: auto; }.metric-row { display: flex; justify-content: space-between; font-size: 0.65rem; color: #cbd5e1; padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }.metric-row:last-child { border-bottom: none; }.tree-container::-webkit-scrollbar { width: 8px; }.tree-container::-webkit-scrollbar-track { background: #0f172a; }.tree-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }.tree-container::-webkit-scrollbar-thumb:hover { background: #475569; }</style>"
    
    html_parts = [css, '<div class="tree-container">', '<div class="level-0">']
    html_parts.append(f'<div class="tree-node gm-node" style="border-left-color: {nodes["gm"]["color"]}"><div class="node-header">Game Model</div><div class="node-score" style="color: {nodes["gm"]["color"]}">{nodes["gm"]["score"]:.1f} <span style="color: {nodes["gm"]["arrow_color"]}">{nodes["gm"]["arrow"]}</span></div><div class="node-target">Target: {nodes["gm"]["kpi"]:.1f}</div></div>')
    html_parts.append('</div><div class="level-1">')
    for name in ['In Possession', 'Out of Possession']:
        html_parts.append(f'<div class="tree-node main-node" style="border-left-color: {nodes[name]["color"]}"><div class="node-header">{name}</div><div class="node-score" style="color: {nodes[name]["color"]}">{nodes[name]["score"]:.1f} <span style="color: {nodes[name]["arrow_color"]}">{nodes[name]["arrow"]}</span></div><div class="node-target">Target: {nodes[name]["kpi"]:.1f}</div></div>')
    html_parts.append(f'</div><div class="level-2"><div class="sub-phase-container">{in_html}</div><div class="sub-phase-container">{out_html}</div></div></div>')
    return "".join(html_parts)

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
    page = st.sidebar.radio("Navigation", ["Performance Tree", "Player Intelligence", "Squad Planner"])
    
    available_teams = get_available_teams()
    if not available_teams: st.error("No teams found."); return
    
    selected_team = st.sidebar.selectbox("Team", available_teams)
    team_data, opp_data, player_data, league_data = load_data(selected_team)
    
    if team_data is None or team_data.empty: st.error("Data load failed."); return
    
    match_options = [f"Match {i+1} ({row['Date']})" for i, row in team_data.iterrows()]
    selected_match_idx = st.sidebar.selectbox("Match", range(len(match_options)), format_func=lambda x: match_options[x])
    
    st.sidebar.markdown("---")
    kpi_mode = st.sidebar.radio("Benchmark", ["Season Avg", "Top 25%", "Custom"])
    custom_target = 80
    if kpi_mode == "Custom": custom_target = st.sidebar.slider("Target", 0, 100, 80)
    
    current_team_row = team_data.iloc[selected_match_idx]
    current_opp_row = opp_data.iloc[selected_match_idx] if not opp_data.empty else pd.Series()
    scores = scoring_engine.calculate_match_scores(current_team_row, current_opp_row)
    
    all_scores_list = [scoring_engine.calculate_match_scores(team_data.iloc[i], opp_data.iloc[i] if not opp_data.empty else pd.Series()) for i in range(len(team_data))]
    all_scores_df = pd.DataFrame(all_scores_list)
    
    if kpi_mode == "Season Avg": kpi_thresholds = all_scores_df.mean().to_dict()
    elif kpi_mode == "Top 25%": kpi_thresholds = all_scores_df.quantile(0.75).to_dict()
    else: kpi_thresholds = {col: custom_target for col in all_scores_df.columns}
    
    if page == "Performance Tree":
        st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
        tree_html = render_full_tree(scores, kpi_thresholds, current_team_row)
        st.markdown(tree_html, unsafe_allow_html=True)

    elif page == "Player Intelligence":
        st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
        st.subheader("PLAYER RADAR ANALYSIS")
        
        if player_data.empty:
            st.warning("Player data not available.")
        else:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                template = st.selectbox("Position Template", ["Forward", "Midfielder", "Defender"])
            
            position_map = {
                "Forward": ['LWF', 'LW', 'LAMF', 'RW', 'RWF', 'RAMF', 'AMF', 'CF'],
                "Midfielder": ['LAMF', 'RAMF', 'AMF', 'CM', 'LCM', 'RCM', 'DM', 'LDM', 'RDM'],
                "Defender": ['CB', 'RCB', 'LCB', 'LB', 'LWB', 'RB', 'RWB']
            }
            
            allowed_positions = position_map[template]
            team_pos_mask = player_data['Position'].apply(lambda x: any(p in str(x) for p in allowed_positions))
            valid_team_players = player_data[team_pos_mask]['Player'].unique()
            
            with col_sel2:
                if len(valid_team_players) == 0:
                    st.warning(f"No {template}s found in team.")
                    player_name = None
                else:
                    player_name = st.selectbox("Select Player", valid_team_players)
            
            compare = st.checkbox("Compare with another player")
            player_name_2 = None
            
            if compare and len(valid_team_players) > 1:
                player_name_2 = st.selectbox("Select Player 2", [p for p in valid_team_players if p != player_name], key="player2")
            elif compare:
                st.warning("Need at least 2 players for comparison.")
            
            if player_name and st.button("Generate Radar"):
                p_vals, raw_vals, labels = radar_chart.get_player_percentiles(player_name, league_data, template)
                if p_vals is not None:
                    if compare and player_name_2:
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.markdown(f"### {player_name}")
                            fig = radar_chart.create_pizza_chart(player_name, p_vals, raw_vals, labels, template)
                            st.pyplot(fig)
                        with col_r2:
                            st.markdown(f"### {player_name_2}")
                            p_vals_2, raw_vals_2, labels_2 = radar_chart.get_player_percentiles(player_name_2, league_data, template)
                            if p_vals_2 is not None:
                                fig2 = radar_chart.create_pizza_chart(player_name_2, p_vals_2, raw_vals_2, labels_2, template)
                                st.pyplot(fig2)
                    else:
                        fig = radar_chart.create_pizza_chart(player_name, p_vals, raw_vals, labels, template)
                        st.pyplot(fig)
                else:
                    st.error("Could not generate chart for selected player.")
            
            st.markdown("---")
            st.subheader("TEAM PLAYER RANKINGS")
            
            if not player_data.empty:
                numeric_cols = player_data.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['Age', 'Height', 'Weight', 'Market value', 'Minutes played', 'Matches played']
                metric_cols = [c for c in numeric_cols if c not in exclude_cols and ('per 90' in c or '%' in c)]
                
                selected_metric = st.selectbox("Select Metric for Rankings", metric_cols)
                ranked = player_data[['Player', 'Position', selected_metric]].sort_values(by=selected_metric, ascending=False).reset_index(drop=True)
                ranked.index += 1
                ranked.index.name = 'Rank'
                
                fig_rank = go.Figure()
                fig_rank.add_trace(go.Bar(
                    y=ranked['Player'][:10], x=ranked[selected_metric][:10], orientation='h',
                    marker=dict(color=ranked[selected_metric][:10], colorscale='Viridis', showscale=True, colorbar=dict(title=selected_metric)),
                    text=ranked[selected_metric][:10].round(2), textposition='outside'
                ))
                fig_rank.update_layout(title=f"Top 10 Players - {selected_metric}", xaxis_title=selected_metric, yaxis_title="Player", template='plotly_dark', height=500, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_rank, use_container_width=True)
                
                with st.expander("View Full Rankings"):
                    st.dataframe(ranked, use_container_width=True)

    elif page == "Squad Planner":
        render_squad_planner(league_data, selected_team)

if __name__ == "__main__":
    main()
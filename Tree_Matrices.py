import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import custom modules
import scoring_engine
import forecasting_model
import radar_chart

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

def render_tree_node(title, score, kpi, metrics=None):
    """Render a single tree node"""
    if score >= 80: border_color = "#10b981"; score_color = "#10b981"
    elif score >= 60: border_color = "#f59e0b"; score_color = "#f59e0b"
    else: border_color = "#ef4444"; score_color = "#ef4444"
    
    
    # Arrow indicator based on score vs KPI
    if score >= kpi:
        arrow = "↑"
        arrow_color = "#10b981"
    else:
        arrow = "↓"
        arrow_color = "#ef4444"
    html = f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-left: 6px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin: 10px auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-weight: 700; font-size: 1.1rem; color: #f1f5f9; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; text-align: center;">{title}</div>
            <div style="font-size: 2.5rem; font-weight: 800; text-align: center; margin: 10px 0; color: {score_color};">{score:.1f} <span style="font-size: 1.5rem; margin-left: 8px; color: {arrow_color};">{arrow}</span></div>
            <div style="font-size: 0.85rem; color: #94a3b8; text-align: center; margin-bottom: 15px;">Target: {kpi:.1f}</div>
    """
    
    if metrics:
        html += '<div style="background: rgba(0,0,0,0.2); border-radius: 6px; padding: 12px; margin-top: 15px; font-size: 0.85rem;">'
        for m_name, m_val in metrics.items():
            html += f'<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span style="color: #cbd5e1; font-weight: 500;">{m_name}</span><span style="color: #60a5fa; font-weight: 600;">{m_val:.1f}</span></div>'
        html += '</div>'
    
    html += "</div>"
    return html

# --- Main App ---

def main():
    st.markdown('<div class="main-header">Team Intelligence Suite</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("CONFIGURATION")
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
    
    # Calculate Scores
    current_team_row = team_data.iloc[selected_match_idx]
    current_opp_row = opp_data.iloc[selected_match_idx] if not opp_data.empty else pd.Series()
    scores = scoring_engine.calculate_match_scores(current_team_row, current_opp_row)
    
    all_scores_list = [scoring_engine.calculate_match_scores(team_data.iloc[i], opp_data.iloc[i] if not opp_data.empty else pd.Series()) for i in range(len(team_data))]
    all_scores_df = pd.DataFrame(all_scores_list)
    
    if kpi_mode == "Season Avg": kpi_thresholds = all_scores_df.mean().to_dict()
    elif kpi_mode == "Top 25%": kpi_thresholds = all_scores_df.quantile(0.75).to_dict()
    else: kpi_thresholds = {col: custom_target for col in all_scores_df.columns}
    
    # Helper to get values
    def get_val(key, default=0): return current_team_row.get(key, default)
    
    # --- TABS ---
    tab_tree, tab_analytics, tab_player = st.tabs(["PERFORMANCE TREE", "DEEP ANALYTICS", "PLAYER INTELLIGENCE"])
    
    # --- 1. PERFORMANCE TREE ---
    with tab_tree:
        if 'tree_state' not in st.session_state:
            st.session_state.tree_state = {'gm': True, 'in': False, 'out': False}
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Level 0: Game Model
        col_gm = st.columns([1, 2, 1])
        with col_gm[1]:
            gm_metrics = {'In Possession': scores['In Possession'], 'Out of Possession': scores['Out of Possession']} if st.session_state.tree_state['gm'] else None
            st.markdown(render_tree_node("Game Model", scores['Game Model'], kpi_thresholds['Game Model'], gm_metrics), unsafe_allow_html=True)
            if st.button("⬍ Click to " + ("collapse" if st.session_state.tree_state['gm'] else "expand"), key="btn_gm", use_container_width=True):
                st.session_state.tree_state['gm'] = not st.session_state.tree_state['gm']
                if not st.session_state.tree_state['gm']:
                    st.session_state.tree_state['in'] = False
                    st.session_state.tree_state['out'] = False
                st.rerun()
        
        # Level 1: Phases
        if st.session_state.tree_state['gm']:
            st.markdown("<div style='text-align:center; padding: 20px;'><div style='width:2px; height:30px; background: rgba(148, 163, 184, 0.3); margin: 0 auto;'></div></div>", unsafe_allow_html=True)
            
            col_phases = st.columns(2)
            
            with col_phases[0]:
                in_metrics = {
                    'Build Up': scores['Build Up'],
                    'Progression': scores['Progression'],
                    'Transition D-A': scores['Transition (D to A)'],
                    'Set Pieces Off': scores['Set Pieces (Off)'],
                    'Output': scores['Output']
                } if st.session_state.tree_state['in'] else None
                
                st.markdown(render_tree_node("In Possession", scores['In Possession'], kpi_thresholds['In Possession'], in_metrics), unsafe_allow_html=True)
                if st.button("⬍ Click to " + ("collapse" if st.session_state.tree_state['in'] else "expand"), key="btn_in", use_container_width=True):
                    st.session_state.tree_state['in'] = not st.session_state.tree_state['in']
                    st.rerun()
            
            with col_phases[1]:
                out_metrics = {
                    'Pressing': scores['Pressing'],
                    'Defensive Block': scores['Defensive Block'],
                    'Transition A-D': scores['Transition (A to D)'],
                    'Set Pieces Def': scores['Set Pieces (Def)'],
                    'Output Against': scores['Output Against']
                } if st.session_state.tree_state['out'] else None
                
                st.markdown(render_tree_node("Out of Possession", scores['Out of Possession'], kpi_thresholds['Out of Possession'], out_metrics), unsafe_allow_html=True)
                if st.button("⬍ Click to " + ("collapse" if st.session_state.tree_state['out'] else "expand"), key="btn_out", use_container_width=True):
                    st.session_state.tree_state['out'] = not st.session_state.tree_state['out']
                    st.rerun()
            
            # Level 2: Sub-phases with raw metrics
            if st.session_state.tree_state['in'] or st.session_state.tree_state['out']:
                st.markdown("<div style='text-align:center; padding: 20px;'><div style='width:2px; height:30px; background: rgba(148, 163, 184, 0.3); margin: 0 auto;'></div></div>", unsafe_allow_html=True)
                
                if st.session_state.tree_state['in']:
                    st.markdown("#### In Possession Phases")
                    cols_in = st.columns(5)
                    
                    bu_metrics = {
                        'Back Pass Acc %': get_val('Percentage of accurate back passes'), 
                        'Lateral Pass Acc %': get_val('Percentage of accurate lateral passes'),
                        'Possession %': get_val('Possession, %'),
                        'Total Pass Acc %': get_val('Percentage of accurate passes')
                    }
                    prog_metrics = {
                        'Forward Pass Acc %': get_val('Percentage of accurate forward passes'),
                        'Progressive Pass Acc %': get_val('Percentage of acccurate progressive passes'),
                        'Final Third Acc %': get_val('Percentage of accurate passes to final third'),
                        'PA Entries': get_val('Total penalty area entries (runs / crosses)')
                    }
                    tda_metrics = {
                        'CA w/ Shots': get_val('Total counterattacks with shots'),
                        'CA Attacks': get_val('Total counterattacks attacks'),
                        'High Recoveries': get_val('Recoveries (high)'),
                        'PPDA': get_val('PPDA')
                    }
                    spo_metrics = {
                        'SP w/ Shots': get_val('Total set pieces with shots'),
                        'Corners w/ Shots': get_val('Total corners with shots'),
                        'Cross Acc %': get_val('Percentage of accurate crosses')
                    }
                    out_metrics_in = {
                        'Goals': get_val('Goals'),
                        'xG': get_val('xG'),
                        'Shots on Target %': get_val('Percentage of shots on target'),
                        'Touches in Box': get_val('Touches in penalty area')
                    }
                    
                    nodes_in = [
                        ("Build Up", scores['Build Up'], kpi_thresholds['Build Up'], bu_metrics),
                        ("Progression", scores['Progression'], kpi_thresholds['Progression'], prog_metrics),
                        ("Transition D-A", scores['Transition (D to A)'], kpi_thresholds['Transition (D to A)'], tda_metrics),
                        ("Set Pieces Off", scores['Set Pieces (Off)'], kpi_thresholds['Set Pieces (Off)'], spo_metrics),
                        ("Output", scores['Output'], kpi_thresholds['Output'], out_metrics_in)
                    ]
                    for idx, (title, score, kpi, metrics) in enumerate(nodes_in):
                        with cols_in[idx]:
                            st.markdown(render_tree_node(title, score, kpi, metrics), unsafe_allow_html=True)
                
                if st.session_state.tree_state['out']:
                    st.markdown("#### Out of Possession Phases")
                    cols_out = st.columns(5)
                    
                    press_metrics = {
                        'PPDA': get_val('PPDA'),
                        'High Recoveries': get_val('Recoveries (high)'),
                        'Def Duels Won %': get_val('Percentage defensive duels  won'),
                        'Interceptions': get_val('Interceptions')
                    }
                    db_metrics = {
                        'Def Duels Won %': get_val('Percentage defensive duels  won'),
                        'Aerial Won %': get_val('Percentage aerial duels  won'),
                        'Tackle Success %': get_val('Percentage successful sliding tackles'),
                        'Clearances': get_val('Clearances')
                    }
                    tad_metrics = {
                        'Ball Losses (High)': get_val('Ball losses (high)'),
                        'Recoveries (Med)': get_val('Recoveries (medium)'),
                        'Recoveries (High)': get_val('Recoveries (high)'),
                        'Fouls': get_val('Fouls')
                    }
                    spd_metrics = {
                        'Shots Against': get_val('Total shots against'),
                        'Aerial Won %': get_val('Percentage aerial duels  won'),
                        'Clearances': get_val('Clearances')
                    }
                    oa_metrics = {
                        'Goals Conceded': get_val('Conceded goals'),
                        'Shots Against': get_val('Total shots against'),
                        'SoT Against %': get_val('Percentage of shots against on target')
                    }
                    
                    nodes_out = [
                        ("Pressing", scores['Pressing'], kpi_thresholds['Pressing'], press_metrics),
                        ("Defensive Block", scores['Defensive Block'], kpi_thresholds['Defensive Block'], db_metrics),
                        ("Transition A-D", scores['Transition (A to D)'], kpi_thresholds['Transition (A to D)'], tad_metrics),
                        ("Set Pieces Def", scores['Set Pieces (Def)'], kpi_thresholds['Set Pieces (Def)'], spd_metrics),
                        ("Output Against", scores['Output Against'], kpi_thresholds['Output Against'], oa_metrics)
                    ]
                    for idx, (title, score, kpi, metrics) in enumerate(nodes_out):
                        with cols_out[idx]:
                            st.markdown(render_tree_node(title, score, kpi, metrics), unsafe_allow_html=True)

    # --- 2. DEEP ANALYTICS ---
    with tab_analytics:
        st.subheader("FORECASTING & TRENDS")
        
        metric_to_predict = st.selectbox("Select Metric", all_scores_df.columns)
        trend_hist, forecast, future_idx = forecasting_model.train_and_predict(all_scores_df, metric_to_predict)
        
        # Model Accuracy
        actual = all_scores_df[metric_to_predict].values
        rmse = np.sqrt(mean_squared_error(actual, trend_hist))
        mae = mean_absolute_error(actual, trend_hist)
        r2 = r2_score(actual, trend_hist)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R² Score", f"{r2:.3f}", help="Coefficient of determination (1.0 = perfect)")
        col_m2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")
        col_m3.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
        col_m4.metric("Model", "Random Forest", help="Ensemble learning method")
        
        # Plot
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=all_scores_df.index, y=actual, mode='lines+markers', name='Actual', line=dict(color='#60a5fa', width=2), marker=dict(size=6)))
        fig_trend.add_trace(go.Scatter(x=all_scores_df.index, y=trend_hist, mode='lines', name='Model Fit', line=dict(color='#f59e0b', width=2, dash='dash')))
        fig_trend.add_trace(go.Scatter(x=future_idx, y=forecast, mode='lines+markers', name='Forecast', line=dict(color='#10b981', width=2), marker=dict(size=8, symbol='diamond')))
        
        std_dev = np.std(actual - trend_hist)
        fig_trend.add_trace(go.Scatter(
            x=np.concatenate([future_idx, future_idx[::-1]]),
            y=np.concatenate([forecast + std_dev, (forecast - std_dev)[::-1]]),
            fill='toself', fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='Confidence Band'
        ))
        
        fig_trend.update_layout(title=f"{metric_to_predict} - Historical & Forecast", xaxis_title="Match Index", yaxis_title="Score", hovermode='x unified', template='plotly_dark', height=500)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.subheader("CORRELATION MATRIX")
        corr = all_scores_df.corr()
        fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
        fig_corr.update_layout(template='plotly_dark', height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- 3. PLAYER INTELLIGENCE ---
    with tab_player:
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
            
            # Comparison Option
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
                        # Two columns for comparison
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
            
            # Player Rankings
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
                    y=ranked['Player'][:10],
                    x=ranked[selected_metric][:10],
                    orientation='h',
                    marker=dict(color=ranked[selected_metric][:10], colorscale='Viridis', showscale=True, colorbar=dict(title=selected_metric)),
                    text=ranked[selected_metric][:10].round(2),
                    textposition='outside'
                ))
                
                fig_rank.update_layout(title=f"Top 10 Players - {selected_metric}", xaxis_title=selected_metric, yaxis_title="Player", template='plotly_dark', height=500, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_rank, use_container_width=True)
                
                with st.expander("View Full Rankings"):
                    st.dataframe(ranked, use_container_width=True)

if __name__ == "__main__":
    main()
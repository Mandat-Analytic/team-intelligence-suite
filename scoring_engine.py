import pandas as pd
import numpy as np

class GameModelScorer:
    """
    Hierarchical scoring system following the tree structure:
    Game Model (100)
    ├── In Possession (50)
    │   ├── Build Up Phase (10)
    │   ├── Progression Phase (10)
    │   ├── Transition D→A (8)
    │   ├── Offensive Set Pieces (7)
    │   └── Output Phase (15)
    └── Out of Possession (50)
        ├── Pressing Phase (10)
        ├── Defensive Block (12)
        ├── Transition A→D (8)
        ├── Defensive Set Pieces (10)
        └── Output Against (10)
    """
    
    def __init__(self):
        self.weights = {
            'in_possession': 0.50,
            'out_possession': 0.50
        }
    
    def safe_get(self, data, possible_keys, default=0):
        """
        Try multiple possible column names and return first match.
        Handles variations in Excel column naming.
        """
        if not isinstance(possible_keys, list):
            possible_keys = [possible_keys]
        
        for key in possible_keys:
            if key in data and pd.notna(data[key]):
                return data[key]
        return default
    
    # ==================== IN POSSESSION PHASES ====================
    
    def score_build_up_phase(self, data):
        """
        Build Up Phase (0-10 points)
        Metrics: Back passes accuracy, lateral passes accuracy, possession %, 
                 pass completion in own/mid third, ball retention
        """
        # Normalize metrics to 0-1 scale
        back_pass_acc = self.safe_get(data, ['Back passes %', 'Percentage of accurate back passes'], 0) / 100
        lateral_pass_acc = self.safe_get(data, ['Lateral passes %', 'Percentage of accurate lateral passes'], 0) / 100
        possession = self.safe_get(data, ['Possession, %'], 0) / 100
        total_pass_acc = self.safe_get(data, ['Passes %', 'Percentage of accurate passes'], 0) / 100
        
        # Ball retention (inverse of losses in low/medium areas)
        total_losses_back = self.safe_get(data, ['Losses Low', 'Ball losses (low)'], 0) + \
                           self.safe_get(data, ['Losses Medium', 'Ball losses (medium)'], 0)
        total_passes_back = self.safe_get(data, ['Back passes Total', 'Total back passes'], 0) + \
                           self.safe_get(data, ['Lateral passes Total', 'Total lateral passes'], 0)
        retention = 1 - (total_losses_back / max(total_passes_back, 1))
        retention = max(0, min(1, retention))
        
        # Weighted scoring
        score = (
            back_pass_acc * 0.25 +
            lateral_pass_acc * 0.20 +
            possession * 0.20 +
            total_pass_acc * 0.20 +
            retention * 0.15
        ) * 10
        
        return round(score, 2)
    
    def score_progression_phase(self, data):
        """
        Progression Phase (0-10 points)
        Metrics: Forward passes accuracy, progressive passes, passes to final third,
                 successful penalty area entries
        """
        forward_pass_acc = self.safe_get(data, ['Forward passes %', 'Percentage of accurate forward passes'], 0) / 100
        progressive_pass_acc = self.safe_get(data, ['Progressive passes %', 'Percentage of acccurate progressive passes'], 0) / 100
        final_third_acc = self.safe_get(data, ['Passes to final third %', 'Percentage of accurate passes to final third'], 0) / 100
        
        # Penalty area entry success rate
        total_entries = self.safe_get(data, ['Penalty area entries Total', 'Total penalty area entries (runs / crosses)'], 0)
        pa_entry_rate = min(1, total_entries / 30)  # Normalize to ~30 entries
        
        # Progressive actions
        prog_passes = self.safe_get(data, ['Progressive passes Total', 'Total progressive passes'], 0)
        prog_rate = min(1, prog_passes / 50)  # Normalize to ~50 progressive passes
        
        score = (
            forward_pass_acc * 0.25 +
            progressive_pass_acc * 0.25 +
            final_third_acc * 0.20 +
            pa_entry_rate * 0.15 +
            prog_rate * 0.15
        ) * 10
        
        return round(score, 2)
    
    def score_transition_d_to_a(self, data):
        """
        Transition Defense to Attack (0-8 points)
        Metrics: Counterattack efficiency, recoveries in high area, match tempo
        """
        # Counterattack efficiency
        ca_attacks = self.safe_get(data, ['Counterattacks Total', 'Total counterattacks attacks'], 0)
        ca_with_shots = self.safe_get(data, ['Counterattacks with shots', 'Total counterattacks with shots'], 0)
        ca_efficiency = (ca_with_shots / max(ca_attacks, 1)) if ca_attacks > 0 else 0
        
        # High recoveries (opportunity creation)
        high_rec = self.safe_get(data, ['Recoveries High', 'Recoveries (high)'], 0)
        high_rec_rate = min(1, high_rec / 20)  # Normalize to ~20 high recoveries
        
        # Quick transition indicator (PPDA - lower is more aggressive)
        ppda = self.safe_get(data, ['PPDA'], 0)
        ppda_score = max(0, 1 - (ppda / 20))  # Lower PPDA = higher score
        
        # Match tempo
        tempo = self.safe_get(data, ['Match tempo'], 0)
        tempo_score = min(1, tempo / 2.0)  # Normalize tempo
        
        score = (
            ca_efficiency * 0.35 +
            high_rec_rate * 0.30 +
            ppda_score * 0.20 +
            tempo_score * 0.15
        ) * 8
        
        return round(score, 2)
    
    def score_offensive_set_pieces(self, data):
        """
        Offensive Set Pieces (0-7 points)
        Metrics: Set piece shot conversion, corner efficiency, cross accuracy
        """
        # Set piece efficiency
        sp_total = self.safe_get(data, ['Set pieces Total', 'Total set pieces'], 0)
        sp_shots = self.safe_get(data, ['Set pieces with shots', 'Total set pieces with shots'], 0)
        sp_efficiency = (sp_shots / max(sp_total, 1)) if sp_total > 0 else 0
        
        # Corner efficiency
        corners = self.safe_get(data, ['Corners Total', 'Total corners'], 0)
        corner_shots = self.safe_get(data, ['Corners with shots', 'Total corners with shots'], 0)
        corner_efficiency = (corner_shots / max(corners, 1)) if corners > 0 else 0
        
        # Free kick efficiency
        fk_total = self.safe_get(data, ['Free kicks Total', 'Total free kicks'], 0)
        fk_shots = self.safe_get(data, ['Free kicks with shots', 'Total free kicks with shots'], 0)
        fk_efficiency = (fk_shots / max(fk_total, 1)) if fk_total > 0 else 0
        
        # Cross accuracy
        cross_acc = self.safe_get(data, ['Crosses %', 'Percentage of accurate crosses'], 0) / 100
        
        score = (
            sp_efficiency * 0.30 +
            corner_efficiency * 0.30 +
            fk_efficiency * 0.20 +
            cross_acc * 0.20
        ) * 7
        
        return round(score, 2)
    
    def score_output_phase(self, data):
        """
        Output Phase (0-15 points)
        Metrics: Goals, xG, shots on target %, positional attack efficiency,
                 touches in box, shot quality
        """
        goals = self.safe_get(data, ['Goals'], 0)
        xg = self.safe_get(data, ['xG'], 0)
        shots_ot_pct = self.safe_get(data, ['Shots %', 'Percentage of shots on target'], 0) / 100
        
        # Goal vs xG efficiency
        xg_efficiency = min(2, goals / max(xg, 0.1))  # Cap at 2x overperformance
        xg_efficiency = xg_efficiency / 2  # Normalize to 0-1
        
        # Positional attack efficiency
        pos_attacks = self.safe_get(data, ['Positional attacks Total', 'Total positional attacks'], 0)
        pos_shots = self.safe_get(data, ['Positional attacks with shots', 'Total positional attacks with shots'], 0)
        pos_efficiency = (pos_shots / max(pos_attacks, 1)) if pos_attacks > 0 else 0
        
        # Box presence
        touches_box = self.safe_get(data, ['Touches in penalty area'], 0)
        box_presence = min(1, touches_box / 40)  # Normalize to ~40 touches
        
        # Offensive duels
        off_duels_won_pct = self.safe_get(data, ['Offensive duels %', 'Percentage offensive duels  won'], 0) / 100
        
        # Shot quality (distance)
        avg_shot_dist = self.safe_get(data, ['Average shot distance'], 0)
        shot_quality = max(0, 1 - (avg_shot_dist / 20))  # Closer shots = higher quality
        
        score = (
            (min(goals, 5) / 5) * 0.25 +  # Normalize goals (5 goals = max score)
            xg_efficiency * 0.20 +
            shots_ot_pct * 0.20 +
            pos_efficiency * 0.15 +
            box_presence * 0.10 +
            off_duels_won_pct * 0.05 +
            shot_quality * 0.05
        ) * 15
        
        return round(min(score, 15), 2)  # Cap at 15
    
    # ==================== OUT OF POSSESSION PHASES ====================
    
    def score_pressing_phase(self, data):
        """
        Pressing Phase (0-10 points)
        Metrics: PPDA (low = high press), high recoveries, defensive duels won %, interceptions
        """
        # PPDA (Passes Allowed Per Defensive Action) - lower is better
        ppda = self.safe_get(data, ['PPDA'], 0)
        ppda_score = max(0, 1 - (ppda / 15))  # Normalize (aggressive press ~8-10 PPDA)
        
        # High recoveries
        high_rec = self.safe_get(data, ['Recoveries High', 'Recoveries (high)'], 0)
        high_rec_rate = min(1, high_rec / 25)
        
        # Defensive duels won
        def_duels_won_pct = self.safe_get(data, ['Defensive duels %', 'Percentage defensive duels  won'], 0) / 100
        
        # Interceptions
        interceptions = self.safe_get(data, ['Interceptions'], 0)
        intercept_rate = min(1, interceptions / 15)
        
        # Overall recovery rate
        total_rec = self.safe_get(data, ['Recoveries Total', 'Total recoveries'], 0)
        recovery_rate = min(1, total_rec / 60)
        
        score = (
            ppda_score * 0.30 +
            high_rec_rate * 0.25 +
            def_duels_won_pct * 0.20 +
            intercept_rate * 0.15 +
            recovery_rate * 0.10
        ) * 10
        
        return round(score, 2)
    
    def score_defensive_block(self, data):
        """
        Defensive Block (0-12 points)
        Metrics: Defensive duels won %, aerial duels won %, sliding tackles %, clearances
        """
        def_duels_won_pct = self.safe_get(data, ['Defensive duels %', 'Percentage defensive duels  won'], 0) / 100
        aerial_won_pct = self.safe_get(data, ['Aerial duels %', 'Percentage aerial duels  won'], 0) / 100
        tackle_success_pct = self.safe_get(data, ['Sliding tackles %', 'Percentage successful sliding tackles'], 0) / 100
        
        # Clearances
        clearances = self.safe_get(data, ['Clearances'], 0)
        clearance_rate = min(1, clearances / 25)
        
        # Low recoveries (defensive organization)
        low_rec = self.safe_get(data, ['Recoveries Low', 'Recoveries (low)'], 0)
        low_rec_rate = min(1, low_rec / 20)
        
        score = (
            def_duels_won_pct * 0.30 +
            aerial_won_pct * 0.25 +
            tackle_success_pct * 0.20 +
            clearance_rate * 0.15 +
            low_rec_rate * 0.10
        ) * 12
        
        return round(score, 2)
    
    def score_transition_a_to_d(self, data):
        """
        Transition Attack to Defense (0-8 points)
        Metrics: Ball losses in high area (negative), quick recoveries, counter-press, fouls
        """
        # Ball losses in high area (negative indicator)
        high_losses = self.safe_get(data, ['Losses High', 'Ball losses (high)'], 0)
        high_loss_penalty = min(1, high_losses / 15)  # More losses = worse
        
        # Medium recoveries (counter-press)
        medium_rec = self.safe_get(data, ['Recoveries Medium', 'Recoveries (medium)'], 0)
        counter_press_rate = min(1, medium_rec / 25)
        
        # High recoveries after loss
        high_rec = self.safe_get(data, ['Recoveries High', 'Recoveries (high)'], 0)
        high_rec_rate = min(1, high_rec / 15)
        
        # Tactical fouls (controlled transition defense)
        fouls = self.safe_get(data, ['Fouls'], 0)
        tactical_fouls = min(1, fouls / 15)
        
        score = (
            (1 - high_loss_penalty) * 0.35 +
            counter_press_rate * 0.30 +
            high_rec_rate * 0.25 +
            tactical_fouls * 0.10
        ) * 8
        
        return round(score, 2)
    
    def score_defensive_set_pieces(self, data):
        """
        Defensive Set Pieces (0-10 points)
        Metrics: Set pieces conceded, aerial duels won %, clearances
        """
        # Inverse of offensive set pieces against (estimate from shots against)
        shots_against = self.safe_get(data, ['Shots against Total', 'Total shots against'], 0)
        sp_defense = max(0, 1 - (shots_against / 20))
        
        # Aerial dominance
        aerial_won_pct = self.safe_get(data, ['Aerial duels %', 'Percentage aerial duels  won'], 0) / 100
        
        # Clearances
        clearances = self.safe_get(data, ['Clearances'], 0)
        clearance_rate = min(1, clearances / 30)
        
        # Defensive duels in set pieces
        def_duels_won_pct = self.safe_get(data, ['Defensive duels %', 'Percentage defensive duels  won'], 0) / 100
        
        score = (
            sp_defense * 0.30 +
            aerial_won_pct * 0.30 +
            clearance_rate * 0.25 +
            def_duels_won_pct * 0.15
        ) * 10
        
        return round(score, 2)
    
    def score_output_against(self, data):
        """
        Output Against (0-10 points)
        Metrics: Goals conceded, shots against, shots on target against %, defensive solidity
        """
        goals_conceded = self.safe_get(data, ['Conceded goals'], 0)
        shots_against = self.safe_get(data, ['Shots against Total', 'Total shots against'], 0)
        shots_ot_against = self.safe_get(data, ['Shots against on target', 'Total shots against on target'], 0)
        shots_ot_against_pct = self.safe_get(data, ['Shots against %', 'Percentage of shots against on target'], 0) / 100
        
        # Goals conceded (inverse scoring)
        goals_score = max(0, 1 - (goals_conceded / 4))  # 0 goals = 1.0, 4+ goals = 0
        
        # Shots against (inverse)
        shots_score = max(0, 1 - (shots_against / 20))  # Fewer shots = better
        
        # Shots on target % (lower is better)
        shot_quality_defense = max(0, 1 - shots_ot_against_pct)
        
        # Overall defensive efficiency
        def_efficiency = (goals_conceded / max(shots_against, 1)) if shots_against > 0 else 0
        def_efficiency_score = max(0, 1 - def_efficiency)
        
        score = (
            goals_score * 0.40 +
            shots_score * 0.25 +
            shot_quality_defense * 0.20 +
            def_efficiency_score * 0.15
        ) * 10
        
        return round(score, 2)
    
    # ==================== AGGREGATION METHODS ====================
    
    def calculate_in_possession_score(self, data):
        """Calculate total In Possession score (0-50 points)"""
        build_up = self.score_build_up_phase(data)
        progression = self.score_progression_phase(data)
        transition_da = self.score_transition_d_to_a(data)
        off_setpieces = self.score_offensive_set_pieces(data)
        output = self.score_output_phase(data)
        
        total = build_up + progression + transition_da + off_setpieces + output
        
        return {
            'total': round(total, 2),
            'build_up': build_up,
            'progression': progression,
            'transition_d_to_a': transition_da,
            'offensive_set_pieces': off_setpieces,
            'output': output
        }
    
    def calculate_out_possession_score(self, data):
        """Calculate total Out of Possession score (0-50 points)"""
        pressing = self.score_pressing_phase(data)
        def_block = self.score_defensive_block(data)
        transition_ad = self.score_transition_a_to_d(data)
        def_setpieces = self.score_defensive_set_pieces(data)
        output_against = self.score_output_against(data)
        
        total = pressing + def_block + transition_ad + def_setpieces + output_against
        
        return {
            'total': round(total, 2),
            'pressing': pressing,
            'defensive_block': def_block,
            'transition_a_to_d': transition_ad,
            'defensive_set_pieces': def_setpieces,
            'output_against': output_against
        }
    
    def calculate_game_model_score(self, data):
        """
        Calculate overall Game Model score (0-100 points)
        
        Returns detailed breakdown of all scores
        """
        in_poss = self.calculate_in_possession_score(data)
        out_poss = self.calculate_out_possession_score(data)
        
        game_model_score = in_poss['total'] + out_poss['total']
        
        return {
            'game_model_score': round(game_model_score, 2),
            'in_possession': in_poss,
            'out_of_possession': out_poss,
            'rating': self._get_rating(game_model_score)
        }
    
    def _get_rating(self, score):
        """Convert numerical score to rating"""
        if score >= 90: return "Exceptional"
        elif score >= 80: return "Excellent"
        elif score >= 70: return "Very Good"
        elif score >= 60: return "Good"
        elif score >= 50: return "Average"
        elif score >= 40: return "Below Average"
        else: return "Poor"
    
    def score_match(self, match_data):
        """
        Score a single match
        """
        return self.calculate_game_model_score(match_data)

def normalize_scores_to_league(team_scores_df, league_scores_df):
    """
    Normalizes team scores using league context (min-max and percentile).
    
    Args:
        team_scores_df: DataFrame of scores for the team matches
        league_scores_df: DataFrame of scores for all league matches
        
    Returns:
        Dict of results with raw scores, min-max scores, and percentiles
    """
    results = {}
    
    for col in team_scores_df.columns:
        if col == 'Rating': continue
        
        team_vals = team_scores_df[col]
        league_vals = league_scores_df[col].dropna()
        
        if league_vals.empty:
            results[col] = {
                'raw': team_vals.iloc[0] if len(team_vals) > 0 else 0,
                'min_max': 50,
                'percentile': 50
            }
            continue
            
        # Min-Max Normalization
        l_min = league_vals.min()
        l_max = league_vals.max()
        
        raw_val = team_vals.iloc[0] if len(team_vals) > 0 else 0
        
        if l_max > l_min:
            min_max_val = (raw_val - l_min) / (l_max - l_min) * 100
        else:
            min_max_val = 50
            
        # Percentile Rank
        percentile_val = (league_vals < raw_val).mean() * 100
        
        results[col] = {
            'raw': round(raw_val, 1),
            'min_max': round(min_max_val, 1),
            'percentile': round(percentile_val, 1)
        }
        
    return results

def calculate_match_scores(team_row, opp_row):
    """
    Wrapper function to maintain compatibility with existing code.
    Merges team and opponent rows to ensure all metrics are available.
    Returns all scores normalized to 0-100 scale.
    """
    # Create merged data dictionary
    data = team_row.to_dict()
    
    # Map opponent stats if missing in team row but present in opp row
    if 'Conceded goals' not in data and 'Goals' in opp_row:
        data['Conceded goals'] = opp_row['Goals']
    if 'Total shots against' not in data and 'Total Shots' in opp_row:
        data['Total shots against'] = opp_row['Total Shots']
    if 'Total shots against on target' not in data and 'Shots on target' in opp_row:
        data['Total shots against on target'] = opp_row['Shots on target']
    
    scorer = GameModelScorer()
    scores = scorer.score_match(data)
    
    # Normalize all scores to 0-100 scale
    return {
        'Game Model': scores['game_model_score'],  # Already 0-100
        'In Possession': (scores['in_possession']['total'] / 50) * 100,  # 0-50 -> 0-100
        'Out of Possession': (scores['out_of_possession']['total'] / 50) * 100,  # 0-50 -> 0-100
        'Build Up': (scores['in_possession']['build_up'] / 10) * 100,  # 0-10 -> 0-100
        'Progression': (scores['in_possession']['progression'] / 10) * 100,  # 0-10 -> 0-100
        'Transition (D to A)': (scores['in_possession']['transition_d_to_a'] / 8) * 100,  # 0-8 -> 0-100
        'Set Pieces (Off)': (scores['in_possession']['offensive_set_pieces'] / 7) * 100,  # 0-7 -> 0-100
        'Output': (scores['in_possession']['output'] / 15) * 100,  # 0-15 -> 0-100
        'Pressing': (scores['out_of_possession']['pressing'] / 10) * 100,  # 0-10 -> 0-100
        'Defensive Block': (scores['out_of_possession']['defensive_block'] / 12) * 100,  # 0-12 -> 0-100
        'Transition (A to D)': (scores['out_of_possession']['transition_a_to_d'] / 8) * 100,  # 0-8 -> 0-100
        'Set Pieces (Def)': (scores['out_of_possession']['defensive_set_pieces'] / 10) * 100,  # 0-10 -> 0-100
        'Output Against': (scores['out_of_possession']['output_against'] / 10) * 100  # 0-10 -> 0-100
    }


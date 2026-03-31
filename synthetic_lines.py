"""
PropEdge V9.2 — Synthetic Line Generator
==========================================
Generates realistic prop lines for 2024-25 season backtest.
Mimics sportsbook line-setting: weighted averages + matchup adjustments.
"""
import pandas as pd
import numpy as np
from config import get_dvp, POS_MAP


def generate_synthetic_line(L30, L10, L5, L3, opp, position, is_home,
                            min30=None, min10=None):
    """
    Generate a sportsbook-style prop line.
    
    Methodology:
    1. Weighted average: 40% L30, 25% L10, 20% L5, 15% L3
    2. Opponent defense adjustment: DVP rank → ±pts (0.12 per rank from neutral)
    3. Home/away: +0.4 home, -0.4 away
    4. Minutes trajectory: if minutes declining, adjust line down proportionally
    5. Round to nearest 0.5 (standard book rounding)
    """
    # Weighted baseline
    L10 = L10 if pd.notna(L10) else L30
    L5 = L5 if pd.notna(L5) else L30
    L3 = L3 if pd.notna(L3) else L30
    base = 0.40 * L30 + 0.25 * L10 + 0.20 * L5 + 0.15 * L3

    # Opponent defense
    pos = POS_MAP.get(str(position), 'Forward')
    dvp = get_dvp(opp, pos)
    def_adj = (dvp - 15) * 0.12

    # Venue
    venue_adj = 0.4 if is_home else -0.4

    # Minutes trajectory
    min_adj = 0
    if pd.notna(min30) and pd.notna(min10) and min30 > 0:
        min_pct = (min10 - min30) / min30
        min_adj = np.clip(min_pct * base * 0.3, -2, 2)

    # Combine and round
    line = base + def_adj + venue_adj + min_adj
    line = round(line * 2) / 2  # Nearest 0.5

    # Floor at 3.5 (books don't offer lines below ~3.5)
    return max(3.5, line)


def generate_season_lines(game_logs_df, season='2024-25'):
    """
    Generate synthetic prop lines for an entire season of game logs.
    Returns DataFrame with prop-like structure.
    """
    df = game_logs_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    # Only process players with sufficient minutes (starter/rotation)
    df = df[df['MIN_NUM'] >= 12].copy()

    rows = []
    for _, row in df.iterrows():
        L30 = row.get('L30_PTS')
        if pd.isna(L30) or L30 is None:
            continue

        L10 = row.get('L10_PTS')
        L5 = row.get('L5_PTS')
        L3 = row.get('L3_PTS')

        opp = row['OPPONENT']
        pos = row.get('PLAYER_POSITION', 'Forward')
        is_home = row.get('IS_HOME', 0) == 1
        min30 = row.get('L30_MIN_NUM')
        min10 = row.get('L10_MIN_NUM')

        line = generate_synthetic_line(L30, L10, L5, L3, opp, pos, is_home, min30, min10)

        team = row.get('GAME_TEAM_ABBREVIATION', '')
        matchup = row.get('MATCHUP', '')

        # Determine Home/Away teams from matchup
        if 'vs.' in matchup:
            home_team = team
            away_team = opp
        else:
            home_team = opp
            away_team = team

        rows.append({
            'Date': row['GAME_DATE'],
            'Player': row['PLAYER_NAME'],
            'Position': pos,
            'Game': f"{away_team} @ {home_team}",
            'Home': home_team,
            'Away': away_team,
            'Line': line,
            'Over Odds': -115,  # Standard juice
            'Under Odds': -115,
            'Books': 7,
            'Min Line': np.nan,
            'Max Line': np.nan,
            'Actual_PTS': row['PTS'],  # For grading
        })

    return pd.DataFrame(rows)

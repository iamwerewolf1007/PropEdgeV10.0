"""
PropEdge V10.0 — Rolling Stats Engine
=======================================
Completely independent from V9.2.

Core guarantees:
  - Never reads pre-computed L*_* CSV columns (they go stale)
  - Career-chronological across both seasons (no season resets)
  - DNP rows excluded from all rolling windows
  - Prediction-time cutoff: strictly < game_date (no lookahead)
  - Post-game display cutoff: ≤ game_date (includes result)

V10.0 new features in extract_prediction_features():
  l10_ewm, l5_ewm        — Exponential decay rolling (recency-weighted)
  usage_l10, usage_l30   — Usage rate rolling (corr=0.885 with PTS)
  fg3a_l10, fg3m_l10     — 3-point volume rolling
  fta_l10, ft_rate_l10   — Free throw volume + rate
  home_l10, away_l10     — Home/away split rolling averages
  home_away_split        — Delta between home and away L10
  b2b_pts_delta          — Player-specific B2B pts vs non-B2B pts
  usage_segment          — 0=role / 1=rotational / 2=star
  line_bucket            — Bookmaker line range 0-5
"""

import pandas as pd
import numpy as np
from config import WINDOWS, ROLL_COLS

SEG_LOW  = 15.0
SEG_HIGH = 22.0

_POS_GROUPS = {
    'Guard':   ['PG','SG','G','G-F','F-G','Guard'],
    'Forward': ['SF','PF','F','F-C','C-F','Forward'],
    'Center':  ['C','Center'],
}

def _pos_group(raw):
    for grp, vals in _POS_GROUPS.items():
        if str(raw) in vals: return grp
    return 'Forward'


# ─── DNP UTILITIES ────────────────────────────────────────────────────────────

def is_dnp_row(row):
    if row.get('DNP',0) == 1: return True
    mn = row.get('MIN_NUM', None)
    if mn is None or (isinstance(mn,float) and np.isnan(mn)): return True
    return float(mn) == 0.0

def filter_played(df):
    if df is None or len(df) == 0: return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if 'DNP' in df.columns:
        mask = mask & (df['DNP'].fillna(0) != 1)
    if 'MIN_NUM' in df.columns:
        mask = mask & (df['MIN_NUM'].fillna(0) > 0)
    return df[mask].copy()


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_combined(file_2425, file_2526):
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    if 'DNP' not in combined.columns: combined['DNP'] = 0
    combined['DNP'] = combined['DNP'].fillna(0).astype(int)
    combined = combined.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)
    return combined

def build_player_index(combined_df):
    return {pname: grp.sort_values('GAME_DATE').reset_index(drop=True)
            for pname, grp in combined_df.groupby('PLAYER_NAME', sort=False)}

def get_prior_games_played(pidx, player_name, before_date_str):
    if player_name not in pidx: return pd.DataFrame()
    ph     = pidx[player_name]
    before = pd.Timestamp(before_date_str)
    return filter_played(ph[ph['GAME_DATE'] < before])


# ─── BATCH-PREDICT CACHE BUILDERS ─────────────────────────────────────────────

def build_b2b_delta(played_df):
    """
    Pre-compute per-player B2B quality-controlled delta.
    Removes star-player confound: elite players play more B2B games,
    making raw B2B look positive. This measures each player vs themselves.
    Returns: {player_name: float delta_pts}
    """
    delta = {}
    for pname, grp in played_df.groupby('PLAYER_NAME'):
        g    = grp.sort_values('GAME_DATE')
        rest = g['GAME_DATE'].diff().dt.days.fillna(99).values
        b2b  = rest == 1
        pts  = g['PTS'].fillna(0).values
        bm   = pts[b2b].mean()  if b2b.sum()  > 0 else np.nan
        nm   = pts[~b2b].mean() if (~b2b).sum() > 0 else np.nan
        delta[pname] = round(float(bm - nm), 2) if not (np.isnan(bm) or np.isnan(nm)) else 0.0
    return delta

def build_dynamic_dvp(played_df):
    """
    Compute DVP ranks from actual CSV data (not season-start hardcoded table).
    Updates automatically as each game is appended.
    Returns: {(team, pos_group): rank_int}
    """
    dvp = {}
    pf  = played_df.copy()
    pf['_pg'] = pf['PLAYER_POSITION'].map(_pos_group)
    for pos in ['Guard','Forward','Center']:
        opp_pts = pf[pf['_pg']==pos].groupby('OPPONENT')['PTS'].mean()
        ranks   = opp_pts.rank(ascending=False).astype(int)
        for team, rank in ranks.items():
            dvp[(team, pos)] = int(rank)
    return dvp


# ─── LIVE FEATURE EXTRACTION ──────────────────────────────────────────────────

def extract_prediction_features(prior_played, line,
                                 b2b_delta=None, dyn_dvp=None,
                                 opp=None, pos=None,
                                 player_name=None, game_date=None):
    """
    Compute all V10.0 prediction features from prior played games only.
    No lookahead — prior_played must already be filtered to < game_date.
    Returns None if fewer than 5 played games.
    """
    if prior_played is None or len(prior_played) < 5:
        return None

    p = prior_played

    def safe_mean(col, n):
        if col not in p.columns: return None
        vals = p.tail(n)[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def safe_std(col, n, min_n=3):
        if col not in p.columns: return 5.0
        vals = p.tail(n)[col].dropna()
        return float(vals.std()) if len(vals) >= min_n else 5.0

    def ewm_val(col, span):
        if col not in p.columns: return None
        s = p[col].dropna()
        if len(s) == 0: return None
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    # ── PTS rolling ───────────────────────────────────────────────────────────
    L30 = safe_mean('PTS',30) or 0.0
    L20 = safe_mean('PTS',20) or L30
    L10 = safe_mean('PTS',10) or L30
    L5  = safe_mean('PTS', 5) or L30
    L3  = safe_mean('PTS', 3) or L30

    # EWMA (more weight on recent games — higher predictive power)
    L10_ewm = ewm_val('PTS',10) or L10
    L5_ewm  = ewm_val('PTS', 5) or L5

    # ── Shooting ──────────────────────────────────────────────────────────────
    def fg_pct(n):
        v = safe_mean('FG_PCT',n)
        if v is None: return None
        return round(v*100,1) if v < 1.5 else round(v,1)

    fg30    = fg_pct(30); fg10 = fg_pct(10)
    fgTrend = round(fg10-fg30,1) if fg30 and fg10 else None
    fga30   = safe_mean('FGA',30); fga10 = safe_mean('FGA',10)

    # ── NEW: 3-point volume ───────────────────────────────────────────────────
    fg3a_l10 = safe_mean('FG3A',10) or 0.0
    fg3m_l10 = safe_mean('FG3M',10) or 0.0

    # ── NEW: Free throw ───────────────────────────────────────────────────────
    fta_l10     = safe_mean('FTA',10) or 0.0
    fga10_safe  = max(fga10 or 8.0, 0.5)
    ft_rate_l10 = round(fta_l10 / fga10_safe, 3)

    # ── NEW: Usage rolling ────────────────────────────────────────────────────
    usage_l10 = safe_mean('USAGE_APPROX',10) or 0.0
    usage_l30 = safe_mean('USAGE_APPROX',30) or 0.0

    # ── Minutes ───────────────────────────────────────────────────────────────
    m30 = safe_mean('MIN_NUM',30); m10 = safe_mean('MIN_NUM',10)
    minTrend = round(m10-m30,1) if m30 and m10 else None

    # ── NEW: Home/Away split rolling ──────────────────────────────────────────
    home_l10 = away_l10 = home_away_split = None
    if 'IS_HOME' in p.columns:
        hg = p[p['IS_HOME']==1].tail(10)
        ag = p[p['IS_HOME']==0].tail(10)
        if len(hg) >= 1: home_l10 = float(hg['PTS'].mean())
        if len(ag) >= 1: away_l10 = float(ag['PTS'].mean())
        if home_l10 and away_l10:
            home_away_split = round(home_l10 - away_l10, 1)
    home_l10        = home_l10    if home_l10    is not None else L10
    away_l10        = away_l10    if away_l10    is not None else L10
    home_away_split = home_away_split if home_away_split is not None else 0.0

    # ── Variance / hit rates ──────────────────────────────────────────────────
    std10 = safe_std('PTS',10)
    recent20 = list(p.tail(20)['PTS'].fillna(0).astype(int).values)
    recent10 = recent20[-10:]
    recent30 = list(p.tail(30)['PTS'].fillna(0).astype(int).values)
    r20h = (list(p.tail(20)['IS_HOME'].fillna(0).values.astype(int))
            if 'IS_HOME' in p.columns else [0]*len(recent20))
    hr10 = round(sum(1 for r in recent10 if r>line)/len(recent10)*100) if recent10 else 50
    hr30 = round(sum(1 for r in recent30 if r>line)/len(recent30)*100) if recent30 else 50
    vol  = round(L30-line,1); trend = round(L5-L30,1)

    # ── Minutes-derived features ──────────────────────────────────────────────
    mp10 = p.tail(10)['MIN_NUM'].replace(0,np.nan).fillna(30.0) if 'MIN_NUM' in p.columns else pd.Series([30.0]*10)
    m10c = max(float(mp10.mean()), 1.0)
    min_cv = float(mp10.std()/m10c) if mp10.mean() > 0 else 1.0
    ppm    = float((p.tail(10)['PTS'].fillna(0)/mp10).mean()) if len(p) > 0 else 0.0
    rmt    = float(p.tail(3)['MIN_NUM'].mean() - mp10.mean()) if 'MIN_NUM' in p.columns and len(p)>=3 else 0.0
    fga10s = p.tail(10)['FGA'].fillna(0) if 'FGA' in p.columns else pd.Series([0.0]*10)
    fpm    = float((fga10s/mp10).mean()) if len(fga10s) > 0 else 0.0

    # ── NEW: B2B quality-controlled delta ─────────────────────────────────────
    if b2b_delta is not None and player_name is not None:
        b2b_pts_delta = b2b_delta.get(player_name, 0.0)
    elif 'MIN_NUM' in p.columns and len(p) > 5:
        dates   = p['GAME_DATE'].sort_values().values
        rests   = np.diff(dates.astype('datetime64[D]').astype(int),
                          prepend=dates[0].astype('datetime64[D]').astype(int)-99)
        is_b2b  = rests == 1
        pts_arr = p['PTS'].fillna(0).values
        bm      = pts_arr[is_b2b].mean()  if is_b2b.sum()  > 0 else np.nan
        nm      = pts_arr[~is_b2b].mean() if (~is_b2b).sum() > 0 else np.nan
        b2b_pts_delta = round(float(bm-nm),2) if not(np.isnan(bm) or np.isnan(nm)) else 0.0
    else:
        b2b_pts_delta = 0.0

    # ── NEW: Usage segment ────────────────────────────────────────────────────
    usage_segment = 2 if usage_l10>=SEG_HIGH else 1 if usage_l10>=SEG_LOW else 0

    # ── NEW: Line bucket ──────────────────────────────────────────────────────
    if   line < 10: line_bucket = 0
    elif line < 15: line_bucket = 1
    elif line < 20: line_bucket = 2
    elif line < 25: line_bucket = 3
    elif line < 30: line_bucket = 4
    else:           line_bucket = 5

    return {
        # Standard rolling (backward compat keys in UPPERCASE)
        'L30':round(L30,1),'L20':round(L20,1),'L10':round(L10,1),
        'L5':round(L5,1),'L3':round(L3,1),
        'fg30':fg30,'fg10':fg10,'fgTrend':fgTrend,
        'fga30':round(fga30,1) if fga30 else None,
        'fga10':round(fga10,1) if fga10 else None,
        'm30':round(m30,1) if m30 else None,
        'm10':round(m10,1) if m10 else None,
        'minTrend':minTrend,'std10':round(std10,1),
        'vol':vol,'trend':trend,'hr10':hr10,'hr30':hr30,
        'recent20':recent20,'recent10':recent10,'r20_homes':r20h,
        'min_cv':round(min_cv,3),'ppm':round(ppm,3),
        'rmt':round(rmt,1),'fpm':round(fpm,3),
        # V10.0 new keys (lowercase, match FEATURES list)
        'l10_ewm':round(L10_ewm,1),'l5_ewm':round(L5_ewm,1),
        'usage_l10':round(usage_l10,2),'usage_l30':round(usage_l30,2),
        'fg3a_l10':round(fg3a_l10,1),'fg3m_l10':round(fg3m_l10,1),
        'fta_l10':round(fta_l10,1),'ft_rate_l10':round(ft_rate_l10,3),
        'home_l10':round(home_l10,1),'away_l10':round(away_l10,1),
        'home_away_split':round(home_away_split,1),
        'b2b_pts_delta':b2b_pts_delta,
        'usage_segment':usage_segment,'line_bucket':line_bucket,
    }


# ─── APPEND ROLLING COLS (for batch0 new game log rows) ───────────────────────

def compute_rolling_for_new_rows(new_df, hist_df):
    """
    Compute L*_* rolling columns for newly appended game rows.
    DNP rows: all rolling cols set to NaN.
    Uses explicit loops — never groupby().apply().
    """
    if 'DNP' not in new_df.columns: new_df['DNP'] = 0
    new_df['DNP'] = new_df['DNP'].fillna(0).astype(int)
    if 'DNP' not in hist_df.columns: hist_df = hist_df.copy(); hist_df['DNP'] = 0

    for w in WINDOWS:
        for c in ROLL_COLS:
            new_df[f'L{w}_{c}'] = np.nan

    new_df = new_df.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)

    for pname in new_df['PLAYER_NAME'].dropna().unique():
        ph_played = filter_played(
            hist_df[hist_df['PLAYER_NAME']==pname].sort_values('GAME_DATE'))
        pn_all = new_df[new_df['PLAYER_NAME']==pname].sort_values('GAME_DATE')

        for i,(idx,row) in enumerate(pn_all.iterrows()):
            if int(row.get('DNP',0))==1 or float(row.get('MIN_NUM',0) or 0)==0:
                continue
            new_played = filter_played(pn_all.iloc[:i]) if i>0 else pd.DataFrame()
            prior = (pd.concat([ph_played,new_played]).sort_values('GAME_DATE')
                     if len(new_played)>0 else ph_played.copy())
            for w in WINDOWS:
                subset  = prior.tail(w)
                min_req = w//2 if w>=100 else min(3,w)
                for c in ROLL_COLS:
                    if c not in prior.columns or len(subset)<min_req:
                        new_df.at[idx,f'L{w}_{c}'] = np.nan
                    else:
                        v = subset[c].mean()
                        new_df.at[idx,f'L{w}_{c}'] = round(float(v),4) if pd.notna(v) else np.nan
    return new_df

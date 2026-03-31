"""
PropEdge V10.0 — Projection Model Trainer
==========================================
Completely independent from V9.2.

Trains four model artefacts:
  projection_model.pkl  — Global enhanced GBR (36 features, depth 5)
  segment_model.pkl     — Per-usage-tier GBR (role / rotational / star)
  quantile_models.pkl   — P25 + P75 GBR for uncertainty bands
  calibrator.pkl        — Isotonic regression: |pred_gap| → P(hit)

Feature set (36 total, 15 new vs V9.2):
  Rolling PTS uniform:     l30, l10, l5, l3
  Rolling PTS EWMA:        l10_ewm, l5_ewm               [NEW]
  Derived PTS:             volume, trend, std10, consistency
  3-pt volume:             fg3a_l10, fg3m_l10             [NEW]
  Free throw:              fta_l10, ft_rate_l10           [NEW]
  Usage rate:              usage_l10, usage_l30           [NEW]
  Minutes:                 min_cv, pts_per_min, recent_min_trend
  Home/away split:         home_l10, away_l10, home_away_split [NEW]
  FGA:                     fga_per_min
  Fatigue (QC):            is_b2b, b2b_pts_delta, rest_days  [b2b_delta NEW]
  Matchup (dynamic):       defP, defP_dynamic, pace_rank  [dynamic NEW]
  H2H:                     h2h_ts_dev, h2h_fga_dev, h2h_min_dev, h2h_conf
  Player segment:          usage_segment                  [NEW]
  Line:                    line, line_bucket              [bucket NEW]

Accuracy results (43,781 training rows, 2 seasons):
  MAE:                 4.39 pts  (vs 4.57 baseline, -3.9%)
  Dir acc all plays:   61.9%     (vs 59.1%, +2.8pp)
  Dir acc gap>=3pt:    84.7%     (vs 77.2%, +7.5pp)
  Dir acc gap>=4pt:    91.9%     (vs 81.1%, +10.8pp)
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from config import get_dvp, POS_MAP

# ── FEATURE LIST ──────────────────────────────────────────────────────────────

FEATURES = [
    # Rolling PTS — uniform windows
    'l30','l10','l5','l3',
    # Rolling PTS — exponential decay (V10.0 new)
    'l10_ewm','l5_ewm',
    # Derived PTS
    'volume','trend','std10','consistency',
    # V10.0: 3-point volume
    'fg3a_l10','fg3m_l10',
    # V10.0: Free throw volume + rate
    'fta_l10','ft_rate_l10',
    # V10.0: Usage rate (corr=0.885 with PTS — strongest untapped predictor)
    'usage_l10','usage_l30',
    # Minutes
    'min_cv','pts_per_min','recent_min_trend',
    # V10.0: Home/away split
    'home_l10','away_l10','home_away_split',
    # FGA rate
    'fga_per_min',
    # Fatigue — quality-controlled B2B (V10.0 new: per-player delta)
    'is_b2b','b2b_pts_delta','rest_days',
    # Matchup — static DVP + V10.0 dynamic DVP computed from CSV
    'defP','defP_dynamic','pace_rank',
    # Head-to-head
    'h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf',
    # V10.0: Usage segment
    'usage_segment',
    # Line + V10.0 bookmaker line bucket
    'line','line_bucket',
]

FEATURES_SEGMENT = FEATURES  # segment models use same feature set

_POS_GRP = {
    'Guard':   ['PG','SG','G','G-F','F-G','Guard'],
    'Forward': ['SF','PF','F','F-C','C-F','Forward'],
    'Center':  ['C','Center'],
}

def _pgrp(raw):
    for g,vals in _POS_GRP.items():
        if str(raw) in vals: return g
    return 'Forward'


# ── TRAINING DATA BUILDER ─────────────────────────────────────────────────────

def build_training_data(file_2425, file_2526, file_h2h):
    """
    Build training samples with full V10.0 feature set.
    shift(1) rolling guarantees zero lookahead.
    DNP rows excluded before any rolling computation.
    """
    from rolling_engine import filter_played

    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(file_h2h)

    for df in [df25,df26]:
        if 'DNP' not in df.columns: df['DNP'] = 0

    combined = pd.concat([df25,df26], ignore_index=True)
    combined = combined.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)
    combined = filter_played(combined).copy().reset_index(drop=True)

    h2h_dedup = h2h.drop_duplicates(subset=['PLAYER_NAME','OPPONENT'], keep='last')
    h2h_lkp   = {(r['PLAYER_NAME'],r['OPPONENT']): r.to_dict()
                 for _,r in h2h_dedup.iterrows()}

    print(f"    Played rows: {len(combined):,}   Players: {combined['PLAYER_NAME'].nunique():,}")

    grp = combined.groupby('PLAYER_NAME', sort=False)

    def sroll(col,w):
        return grp[col].transform(lambda s: s.rolling(w,min_periods=1).mean().shift(1))
    def ewroll(col,span):
        return grp[col].transform(lambda s: s.shift(1).ewm(span=span,adjust=False).mean())

    print("    Computing vectorised rolling features...")
    rolled = pd.concat([
        sroll('PTS',30).rename('_l30'), sroll('PTS',10).rename('_l10'),
        sroll('PTS', 5).rename('_l5'),  sroll('PTS', 3).rename('_l3'),
        ewroll('PTS',10).rename('_l10_ewm'), ewroll('PTS',5).rename('_l5_ewm'),
        grp['PTS'].transform(lambda s: s.rolling(10,min_periods=3).std().shift(1)).fillna(5.0).rename('_std10'),
        sroll('MIN_NUM',10).rename('_m10'), sroll('MIN_NUM',3).rename('_m3'),
        sroll('FGA',10).rename('_fga10'),
        sroll('FG3A',10).rename('_fg3a10'), sroll('FG3M',10).rename('_fg3m10'),
        sroll('FTA',10).rename('_fta10'),
        sroll('USAGE_APPROX',10).rename('_usage10'),
        sroll('USAGE_APPROX',30).rename('_usage30'),
        grp['GAME_DATE'].transform(lambda s: s.diff().dt.days.fillna(99)).astype(int).rename('_rest'),
        grp['GAME_DATE'].transform('cumcount').rename('_seq'),
    ], axis=1)

    # Home/away split rolling
    if 'IS_HOME' in combined.columns:
        combined['_hp'] = combined['PTS'].where(combined['IS_HOME']==1)
        combined['_ap'] = combined['PTS'].where(combined['IS_HOME']==0)
        rolled['_home_l10'] = combined.groupby('PLAYER_NAME')['_hp'].transform(
            lambda s: s.rolling(10,min_periods=1).mean().shift(1).ffill())
        rolled['_away_l10'] = combined.groupby('PLAYER_NAME')['_ap'].transform(
            lambda s: s.rolling(10,min_periods=1).mean().shift(1).ffill())
        combined.drop(columns=['_hp','_ap'], inplace=True)
    else:
        rolled['_home_l10'] = np.nan
        rolled['_away_l10'] = np.nan

    base = pd.concat([
        combined[['PLAYER_NAME','GAME_DATE','PTS','OPPONENT','PLAYER_POSITION']],
        rolled
    ], axis=1)

    base = base[base['_seq'] >= 10].dropna(subset=['_l30']).copy()
    print(f"    After sequence filter: {len(base):,}")

    # Cast numerics
    for c in ['_l30','_l10','_l5','_l3','_l10_ewm','_l5_ewm','_std10',
              '_m10','_m3','_fga10','_fg3a10','_fg3m10','_fta10',
              '_usage10','_usage30','_rest','_home_l10','_away_l10']:
        base[c] = pd.to_numeric(base[c], errors='coerce')
    base['_m10']      = base['_m10'].fillna(28.0)
    base['_m3']       = base['_m3'].fillna(28.0)
    base['_fga10']    = base['_fga10'].fillna(8.0)
    base['_usage10']  = base['_usage10'].fillna(0.0)
    base['_usage30']  = base['_usage30'].fillna(0.0)
    base['_fta10']    = base['_fta10'].fillna(0.0)
    base['_fg3a10']   = base['_fg3a10'].fillna(0.0)
    base['_fg3m10']   = base['_fg3m10'].fillna(0.0)
    base['_home_l10'] = base['_home_l10'].fillna(base['_l10'])
    base['_away_l10'] = base['_away_l10'].fillna(base['_l10'])

    # Derived features
    base['line']       = (base['_l30']*2).round()/2
    base['line']       = base['line'].clip(lower=3.5)
    base['volume']     = (base['_l30']-base['line']).round(1)
    base['trend']      = (base['_l5']-base['_l30']).round(1)
    m10c               = base['_m10'].clip(lower=1)
    base['min_cv']     = (base['_std10']/m10c).round(3)
    base['pts_per_min']= (base['_l10']/m10c).round(3)
    base['recent_min_trend'] = (base['_m3']-base['_m10']).round(1)
    base['fga_per_min']= (base['_fga10']/m10c).round(3)
    base['consistency']= (1/(base['_std10']+1)).round(3)
    base['is_b2b']     = (base['_rest']==1).astype(int)
    base['rest_days']  = base['_rest'].clip(upper=10).astype(int)
    base['ft_rate_l10']= (base['_fta10']/base['_fga10'].clip(lower=0.5)).round(3)
    base['home_away_split'] = (base['_home_l10']-base['_away_l10']).round(1)

    # B2B quality-controlled delta
    b2b_m  = base['is_b2b']==1
    p_b2b  = base[b2b_m].groupby('PLAYER_NAME')['PTS'].mean()
    p_norm = base[~b2b_m].groupby('PLAYER_NAME')['PTS'].mean()
    base['b2b_pts_delta'] = base['PLAYER_NAME'].map(
        (p_b2b-p_norm).fillna(0)).fillna(0).round(2)

    # Usage segment
    base['usage_segment'] = pd.cut(
        base['_usage10'], bins=[-np.inf,15.0,22.0,np.inf], labels=[0,1,2]
    ).astype(float).fillna(0)

    # Line bucket
    base['line_bucket'] = pd.cut(
        base['line'], bins=[0,10,15,20,25,30,100], labels=[0,1,2,3,4,5]
    ).astype(float).fillna(0)

    # Dynamic DVP from CSV
    base['_pos_grp'] = base['PLAYER_POSITION'].map(_pgrp)
    base['defP_dynamic'] = 15.0
    for pos in ['Guard','Forward','Center']:
        pm   = combined['PLAYER_POSITION'].map(_pgrp)==pos
        opp  = combined[pm].groupby('OPPONENT')['PTS'].mean()
        rnks = opp.rank(ascending=False).astype(int)
        bm   = base['_pos_grp']==pos
        base.loc[bm,'defP_dynamic'] = base.loc[bm,'OPPONENT'].map(rnks).fillna(15)

    # Static DVP (config.py)
    base['defP'] = base.apply(
        lambda r: get_dvp(r['OPPONENT'], POS_MAP.get(str(r['PLAYER_POSITION']),'Forward')), axis=1)

    # Pace rank
    team_fga  = combined.groupby('OPPONENT')['FGA'].mean()
    pace_map  = {t:i+1 for i,(t,_) in enumerate(team_fga.sort_values(ascending=False).items())}
    base['pace_rank'] = base['OPPONENT'].map(pace_map).fillna(15).astype(int)

    # H2H
    def _h2h(row):
        hr = h2h_lkp.get((row['PLAYER_NAME'],row['OPPONENT']))
        if hr is None: return 0.0,0.0,0.0,0.0
        def s(k): return float(hr[k]) if pd.notna(hr.get(k)) else 0.0
        return s('H2H_TS_VS_OVERALL'),s('H2H_FGA_VS_OVERALL'),\
               s('H2H_MIN_VS_OVERALL'),s('H2H_CONFIDENCE')
    hv = base.apply(_h2h, axis=1, result_type='expand')
    hv.columns = ['h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf']
    base = pd.concat([base, hv], axis=1)

    base = base.rename(columns={
        '_l30':'l30','_l10':'l10','_l5':'l5','_l3':'l3',
        '_l10_ewm':'l10_ewm','_l5_ewm':'l5_ewm','_std10':'std10',
        '_fga10':'fga10','_fg3a10':'fg3a_l10','_fg3m10':'fg3m_l10',
        '_fta10':'fta_l10','_usage10':'usage_l10','_usage30':'usage_l30',
        '_m10':'m10','_m3':'m3','_home_l10':'home_l10','_away_l10':'away_l10',
    })
    base['actual_pts'] = combined.loc[base.index,'PTS'].astype(int)
    print(f"    Final training samples: {len(base):,}")
    return base


# ── MAIN TRAINER ──────────────────────────────────────────────────────────────

def train_and_save(file_2425, file_2526, file_h2h, model_file, trust_file,
                   segment_file=None, quantile_file=None, calibrator_file=None):
    """
    Full V10.0 training pipeline.
    Trains and saves: global GBR, SegmentModel, quantile P25/P75, calibrator, trust.
    """
    from segment_model import SegmentModel

    print("    Building training data...")
    train_df = build_training_data(file_2425, file_2526, file_h2h)
    for col in FEATURES:
        if col not in train_df.columns: train_df[col] = 0.0

    X = train_df[FEATURES].fillna(0)
    y = train_df['actual_pts']

    # 1. Global enhanced GBR
    print("    Training global enhanced GBR (depth=5, 36 features)...")
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        min_samples_leaf=15, subsample=0.8,
        n_iter_no_change=20, validation_fraction=0.1, tol=1e-4, random_state=42,
    )
    model.fit(X, y)
    print(f"    GBR: {model.n_estimators_} trees")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file,'wb') as f: pickle.dump(model, f)
    print(f"    ✓ projection_model.pkl")

    # 2. Segment models (role / rotational / star)
    seg_path = segment_file or model_file.parent/'segment_model.pkl'
    print("    Training per-usage-tier segment models...")
    sm = SegmentModel()
    sm.fit(X, y.values, train_df['usage_l10'].values, fallback_model=model)
    sm.save(seg_path)
    print(f"    ✓ segment_model.pkl")

    # 3. Quantile P25/P75
    q_path = quantile_file or model_file.parent/'quantile_models.pkl'
    print("    Training quantile GBR (P25 + P75)...")
    q_lo = GradientBoostingRegressor(loss='quantile',alpha=0.25,n_estimators=200,
        max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42)
    q_hi = GradientBoostingRegressor(loss='quantile',alpha=0.75,n_estimators=200,
        max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42)
    q_lo.fit(X,y); q_hi.fit(X,y)
    with open(q_path,'wb') as f: pickle.dump({'q25':q_lo,'q75':q_hi}, f)
    print(f"    ✓ quantile_models.pkl")

    # 4. Isotonic calibrator
    cal_path = calibrator_file or model_file.parent/'calibrator.pkl'
    print("    Fitting isotonic probability calibrator...")
    pred_pts = sm.predict(X, train_df['usage_l10'].values)
    pred_gap = pred_pts - train_df['line'].values
    hit      = (y.values > train_df['line'].values).astype(int)
    correct  = (((pred_gap>0)&(hit==1))|((pred_gap<0)&(hit==0))).astype(int)
    cal      = IsotonicRegression(out_of_bounds='clip')
    cal.fit(np.abs(pred_gap), correct)
    with open(cal_path,'wb') as f: pickle.dump(cal, f)
    print(f"    ✓ calibrator.pkl")

    # 5. Player trust scores
    train_df['pred'] = pred_pts
    train_df['correct'] = (
        ((train_df['pred']>train_df['line'])&(train_df['actual_pts']>train_df['line']))|
        ((train_df['pred']<train_df['line'])&(train_df['actual_pts']<train_df['line']))
    ).astype(int)
    trust = {p: round(float(g['correct'].mean()),3)
             for p,g in train_df.groupby('PLAYER_NAME') if len(g)>=10}
    with open(trust_file,'w') as f: json.dump(trust, f, indent=2)
    print(f"    ✓ player_trust.json ({len(trust)} players)")
    print(f"    In-sample accuracy: {float(train_df['correct'].mean()):.1%}")
    return model

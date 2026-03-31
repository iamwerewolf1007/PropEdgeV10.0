"""
PropEdge V9.2 — H2H Database Builder (Vectorised)
"""
import pandas as pd
import numpy as np


def build_h2h(file_2425, file_2526, output_file):
    use_cols = ['PLAYER_ID','PLAYER_NAME','GAME_DATE','OPPONENT','IS_HOME','WL',
                'PTS','MIN_NUM','FGM','FGA','FG3M','FG3A','FTM','FTA',
                'REB','AST','STL','BLK','TOV','PLUS_MINUS',
                'TRUE_SHOOTING_PCT','USAGE_APPROX','PTS_REB_AST',
                'GAME_TEAM_ABBREVIATION']
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'], usecols=use_cols)
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'], usecols=use_cols)
    df25 = df25.assign(_SEASON='2024-25')
    df26 = df26.assign(_SEASON='2025-26')
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    combined = combined.sort_values(['PLAYER_ID','GAME_DATE']).reset_index(drop=True)
    latest_date = combined['GAME_DATE'].max()
    keys = ['PLAYER_ID','PLAYER_NAME','OPPONENT']

    overall = combined.groupby('PLAYER_ID').agg(
        OVR_PTS=('PTS','mean'),OVR_FGA=('FGA','mean'),OVR_FTA=('FTA','mean'),
        OVR_MIN=('MIN_NUM','mean'),OVR_TS=('TRUE_SHOOTING_PCT','mean'),
        OVR_USG=('USAGE_APPROX','mean'))

    grp = combined.groupby(keys)
    agg = grp.agg(
        H2H_GAMES=('PTS','count'),H2H_AVG_PTS=('PTS','mean'),H2H_MEDIAN_PTS=('PTS','median'),
        H2H_STD_PTS=('PTS','std'),H2H_MIN_PTS=('PTS','min'),H2H_MAX_PTS=('PTS','max'),
        H2H_AVG_FGA=('FGA','mean'),H2H_AVG_FTA=('FTA','mean'),
        H2H_AVG_MIN=('MIN_NUM','mean'),H2H_AVG_USAGE=('USAGE_APPROX','mean'),
        H2H_AVG_PLUS_MINUS=('PLUS_MINUS','mean'),
        H2H_AVG_REB=('REB','mean'),H2H_AVG_AST=('AST','mean'),
        H2H_AVG_STL=('STL','mean'),H2H_AVG_BLK=('BLK','mean'),
        H2H_AVG_TOV=('TOV','mean'),H2H_AVG_PTS_REB_AST=('PTS_REB_AST','mean'),
        TOTAL_FGM=('FGM','sum'),TOTAL_FGA=('FGA','sum'),
        TOTAL_FG3M=('FG3M','sum'),TOTAL_FG3A=('FG3A','sum'),
        TOTAL_FTM=('FTM','sum'),TOTAL_FTA=('FTA','sum'),TOTAL_PTS=('PTS','sum'),
        LAST_DATE=('GAME_DATE','max'),TEAM=('GAME_TEAM_ABBREVIATION','last'),
    ).reset_index()

    def tail_agg(g, n):
        def _t(x):
            t = x.sort_values('GAME_DATE').tail(n)
            fgm,fga,fta,pts = t['FGM'].sum(),t['FGA'].sum(),t['FTA'].sum(),t['PTS'].sum()
            tsa = 2*(fga+0.44*fta)
            return pd.Series({f'L{n}_AVG_PTS':t['PTS'].mean(),f'L{n}_AVG_FGA':t['FGA'].mean(),
                f'L{n}_GAMES':len(t),f'L{n}_FG_PCT':(fgm/fga*100) if fga>0 else np.nan,
                f'L{n}_TS_PCT':(pts/tsa*100) if tsa>0 else np.nan})
        return g.apply(_t).reset_index()

    l3=tail_agg(grp,3); l5=tail_agg(grp,5)
    cs=combined[combined['_SEASON']=='2025-26'].groupby(keys).agg(CS_GAMES=('PTS','count'),CS_AVG_PTS=('PTS','mean')).reset_index()
    ps=combined[combined['_SEASON']=='2024-25'].groupby(keys).agg(PS_GAMES=('PTS','count'),PS_AVG_PTS=('PTS','mean')).reset_index()
    hm=combined[combined['IS_HOME']==1].groupby(keys).agg(HOME_AVG_PTS=('PTS','mean')).reset_index()
    aw=combined[combined['IS_HOME']==0].groupby(keys).agg(AWAY_AVG_PTS=('PTS','mean')).reset_index()
    wn=combined[combined['WL']=='W'].groupby(keys).agg(WINS=('PTS','count'),WIN_AVG_PTS=('PTS','mean')).reset_index()
    lo=combined[combined['WL']=='L'].groupby(keys).agg(LOSSES=('PTS','count'),LOSS_AVG_PTS=('PTS','mean')).reset_index()
    bl=combined[combined['PLUS_MINUS'].abs()>15].groupby(keys).agg(BLOWOUT_GAMES=('PTS','count')).reset_index()

    h = agg
    for m in [l3,l5,cs,ps,hm,aw,wn,lo,bl]:
        h = h.merge(m, on=keys, how='left')
    h = h.merge(overall, on='PLAYER_ID', how='left')

    h['H2H_FG_PCT']=np.where(h['TOTAL_FGA']>0,(h['TOTAL_FGM']/h['TOTAL_FGA']*100).round(1),np.nan)
    h['H2H_FG3_PCT']=np.where(h['TOTAL_FG3A']>0,(h['TOTAL_FG3M']/h['TOTAL_FG3A']*100).round(1),np.nan)
    h['H2H_FT_PCT']=np.where(h['TOTAL_FTA']>0,(h['TOTAL_FTM']/h['TOTAL_FTA']*100).round(1),np.nan)
    tsa=2*(h['TOTAL_FGA']+0.44*h['TOTAL_FTA'])
    h['H2H_TS_PCT']=np.where(tsa>0,(h['TOTAL_PTS']/tsa*100).round(1),np.nan)
    h['H2H_EFG_PCT']=np.where(h['TOTAL_FGA']>0,((h['TOTAL_FGM']+0.5*h['TOTAL_FG3M'])/h['TOTAL_FGA']*100).round(1),np.nan)
    h['H2H_FGA_VS_OVERALL']=(h['H2H_AVG_FGA']-h['OVR_FGA']).round(1)
    h['H2H_FTA_VS_OVERALL']=(h['H2H_AVG_FTA']-h['OVR_FTA']).round(1)
    h['H2H_MIN_VS_OVERALL']=(h['H2H_AVG_MIN']-h['OVR_MIN']).round(1)
    h['H2H_USAGE_VS_OVERALL']=(h['H2H_AVG_USAGE']-h['OVR_USG']).round(2)
    h['H2H_TS_VS_OVERALL']=(h['H2H_TS_PCT']-h['OVR_TS']*100).round(1)
    h['H2H_PTS_TREND']=(h['L3_AVG_PTS']-h['H2H_AVG_PTS']).round(1)
    h['H2H_SEASON_DRIFT']=(h['CS_AVG_PTS']-h['PS_AVG_PTS']).round(1)
    h['H2H_HOME_AWAY_DIFF']=(h['HOME_AVG_PTS']-h['AWAY_AVG_PTS']).round(1)
    h['H2H_WL_PTS_DIFF']=(h['WIN_AVG_PTS']-h['LOSS_AVG_PTS']).round(1)
    h['DAYS_SINCE_LAST_H2H']=(latest_date-h['LAST_DATE']).dt.days
    h['_sf']=(h['H2H_GAMES']/8.0).clip(upper=1.0)
    h['_cv']=np.where((h['H2H_AVG_PTS']>0)&(h['H2H_GAMES']>=2),h['H2H_STD_PTS']/h['H2H_AVG_PTS'],1.0)
    h['H2H_CONFIDENCE']=(0.6*h['_sf']+0.4*(1.0-h['_cv']).clip(lower=0)).round(3)
    h['H2H_PREDICTABILITY']=np.where(h['_cv']>0,(1.0/h['_cv']).round(2),np.nan)
    days=h['DAYS_SINCE_LAST_H2H']
    h['H2H_RECENCY_WEIGHT']=np.select([days<=14,days<=60,days<=120,days<=200],[1.0,0.8,0.6,0.4],default=0.2)
    mask=h['CS_GAMES'].fillna(0)>=2
    h.loc[mask,'H2H_RECENCY_WEIGHT']=(h.loc[mask,'H2H_RECENCY_WEIGHT']+0.15).clip(upper=1.0)
    pts_dev=h['H2H_AVG_PTS']-h['OVR_PTS']
    ts_dev=h['H2H_TS_VS_OVERALL'].abs().fillna(0)
    fga_dev=h['H2H_FGA_VS_OVERALL'].abs().fillna(0)
    min_dev=h['H2H_MIN_VS_OVERALL'].abs().fillna(0)
    h['H2H_SCORING_PROFILE']=np.select(
        [pts_dev.abs()<1.5,(min_dev>2.5)&(min_dev>=fga_dev)&(min_dev>=ts_dev),
         (fga_dev>1.5)&(fga_dev>=ts_dev),ts_dev>3.0],
        ['NEUTRAL','MINUTES','VOLUME','EFFICIENCY'],default='MIXED')

    rename={'CS_GAMES':'H2H_GAMES_CURRENT_SEASON','PS_GAMES':'H2H_GAMES_PRIOR_SEASON',
        'L3_GAMES':'L3_H2H_GAMES','L5_GAMES':'L5_H2H_GAMES',
        'L3_AVG_PTS':'L3_H2H_AVG_PTS','L5_AVG_PTS':'L5_H2H_AVG_PTS',
        'CS_AVG_PTS':'H2H_CURRENT_SEASON_AVG_PTS','HOME_AVG_PTS':'H2H_HOME_AVG_PTS',
        'AWAY_AVG_PTS':'H2H_AWAY_AVG_PTS','L3_AVG_FGA':'L3_H2H_AVG_FGA',
        'L3_FG_PCT':'L3_H2H_FG_PCT','L3_TS_PCT':'L3_H2H_TS_PCT',
        'WINS':'H2H_WINS','LOSSES':'H2H_LOSSES','WIN_AVG_PTS':'H2H_WIN_AVG_PTS',
        'LOSS_AVG_PTS':'H2H_LOSS_AVG_PTS','BLOWOUT_GAMES':'H2H_BLOWOUT_GAMES'}
    out_cols=['PLAYER_ID','PLAYER_NAME','TEAM','OPPONENT',
        'H2H_GAMES','CS_GAMES','PS_GAMES','DAYS_SINCE_LAST_H2H','L3_GAMES','L5_GAMES',
        'H2H_AVG_PTS','H2H_MEDIAN_PTS','H2H_STD_PTS','H2H_MIN_PTS','H2H_MAX_PTS',
        'L3_AVG_PTS','L5_AVG_PTS','H2H_PTS_TREND','CS_AVG_PTS','H2H_SEASON_DRIFT',
        'HOME_AVG_PTS','AWAY_AVG_PTS','H2H_HOME_AWAY_DIFF','H2H_WL_PTS_DIFF',
        'H2H_AVG_FGA','H2H_FGA_VS_OVERALL','H2H_AVG_FTA','H2H_FTA_VS_OVERALL',
        'H2H_AVG_MIN','H2H_MIN_VS_OVERALL','H2H_USAGE_VS_OVERALL','L3_AVG_FGA',
        'H2H_FG_PCT','H2H_FG3_PCT','H2H_FT_PCT','H2H_TS_PCT','H2H_EFG_PCT',
        'H2H_TS_VS_OVERALL','L3_FG_PCT','L3_TS_PCT',
        'H2H_AVG_PLUS_MINUS','WINS','LOSSES','WIN_AVG_PTS','LOSS_AVG_PTS','BLOWOUT_GAMES',
        'H2H_AVG_REB','H2H_AVG_AST','H2H_AVG_STL','H2H_AVG_BLK','H2H_AVG_TOV','H2H_AVG_PTS_REB_AST',
        'H2H_CONFIDENCE','H2H_SCORING_PROFILE','H2H_PREDICTABILITY','H2H_RECENCY_WEIGHT']
    final=h[out_cols].rename(columns=rename)
    final=final.fillna({'H2H_GAMES_CURRENT_SEASON':0,'H2H_GAMES_PRIOR_SEASON':0,
        'H2H_WINS':0,'H2H_LOSSES':0,'H2H_BLOWOUT_GAMES':0})
    final=final.sort_values(['PLAYER_NAME','OPPONENT']).reset_index(drop=True)
    final[final.select_dtypes('float64').columns]=final.select_dtypes('float64').round(2)
    final.to_csv(output_file, index=False)
    print(f"  ✓ H2H: {len(final)} pairs → {output_file.name}")
    return len(final)

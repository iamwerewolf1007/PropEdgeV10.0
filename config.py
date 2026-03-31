"""
PropEdge V10.0 — Configuration & Shared Constants
===================================================
Completely independent from V9.2.
Repo:       git@github.com:iamwerewolf1007/PropEdgeV10.0.git
Working dir: ~/Documents/GitHub/PropEdgeV10.0
"""
from pathlib import Path
from datetime import timezone, timedelta, datetime

VERSION    = 'V10.0'

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
SOURCE_DIR = ROOT / 'source-files'
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'
LOG_DIR    = ROOT / 'logs'
DAILY_DIR  = ROOT / 'daily'
MASTER_DIR = ROOT / 'master'

# Source data files (place these in source-files/ before first run)
FILE_GL_2425 = SOURCE_DIR / 'nba_gamelogs_2024_25.csv'
FILE_GL_2526 = SOURCE_DIR / 'nba_gamelogs_2025_26.csv'
FILE_H2H     = SOURCE_DIR / 'h2h_database.csv'
FILE_PROPS   = SOURCE_DIR / 'PropEdge_-_Match_and_Player_Prop_lines_.xlsx'

# Model files (written by model_trainer.py, read by batch_predict.py)
FILE_MODEL       = MODEL_DIR / 'projection_model.pkl'    # Global enhanced GBR
FILE_TRUST       = MODEL_DIR / 'player_trust.json'       # Per-player directional accuracy
FILE_SEG_MODELS  = MODEL_DIR / 'segment_model.pkl'       # Per-usage-tier GBR (role/rot/star)
FILE_Q_MODELS    = MODEL_DIR / 'quantile_models.pkl'     # P25 + P75 uncertainty bounds
FILE_CALIBRATOR  = MODEL_DIR / 'calibrator.pkl'          # Isotonic gap → P(hit)

# Live data files
TODAY_JSON  = DATA_DIR / 'today.json'
SEASON_2425 = DATA_DIR / 'season_2024_25.json'
SEASON_2526 = DATA_DIR / 'season_2025_26.json'
AUDIT_LOG   = DATA_DIR / 'audit_log.csv'

# ─── REPO / GIT ───────────────────────────────────────────────────────────────
REPO_DIR   = Path.home() / 'Documents' / 'GitHub' / 'PropEdgeV10.0'
GIT_REMOTE = 'git@github.com:iamwerewolf1007/PropEdgeV10.0.git'

# ─── API ──────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = 'c0bab20a574208a41a6e0d930cdaf313'
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
SPORT         = 'basketball_nba'
CREDIT_ALERT  = 170

# ─── TIMEZONE ─────────────────────────────────────────────────────────────────
def _offset(hours): return timezone(timedelta(hours=hours))
def _is_dst():      m = datetime.now(timezone.utc).month; return 3 <= m <= 10

def get_et(): return _offset(-4 if _is_dst() else -5)
def get_uk(): return _offset(1  if _is_dst() else 0)

def today_et():  return datetime.now(get_et()).strftime('%Y-%m-%d')
def now_uk():    return datetime.now(get_uk())
def now_utc():   return datetime.now(timezone.utc)

# ─── TEAM MAPS ────────────────────────────────────────────────────────────────
TEAM_ABR = {
    'Atlanta Hawks':'ATL','Boston Celtics':'BOS','Brooklyn Nets':'BKN',
    'Charlotte Hornets':'CHA','Chicago Bulls':'CHI','Cleveland Cavaliers':'CLE',
    'Dallas Mavericks':'DAL','Denver Nuggets':'DEN','Detroit Pistons':'DET',
    'Golden State Warriors':'GSW','Houston Rockets':'HOU','Indiana Pacers':'IND',
    'LA Clippers':'LAC','Los Angeles Clippers':'LAC','Los Angeles Lakers':'LAL',
    'Memphis Grizzlies':'MEM','Miami Heat':'MIA','Milwaukee Bucks':'MIL',
    'Minnesota Timberwolves':'MIN','New Orleans Pelicans':'NOP',
    'New York Knicks':'NYK','Oklahoma City Thunder':'OKC','Orlando Magic':'ORL',
    'Philadelphia 76ers':'PHI','Phoenix Suns':'PHX','Portland Trail Blazers':'POR',
    'Sacramento Kings':'SAC','San Antonio Spurs':'SAS','Toronto Raptors':'TOR',
    'Utah Jazz':'UTA','Washington Wizards':'WAS',
}
TEAM_FULL = {v: k for k, v in TEAM_ABR.items()}

def resolve_abr(full_name):
    return TEAM_ABR.get(full_name, full_name[:3].upper())

# ─── DVP RANKINGS (position-specific, updated at season start) ────────────────
DVP_RAW = {
    'BOS':{'PG':4,'SG':1,'SF':1,'PF':3,'C':1},
    'DET':{'PG':1,'SG':7,'SF':10,'PF':17,'C':5},
    'GSW':{'PG':16,'SG':16,'SF':16,'PF':11,'C':21},
    'ATL':{'PG':12,'SG':27,'SF':27,'PF':22,'C':17},
    'HOU':{'PG':7,'SG':3,'SF':8,'PF':1,'C':7},
    'BKN':{'PG':9,'SG':26,'SF':17,'PF':23,'C':19},
    'MEM':{'PG':24,'SG':20,'SF':21,'PF':25,'C':23},
    'LAC':{'PG':19,'SG':21,'SF':6,'PF':2,'C':12},
    'DAL':{'PG':13,'SG':29,'SF':12,'PF':26,'C':27},
    'CLE':{'PG':15,'SG':10,'SF':15,'PF':16,'C':20},
    'CHA':{'PG':5,'SG':13,'SF':2,'PF':14,'C':4},
    'DEN':{'PG':14,'SG':8,'SF':20,'PF':19,'C':10},
    'IND':{'PG':29,'SG':14,'SF':28,'PF':13,'C':24},
    'LAL':{'PG':11,'SG':15,'SF':22,'PF':8,'C':6},
    'MIA':{'PG':27,'SG':18,'SF':19,'PF':27,'C':14},
    'CHI':{'PG':20,'SG':17,'SF':29,'PF':28,'C':26},
    'NOP':{'PG':21,'SG':25,'SF':23,'PF':20,'C':25},
    'UTA':{'PG':30,'SG':30,'SF':24,'PF':30,'C':22},
    'SAC':{'PG':22,'SG':28,'SF':14,'PF':18,'C':29},
    'POR':{'PG':18,'SG':22,'SF':26,'PF':15,'C':28},
    'WAS':{'PG':26,'SG':23,'SF':30,'PF':29,'C':30},
    'OKC':{'PG':2,'SG':11,'SF':13,'PF':4,'C':8},
    'NYK':{'PG':3,'SG':6,'SF':9,'PF':7,'C':2},
    'PHI':{'PG':8,'SG':24,'SF':18,'PF':24,'C':15},
    'PHX':{'PG':6,'SG':2,'SF':7,'PF':9,'C':16},
    'MIN':{'PG':25,'SG':4,'SF':4,'PF':10,'C':13},
    'ORL':{'PG':23,'SG':12,'SF':3,'PF':12,'C':11},
    'TOR':{'PG':17,'SG':9,'SF':5,'PF':6,'C':9},
    'SAS':{'PG':10,'SG':5,'SF':11,'PF':5,'C':18},
    'MIL':{'PG':28,'SG':19,'SF':25,'PF':21,'C':3},
}

POS_MAP = {
    'G':'Guard','F':'Forward','C':'Center','G-F':'Guard','F-G':'Guard',
    'PG':'Guard','SG':'Guard','F-C':'Forward','C-F':'Center','SF':'Forward',
    'PF':'Forward','Guard':'Guard','Forward':'Forward','Center':'Center',
}

def get_dvp(team, pos, fallback=15):
    if team not in DVP_RAW: return fallback
    d = DVP_RAW[team]
    pos = POS_MAP.get(str(pos), str(pos))
    if pos == 'Guard':   return round((d['PG'] + d['SG']) / 2)
    elif pos == 'Center': return d['C']
    else:                 return round((d['SF'] + d['PF']) / 2)

def get_def_overall(team, fallback=15):
    if team not in DVP_RAW: return fallback
    return round(sum(DVP_RAW[team].values()) / 5)

# ─── POSITION WEIGHTS (10-signal Engine A) ────────────────────────────────────
POS_WEIGHTS = {
    'Guard':   {1:3.0,2:2.5,3:2.0,4:2.0,5:1.0,6:1.5,7:1.2,8:0.5,9:1.5,10:1.0},
    'Forward': {1:3.0,2:2.5,3:2.0,4:1.5,5:1.5,6:1.5,7:1.0,8:0.5,9:1.0,10:0.75},
    'Center':  {1:2.5,2:2.0,3:2.0,4:1.0,5:1.5,6:2.5,7:1.0,8:1.0,9:0.5,10:1.5},
}

# ─── ROLLING WINDOWS & COLUMNS ────────────────────────────────────────────────
WINDOWS   = [3,5,10,20,30,50,100,200]
ROLL_COLS = [
    'MIN_NUM','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
    'FTM','FTA','FT_PCT','OREB','DREB','REB','AST','STL',
    'BLK','TOV','PF','PTS','PLUS_MINUS',
    'WL_WIN','WL_LOSS','IS_HOME',
    'EFF_FG_PCT','TRUE_SHOOTING_PCT','USAGE_APPROX','FANTASY_PTS',
    'PTS_REB_AST','PTS_REB','PTS_AST','REB_AST',
    'DOUBLE_DOUBLE','TRIPLE_DOUBLE',
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def american_to_decimal(odds):
    if odds is None: return None
    try: odds = float(odds)
    except: return None
    if odds == 0: return None
    return round(odds/100+1,3) if odds > 0 else round(100/abs(odds)+1,3)

def clean_json(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):           return {k: clean_json(v) for k,v in obj.items()}
    if isinstance(obj, list):           return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):     return int(obj)
    if isinstance(obj, np.floating):    return None if np.isnan(obj) else round(float(obj),4)
    if isinstance(obj, np.bool_):       return bool(obj)
    if isinstance(obj, np.ndarray):     return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, float) and obj != obj: return None
    return obj

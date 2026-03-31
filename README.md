# PropEdge V10.0 — NBA Player Points Prop Prediction Engine

**Completely independent from V9.2.**
Repo: `git@github.com:iamwerewolf1007/PropEdgeV10.0.git`
Working dir: `~/Documents/GitHub/PropEdgeV10.0`

---

## What's new in V10.0

### Model improvements (vs V9.2 baseline)

| Metric | V9.2 | V10.0 | Gain |
|---|---|---|---|
| MAE | 4.57 pts | **4.39 pts** | −0.18 pts (−3.9%) |
| Direction acc — all plays | 59.1% | **61.9%** | +2.8pp |
| Direction acc — gap ≥ 1pt | 65.3% | **69.7%** | +4.4pp |
| Direction acc — gap ≥ 2pt | 71.6% | **77.3%** | +5.7pp |
| Direction acc — gap ≥ 3pt | 77.2% | **84.7%** | +7.5pp |
| Direction acc — gap ≥ 4pt | 81.1% | **91.9%** | +10.8pp |
| Star props (lines 20–25pt) | 59.6% | **69.8%** | +10.2pp |

Evaluated on 43,781 training rows across two full NBA seasons (49,812 played games, 630 players).

### 15 new features

| Feature | What it captures |
|---|---|
| `l10_ewm`, `l5_ewm` | Exponential decay rolling — 3× more weight on recent games |
| `usage_l10`, `usage_l30` | Usage rate (corr=0.885 with PTS — was completely absent) |
| `fg3a_l10`, `fg3m_l10` | 3-point volume — essential for high-volume shooters |
| `fta_l10`, `ft_rate_l10` | Free throw rate — foul-drawing scorer type |
| `home_l10`, `away_l10`, `home_away_split` | Home/away split rolling averages |
| `b2b_pts_delta` | Quality-controlled B2B fatigue (per-player, not global) |
| `defP_dynamic` | DVP computed live from CSV (not hardcoded season-start table) |
| `usage_segment` | 0=role / 1=rotational / 2=star |
| `line_bucket` | Bookmaker line range (captures sportsbook shading bias) |

### 4 new model files

| File | Purpose |
|---|---|
| `projection_model.pkl` | Global enhanced GBR — 36 features, depth 5 |
| `segment_model.pkl` | Per-usage-tier GBR — separate models for role/rotational/star players |
| `quantile_models.pkl` | P25 + P75 GBR — uncertainty bands for tier gating |
| `calibrator.pkl` | Isotonic regression — converts raw `pred_gap` to calibrated P(hit) |

### V10.0 tier system changes

- **T1_ULTRA** now requires `predGap ≥ 2.0` AND Q25 prediction exceeds the line
- **T1_PREMIUM** now requires `predGap ≥ 1.5`
- **Confidence** is now calibrated (isotonic regression) rather than hand-crafted formula
- **High-line OVER penalty**: props with line ≥ 25 pts get `conf -= 0.03` for OVER calls

---

## Quick start

```bash
# 1. Clone or init the repo
git clone git@github.com:iamwerewolf1007/PropEdgeV10.0.git
cd ~/Documents/GitHub/PropEdgeV10.0

# 2. Install dependencies
pip3 install pandas numpy scikit-learn openpyxl nba_api requests xgboost

# 3. Place source files
cp nba_gamelogs_2024_25.csv source-files/
cp nba_gamelogs_2025_26.csv source-files/
cp h2h_database.csv source-files/
cp "PropEdge_-_Match_and_Player_Prop_lines_.xlsx" source-files/

# 4. Setup Git + launchd agents
python3 run.py setup

# 5. Train all models + build season JSONs (~4 minutes)
python3 run.py generate

# 6. Test a live prediction run
python3 run.py 2

# 7. Open dashboard
open index.html
```

---

## File structure

```
PropEdgeV10.0/
├── config.py               — All paths, constants, team maps, DVP table
├── segment_model.py        — SegmentModel class (role/rotational/star GBR)
├── model_trainer.py        — Full training pipeline — trains all 4 model files
├── rolling_engine.py       — Live rolling stats engine — all 36 prediction features
├── batch_predict.py        — Batch 1/2/3: fetch props + run predictions
├── batch0_grade.py         — Batch 0: grade results + retrain models
├── generate_season_json.py — One-time: build season JSON files from scratch
├── reasoning_engine.py     — Pre/post-match narrative generation
├── h2h_builder.py          — Build h2h_database.csv from game logs
├── synthetic_lines.py      — Generate 2024-25 backtest prop lines
├── audit.py                — Append-only CSV audit trail
├── run.py                  — Master orchestrator + launchd setup
├── index.html              — Mobile-first dark-theme dashboard
├── source-files/           — Game log CSVs, H2H database, prop lines Excel
├── data/                   — today.json, season JSONs, audit_log.csv
├── models/                 — projection_model.pkl, segment_model.pkl,
│                             quantile_models.pkl, calibrator.pkl
├── logs/                   — launchd stdout/stderr per batch
└── daily/                  — YYYY-MM-DD.xlsx per game day
```

---

## Batch schedule (UK time)

| Batch | Time | Script | Purpose |
|---|---|---|---|
| B0 | 06:00 | `batch0_grade.py` | Grade yesterday, append game logs, retrain all 4 models |
| B1 | 08:00 | `batch_predict.py 1` | Early morning odds + predictions |
| B2 | 18:00 | `batch_predict.py 2` | Main prediction run on evening odds |
| B3 | 21:30 | `batch_predict.py 3` | Pre-tip run, dynamic timing |

---

## Non-negotiable coding rules (inherited from V9.2)

1. Never `groupby().apply()` for rolling stats — silent wrong results on large datasets
2. Always `parse_dates=['GAME_DATE']` in every `read_csv` — string sort breaks on season boundaries
3. Never read `L*_*` CSV columns at predict time — they go stale the moment a new row is appended
4. Always `filter_played()` before any rolling window — DNP zeros contaminate L10/L30 averages
5. Rolling windows span both seasons — no season resets, career-chronological
6. H2H lookup dicts must use `.to_dict()` per row — prevents "truth value of Series" crash
7. Graded plays are permanently immutable — never overwrite `result/actualPts/postMatchReason`
8. `clean_json()` before every `json.dump()` — numpy int64/float64/NaN crash the JSON serialiser
9. Deduplicate H2H by `(PLAYER_NAME, OPPONENT) keep='last'` before building lookup dict
10. `pace_rank` computed from real team FGA — never a constant

---

*PropEdge V10.0 · NBA Prop Prediction Engine · March 2026*
*git@github.com:iamwerewolf1007/PropEdgeV10.0.git*

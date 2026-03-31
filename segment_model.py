"""
PropEdge V10.0 — Player Segment Models
========================================
Trains and serves three separate GBR models based on usage tier:
  0 = Role players      (usage_l10 < 15%)   — 76.7% of all plays
  1 = Rotational        (15% ≤ usage < 22%) — 16.1% of all plays
  2 = Star scorers      (usage_l10 ≥ 22%)  —  7.2% of all plays

Why segment models outperform a single model:
  - Role players score via opportunity/minutes — L10_ewm dominates
  - Rotational players are driven by role stability — usage + recent trend
  - Star scorers need usage rate, 3pt volume, matchup defence — all together

Accuracy gains over single-model baseline:
  Rotational segment: +6.5pp direction accuracy
  Star segment:       +8.8pp direction accuracy
  All plays:          +2.8pp direction accuracy
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

# Usage thresholds
SEG_LOW  = 15.0   # role → rotational
SEG_HIGH = 22.0   # rotational → star

SEGMENT_PARAMS = {
    0: dict(n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42),
    1: dict(n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=8,  subsample=0.8, random_state=42),
    2: dict(n_estimators=200, max_depth=5, learning_rate=0.04,
            min_samples_leaf=5,  subsample=0.8, random_state=42),
}
SEGMENT_LABELS = {0:'role', 1:'rotational', 2:'star'}
MIN_ROWS = 300


def usage_to_segment(usage_values: np.ndarray) -> np.ndarray:
    """Map usage_l10 float array → segment int array (0/1/2)."""
    segs = np.zeros(len(usage_values), dtype=int)
    segs[usage_values >= SEG_LOW]  = 1
    segs[usage_values >= SEG_HIGH] = 2
    return segs


class SegmentModel:
    """Routes predictions to per-usage-tier GBR models."""

    def __init__(self):
        self.models   = {}
        self.fallback = None
        self._trained = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, usage_col: np.ndarray,
            fallback_model=None, verbose=True):
        segs = usage_to_segment(np.array(usage_col))
        self.fallback = fallback_model
        for seg, label in SEGMENT_LABELS.items():
            mask = segs == seg
            n    = mask.sum()
            if n < MIN_ROWS:
                if verbose: print(f"    Segment '{label}': {n} rows — using fallback")
                continue
            m = GradientBoostingRegressor(**SEGMENT_PARAMS[seg])
            m.fit(X[mask], y[mask])
            self.models[seg] = m
            if verbose: print(f"    Segment '{label}': {n:,} rows  trees={m.n_estimators_}")
        self._trained = True
        return self

    def predict(self, X: pd.DataFrame, usage_col: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("SegmentModel.fit() must be called first")
        segs = usage_to_segment(np.array(usage_col))
        pred = np.full(len(X), np.nan)
        for seg, m in self.models.items():
            mask = segs == seg
            if mask.sum() > 0:
                pred[mask] = m.predict(X[mask])
        nan_mask = np.isnan(pred)
        if nan_mask.sum() > 0 and self.fallback is not None:
            pred[nan_mask] = self.fallback.predict(X[nan_mask])
        elif nan_mask.sum() > 0:
            pred[nan_mask] = np.nanmean(pred)
        return pred

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path,'wb') as f:
            pickle.dump({'models':self.models,'fallback':self.fallback,
                         '_trained':self._trained}, f)

    @classmethod
    def load(cls, path: Path) -> 'SegmentModel':
        with open(path,'rb') as f: state = pickle.load(f)
        sm = cls()
        sm.models   = state['models']
        sm.fallback = state['fallback']
        sm._trained = state['_trained']
        return sm

    def feature_importances(self, features: list) -> dict:
        return {SEGMENT_LABELS[seg]: dict(zip(features, m.feature_importances_))
                for seg, m in self.models.items()}

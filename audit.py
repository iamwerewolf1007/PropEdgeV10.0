"""
PropEdge V9.2 — Audit Log
Append-only integrity logging. Never deletes records.
"""
import csv
from pathlib import Path
from config import AUDIT_LOG, now_uk

def _ts():
    return now_uk().strftime('%Y-%m-%d %H:%M:%S')

def _append(row: dict):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not AUDIT_LOG.exists()
    with open(AUDIT_LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ts','batch','event','file','rows_before','rows_after','detail'])
        if write_header: w.writeheader()
        w.writerow(row)

def log_event(batch, event, file='', rows_before=None, rows_after=None, detail=''):
    _append({'ts':_ts(),'batch':batch,'event':event,'file':str(file),
             'rows_before':rows_before if rows_before is not None else '',
             'rows_after': rows_after  if rows_after  is not None else '',
             'detail':str(detail)[:500]})

def log_file_state(batch, filepath, label):
    p = Path(filepath)
    rows = ''
    if p.exists():
        try:
            import pandas as pd
            rows = len(pd.read_csv(p, usecols=[0]))
        except: pass
    log_event(batch, f'FILE_STATE_{label}', file=p.name, rows_before=rows)

def log_batch_summary(batch, **kwargs):
    detail = ', '.join(f'{k}={v}' for k,v in kwargs.items())
    log_event(batch, 'BATCH_SUMMARY', detail=detail)

def verify_no_deletion(batch, filepath, rows_before, rows_after, context):
    if rows_after < rows_before:
        msg = f'DELETION DETECTED: {rows_before} → {rows_after} in {context}'
        log_event(batch, 'DELETION_ALERT', file=str(filepath),
                  rows_before=rows_before, rows_after=rows_after, detail=msg)
        print(f"  ⚠ AUDIT: {msg}")
    else:
        log_event(batch, 'INTEGRITY_OK', file=str(filepath),
                  rows_before=rows_before, rows_after=rows_after, detail=context)

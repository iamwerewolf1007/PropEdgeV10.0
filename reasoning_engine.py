"""
PropEdge V10.0 — Reasoning Engine
====================================
Generates pre-match and post-match narrative for each prediction.

V10.0 additions:
  - generate_pre_match_reason: surfaces usage rate, home/away split,
    3-pt volume, quantile bands, B2B quality delta, high-line risk
  - generate_post_match_reason: now accepts optional box_data dict,
    computes loss_type internally, returns (narrative_str, loss_type_str)
"""


def _dvp_desc(rank):
    if rank <= 5:   return "one of the toughest defences in the league"
    if rank <= 10:  return "a top-10 defence"
    if rank <= 15:  return "a middle-of-the-pack defence"
    if rank <= 22:  return "a below-average defence"
    return "one of the weakest defences in the league"

def _pace_desc(rank):
    if rank <= 5:   return "a high-pace offence (lots of possessions)"
    if rank <= 10:  return "an above-average pace"
    if rank <= 20:  return "a mid-range pace"
    return "a slow-paced offence"

def _h2h_avg(h2h_str):
    try: return float(str(h2h_str).split('(')[0].strip())
    except: return None

def _last_name(player):
    parts = str(player).strip().split()
    return parts[-1] if parts else player

def _sign(v):
    return '+' if v > 0 else ''


# ─── PRE-MATCH ────────────────────────────────────────────────────────────────

def generate_pre_match_reason(play):
    """
    5-part pre-match narrative:
    Lead → Matchup context → Signal audit → Model projection → Risk
    """
    direction  = play.get('dir', 'OVER')
    is_over    = 'UNDER' not in direction
    is_lean    = 'LEAN' in direction
    line       = float(play.get('line', 0) or 0)
    name       = _last_name(play.get('player', ''))

    L30   = float(play.get('l30', 0) or 0)
    L10   = float(play.get('l10', L30) or L30)
    L5    = float(play.get('l5',  L30) or L30)
    L3    = float(play.get('l3',  L30) or L30)
    vol   = float(play.get('volume', L30 - line) or 0)
    trend = float(play.get('trend',  L5 - L30)   or 0)
    std10 = float(play.get('std10', 5.0) or 5.0)
    flags        = int(play.get('flags', 0) or 0)
    flag_details = play.get('flagDetails', [])
    conf         = float(play.get('conf', 0.55) or 0.55)
    defP         = int(play.get('defP', 15) or 15)
    pace         = int(play.get('pace', 15) or 15)
    fgTrend      = play.get('fgTrend')
    minTrend     = play.get('minTrend')
    m10          = play.get('minL10')
    m30          = play.get('minL30')
    hr30         = int(play.get('hr30', 50) or 50)
    hr10         = int(play.get('hr10', 50) or 50)
    recent       = play.get('recent', [])
    predPts      = play.get('predPts')
    predGap      = play.get('predGap')
    h2h_str      = play.get('h2h', '')
    h2hG         = int(play.get('h2hG', 0) or 0)
    h2hTsDev     = float(play.get('h2hTsDev', 0) or 0)
    tl           = play.get('tierLabel', 'T3')

    # V10.0
    usage_l10       = float(play.get('usage_l10', 0) or 0)
    fg3a_l10        = float(play.get('fg3a_l10', 0) or 0)
    home_l10        = play.get('home_l10')
    away_l10        = play.get('away_l10')
    home_away_split = float(play.get('home_away_split', 0) or 0)
    is_home         = play.get('isHome', False)
    pred_q25        = play.get('predQ25')
    pred_q75        = play.get('predQ75')
    b2b_pts_delta   = float(play.get('b2b_pts_delta', 0) or 0)
    usage_segment   = int(play.get('usage_segment', 0) or 0)

    h2h_avg = _h2h_avg(h2h_str) if h2hG >= 3 else None
    use_h2h = h2h_avg is not None and h2hG >= 3

    agrees      = [f for f in flag_details if f.get('agrees')]
    disagrees   = [f for f in flag_details if not f.get('agrees')]
    agree_names = [f['name'] for f in agrees]
    dis_names   = [f['name'] for f in disagrees]

    parts = []

    # ── S1: Lead ──────────────────────────────────────────────────────────────
    candidates = []
    if abs(vol) >= 2:
        sup = (is_over and vol > 0) or (not is_over and vol < 0)
        candidates.append((abs(vol) * (1.5 if sup else 1.1), 'vol',
            f"{name}'s L30 of {L30:.1f} pts sits {abs(vol):.1f} "
            f"{'above' if vol > 0 else 'below'} the {line} line"))
    if use_h2h and h2h_avg and abs(h2h_avg - line) >= 2:
        sup = (is_over and h2h_avg > line) or (not is_over and h2h_avg < line)
        gap = h2h_avg - line
        candidates.append((abs(gap) * (1.4 if sup else 1.1), 'h2h',
            f"{name} averages {h2h_avg:.1f} pts in {h2hG} meetings "
            f"({_sign(gap)}{gap:.1f} vs line)"))
    if abs(trend) >= 2:
        sup = (is_over and trend > 0) or (not is_over and trend < 0)
        candidates.append((abs(trend) * (1.3 if sup else 1.0), 'trend',
            f"recent scoring {'trending up' if trend > 0 else 'trending down'} "
            f"{abs(trend):.1f} pts (L5 vs L30)"))
    if std10 <= 5:
        candidates.append((6 - std10, 'consistency',
            f"low scoring variance (\u03c3={std10:.1f}) \u2014 highly predictable output"))
    if usage_l10 >= 22 and usage_segment >= 2 and abs(vol) >= 1.5:
        sup = (is_over and vol > 0) or (not is_over and vol < 0)
        candidates.append((abs(vol) * (1.6 if sup else 1.2), 'usage',
            f"{name} carrying {usage_l10:.0f}% usage rate with L30 of {L30:.1f} "
            f"pts vs {line} line"))
    if home_l10 and away_l10 and abs(home_away_split) >= 4:
        venue   = 'home' if is_home else 'away'
        loc_avg = home_l10 if is_home else away_l10
        sup = (is_over and ((is_home and home_away_split > 0) or
                            (not is_home and home_away_split < 0))) or \
              (not is_over and ((is_home and home_away_split < 0) or
                                (not is_home and home_away_split > 0)))
        candidates.append((abs(home_away_split) * (1.3 if sup else 1.0), 'venue',
            f"{name} averages {loc_avg:.1f} pts in {venue} games vs {line} line "
            f"({_sign(home_away_split)}{abs(home_away_split):.1f} home/away split)"))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        lead = candidates[0][2]
    else:
        lead = f"{name}'s L30 average of {L30:.1f} pts vs the {line} line"

    dir_word = direction.replace('LEAN ', 'lean ')
    parts.append(f"{lead[0].upper()}{lead[1:]} \u2014 "
                 f"{'supports' if flags >= 6 else 'marginally supports'} the {dir_word}.")

    # ── S2: Matchup context ────────────────────────────────────────────────────
    ctx = []
    if use_h2h and h2h_avg and abs(h2h_avg - L30) >= 1.5:
        d = h2h_avg - L30
        ctx.append(f"H2H avg {abs(d):.1f} pts {'above' if d > 0 else 'below'} season L30")
    if abs(h2hTsDev) >= 3 and use_h2h:
        ctx.append(f"TS% shifts {_sign(h2hTsDev)}{h2hTsDev:.1f}% in this matchup")
    if fgTrend and abs(fgTrend) >= 3:
        ctx.append(f"FG% {_sign(fgTrend)}{fgTrend:.1f}% trending (L10 vs L30)")
    if m10 and m30 and abs(m10 - m30) >= 2:
        ctx.append(f"minutes {'up' if m10 > m30 else 'down'} {abs(m10 - m30):.1f} "
                   f"recently (L10={m10:.1f} vs L30={m30:.1f})")
    if fg3a_l10 >= 6:
        ctx.append(f"averaging {fg3a_l10:.1f} three-point attempts (high-variance scorer)")
    if home_l10 and away_l10 and abs(home_away_split) >= 3 \
            and (not candidates or candidates[0][1] != 'venue'):
        venue   = 'home' if is_home else 'away'
        loc_avg = home_l10 if is_home else away_l10
        ctx.append(f"{venue} avg {loc_avg:.1f} pts "
                   f"({_sign(home_away_split)}{abs(home_away_split):.1f} split)")

    matchup = f"Opponent is {_dvp_desc(defP)} at {_pace_desc(pace)}."
    if ctx:
        parts.append(f"{'; '.join(c[0].upper() + c[1:] for c in ctx[:2])}. {matchup}")
    else:
        parts.append(matchup)

    # ── S3: Signal audit ──────────────────────────────────────────────────────
    if flags >= 8 and not disagrees:
        parts.append(
            f"Full consensus: {flags}/10 signals aligned ({', '.join(agree_names[:4])}).")
    elif flags >= 6:
        parts.append(
            f"{flags}/10 signals agree \u2014 {', '.join(agree_names[:4])} support the "
            f"{dir_word.split()[-1].upper()}; {', '.join(dis_names[:3])} "
            f"{'dissent' if len(disagrees) > 1 else 'dissents'}.")
    else:
        parts.append(
            f"Mixed signals: {flags}/10 agree ({', '.join(agree_names[:3])}); "
            f"counter-signals: {', '.join(dis_names[:3])}.")

    # ── S4: Model projection ──────────────────────────────────────────────────
    if predPts is not None:
        gap_dir = 'above' if predPts > line else 'below'
        q_note  = ''
        if pred_q25 and pred_q75:
            if is_over and pred_q25 > line:
                q_note = (f" Even the P25 bound ({pred_q25:.1f}) exceeds the line "
                          f"\u2014 high-conviction band.")
            elif not is_over and pred_q75 < line:
                q_note = (f" P75 bound ({pred_q75:.1f}) still below line "
                          f"\u2014 high-conviction band.")
            else:
                q_note = f" P25\u2013P75 range: {pred_q25:.1f}\u2013{pred_q75:.1f} pts."
        parts.append(
            f"Projection model targets {predPts:.1f} pts "
            f"({predGap:.1f} pts {gap_dir} line; "
            f"{int(conf * 100)}% calibrated confidence [{tl}]).{q_note}")

    # ── S5: Risk ──────────────────────────────────────────────────────────────
    risks = []
    l3_vs_l30 = L3 - L30
    if is_over and l3_vs_l30 < -4:
        risks.append(
            f"L3 has dropped to {L3:.1f} ({abs(l3_vs_l30):.1f} below L30) "
            f"\u2014 slump makes over vulnerable")
    elif not is_over and l3_vs_l30 > 4:
        last  = recent[0] if recent else None
        extra = f" after a {last}-pt outing" if last else ''
        risks.append(
            f"L3 surged to {L3:.1f}{extra} \u2014 momentum could threaten the under")
    if std10 > 7 and not risks:
        risks.append(
            f"high variance (\u03c3={std10:.1f}) makes this difficult to call with confidence")
    if is_over and hr30 < 42 and not risks:
        risks.append(f"only {hr30}% hit rate over L30 \u2014 line may be set fairly")
    elif not is_over and hr30 > 58 and not risks:
        risks.append(f"{hr30}% over-rate on L30 could threaten the under")
    if m10 and m30 and m10 - m30 < -3 and not risks:
        risks.append(
            f"minutes trending down ({m10:.1f} L10 vs {m30:.1f} L30) "
            f"\u2014 role reduction risk")
    if play.get('is_b2b') and abs(b2b_pts_delta) >= 2 and not risks:
        d_str = 'drops' if b2b_pts_delta < 0 else 'rises'
        risks.append(
            f"B2B game \u2014 {name} historically {d_str} "
            f"{abs(b2b_pts_delta):.1f} pts on zero rest")
    if is_over and line >= 25 and not risks:
        risks.append(
            f"props at {line}+ pts are systematically shaded by sportsbooks "
            f"\u2014 lower base hit rate for OVER")

    if risks:
        parts.append(f"Risk: {risks[0]}.")

    result = ' '.join(p for p in parts if p.strip())
    if is_lean:
        result = '[Low conviction \u2014 lean only] ' + result
    return result


# ─── POST-MATCH ───────────────────────────────────────────────────────────────

def generate_post_match_reason(play, box_data=None):
    """
    7-part post-match narrative.
    Returns (narrative_str, loss_type_str).

    box_data (optional dict) may contain:
        actual_min     — minutes played
        actual_fg_pct  — FG% as percentage (0–100)

    These are merged into the play dict so the function works whether
    called from batch0 (with a full play dict) or from generate_season_json
    (with separate box_data).
    """
    # Merge box_data into a working copy so we don't need to pass it around
    p = dict(play)
    if box_data:
        # Normalise key names — box_data uses snake_case, play uses camelCase
        if 'actual_min' in box_data and 'actualMin' not in p:
            p['actualMin'] = box_data['actual_min']
        if 'actual_fg_pct' in box_data and 'actualFgPct' not in p:
            p['actualFgPct'] = box_data['actual_fg_pct']
        # Also accept camelCase directly
        for k in ('actualMin', 'actualFgPct'):
            if k in box_data and k not in p:
                p[k] = box_data[k]

    result_val = p.get('result', '')
    actual     = p.get('actualPts')
    line       = float(p.get('line', 0) or 0)
    name       = _last_name(p.get('player', ''))
    direction  = p.get('dir', 'OVER')
    is_over    = 'UNDER' not in direction

    if actual is None:
        return '', None

    delta    = actual - line
    hit      = (actual > line and is_over) or (actual <= line and not is_over)

    predPts    = p.get('predPts')
    minL10     = p.get('minL10')
    actual_min = p.get('actualMin')
    fgL10      = p.get('fgL10')
    actual_fg  = p.get('actualFgPct')
    pred_q25   = p.get('predQ25')
    pred_q75   = p.get('predQ75')
    flags      = p.get('flagDetails', [])
    agrees     = [f['name'] for f in flags if f.get('agrees')]
    disagrees  = [f['name'] for f in flags if not f.get('agrees')]

    parts = []

    # S1: Outcome
    outcome  = 'HIT' if hit else 'MISS'
    delta_str = f"{_sign(delta)}{delta:.1f}"
    parts.append(
        f"{outcome} \u2014 {name} scored {actual} pts ({delta_str} vs {line} line). "
        f"Direction was {'correct' if hit else 'incorrect'}.")

    # S2: Minutes
    if actual_min is not None and minL10 is not None:
        min_diff = actual_min - minL10
        if abs(min_diff) >= 3:
            flag = 'above' if min_diff > 0 else 'below'
            parts.append(
                f"Minutes: played {actual_min:.0f} mins \u2014 {abs(min_diff):.1f} {flag} "
                f"expected L10 average of {minL10:.1f}.")
        else:
            parts.append(
                f"Minutes: {actual_min:.0f} mins (in line with L10 average of {minL10:.1f}).")

    # S3: Efficiency
    if actual_fg is not None and fgL10 is not None:
        fg_diff = actual_fg - fgL10
        if abs(fg_diff) >= 5:
            tag = 'hot shooting' if fg_diff > 0 else 'cold shooting'
            parts.append(
                f"Efficiency: {tag} \u2014 {actual_fg:.1f}% FG "
                f"({_sign(fg_diff)}{fg_diff:.1f}% vs L10 {fgL10:.1f}%).")
        else:
            parts.append(
                f"Efficiency: FG% ({actual_fg:.1f}%) was consistent with L10 "
                f"baseline ({fgL10:.1f}%).")

    # S4: Signal audit
    if hit:
        if agrees:
            parts.append(f"Signals validated: {', '.join(agrees[:4])} all called it correctly.")
        else:
            parts.append("Result correct despite mixed signal picture.")
    else:
        if disagrees:
            parts.append(
                f"Counter-signals proved correct: {', '.join(disagrees[:3])} were the tell.")
        else:
            parts.append(
                "Signals were broadly aligned but the outcome didn't follow \u2014 "
                "variance event.")

    # S5: Model accuracy
    if predPts is not None:
        err        = abs(predPts - actual)
        dir_right  = (predPts > line) == (actual > line)
        q_note     = ''
        if pred_q25 is not None and pred_q75 is not None:
            in_band = pred_q25 <= actual <= pred_q75
            q_note  = (f" Actual {'was' if in_band else 'fell outside'} the "
                       f"P25\u2013P75 band ({pred_q25:.1f}\u2013{pred_q75:.1f}).")
        parts.append(
            f"Model: projected {predPts:.1f} pts, actual {actual} \u2014 "
            f"error {err:.1f} pts, direction "
            f"{'correct' if dir_right else 'wrong'}.{q_note}")

    # S6: Loss classification (computed here, not read from play)
    loss_type = None
    if result_val == 'LOSS':
        if abs(delta) <= 2:
            loss_type = 'CLOSE_CALL'
            parts.append(
                "Loss type: CLOSE CALL \u2014 within 2 pts of line, "
                "no model adjustment needed.")
        elif actual_min is not None and minL10 is not None and actual_min < minL10 - 4:
            loss_type = 'MINUTES_SHORTFALL'
            parts.append(
                "Loss type: MINUTES SHORTFALL \u2014 playing time was the issue, "
                "not the scoring model.")
        elif actual_fg is not None and fgL10 is not None and abs(actual_fg - fgL10) >= 5:
            loss_type = 'SHOOTING_VARIANCE'
            parts.append(
                "Loss type: SHOOTING VARIANCE \u2014 opportunity existed but "
                "efficiency diverged from expected.")
        elif abs(delta) > 8:
            loss_type = 'BLOWOUT_EFFECT'
            parts.append(
                "Loss type: BLOWOUT EFFECT \u2014 game script invalidated the prop "
                "(starters likely pulled).")
        elif len(agrees) >= 7:
            loss_type = 'MODEL_FAILURE'
            parts.append(
                "Loss type: MODEL FAILURE \u2014 signals were systematically "
                "misleading for this matchup.")
        else:
            loss_type = 'MODEL_FAILURE'
    elif result_val == 'WIN':
        loss_type = 'MODEL_CORRECT'

    # S7: Learning note
    err_val = abs(predPts - actual) if predPts is not None else None
    if hit and err_val is not None and err_val < 2:
        parts.append("Learning: clean prediction \u2014 role confirmation for future props.")
    elif hit and err_val is not None and err_val >= 5:
        parts.append(
            "Learning: correct direction but model underestimated by a wide margin "
            "\u2014 usage or pace may have been a factor.")
    elif not hit and abs(delta) <= 2:
        parts.append("Learning: close call, no systematic issue flagged.")
    elif not hit and actual_min is not None and minL10 is not None and actual_min < minL10 - 4:
        parts.append(
            "Learning: minutes risk materialised \u2014 flag as a concern for "
            "future props when playing time is uncertain.")

    narrative = ' '.join(part for part in parts if part.strip())
    return narrative, loss_type

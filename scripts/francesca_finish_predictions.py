"""Francesca's predicted vs. actual finish on all her Regional+ (RYC/RJCC/ROC/SYC/NAC)
events, leakage-free. For each event: fit on bouts BEFORE that event's month, rank the
field by predicted strength g, and report Francesca's predicted rank vs her actual place.
Current fixed-effect model vs. hierarchical (lam_s=3, lam_cm=3)."""
from __future__ import annotations
import re, sqlite3
import numpy as np
from analytics.features import load_dataset
from analytics import model as M, forecast as FC
from analytics.validate import _subset

F = 100835605
DB = "fencing_full.db"
SERIOUS = re.compile(r"\bNAC\b|North American Cup|Summer Nationals|\bSYC\b|Super Youth|"
                     r"\bROC\b|\bRYC\b|\bRJCC?\b|Regional", re.I)


def strengths(fm, ids, club_a, by, ey):
    rs = FC.recent_skill(fm); hier = fm.config.hier_club; C = len(fm.cm) if hier else len(fm.c)
    out = {}
    for x in ids:
        ci = club_a.get(x, -1)
        base = (rs[x] if x in rs else (fm.cm[ci] if 0 <= ci < C else 0.0)) if hier \
            else rs.get(x, 0.0) + (fm.c[ci] if 0 <= ci < C else 0.0)
        out[x] = base + fm.beta_age * ((ey - by[x]) if by.get(x) else 0.0)
    return out


def run():
    ds = load_dataset(DB, core_only=True, since="2022-09", birth_min=2013, birth_max=2018)
    month_of = {mon: i for i, mon in enumerate(ds.months)}
    club_a = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_a))
    club_a.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_b))
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    rows = conn.execute(
        """SELECT fer.event_id, e.event_date, e.name, fer.placement, fer.field_size
           FROM fencer_event_results fer JOIN events e ON e.id=fer.event_id
           WHERE fer.fencer_id=? AND fer.placement IS NOT NULL AND e.event_date IS NOT NULL
           ORDER BY e.event_date""", (F,)).fetchall()
    cfgs = {"cur": M.ModelConfig(rank=0, lam_s=0.05, lam_time=400),
            "hier": M.ModelConfig(rank=0, hier_club=True, lam_s=3.0, lam_cm=3.0, lam_time=400)}
    fits: dict = {}
    print(f"{'date':11s} {'event':30s} {'fld':>4} {'actual':>6} {'pred:cur':>8} {'pred:hier':>9}")
    err = {"cur": [], "hier": []}
    for eid, edate, name, place, fsize in rows:
        if not name or not SERIOUS.search(name):
            continue
        m = month_of.get(edate[:7])
        if not m:
            continue
        field = conn.execute("SELECT fencer_id, placement FROM fencer_event_results "
                             "WHERE event_id=? AND placement IS NOT NULL", (eid,)).fetchall()
        ids = [r[0] for r in field]
        if F not in ids or len(ids) < 5:
            continue
        preds = {}
        for k, cfg in cfgs.items():
            key = (k, m)
            if key not in fits:
                fits[key] = M.fit(_subset(ds, ds.bouts["month_idx"].to_numpy() < m), cfg)
            g = strengths(fits[key], ids, club_a, by, int(edate[:4]) + 0.5)
            preds[k] = 1 + sum(1 for i in ids if g[i] > g[F])   # rank among field (1=best)
            err[k].append(abs(preds[k] - place))
        print(f"{edate:11s} {name[:30]:30s} {len(ids):>4} {place:>6} {preds['cur']:>8} {preds['hier']:>9}")
    n = len(err["cur"])
    print(f"\nRegional+ events: {n}")
    for k in ("cur", "hier"):
        a = np.array(err[k])
        print(f"  {k:5s}  mean |pred-actual| finish error = {a.mean():.2f}   median = {np.median(a):.1f}")


if __name__ == "__main__":
    run()

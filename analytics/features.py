"""Build the calibration-ready per-bout design matrix for the rating model.

Maps the raw `bouts`/`events`/`fencers` tables onto the quantities the model in
`analytics.tex` consumes:

    Z_b = (score_i - score_j) / target          (oriented i=fencer_a, j=fencer_b)
    target = 5 (Pool); DE = 10 if age_group in {Y8,Y10} else 15
    month index m   from events.event_date
    1{DE}           bout_type != 'Pool'
    age_i - age_j   (event_year + (month-1)/12) - birth_year
    club_i, club_j  stationary fencers.club  (sigma split: pool vs de)

Coverage note: birth_year exists only for *scraped* fencers, so the age covariate is
complete only on bouts between two scraped fencers. `core_only=True` restricts to that
well-observed subgraph (recommended for fitting); the un-scraped opponents are mostly
single-bout leaves with no estimable parameters anyway.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

YOUTH_DE_TO_10 = {"Y8", "Y10"}


@dataclass
class Dataset:
    bouts: pd.DataFrame            # one row per bout with model features
    months: list[str]             # ordered 'YYYY-MM'; index = month_idx
    fencer_ids: list[int]         # ordered; index = fencer_idx
    club_names: list[str]         # ordered; index = club_idx (unknown club -> -1, not listed)
    popular_clubs: list[str] = field(default_factory=list)  # top-N by bouts; club_pop id = index, 'other' = N, unknown = -1

    def summary(self) -> dict:
        b = self.bouts
        return {
            "bouts": len(b),
            "fencers": len(self.fencer_ids),
            "months": len(self.months),
            "month_span": (self.months[0], self.months[-1]) if self.months else None,
            "clubs": len(self.club_names),
            "pool_bouts": int((~b["de"]).sum()),
            "de_bouts": int(b["de"].sum()),
            "with_age_diff": int(b["age_diff"].notna().sum()),
            "age_diff_pct": round(100 * b["age_diff"].notna().mean(), 1) if len(b) else 0.0,
            "both_clubs_known": int((b["club_a"].ge(0) & b["club_b"].ge(0)).sum()),
        }


def _target(bout_type: str, age_group: Optional[str]) -> int:
    if bout_type == "Pool":
        return 5
    return 10 if age_group in YOUTH_DE_TO_10 else 15


def load_dataset(
    db_path: str,
    *,
    weapon: Optional[str] = "epee",
    core_only: bool = True,
    since: Optional[str] = None,
    min_bouts: int = 0,
    popular_clubs: int = 40,
    birth_min: Optional[int] = None,
    birth_max: Optional[int] = None,
) -> Dataset:
    """Load bouts into a model-ready design frame.

    weapon: keep only bouts in events of this weapon ('epee'|'foil'|'saber'). The bouts
        table mixes all three; ratings are weapon-specific, so this MUST be set. Defaults
        to 'epee' (the focal fencer's weapon). None = no filter (all weapons — usually wrong).
    core_only: keep only bouts where BOTH fencers are fully scraped ('done') — the
        subgraph with complete features (birth year + full histories).
    since: keep only bouts on/after this 'YYYY-MM' (or 'YYYY-MM-DD') — trims the near-
        empty early months that predate the relevant competitive era. None = all.
    min_bouts: after selection, iteratively drop fencers with fewer than this many
        bouts (0 = no filter).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = "WHERE e.event_date IS NOT NULL"
    params: list = []
    if weapon is not None:
        where += " AND e.weapon = ?"
        params.append(weapon)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT
                b.event_id, b.fencer_a_id, b.fencer_b_id, b.bout_type,
                b.fencer_a_score, b.fencer_b_score, b.winner_id,
                e.event_date, e.age_group,
                fa.scrape_status AS a_status, fb.scrape_status AS b_status,
                fa.birth_year AS a_birth, fb.birth_year AS b_birth,
                fa.club AS a_club, fb.club AS b_club
            FROM bouts b
            JOIN events e   ON e.id = b.event_id
            JOIN fencers fa ON fa.id = b.fencer_a_id
            JOIN fencers fb ON fb.id = b.fencer_b_id
            {where}
            """,
            conn, params=params,
        )
    finally:
        conn.close()

    if core_only:
        df = df[(df["a_status"] == "done") & (df["b_status"] == "done")].copy()

    # Birth-year band: restrict the model population to the relevant age cohort (both
    # fencers in-band). The focal fences youth events, so the wider graph of teens/adults
    # 2 hops away is irrelevant; this is the real "who matters" filter.
    if birth_min is not None:
        df = df[(df["a_birth"] >= birth_min) & (df["b_birth"] >= birth_min)].copy()
    if birth_max is not None:
        df = df[(df["a_birth"] <= birth_max) & (df["b_birth"] <= birth_max)].copy()

    # Month index
    df["month"] = df["event_date"].str.slice(0, 7)
    df = df[df["month"].str.match(r"^\d{4}-\d{2}$", na=False)].copy()

    if since is not None:
        df = df[df["event_date"] >= since].copy()

    # Optional min-bouts pruning (iterative: dropping a fencer can drop others below threshold)
    if min_bouts > 0:
        while True:
            counts = pd.concat([df["fencer_a_id"], df["fencer_b_id"]]).value_counts()
            keep = set(counts[counts >= min_bouts].index)
            mask = df["fencer_a_id"].isin(keep) & df["fencer_b_id"].isin(keep)
            if mask.all():
                break
            df = df[mask].copy()
            if df.empty:
                break

    # Target + normalized score differential Z (oriented a - b)
    df["target"] = [_target(bt, ag) for bt, ag in zip(df["bout_type"], df["age_group"])]
    df["z"] = (df["fencer_a_score"] - df["fencer_b_score"]) / df["target"]
    df["de"] = df["bout_type"] != "Pool"
    df["sigma_type"] = np.where(df["de"], "de", "pool")

    # Age (fractional, born Jan 1 approximation) and age difference a - b
    ev_year = df["event_date"].str.slice(0, 4).astype(int)
    ev_month = df["event_date"].str.slice(5, 7).astype(int)
    age_frac = ev_year + (ev_month - 1) / 12.0
    df["age_a"] = age_frac - df["a_birth"]
    df["age_b"] = age_frac - df["b_birth"]
    df["age_diff"] = df["age_a"] - df["age_b"]   # NaN if either birth year missing

    # Index maps
    months = sorted(df["month"].unique().tolist())
    month_idx = {m: i for i, m in enumerate(months)}
    df["month_idx"] = df["month"].map(month_idx)

    fencer_ids = sorted(set(df["fencer_a_id"]) | set(df["fencer_b_id"]))
    fencer_idx = {f: i for i, f in enumerate(fencer_ids)}
    df["a_idx"] = df["fencer_a_id"].map(fencer_idx)
    df["b_idx"] = df["fencer_b_id"].map(fencer_idx)

    # Stationary club ids; unknown/blank -> -1 (caller decides whether to model it)
    def _norm_club(c):
        return c if (isinstance(c, str) and c.strip()) else None
    fencer_club = {}
    for fid, ca, cb in zip(df["fencer_a_id"], df["a_club"], df["b_club"]):
        fencer_club.setdefault(fid, _norm_club(ca))
    for fid, cb in zip(df["fencer_b_id"], df["b_club"]):
        fencer_club.setdefault(fid, _norm_club(cb))
    club_names = sorted({c for c in fencer_club.values() if c is not None})
    club_idx = {c: i for i, c in enumerate(club_names)}
    df["club_a"] = df["fencer_a_id"].map(lambda f: club_idx.get(fencer_club.get(f), -1))
    df["club_b"] = df["fencer_b_id"].map(lambda f: club_idx.get(fencer_club.get(f), -1))

    # Popular clubs (top-N by bout appearances) for the club-pair interaction term;
    # everything else collapses to 'other' (id N); unknown club -> -1.
    pop_counts = Counter()
    for fa, fb in zip(df["fencer_a_id"], df["fencer_b_id"]):
        ca, cb = fencer_club.get(fa), fencer_club.get(fb)
        if ca:
            pop_counts[ca] += 1
        if cb:
            pop_counts[cb] += 1
    pop_list = [c for c, _ in pop_counts.most_common(popular_clubs)]
    pop_idx = {c: i for i, c in enumerate(pop_list)}
    other_id = len(pop_list)
    def _pop(f):
        c = fencer_club.get(f)
        if not c:
            return -1
        return pop_idx.get(c, other_id)
    df["club_pop_a"] = df["fencer_a_id"].map(_pop)
    df["club_pop_b"] = df["fencer_b_id"].map(_pop)

    cols = [
        "event_id", "month", "month_idx", "fencer_a_id", "fencer_b_id",
        "a_idx", "b_idx", "z", "target", "de", "sigma_type",
        "age_a", "age_b", "age_diff", "club_a", "club_b", "club_pop_a", "club_pop_b",
        "winner_id",
    ]
    return Dataset(
        bouts=df[cols].reset_index(drop=True),
        months=months,
        fencer_ids=fencer_ids,
        club_names=club_names,
        popular_clubs=pop_list,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fencing.db"
    for core in (True, False):
        ds = load_dataset(path, core_only=core)
        print(f"\n=== core_only={core} ===")
        for k, v in ds.summary().items():
            print(f"  {k}: {v}")

"""Read-only data layer for the Fencing Explorer.

Loads `fencing.db` into pandas once (Streamlit-cached) and exposes
fencer-centric helpers. No writes, ever.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "fencing.db"
FRANCESCA_ID = 100835605
POOL = "Pool"

# Round ordering for tidy sorting/display
BOUT_TYPE_ORDER = ["Pool", "T256", "T128", "T64", "T32", "T16", "T8", "T4", "T2"]


# --------------------------------------------------------------------------
# Name prettifying — fencingtracker stores opponents as "SURNAME Given".
# --------------------------------------------------------------------------

def _title_token(tok: str) -> str:
    return "-".join(p.capitalize() for p in tok.split("-"))


def pretty_name(raw: str | None) -> str:
    """Render "SURNAME Given" as "Given Surname"; pass through anything else."""
    if not raw or not raw.strip():
        return "Unknown"
    raw = raw.strip()
    surname, given = [], []
    in_surname = True
    for tok in raw.split():
        letters = [c for c in tok if c.isalpha()]
        is_upper = bool(letters) and all(c.isupper() for c in letters)
        if in_surname and is_upper:
            surname.append(tok)
        else:
            in_surname = False
            given.append(tok)
    if not surname or not given:
        return raw.title() if raw.isupper() else raw
    return " ".join(given) + " " + " ".join(_title_token(t) for t in surname)


# --------------------------------------------------------------------------
# Raw tables
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading database…")
def load_tables() -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run the scraper first.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return {
            "fencers": pd.read_sql_query("SELECT * FROM fencers", conn),
            "events": pd.read_sql_query("SELECT * FROM events", conn),
            "bouts": pd.read_sql_query("SELECT * FROM bouts", conn),
            "results": pd.read_sql_query("SELECT * FROM fencer_event_results", conn),
        }
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def fencer_table() -> pd.DataFrame:
    f = load_tables()["fencers"].copy()
    f["display"] = f["name"].map(pretty_name)
    return f


@st.cache_resource(show_spinner=False)
def name_map() -> dict[int, str]:
    # cache_resource, not cache_data: returns the *same* dict object every call,
    # with no pickle round-trip. display_name() is called thousands of times per
    # render (selectbox format_func), so this must be a plain lookup. Read-only.
    f = fencer_table()
    return dict(zip(f["id"], f["display"]))


def display_name(fid: int) -> str:
    return name_map().get(fid, f"#{fid}")


@st.cache_data(show_spinner=False)
def scraped_fencers() -> pd.DataFrame:
    """The 1,002 fencers with fully scraped histories — valid focal fencers."""
    f = fencer_table()
    return f[f["scrape_status"] == "done"].sort_values("display").reset_index(drop=True)


# --------------------------------------------------------------------------
# Season
# --------------------------------------------------------------------------

def _season(ts) -> str | None:
    """US fencing season runs Aug–Jul. 'May 2026' -> '2025-26'."""
    if pd.isna(ts):
        return None
    y = ts.year
    return f"{y}-{(y + 1) % 100:02d}" if ts.month >= 8 else f"{y - 1}-{y % 100:02d}"


# --------------------------------------------------------------------------
# Perspective bouts — every bout doubled, once per fencer's point of view.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Building bout index…")
def perspective_bouts() -> pd.DataFrame:
    t = load_tables()
    bouts = t["bouts"]

    side_a = bouts.rename(columns={
        "fencer_a_id": "fencer_id", "fencer_b_id": "opponent_id",
        "fencer_a_score": "score_for", "fencer_b_score": "score_against",
    })
    side_b = bouts.rename(columns={
        "fencer_b_id": "fencer_id", "fencer_a_id": "opponent_id",
        "fencer_b_score": "score_for", "fencer_a_score": "score_against",
    })
    cols = ["event_id", "fencer_id", "opponent_id", "bout_type", "bout_seq",
            "score_for", "score_against", "winner_id"]
    persp = pd.concat([side_a[cols], side_b[cols]], ignore_index=True)

    persp["won"] = persp["winner_id"] == persp["fencer_id"]
    persp["result"] = persp["won"].map({True: "Win", False: "Loss"})
    persp["margin"] = persp["score_for"] - persp["score_against"]
    persp["is_de"] = persp["bout_type"] != POOL
    persp["phase"] = persp["is_de"].map({True: "Direct Elimination", False: "Pool"})

    events = t["events"].rename(columns={"id": "event_id", "name": "event_name"})
    persp = persp.merge(
        events[["event_id", "event_name", "classification", "weapon",
                "gender", "age_group", "rating_level", "event_date"]],
        on="event_id", how="left",
    )
    persp["event_date"] = pd.to_datetime(persp["event_date"], errors="coerce")
    persp["season"] = persp["event_date"].map(_season)
    persp["opponent_name"] = persp["opponent_id"].map(name_map()).fillna("Unknown")
    return persp


# --------------------------------------------------------------------------
# Fencer-centric helpers (cheap pandas slices — no caching needed)
# --------------------------------------------------------------------------

def focal_bouts(focal_id: int) -> pd.DataFrame:
    p = perspective_bouts()
    return p[p["fencer_id"] == focal_id].copy()


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    wins = int(df["won"].sum())
    return {
        "bouts": n,
        "wins": wins,
        "losses": n - wins,
        "win_pct": (wins / n * 100.0) if n else 0.0,
        "opponents": int(df["opponent_id"].nunique()),
        "events": int(df["event_id"].nunique()),
        "touches_for": int(df["score_for"].sum()),
        "touches_against": int(df["score_against"].sum()),
    }


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season summary (fencingtracker-style): events, bouts, W/L, win %, indicator."""
    d = df[df["season"].notna()]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("season").agg(
        events=("event_id", "nunique"), bouts=("won", "size"), wins=("won", "sum"),
        tf=("score_for", "sum"), ta=("score_against", "sum"),
    ).reset_index()
    g["wins"] = g["wins"].astype(int)
    g["losses"] = g["bouts"] - g["wins"]
    g["win_pct"] = g["wins"] / g["bouts"] * 100.0
    g["indicator"] = (g["tf"] - g["ta"]).astype(int)
    return g.sort_values("season", ascending=False).reset_index(drop=True)


def opponent_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per opponent in `df`, with the focal fencer's record vs them."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("opponent_id").agg(
        bouts=("won", "size"),
        wins=("won", "sum"),
        last_met=("event_date", "max"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    g["wins"] = g["wins"].astype(int)
    g["losses"] = g["bouts"] - g["wins"]
    g["win_pct"] = g["wins"] / g["bouts"] * 100.0
    g["opponent"] = g["opponent_id"].map(name_map()).fillna("Unknown")
    return g.sort_values(["bouts", "win_pct"], ascending=[False, False]).reset_index(drop=True)


def event_history(focal_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """One row per event in `df`, with seed/placement from fencer_event_results."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("event_id").agg(
        bouts=("won", "size"),
        wins=("won", "sum"),
        event_name=("event_name", "first"),
        classification=("classification", "first"),
        weapon=("weapon", "first"),
        event_date=("event_date", "first"),
    ).reset_index()
    g["wins"] = g["wins"].astype(int)
    g["losses"] = g["bouts"] - g["wins"]

    results = load_tables()["results"]
    r = results[results["fencer_id"] == focal_id][
        ["event_id", "seed", "placement", "field_size"]
    ]
    g = g.merge(r, on="event_id", how="left")
    return g.sort_values("event_date", ascending=False, na_position="last").reset_index(drop=True)


def common_opponents(focal_id: int, opp_id: int) -> pd.DataFrame:
    """Opponents that both fencers have faced — for indirect comparison.

    Uses the full (unfiltered) bout graph.
    """
    p = perspective_bouts()
    focal = p[p["fencer_id"] == focal_id]
    other = p[p["fencer_id"] == opp_id]
    shared = (set(focal["opponent_id"]) & set(other["opponent_id"])) - {focal_id, opp_id}
    if not shared:
        return pd.DataFrame()

    rows = []
    for sid in shared:
        fb = focal[focal["opponent_id"] == sid]
        ob = other[other["opponent_id"] == sid]
        f_w = int(fb["won"].sum())
        o_w = int(ob["won"].sum())
        rows.append({
            "opponent": display_name(sid),
            "focal_bouts": len(fb),
            "focal_wins": f_w,
            "focal_losses": len(fb) - f_w,
            "focal_win_pct": f_w / len(fb) * 100.0,
            "other_bouts": len(ob),
            "other_wins": o_w,
            "other_losses": len(ob) - o_w,
            "other_win_pct": o_w / len(ob) * 100.0,
        })
    return pd.DataFrame(rows).sort_values("focal_bouts", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def opponent_options(focal_id: int) -> tuple[list[int], dict[int, int]]:
    """Opponent picker options for `focal_id`.

    Returns (ordered_ids, faced_counts):
      - ordered_ids: opponents the focal has faced (by bout count desc) first,
        then every other fencer with bouts, alphabetically.
      - faced_counts: id -> number of bouts vs focal (for labels).
    """
    p = perspective_bouts()
    faced = p[p["fencer_id"] == focal_id]["opponent_id"].value_counts()
    faced_counts = {int(k): int(v) for k, v in faced.items()}
    faced_ids = list(faced_counts.keys())

    nm = name_map()
    everyone = set(p["fencer_id"].unique()) | set(p["opponent_id"].unique())
    others = sorted(
        everyone - set(faced_ids) - {focal_id},
        key=lambda i: nm.get(i, "").lower(),
    )
    return faced_ids + others, faced_counts


# --------------------------------------------------------------------------
# Upcoming events (preregistration) + scouting
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def upcoming_tables() -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        ev = pd.read_sql_query("SELECT * FROM upcoming_events", conn)
        reg = pd.read_sql_query("SELECT * FROM upcoming_event_registrants", conn)
    except Exception:
        ev, reg = pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()
    return {"events": ev, "registrants": reg}


def focal_upcoming_events(focal_id: int) -> pd.DataFrame:
    """Upcoming events the focal fencer is registered for (most recent date first)."""
    t = upcoming_tables()
    ev, reg = t["events"], t["registrants"]
    if ev.empty or reg.empty:
        return pd.DataFrame()
    ids = reg.loc[reg["fencer_id"] == focal_id, "event_id"].unique()
    out = ev[ev["event_id"].isin(ids)].copy()
    return out.sort_values("event_date").reset_index(drop=True)


def upcoming_field_ids(focal_id: int) -> set[int]:
    """All fencers registered alongside the focal in their upcoming events."""
    t = upcoming_tables()
    ev = focal_upcoming_events(focal_id)
    reg = t["registrants"]
    if ev.empty or reg.empty:
        return set()
    fid = set(reg.loc[reg["event_id"].isin(ev["event_id"]), "fencer_id"])
    return fid - {focal_id}


def fencer_record(fid: int) -> pd.DataFrame:
    """A fencer's own bouts (their point of view) — backing data for scouting.

    Complete for scraped fencers; partial (vs scraped opponents only) otherwise."""
    p = perspective_bouts()
    return p[p["fencer_id"] == fid].copy()


def longest_streak(df_sorted: pd.DataFrame, win: bool = True) -> int:
    best = cur = 0
    for w in df_sorted["won"]:
        if bool(w) is win:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

"""Fencing Explorer — a fencer-centric Streamlit app over fencing.db.

Run:  streamlit run explorer/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `explorer` importable whether launched via `streamlit run` or AppTest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from explorer import analytics_bridge as ab, charts, data

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="Fencing Explorer", layout="wide")

# Win/loss row tints for the bout-log table (consumed by a pandas Styler,
# which Streamlit renders natively inside st.dataframe).
WIN_BG, LOSS_BG = "#dcfce7", "#fee2e2"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, weapons, ages, bout_mode, date_range) -> pd.DataFrame:
    out = df
    if weapons:
        out = out[out["weapon"].isin(weapons)]
    if ages:
        out = out[out["age_group"].isin(ages)]
    if bout_mode == "Pool only":
        out = out[~out["is_de"]]
    elif bout_mode == "DE only":
        out = out[out["is_de"]]
    if date_range and len(date_range) == 2:
        lo = pd.Timestamp(date_range[0])
        hi = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[out["event_date"].isna() | out["event_date"].between(lo, hi)]
    return out


def form_chips(df: pd.DataFrame, n: int = 15) -> str:
    """Recent W/L as native Streamlit colored-background tokens (no raw HTML)."""
    recent = df.sort_values("event_date", na_position="first").tail(n)
    if recent.empty:
        return "_No bouts._"
    return " ".join(
        ":green-background[ W ]" if won else ":red-background[ L ]"
        for won in recent["won"]
    )


def show_chart(chart) -> None:
    """Render an Altair chart, or a placeholder when there is no data.

    Must stay a plain function call at every call site — never a bare
    `... if ... else ...` expression. Streamlit's "magic" auto-displays bare
    expression statements, which would dump the returned DeltaGenerator object
    onto the page as text.
    """
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("—")


def _bout_log_table(df: pd.DataFrame) -> pd.DataFrame:
    log = df.sort_values("event_date", ascending=False, na_position="last")
    return pd.DataFrame({
        "Date": log["event_date"].dt.strftime("%Y-%m-%d").fillna("—"),
        "Event": log["event_name"].fillna("—"),
        "Level": log["classification"].fillna("—"),
        "Round": log["bout_type"],
        "Score": (log["score_for"].astype(int).astype(str) + " – "
                  + log["score_against"].astype(int).astype(str)),
        "Result": log["result"],
    })


def _style_by_result(row):
    bg = WIN_BG if row["Result"] == "Win" else LOSS_BG
    return [f"background-color: {bg}"] * len(row)


# --------------------------------------------------------------------------
# Tab renderers
# --------------------------------------------------------------------------

def render_overview(focal_id: int, focal_name: str, fdf: pd.DataFrame) -> None:
    st.subheader(f"{focal_name} — Overview")

    # Model-estimated standing among her true peer cohort (the lowest level she enters).
    byr = ab.birth_year_rank(focal_id)
    cohort = ab.eligibility_cohort_rank(focal_id)
    if byr:
        c = st.columns(3)
        c[0].metric("Estimated ability", f"{byr['skill']:+.2f}")
        c[1].metric(f"Rank among born-{byr['year']}", f"#{byr['rank']} / {byr['n']}")
        c[2].metric("Percentile (birth year)", f"{byr['pct']:.0f}%")
        gl = {"W": "women", "M": "men"}.get(byr.get("gender"), "")
        cap = (f"Club-adjusted ability (skill + club effect, age-agnostic) vs. rated "
               f"**{gl} épée fencers born {byr['year']}** with ≥{ab.MIN_COHORT_BOUTS} bouts "
               f"(her true same-age peers).")
        if cohort:
            cap += (f"  For reference, among everyone eligible for **{cohort['level']}** "
                    f"(born {cohort['floor']}+): #{cohort['rank']} / {cohort['n']} "
                    f"({cohort['pct']:.0f}th pct).")
        st.caption(cap)

    st.markdown("##### Estimated ability over time")
    tchart = charts.skill_trajectory(ab.skill_trajectory(focal_id))
    show_chart(tchart)

    st.divider()
    st.caption("The rest of this page is factual and reflects the sidebar filters.")
    if fdf.empty:
        st.info("No bouts match the current filters.")
        return

    s = data.summarize(fdf)
    cols = st.columns(6)
    cols[0].metric("Bouts", s["bouts"])
    cols[1].metric("Wins", s["wins"])
    cols[2].metric("Losses", s["losses"])
    cols[3].metric("Win rate", f"{s['win_pct']:.0f}%")
    cols[4].metric("Opponents", s["opponents"])
    cols[5].metric("Events", s["events"])

    st.markdown("##### Results by season")
    stab = data.season_table(fdf)
    if stab.empty:
        st.caption("—")
    else:
        st.dataframe(
            pd.DataFrame({
                "Season": stab["season"], "Events": stab["events"], "Bouts": stab["bouts"],
                "W": stab["wins"], "L": stab["losses"], "Win %": stab["win_pct"],
                "Indicator": stab["indicator"],
            }),
            width="stretch", hide_index=True,
            column_config={
                "Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                "Indicator": st.column_config.NumberColumn(format="%+d", help="Touches scored − received"),
            },
        )

    st.markdown("##### Touch-margin distribution")
    show_chart(charts.margin_histogram(fdf))

    st.markdown("##### Recent form")
    st.caption("Most recent bouts, oldest → newest.")
    st.markdown(form_chips(fdf))


def render_head_to_head(focal_id: int, focal_name: str, opp_id: int,
                        fdf: pd.DataFrame, focal_all: pd.DataFrame) -> None:
    opp_name = data.display_name(opp_id)
    shown = fdf[fdf["opponent_id"] == opp_id]
    ever = focal_all[focal_all["opponent_id"] == opp_id]

    if shown.empty:
        if not ever.empty:
            st.info(f"{focal_name} and {opp_name} have fenced, "
                    f"but no bouts match the current filters.")
            return
        st.info(f"{focal_name} and {opp_name} have never fenced. "
                "Comparing through common opponents instead.")
        render_common_opponents(focal_id, focal_name, opp_id, opp_name)
        return

    n = len(shown)
    fw = int(shown["won"].sum())
    fl = n - fw
    ordered = shown.sort_values("event_date", na_position="first")

    st.subheader(f"{focal_name}  vs.  {opp_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{focal_name} wins", fw)
    c2.metric("Bouts", n)
    c3.metric(f"{opp_name} wins", fl)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win rate", f"{fw / n * 100:.0f}%")
    c2.metric("Avg touches for", f"{shown['score_for'].mean():.1f}")
    c3.metric("Avg touches against", f"{shown['score_against'].mean():.1f}")
    c4.metric("Longest win streak", data.longest_streak(ordered, win=True))

    chart = charts.rivalry_timeline(shown, focal_name)
    if chart is not None:
        st.markdown("##### Rivalry timeline")
        st.caption("Cumulative touch margin — above zero means the focal fencer is ahead.")
        st.altair_chart(chart, width="stretch")

    st.markdown("##### Every bout")
    table = _bout_log_table(shown)
    st.dataframe(
        table.style.apply(_style_by_result, axis=1),
        width="stretch", hide_index=True,
    )


def render_common_opponents(focal_id: int, focal_name: str,
                            opp_id: int, opp_name: str) -> None:
    co = data.common_opponents(focal_id, opp_id)
    if co.empty:
        st.warning("They share no common opponents either — no comparison possible.")
        return

    f_w, f_l = int(co["focal_wins"].sum()), int(co["focal_losses"].sum())
    o_w, o_l = int(co["other_wins"].sum()), int(co["other_losses"].sum())
    f_total, o_total = f_w + f_l, o_w + o_l

    st.markdown(f"##### Across {len(co)} common opponents")
    c1, c2 = st.columns(2)
    c1.metric(
        f"{focal_name} vs. shared opponents",
        f"{f_w}–{f_l}",
        f"{(f_w / f_total * 100) if f_total else 0:.0f}% win rate",
    )
    c2.metric(
        f"{opp_name} vs. shared opponents",
        f"{o_w}–{o_l}",
        f"{(o_w / o_total * 100) if o_total else 0:.0f}% win rate",
    )
    st.caption("A rough strength proxy — not a rating model. The two fencers may "
               "have met these opponents at different times and ages.")

    table = pd.DataFrame({
        "Common opponent": co["opponent"],
        f"{focal_name} (W–L)": co["focal_wins"].astype(str) + "–" + co["focal_losses"].astype(str),
        f"{focal_name} win %": co["focal_win_pct"],
        f"{opp_name} (W–L)": co["other_wins"].astype(str) + "–" + co["other_losses"].astype(str),
        f"{opp_name} win %": co["other_win_pct"],
    })
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            f"{focal_name} win %": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100),
            f"{opp_name} win %": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100),
        },
    )


def render_opponents_tab(focal_id: int, focal_name: str,
                         fdf: pd.DataFrame, focal_all: pd.DataFrame) -> None:
    st.subheader(f"{focal_name} — Opponents & scouting")
    st.caption("Everyone the focal fencer has faced **plus** the fields of her upcoming "
               "events, ranked by estimated skill. Click a row — or look anyone up below — "
               "for the head-to-head and the data behind their skill.")

    all_ids, faced_counts = data.opponent_options(focal_id)
    names = data.name_map()

    def _lookup_label(i):
        if i is None:
            return "— look up any fencer —"
        nm = names.get(i, f"#{i}")
        c = faced_counts.get(i)
        return f"{nm}  ·  {c} bout{'s' if c != 1 else ''}" if c else f"{nm}  ·  not yet fenced"

    picked = st.selectbox("Look up any fencer (head-to-head / scouting)",
                          [None] + all_ids, format_func=_lookup_label, key="opp_lookup")

    lb = data.opponent_leaderboard(focal_all)          # career record vs each faced opponent
    faced = {int(r["opponent_id"]): r for _, r in lb.iterrows()} if not lb.empty else {}
    upcoming = data.upcoming_field_ids(focal_id)
    skill = ab.recent_skill_map()
    skills_arr = np.fromiter(skill.values(), dtype=float)

    ids = set(faced) | upcoming | {focal_id}      # include the focal herself, ranked by skill
    if not ids:
        st.info("No opponents to show yet.")
        return
    rows = []
    for oid in ids:
        is_focal = oid == focal_id
        r = faced.get(oid)
        sk = skill.get(oid)
        rows.append({
            "opponent_id": oid,
            "Opponent": data.display_name(oid) + ("  ★ (focal)" if is_focal else ""),
            "Est. skill": sk,
            "Pctile": ab.skill_percentile(sk, skills_arr),
            "Upcoming": (not is_focal) and (oid in upcoming),
            "Bouts": 0 if is_focal else (int(r["bouts"]) if r is not None else 0),
            "Record": "—" if is_focal else (f"{int(r['wins'])}–{int(r['losses'])}" if r is not None else "—"),
            "Win %": None if is_focal else (float(r["win_pct"]) if r is not None else None),
        })
    tbl = pd.DataFrame(rows).sort_values(
        "Est. skill", ascending=False, na_position="last").reset_index(drop=True)

    selection = st.dataframe(
        tbl.drop(columns=["opponent_id"]), width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Est. skill": st.column_config.NumberColumn(
                format="%.2f", help="Club-adjusted ability (skill + club effect, age-agnostic); "
                                    "blank if outside the rated cohort."),
            "Pctile": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100, help="Skill percentile among rated peers"),
            "Upcoming": st.column_config.CheckboxColumn(help="In one of her upcoming event fields"),
            "Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        },
    )

    opp_id = None
    if selection.selection.rows:                       # a table click wins
        opp_id = int(tbl.iloc[selection.selection.rows[0]]["opponent_id"])
    elif picked is not None:                            # else the look-up box
        opp_id = picked
    if opp_id is not None:
        st.divider()
        render_scouting(focal_id, focal_name, opp_id, fdf, focal_all)


def render_scouting(focal_id: int, focal_name: str, opp_id: int,
                    fdf: pd.DataFrame, focal_all: pd.DataFrame) -> None:
    """Backing data behind a fencer's skill: their record/form, then the head-to-head."""
    name = data.display_name(opp_id)
    skill = ab.recent_skill_map()
    sk = skill.get(opp_id)
    pct = ab.skill_percentile(sk, np.fromiter(skill.values(), dtype=float))
    rec = data.fencer_record(opp_id)

    st.markdown(f"### {name}")
    cols = st.columns(4)
    cols[0].metric("Est. skill", f"{sk:+.2f}" if sk is not None else "—")
    cols[1].metric("Skill percentile", f"{pct:.0f}%" if pct is not None else "—")
    if not rec.empty:
        s = data.summarize(rec)
        cols[2].metric("Bouts on record", s["bouts"])
        cols[3].metric("Win rate", f"{s['win_pct']:.0f}%")
        st.caption("Their recent form (own bouts, oldest → newest):")
        st.markdown(form_chips(rec))
    else:
        st.caption("No bout record on file (unrated / not yet scraped).")
    if opp_id == focal_id:
        st.caption("This is the focal fencer — shown for her skill ranking among the field.")
        return

    mw = ab.matchup_winprob(focal_id, opp_id)
    if mw and mw["focal_rated"]:
        st.markdown("##### Predicted matchup")
        m = st.columns(2)
        m[0].metric(f"P({focal_name} wins) — pool to 5", f"{mw['p_pool'] * 100:.0f}%")
        m[1].metric("— in a DE", f"{mw['p_de'] * 100:.0f}%")
        if not mw["opp_rated"]:
            st.caption("Opponent is unrated — uses population-average skill, so treat as rough.")

    st.divider()
    render_head_to_head(focal_id, focal_name, opp_id, fdf, focal_all)


def render_upcoming_tab(focal_id: int, focal_name: str) -> None:
    ev = data.focal_upcoming_events(focal_id)
    if ev.empty:
        st.info(f"No upcoming registered events on file for {focal_name}.")
        return

    labels = {int(r.event_id): f"{r.event_name} — {r.tournament_name}"
              f"  ({r.event_date or '?'})" for r in ev.itertuples()}
    eid = int(st.selectbox("Upcoming event", ev["event_id"].astype(int).tolist(),
                           format_func=lambda i: labels.get(i, str(i)), key="upcoming_event"))
    df = ab.event_placement_df(eid)
    if df.empty:
        st.info("Not enough rated fencers in this field to project a result.")
        return
    n = len(df)
    ev_name = labels[eid]
    st.subheader(f"{ev_name} — projected results")
    st.caption(f"{n} registered fencers · Monte-Carlo of pools → DE, sorted by expected "
               "finish. **Projected finish reflects both skill *and age*** — younger "
               "fencers (later birth year) are at a real, model-estimated disadvantage, so "
               "a high-skill but young fencer can project lower than her raw skill. Select a "
               "row for its cumulative finish distribution.")

    frow = df[df["fencer_id"] == focal_id]
    if not frow.empty:
        r = frow.iloc[0]
        c = st.columns(4)
        c[0].metric(f"{focal_name} proj. rank", int(r["proj_rank"]))
        c[1].metric("Expected finish", f"{r['exp_finish']:.0f}")
        c[2].metric("Median finish", int(r["p50"]))
        c[3].metric("Likely range (P10–P90)", f"{int(r['p10'])}–{int(r['p90'])}")

    table = pd.DataFrame({
        "Proj. rank": df["proj_rank"], "Fencer": df["fencer"], "Born": df["born"],
        "Skill": df["skill"], "Exp. finish": df["exp_finish"],
        "Finish P10·25·50·75·90": [f"{a} · {b} · {c} · {d} · {e}" for a, b, c, d, e
                                   in zip(df["p10"], df["p25"], df["p50"], df["p75"], df["p90"])],
    })
    selection = st.dataframe(
        table, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Born": st.column_config.NumberColumn(format="%d", help="Birth year — younger = disadvantaged"),
            "Skill": st.column_config.NumberColumn(
                format="%.2f", help="Club-adjusted ability (skill + club effect, age-agnostic); "
                                    "projected finish also factors in age"),
            "Exp. finish": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    chosen = (int(df.iloc[selection.selection.rows[0]]["fencer_id"])
              if selection.selection.rows else focal_id)
    samples = ab.placement_samples(eid, chosen)
    show_chart(charts.placement_density_quintiles(samples, n, data.display_name(chosen)))


def render_events_tab(focal_id: int, focal_name: str, fdf: pd.DataFrame) -> None:
    eh = data.event_history(focal_id, fdf)
    if eh.empty:
        st.info("No bouts match the current filters.")
        return

    st.subheader(f"{focal_name} — Competition history")
    st.caption(f"{len(eh)} events in view.")

    table = pd.DataFrame({
        "Date": eh["event_date"],
        "Event": eh["event_name"].fillna("—"),
        "Classification": eh["classification"].fillna("—"),
        "Weapon": eh["weapon"].fillna("—"),
        "Seed": eh["seed"],
        "Place": eh["placement"],
        "Field": eh["field_size"],
        "Bouts": eh["bouts"],
        "W": eh["wins"],
        "L": eh["losses"],
    })
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Seed": st.column_config.NumberColumn(format="%d"),
            "Place": st.column_config.NumberColumn(format="%d"),
            "Field": st.column_config.NumberColumn(format="%d"),
        },
    )

    chart = charts.placement_over_time(eh)
    if chart is not None:
        st.markdown("##### Finishing position over time")
        st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.title("Fencing Explorer")

    scraped = data.scraped_fencers()
    focal_ids = scraped["id"].tolist()
    if not focal_ids:
        st.error("No fully-scraped fencers in the database.")
        st.stop()

    names = dict(zip(scraped["id"], scraped["display"]))
    default_idx = focal_ids.index(data.FRANCESCA_ID) if data.FRANCESCA_ID in focal_ids else 0

    # ---- Sidebar ----
    st.sidebar.header("Focal fencer")
    focal_id = st.sidebar.selectbox(
        "Fencer", focal_ids, index=default_idx,
        format_func=lambda i: names.get(i, str(i)), label_visibility="collapsed",
    )
    focal_name = names.get(focal_id, str(focal_id))

    focal_all = data.focal_bouts(focal_id)
    frow = data.fencer_table().set_index("id").loc[focal_id]
    career = data.summarize(focal_all)
    span = ""
    if not focal_all["event_date"].isna().all():
        lo = focal_all["event_date"].min()
        hi = focal_all["event_date"].max()
        span = f"{lo:%b %Y} – {hi:%b %Y}"

    st.sidebar.markdown(f"**{focal_name}**")
    st.sidebar.caption(frow["club"] or "Club unknown")
    st.sidebar.markdown(
        f"{career['bouts']} bouts · {career['win_pct']:.0f}% win rate"
        + (f"  \n{span}" if span else "")
    )

    st.sidebar.divider()
    st.sidebar.header("Filters")
    weapon_opts = sorted(focal_all["weapon"].dropna().unique())
    sel_weapons = st.sidebar.multiselect("Weapon", weapon_opts, default=weapon_opts)

    age_opts = sorted(focal_all["age_group"].dropna().unique())
    sel_ages = st.sidebar.multiselect("Age group", age_opts, default=age_opts)

    bout_mode = st.sidebar.radio("Bouts", ["All", "Pool only", "DE only"], horizontal=True)

    date_range = None
    if not focal_all["event_date"].isna().all():
        dmin = focal_all["event_date"].min().date()
        dmax = focal_all["event_date"].max().date()
        date_range = st.sidebar.date_input(
            "Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax,
        )

    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(focal_ids):,} scraped fencers · "
        f"{len(data.perspective_bouts()) // 2:,} bouts in the database."
    )

    fdf = apply_filters(focal_all, sel_weapons, sel_ages, bout_mode, date_range)

    # ---- Tabs ----
    tab_overview, tab_upcoming, tab_opponents, tab_events = st.tabs(
        ["Overview", "Upcoming", "Opponents", "Events"]
    )
    with tab_overview:
        render_overview(focal_id, focal_name, fdf)
    with tab_upcoming:
        render_upcoming_tab(focal_id, focal_name)
    with tab_opponents:
        render_opponents_tab(focal_id, focal_name, fdf, focal_all)
    with tab_events:
        render_events_tab(focal_id, focal_name, fdf)


main()

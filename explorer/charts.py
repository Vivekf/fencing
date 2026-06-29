"""Altair chart builders for the Fencing Explorer.

Every chart returns an `alt.Chart` (or None if there's no data). Consistent
color encoding throughout: Win = green, Loss = red.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

WIN_COLOR = "#16a34a"
LOSS_COLOR = "#dc2626"
ACCENT = "#1d4ed8"

_RESULT_SCALE = alt.Scale(domain=["Win", "Loss"], range=[WIN_COLOR, LOSS_COLOR])
_RESULT_COLOR = alt.Color("result:N", scale=_RESULT_SCALE, title="Result")


def win_rate_by_season(df: pd.DataFrame) -> alt.Chart | None:
    """Line of win % per season, with bout volume as point size."""
    d = df.dropna(subset=["season"])
    if d.empty:
        return None
    g = d.groupby("season").agg(
        bouts=("won", "size"), wins=("won", "sum")
    ).reset_index()
    g["win_pct"] = g["wins"] / g["bouts"] * 100.0

    base = alt.Chart(g).encode(
        x=alt.X("season:O", title="Season", sort=None)
    )
    line = base.mark_line(color=ACCENT, strokeWidth=2.5).encode(
        y=alt.Y("win_pct:Q", title="Win %", scale=alt.Scale(domain=[0, 100])),
    )
    points = base.mark_point(color=ACCENT, filled=True).encode(
        y="win_pct:Q",
        size=alt.Size("bouts:Q", title="Bouts", scale=alt.Scale(range=[40, 320])),
        tooltip=[
            alt.Tooltip("season:O", title="Season"),
            alt.Tooltip("bouts:Q", title="Bouts"),
            alt.Tooltip("wins:Q", title="Wins"),
            alt.Tooltip("win_pct:Q", title="Win %", format=".1f"),
        ],
    )
    return (line + points).properties(height=280)


def outcome_bars(df: pd.DataFrame, field: str, title: str) -> alt.Chart | None:
    """Horizontal stacked W/L bars grouped by a categorical `field`."""
    d = df.dropna(subset=[field])
    if d.empty:
        return None
    g = d.groupby([field, "result"]).size().reset_index(name="bouts")
    return (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("bouts:Q", title="Bouts", stack="zero"),
            y=alt.Y(f"{field}:N", title=None, sort="-x"),
            color=_RESULT_COLOR,
            order=alt.Order("result:N", sort="descending"),
            tooltip=[
                alt.Tooltip(f"{field}:N", title=title),
                alt.Tooltip("result:N", title="Result"),
                alt.Tooltip("bouts:Q", title="Bouts"),
            ],
        )
        .properties(height=max(120, 42 * g[field].nunique()), title=title)
    )


def margin_histogram(df: pd.DataFrame) -> alt.Chart | None:
    """Distribution of touch margin (touches for − against)."""
    if df.empty:
        return None
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("margin:Q", bin=alt.Bin(step=1), title="Touch margin (for − against)"),
            y=alt.Y("count():Q", title="Bouts"),
            color=_RESULT_COLOR,
            tooltip=[
                alt.Tooltip("margin:Q", bin=alt.Bin(step=1), title="Margin"),
                alt.Tooltip("count():Q", title="Bouts"),
            ],
        )
        .properties(height=280)
    )


def rivalry_timeline(h2h: pd.DataFrame, focal_name: str) -> alt.Chart | None:
    """Cumulative head-to-head touch margin across the rivalry."""
    if h2h.empty:
        return None
    d = h2h.sort_values("event_date", na_position="last").reset_index(drop=True).copy()
    d["bout_no"] = range(1, len(d) + 1)
    d["cumulative_margin"] = d["margin"].cumsum()
    d["date_label"] = d["event_date"].dt.date.astype(str)

    line = (
        alt.Chart(d)
        .mark_area(
            line={"color": ACCENT, "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#dbeafe", offset=0),
                    alt.GradientStop(color="#93c5fd", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
            opacity=0.5,
        )
        .encode(
            x=alt.X("bout_no:Q", title="Bout number", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("cumulative_margin:Q", title=f"Cumulative margin ({focal_name})"),
            tooltip=[
                alt.Tooltip("bout_no:Q", title="Bout #"),
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("event_name:N", title="Event"),
                alt.Tooltip("score_for:Q", title="For"),
                alt.Tooltip("score_against:Q", title="Against"),
                alt.Tooltip("cumulative_margin:Q", title="Cumulative"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#94a3b8", strokeDash=[4, 4]
    ).encode(y="y:Q")
    return (zero + line).properties(height=260)


# Field-quintile shading: Q1 (top fifth, best) green → Q5 (bottom fifth) red.
_QUINTILE_COLORS = ["#15803d", "#86efac", "#fde68a", "#fca5a5", "#dc2626"]


def placement_density_quintiles(samples, n_field: int, fencer_name: str) -> alt.Chart | None:
    """Smoothed (Gaussian-KDE) finish histogram, shaded by which field quintile each
    finishing position falls in (Q1 = top fifth … Q5 = bottom fifth)."""
    if samples is None or len(samples) == 0:
        return None
    s = np.asarray(samples, float)
    xs = np.arange(1, n_field + 1)
    bw = max(1.5, n_field / 40.0)
    dens = np.exp(-0.5 * ((xs[:, None] - s[None, :]) / bw) ** 2).mean(1)
    q = np.minimum(4, ((xs - 1) * 5) // n_field).astype(int)
    d = pd.DataFrame({"place": xs, "density": dens / dens.sum(),
                      "Quintile": [f"Q{i + 1}" for i in q]})
    order = [f"Q{i}" for i in range(1, 6)]
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("place:Q", title=f"Finish position (1 – {n_field})",
                    scale=alt.Scale(domain=[1, n_field])),
            y=alt.Y("density:Q", title="Relative likelihood", axis=alt.Axis(labels=False)),
            color=alt.Color("Quintile:N", sort=order,
                            scale=alt.Scale(domain=order, range=_QUINTILE_COLORS),
                            legend=alt.Legend(title="Field quintile")),
            tooltip=[alt.Tooltip("place:Q", title="Place"), alt.Tooltip("Quintile:N")],
        )
        .properties(height=380, title=f"{fencer_name} — projected finish (shaded by field quintile)")
    )


def experience_band(data: dict, focal_name: str) -> alt.Chart | None:
    """Skill vs. cumulative serious (RYC+) experience: cohort points, fitted log curve +
    ±1σ band, focal fencer highlighted in red."""
    if not data or data["points"].empty:
        return None
    pts, line, foc = data["points"], data["line"], data["focal"]
    xax = alt.X("ryc:Q", title="Cumulative serious (RYC+) bouts (log scale)",
                scale=alt.Scale(type="log"))
    tip = [alt.Tooltip("ryc:Q", title="RYC+ bouts"), alt.Tooltip("skill:Q", format="+.2f")]
    band = alt.Chart(line).mark_area(opacity=0.18, color="#6b7280").encode(x=xax, y="lo:Q", y2="hi:Q")
    fit = alt.Chart(line).mark_line(color="#111827", strokeWidth=2.5).encode(
        x=xax, y=alt.Y("expected:Q", title="Skill (s + club), age-adjusted"))
    scat = alt.Chart(pts).mark_circle(size=24, opacity=0.30, color=ACCENT).encode(x=xax, y="skill:Q", tooltip=tip)
    fp = alt.Chart(pd.DataFrame([foc])).mark_point(
        size=240, filled=True, color=LOSS_COLOR, stroke="white", strokeWidth=1.5).encode(
        x=xax, y="skill:Q", tooltip=tip)
    return (band + scat + fit + fp).properties(
        height=320, title=f"{focal_name} — skill vs. serious experience")


def skill_trajectory(traj: pd.DataFrame) -> alt.Chart | None:
    """A fencer's monthly estimated skill over time."""
    if traj is None or traj.empty:
        return None
    return (
        alt.Chart(traj)
        .mark_line(color=ACCENT, strokeWidth=2.5, point=alt.OverlayMarkDef(color=ACCENT, size=45))
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("skill:Q", title="Estimated skill"),
            tooltip=[alt.Tooltip("month:T", title="Month"),
                     alt.Tooltip("skill:Q", title="Skill", format="+.2f")],
        )
        .properties(height=240)
    )


def placement_over_time(events_df: pd.DataFrame) -> alt.Chart | None:
    """Finishing position as a percentile of field size, per event over time."""
    d = events_df.dropna(subset=["placement", "field_size", "event_date"]).copy()
    d = d[d["field_size"] > 0]
    if d.empty:
        return None
    d["finish_pctile"] = d["placement"] / d["field_size"] * 100.0
    return (
        alt.Chart(d)
        .mark_circle(size=90, color=ACCENT, opacity=0.7)
        .encode(
            x=alt.X("event_date:T", title="Date"),
            y=alt.Y(
                "finish_pctile:Q",
                title="Finish (percentile — lower is better)",
                scale=alt.Scale(domain=[0, 100], reverse=True),
            ),
            tooltip=[
                alt.Tooltip("event_name:N", title="Event"),
                alt.Tooltip("event_date:T", title="Date"),
                alt.Tooltip("placement:Q", title="Place"),
                alt.Tooltip("field_size:Q", title="Field"),
            ],
        )
        .properties(height=260)
    )

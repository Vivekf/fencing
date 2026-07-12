"""Connectivity-gated, anchor-rooted expansion of the scrape-core.

The *scrape-core* is the set of fully-scraped fencers we treat as the focal fencer's
relevant competitive world. A fencer is admitted iff ALL hold:
  - within `radius` hops of the anchors (focal + upcoming-event field) in the bout graph,
  - has fenced >= `k` distinct fencers already in the core (density / connectivity),
  - is not a known-male who never fenced the focal (male-pruning),
  - has a scrapable profile.

This replaces the earlier blind hop-budget, which seeded a 2-hop expansion from every
fencer and ballooned the dataset. Here distance bounds reach; connectivity decides
relevance, so one-off peripheral opponents are skipped while well-connected fencers
(even a few hops out) are kept.

`compute_core` is read-only and used both to size a dry-run and to drive scraping.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from . import db, scraper
from .http import HttpClient

log = logging.getLogger(__name__)


@dataclass
class FrontierStats:
    expanded: int = 0           # fencers scraped this run
    bouts_inserted: int = 0
    fencers_discovered: int = 0
    errors: int = 0
    core_size: int = 0          # size of the target scrape-core at finish
    targets_remaining: int = 0  # core members still not 'done' (if capped)
    capped: bool = False


def _load_graph(conn: sqlite3.Connection) -> dict[int, set]:
    adj: dict[int, set] = defaultdict(set)
    for a, b in conn.execute("SELECT fencer_a_id, fencer_b_id FROM bouts"):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _anchors(conn: sqlite3.Connection, focal_id: int) -> set:
    ids = {focal_id}
    for (fid,) in conn.execute("SELECT DISTINCT fencer_id FROM upcoming_event_registrants"):
        ids.add(fid)
    return ids


YOUTH_AGE_GROUPS = {"Y8", "Y10", "Y12", "Y14"}


def _youth_fraction(conn: sqlite3.Connection) -> dict[int, float]:
    """Per-fencer fraction of bouts in youth events (Y8-Y14). Knowable pre-scrape from
    the event age-groups, so it can gate expansion to the focal's age world without
    needing each candidate's (post-scrape) birth year."""
    tally: dict[int, list] = defaultdict(lambda: [0, 0])
    for a, b, g in conn.execute(
        "SELECT bb.fencer_a_id, bb.fencer_b_id, e.age_group "
        "FROM bouts bb JOIN events e ON e.id = bb.event_id"
    ):
        y = 1 if g in YOUTH_AGE_GROUPS else 0
        for f in (a, b):
            tally[f][0] += y
            tally[f][1] += 1
    return {f: (v[0] / v[1] if v[1] else None) for f, v in tally.items()}


def _fenced_focal(conn: sqlite3.Connection, focal_id: int) -> set:
    s = set()
    for a, b in conn.execute(
        "SELECT fencer_a_id, fencer_b_id FROM bouts WHERE fencer_a_id=? OR fencer_b_id=?",
        (focal_id, focal_id),
    ):
        s.add(a)
        s.add(b)
    s.discard(focal_id)
    return s


def compute_core(
    conn: sqlite3.Connection, focal_id: int, k: int, radius: int,
    adj: Optional[dict] = None, youth_frac_min: float = 0.0,
):
    """Return (core, targets, adj).

    core    : set of fencer ids in the target scrape-core
    targets : core members not yet 'done' (scrapable), ordered by core-connectivity desc

    `youth_frac_min` (>0) gates candidates to fencers whose share of youth-event bouts
    is at least that — keeping expansion within the focal's age world (the wider teen/
    adult graph 2 hops away is irrelevant). Anchors (focal + field) are always kept.
    """
    if adj is None:
        adj = _load_graph(conn)
    anchors = {a for a in _anchors(conn, focal_id) if a in adj}
    youth = _youth_fraction(conn) if youth_frac_min > 0 else {}

    # Hop distance from anchors (bounded at `radius`).
    dist = {a: 0 for a in anchors}
    dq = deque(anchors)
    while dq:
        u = dq.popleft()
        if dist[u] >= radius:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    within = set(dist)

    status, prof, gender = {}, {}, {}
    for fid, st, hp, g in conn.execute(
        "SELECT id, scrape_status, has_profile, gender FROM fencers"
    ):
        status[fid] = st
        prof[fid] = hp
        gender[fid] = g
    fenced_focal = _fenced_focal(conn, focal_id)

    def male_ok(x):
        return not (gender.get(x) == "M" and x not in fenced_focal and x != focal_id)

    def youth_ok(x):
        if youth_frac_min <= 0 or x in anchors:
            return True
        yf = youth.get(x)
        return yf is None or yf >= youth_frac_min   # keep unknown-age (rare) by default

    # Iterative density growth within the radius ball.
    core = set(anchors)
    cand = [x for x in within if x not in core and prof.get(x) and male_ok(x) and youth_ok(x)]
    while True:
        newly = [x for x in cand if len(adj[x] & core) >= k]
        if not newly:
            break
        core |= set(newly)
        newly_set = set(newly)
        cand = [x for x in cand if x not in newly_set]

    targets = [x for x in core if status.get(x) == "discovered" and prof.get(x)]
    targets.sort(key=lambda x: len(adj[x] & core), reverse=True)  # most-connected first
    return core, targets, adj


def expand_frontier(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    focal_id: int,
    k: int = 3,
    radius: int = 2,
    max_new: int = 2000,
    youth_frac_min: float = 0.0,
    progress: bool = True,
) -> FrontierStats:
    """Grow the scrape-core by scraping connectivity-qualified fencers, up to `max_new`.

    Recomputes the core after each batch, since scraping reveals new fencers that may
    themselves qualify; converges when no undone core member remains."""
    stats = FrontierStats()
    # max_new: None = unlimited, 0 (or negative) = expansion disabled, N>0 = cap at N.
    # (Previously 0 fell through the falsy `if max_new` checks and meant *unlimited* — a
    # footgun. An explicit cap of 0 now does nothing, matching the intuitive reading.)
    if max_new is not None and max_new <= 0:
        core, _, _ = compute_core(conn, focal_id, k, radius, youth_frac_min=youth_frac_min)
        stats.core_size = len(core)
        return stats

    reset = db.reset_in_progress(conn)
    if reset:
        conn.commit()

    bar = None
    while True:
        core, targets, _ = compute_core(conn, focal_id, k, radius, youth_frac_min=youth_frac_min)
        stats.core_size = len(core)
        if not targets:
            break
        if max_new and stats.expanded >= max_new:
            stats.capped = True
            stats.targets_remaining = len(targets)
            break

        remaining = (max_new - stats.expanded) if max_new else len(targets)
        batch = targets[:remaining]
        if progress and bar is None:
            from tqdm import tqdm
            bar = tqdm(total=min(max_new, len(targets)) if max_new else len(targets),
                       desc="core-expand", unit="fencer")

        for fid in batch:
            row = conn.execute("SELECT slug FROM fencers WHERE id=?", (fid,)).fetchone()
            slug = row[0] if row else None
            db.set_fencer_status(conn, fid, "in_progress")
            conn.commit()
            try:
                sub = scraper.scrape_fencer_history(
                    conn, client, fencer_id=fid, slug=slug,
                    enqueue_new_opponents=False, opponent_depth=None,
                    opponent_hops=None, use_cache=True,
                )
                db.set_fencer_status(conn, fid, "done", history_pages=1)
                stats.expanded += 1
                stats.bouts_inserted += sub.bouts_inserted
                stats.fencers_discovered += sub.fencers_discovered
                if bar is not None:
                    bar.update(1)
            except Exception as exc:
                stats.errors += 1
                db.set_fencer_status(conn, fid, "error", error_message=f"{type(exc).__name__}: {exc}")
                log.exception("Error expanding fencer %s", fid)
            finally:
                conn.commit()
            if max_new and stats.expanded >= max_new:
                break
        # loop recomputes the core with the now-larger graph

    if bar is not None:
        bar.close()
    return stats

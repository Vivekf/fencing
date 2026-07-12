"""SQLite schema and helpers for the fencing tracker."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS fencers (
    id              INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    slug            TEXT,
    club            TEXT,
    has_profile     INTEGER NOT NULL DEFAULT 1,  -- 0 for legacy fencers (4-5 digit IDs, no /p/ URL)
    gender          TEXT,                         -- 'M'|'W' inferred from single-gender events; NULL if mixed-only/unknown
    birth_year      INTEGER,                      -- from the profile hero header; for the model's age covariate
    bfs_depth       INTEGER,                      -- min hops from the bootstrap focal fencer (informational)
    scrape_hops     INTEGER NOT NULL DEFAULT 0,   -- remaining hops to scrape around a newly-found fencer (>=1 = queued for expansion)
    scrape_status   TEXT    NOT NULL DEFAULT 'discovered',
    history_pages   INTEGER,
    last_scraped_at TEXT,
    discovered_at   TEXT    NOT NULL,
    error_message   TEXT
);
CREATE INDEX IF NOT EXISTS idx_fencers_status ON fencers(scrape_status);
CREATE INDEX IF NOT EXISTS idx_fencers_depth  ON fencers(bfs_depth);
-- idx_fencers_hops is created in init_schema() after the scrape_hops migration,
-- so this script stays valid against DBs that predate the column.

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY,
    name            TEXT,
    classification  TEXT,
    weapon          TEXT,
    gender          TEXT,
    age_group       TEXT,
    rating_level    TEXT,
    event_date      TEXT,
    raw_date        TEXT,
    first_seen_at   TEXT NOT NULL,
    results_ingested_at TEXT           -- set when the whole field was ingested from /results
);

CREATE TABLE IF NOT EXISTS bouts (
    event_id         INTEGER NOT NULL,
    fencer_a_id      INTEGER NOT NULL,
    fencer_b_id      INTEGER NOT NULL,
    bout_type        TEXT    NOT NULL,
    bout_seq         INTEGER NOT NULL DEFAULT 1,  -- 1 unless same pair meets again in same bout_type (e.g. Y8 double round-robin pools)
    fencer_a_score   INTEGER NOT NULL,
    fencer_b_score   INTEGER NOT NULL,
    winner_id        INTEGER NOT NULL,
    source_fencer_id INTEGER NOT NULL,
    PRIMARY KEY (event_id, fencer_a_id, fencer_b_id, bout_type, bout_seq),
    FOREIGN KEY (event_id)    REFERENCES events(id),
    FOREIGN KEY (fencer_a_id) REFERENCES fencers(id),
    FOREIGN KEY (fencer_b_id) REFERENCES fencers(id),
    FOREIGN KEY (winner_id)   REFERENCES fencers(id)
);
CREATE INDEX IF NOT EXISTS idx_bouts_event ON bouts(event_id);
CREATE INDEX IF NOT EXISTS idx_bouts_a     ON bouts(fencer_a_id);
CREATE INDEX IF NOT EXISTS idx_bouts_b     ON bouts(fencer_b_id);

CREATE TABLE IF NOT EXISTS fencer_event_results (
    fencer_id     INTEGER NOT NULL,
    event_id      INTEGER NOT NULL,
    seed          INTEGER,
    placement     INTEGER,
    field_size    INTEGER,
    rating_earned TEXT,
    PRIMARY KEY (fencer_id, event_id),
    FOREIGN KEY (fencer_id) REFERENCES fencers(id),
    FOREIGN KEY (event_id)  REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS scrape_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fencer_id      INTEGER,
    url            TEXT NOT NULL,
    status_code    INTEGER,
    bouts_added    INTEGER,
    fencers_added  INTEGER,
    duration_ms    INTEGER,
    error_message  TEXT,
    started_at     TEXT NOT NULL,
    FOREIGN KEY (fencer_id) REFERENCES fencers(id)
);

-- One row per upcoming (preregistered) event a tracked fencer is entered in.
-- These come from the /event/{id} preregistration roster pages, whose id namespace
-- is DISTINCT from the historical events.id namespace (/event/{id}/results). The
-- roster is a moving target (registrations change), so it's refreshed on re-scrape.
-- Deliberately NO strength/seeding/win-probability columns: that roster page ranks
-- fencers by fencingtracker's own conservative-estimate model, which we don't trust.
CREATE TABLE IF NOT EXISTS upcoming_events (
    event_id        INTEGER PRIMARY KEY,         -- fencingtracker preregistration /event/{id}
    tournament_name TEXT,
    event_name      TEXT,                         -- "Youth 10 Women's Epee (Y10WE)"
    classification  TEXT,
    weapon          TEXT,                         -- 'epee' | 'foil' | 'saber'
    gender          TEXT,                         -- 'M' | 'W' | 'X'
    age_group       TEXT,                         -- 'Y10' | 'Y12' | …
    venue           TEXT,
    location        TEXT,
    start_datetime  TEXT,                         -- raw e.g. 'Sunday, July 5, 2026 at 2:00 PM'
    event_date      TEXT,                         -- ISO 'YYYY-MM-DD' if parsable
    field_size      INTEGER,                      -- count of registered fencers
    first_seen_at   TEXT NOT NULL,
    last_scraped_at TEXT NOT NULL
);

-- The field: one row per fencer registered for an upcoming event. The focal fencer
-- appears here too (so "events Francesca is in" = registrants WHERE fencer_id=her).
-- Factual identity only — no strength/seed numbers.
CREATE TABLE IF NOT EXISTS upcoming_event_registrants (
    event_id   INTEGER NOT NULL,
    fencer_id  INTEGER NOT NULL,
    name       TEXT,
    club       TEXT,
    PRIMARY KEY (event_id, fencer_id),
    FOREIGN KEY (event_id)  REFERENCES upcoming_events(event_id),
    FOREIGN KEY (fencer_id) REFERENCES fencers(id)
);
CREATE INDEX IF NOT EXISTS idx_upcoming_reg_event  ON upcoming_event_registrants(event_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_reg_fencer ON upcoming_event_registrants(fencer_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Migrations for databases created before a column existed. CREATE TABLE
    # IF NOT EXISTS won't add columns to a pre-existing table, so do it explicitly.
    if not _column_exists(conn, "fencers", "scrape_hops"):
        conn.execute("ALTER TABLE fencers ADD COLUMN scrape_hops INTEGER NOT NULL DEFAULT 0")
    if not _column_exists(conn, "fencers", "gender"):
        conn.execute("ALTER TABLE fencers ADD COLUMN gender TEXT")
    if not _column_exists(conn, "fencers", "birth_year"):
        conn.execute("ALTER TABLE fencers ADD COLUMN birth_year INTEGER")
    if not _column_exists(conn, "events", "results_ingested_at"):
        conn.execute("ALTER TABLE events ADD COLUMN results_ingested_at TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fencers_hops ON fencers(scrape_hops)")
    conn.commit()


# SQL fragment for "this fencer is worth scraping given male-pruning": keep women,
# unknown-gender (mixed-only) fencers, and any fencer who has fenced the focal
# directly; skip known-male fencers otherwise. `f` must alias the fencers table.
_KEEP_GENDER_SQL = (
    "(f.gender IS NULL OR f.gender != 'M' OR EXISTS ("
    "  SELECT 1 FROM bouts b WHERE (b.fencer_a_id = f.id AND b.fencer_b_id = :focal)"
    "                            OR (b.fencer_b_id = f.id AND b.fencer_a_id = :focal)))"
)


def backfill_gender(conn: sqlite3.Connection) -> int:
    """Infer fencer gender from the single-gender events they've competed in.

    'M' if they appear only in Men's events, 'W' if only Women's; left NULL when they
    appear in mixed events only, in both, or have no bouts. One-time / idempotent.
    Returns the number of rows set to a non-NULL gender.
    """
    conn.execute(
        """
        WITH g AS (
            SELECT fid, SUM(e.gender='M') AS m, SUM(e.gender='W') AS w
            FROM (
                SELECT fencer_a_id AS fid, event_id FROM bouts
                UNION ALL
                SELECT fencer_b_id AS fid, event_id FROM bouts
            ) bb
            JOIN events e ON e.id = bb.event_id
            GROUP BY fid
        )
        UPDATE fencers SET gender = (
            SELECT CASE
                WHEN g.m > 0 AND g.w = 0 THEN 'M'
                WHEN g.w > 0 AND g.m = 0 THEN 'W'
                ELSE NULL END
            FROM g WHERE g.fid = fencers.id
        )
        WHERE id IN (SELECT fid FROM g)
        """
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM fencers WHERE gender IS NOT NULL").fetchone()[0]


def ensure_fencer(
    conn: sqlite3.Connection,
    fencer_id: int,
    name: str,
    slug: Optional[str] = None,
    club: Optional[str] = None,
    bfs_depth: Optional[int] = None,
    has_profile: bool = True,
    scrape_hops: Optional[int] = None,
    gender: Optional[str] = None,
) -> bool:
    """Insert if new (status='discovered'); update name/slug/club if better data arrived.

    `scrape_hops`, when given and >= 1, queues the fencer for frontier expansion
    (raised to the max of any existing value). It is only applied to fencers that
    are not already 'done' — a fully-scraped fencer is refreshed by staleness, not
    re-expanded. `gender` ('M'/'W', typically the discovering event's gender) is
    recorded only when single-gender and not already known. Returns True if a new
    fencer row was inserted.
    """
    seed_hops = scrape_hops if (scrape_hops is not None and scrape_hops >= 1) else None
    known_gender = gender if gender in ("M", "W") else None
    cur = conn.execute(
        "SELECT id, bfs_depth, scrape_status, scrape_hops, gender FROM fencers WHERE id = ?",
        (fencer_id,),
    )
    row = cur.fetchone()
    if row is None:
        # Legacy (no-profile) fencers are stored as 'skipped' so they're not picked up
        # for scraping, but their bouts are still recorded.
        status = "discovered" if has_profile else "skipped"
        hops = seed_hops if (seed_hops and has_profile) else 0
        conn.execute(
            """
            INSERT INTO fencers
                (id, name, slug, club, has_profile, gender, bfs_depth, scrape_hops,
                 scrape_status, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fencer_id, name, slug, club, int(has_profile), known_gender, bfs_depth,
             hops, status, now_iso()),
        )
        return True
    # Update mutable fields if we have better info. Don't downgrade depth.
    updates = []
    params: list = []
    if name:
        updates.append("name = ?")
        params.append(name)
    if slug:
        updates.append("slug = ?")
        params.append(slug)
    if club:
        updates.append("club = ?")
        params.append(club)
    if known_gender and row["gender"] is None:
        updates.append("gender = ?")
        params.append(known_gender)
    if bfs_depth is not None and (row["bfs_depth"] is None or bfs_depth < row["bfs_depth"]):
        updates.append("bfs_depth = ?")
        params.append(bfs_depth)
    # Queue (or raise the budget) for expansion — but never re-queue a done fencer.
    if seed_hops and row["scrape_status"] != "done" and seed_hops > (row["scrape_hops"] or 0):
        updates.append("scrape_hops = ?")
        params.append(seed_hops)
    if updates:
        params.append(fencer_id)
        conn.execute(f"UPDATE fencers SET {', '.join(updates)} WHERE id = ?", params)
    return False


def set_fencer_status(
    conn: sqlite3.Connection,
    fencer_id: int,
    status: str,
    *,
    history_pages: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    sets = ["scrape_status = ?", "last_scraped_at = ?"]
    params: list = [status, now_iso()]
    if history_pages is not None:
        sets.append("history_pages = ?")
        params.append(history_pages)
    if error_message is not None:
        sets.append("error_message = ?")
        params.append(error_message)
    params.append(fencer_id)
    conn.execute(f"UPDATE fencers SET {', '.join(sets)} WHERE id = ?", params)


def set_fencer_birth_year(conn: sqlite3.Connection, fencer_id: int, birth_year: Optional[int]) -> None:
    if birth_year is None:
        return
    conn.execute("UPDATE fencers SET birth_year = ? WHERE id = ?", (birth_year, fencer_id))


def upsert_event(
    conn: sqlite3.Connection,
    event_id: int,
    name: Optional[str],
    classification: Optional[str],
    weapon: Optional[str],
    gender: Optional[str],
    age_group: Optional[str],
    rating_level: Optional[str],
    event_date: Optional[str],
    raw_date: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT INTO events (id, name, classification, weapon, gender, age_group,
                            rating_level, event_date, raw_date, first_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name           = COALESCE(excluded.name, events.name),
            classification = COALESCE(excluded.classification, events.classification),
            weapon         = COALESCE(excluded.weapon, events.weapon),
            gender         = COALESCE(excluded.gender, events.gender),
            age_group      = COALESCE(excluded.age_group, events.age_group),
            rating_level   = COALESCE(excluded.rating_level, events.rating_level),
            event_date     = COALESCE(excluded.event_date, events.event_date),
            raw_date       = COALESCE(excluded.raw_date, events.raw_date)
        """,
        (
            event_id, name, classification, weapon, gender, age_group,
            rating_level, event_date, raw_date, now_iso(),
        ),
    )


def insert_bout(
    conn: sqlite3.Connection,
    event_id: int,
    fencer_a_id: int,
    fencer_b_id: int,
    fencer_a_score: int,
    fencer_b_score: int,
    winner_id: int,
    bout_type: str,
    bout_seq: int,
    source_fencer_id: int,
) -> bool:
    """Insert if new. Canonical ordering required: fencer_a_id < fencer_b_id.

    Returns True if a row was inserted (False if duplicate).
    """
    assert fencer_a_id < fencer_b_id, "Caller must canonicalize fencer ordering"
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO bouts
            (event_id, fencer_a_id, fencer_b_id, bout_type, bout_seq,
             fencer_a_score, fencer_b_score, winner_id, source_fencer_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id, fencer_a_id, fencer_b_id, bout_type, bout_seq,
            fencer_a_score, fencer_b_score, winner_id, source_fencer_id,
        ),
    )
    return cur.rowcount > 0


def set_event_results_ingested(conn: sqlite3.Connection, event_id: int,
                               ts: Optional[str] = None) -> None:
    """Mark an event as having had its whole field ingested from /event/{id}/results."""
    conn.execute("UPDATE events SET results_ingested_at = ? WHERE id = ?",
                 (ts or now_iso(), event_id))


def event_results_ingested(conn: sqlite3.Connection, event_id: int) -> bool:
    row = conn.execute("SELECT results_ingested_at FROM events WHERE id = ?",
                       (event_id,)).fetchone()
    return bool(row and row[0])


def upsert_fencer_event_result(
    conn: sqlite3.Connection,
    fencer_id: int,
    event_id: int,
    seed: Optional[int],
    placement: Optional[int],
    field_size: Optional[int],
    rating_earned: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT INTO fencer_event_results
            (fencer_id, event_id, seed, placement, field_size, rating_earned)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fencer_id, event_id) DO UPDATE SET
            seed          = COALESCE(excluded.seed, fencer_event_results.seed),
            placement     = COALESCE(excluded.placement, fencer_event_results.placement),
            field_size    = COALESCE(excluded.field_size, fencer_event_results.field_size),
            rating_earned = COALESCE(excluded.rating_earned, fencer_event_results.rating_earned)
        """,
        (fencer_id, event_id, seed, placement, field_size, rating_earned),
    )


def upsert_upcoming_event(
    conn: sqlite3.Connection,
    event_id: int,
    tournament_name: Optional[str],
    event_name: Optional[str],
    classification: Optional[str],
    weapon: Optional[str],
    gender: Optional[str],
    age_group: Optional[str],
    venue: Optional[str],
    location: Optional[str],
    start_datetime: Optional[str],
    event_date: Optional[str],
    field_size: Optional[int],
) -> None:
    now = now_iso()
    conn.execute(
        """
        INSERT INTO upcoming_events
            (event_id, tournament_name, event_name, classification, weapon, gender,
             age_group, venue, location, start_datetime, event_date, field_size,
             first_seen_at, last_scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            tournament_name = COALESCE(excluded.tournament_name, upcoming_events.tournament_name),
            event_name      = COALESCE(excluded.event_name, upcoming_events.event_name),
            classification  = COALESCE(excluded.classification, upcoming_events.classification),
            weapon          = COALESCE(excluded.weapon, upcoming_events.weapon),
            gender          = COALESCE(excluded.gender, upcoming_events.gender),
            age_group       = COALESCE(excluded.age_group, upcoming_events.age_group),
            venue           = COALESCE(excluded.venue, upcoming_events.venue),
            location        = COALESCE(excluded.location, upcoming_events.location),
            start_datetime  = COALESCE(excluded.start_datetime, upcoming_events.start_datetime),
            event_date      = COALESCE(excluded.event_date, upcoming_events.event_date),
            field_size      = excluded.field_size,
            last_scraped_at = excluded.last_scraped_at
        """,
        (
            event_id, tournament_name, event_name, classification, weapon, gender,
            age_group, venue, location, start_datetime, event_date, field_size, now, now,
        ),
    )


def replace_upcoming_registrants(
    conn: sqlite3.Connection,
    event_id: int,
    registrants: Iterable[tuple[int, Optional[str], Optional[str]]],
) -> int:
    """Replace the registrant list for an upcoming event (roster is a moving target).

    `registrants` is an iterable of (fencer_id, name, club). Returns the count written.
    """
    conn.execute("DELETE FROM upcoming_event_registrants WHERE event_id = ?", (event_id,))
    rows = [(event_id, fid, name, club) for (fid, name, club) in registrants]
    conn.executemany(
        """
        INSERT OR REPLACE INTO upcoming_event_registrants (event_id, fencer_id, name, club)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def log_scrape(
    conn: sqlite3.Connection,
    fencer_id: Optional[int],
    url: str,
    *,
    status_code: Optional[int] = None,
    bouts_added: int = 0,
    fencers_added: int = 0,
    duration_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO scrape_log
            (fencer_id, url, status_code, bouts_added, fencers_added,
             duration_ms, error_message, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fencer_id, url, status_code, bouts_added, fencers_added,
         duration_ms, error_message, now_iso()),
    )


def next_to_scrape(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Return the next discovered fencer in BFS order (lowest depth first, then ID).

    Only returns fencers with `has_profile = 1` (modern /p/ URLs).
    """
    cur = conn.execute(
        """
        SELECT id, name, slug, bfs_depth
        FROM fencers
        WHERE scrape_status = 'discovered' AND has_profile = 1
        ORDER BY bfs_depth ASC, id ASC
        LIMIT 1
        """
    )
    return cur.fetchone()


def seed_fencer_for_expansion(conn: sqlite3.Connection, fencer_id: int, hops: int) -> None:
    """Queue a known-but-unscraped fencer for frontier expansion.

    Raises scrape_hops to `hops` (never lowers it), and only for profiled fencers
    that aren't already 'done'. A no-op otherwise.
    """
    if hops < 1:
        return
    conn.execute(
        """
        UPDATE fencers SET scrape_hops = MAX(scrape_hops, ?)
        WHERE id = ? AND has_profile = 1 AND scrape_status != 'done'
        """,
        (hops, fencer_id),
    )


def set_scrape_hops(conn: sqlite3.Connection, fencer_id: int, hops: int) -> None:
    conn.execute("UPDATE fencers SET scrape_hops = ? WHERE id = ?", (hops, fencer_id))


def next_to_expand(
    conn: sqlite3.Connection, focal_id: Optional[int] = None
) -> Optional[sqlite3.Row]:
    """Next fencer queued for expansion: most hops remaining first (closest to a new
    entrant), then lowest id. Only profiled, not-yet-scraped fencers. When `focal_id`
    is given, known-male fencers who never fenced the focal are skipped."""
    gate = f"AND {_KEEP_GENDER_SQL}" if focal_id is not None else ""
    cur = conn.execute(
        f"""
        SELECT f.id, f.name, f.slug, f.scrape_hops
        FROM fencers f
        WHERE f.scrape_status = 'discovered' AND f.has_profile = 1 AND f.scrape_hops >= 1
          {gate}
        ORDER BY f.scrape_hops DESC, f.id ASC
        LIMIT 1
        """,
        {"focal": focal_id},
    )
    return cur.fetchone()


def count_pending_expansion(
    conn: sqlite3.Connection, focal_id: Optional[int] = None
) -> int:
    gate = f"AND {_KEEP_GENDER_SQL}" if focal_id is not None else ""
    return conn.execute(
        f"""
        SELECT COUNT(*) FROM fencers f
        WHERE f.scrape_status = 'discovered' AND f.has_profile = 1 AND f.scrape_hops >= 1
          {gate}
        """,
        {"focal": focal_id},
    ).fetchone()[0]


def fencers_to_refresh(
    conn: sqlite3.Connection, cutoff_iso: str, focal_id: Optional[int] = None
) -> list[sqlite3.Row]:
    """Already-scraped fencers whose history is stale (last_scraped_at < cutoff, or
    never recorded), plus the focal fencer always. Known-male fencers who never fenced
    the focal are skipped (we don't maintain the men's graph). Oldest first."""
    gate = f"AND (f.id = :focal OR {_KEEP_GENDER_SQL})" if focal_id is not None else ""
    return conn.execute(
        f"""
        SELECT f.id, f.name, f.slug, f.last_scraped_at
        FROM fencers f
        WHERE f.has_profile = 1 AND f.scrape_status = 'done'
          AND (f.last_scraped_at IS NULL OR f.last_scraped_at < :cutoff OR f.id = :focal)
          {gate}
        ORDER BY (f.id = :focal) DESC, f.last_scraped_at ASC
        """,
        {"cutoff": cutoff_iso, "focal": focal_id if focal_id is not None else -1},
    ).fetchall()


def count_fencers(conn: sqlite3.Connection, *, status: Optional[str] = None) -> int:
    if status is None:
        return conn.execute("SELECT COUNT(*) FROM fencers").fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM fencers WHERE scrape_status = ?", (status,)
    ).fetchone()[0]


def reset_in_progress(conn: sqlite3.Connection) -> int:
    """Reset any 'in_progress' fencers back to 'discovered' (e.g., after a crash)."""
    cur = conn.execute(
        "UPDATE fencers SET scrape_status = 'discovered' WHERE scrape_status = 'in_progress'"
    )
    return cur.rowcount

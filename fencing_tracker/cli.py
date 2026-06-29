"""Command-line interface.

Usage:
    python -m fencing_tracker init-db [--db PATH]
    python -m fencing_tracker scrape --focal ID --slug SLUG --name NAME [--cap 1000]
    python -m fencing_tracker status [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import bfs, db, frontier, scraper, upcoming, updater
from .http import HttpClient

FRANCESCA_ID = 100835605
FRANCESCA_SLUG = "Francesca-Farias"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default="fencing.db", help="SQLite db path (default fencing.db)")


def cmd_init_db(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    try:
        db.init_schema(conn)
        print(f"Schema initialized at {args.db}")
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            print(f"  - {r[0]}")
    finally:
        conn.close()
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    conn = db.connect(args.db)
    db.init_schema(conn)
    client = HttpClient(cache_dir=args.cache_dir, delay_seconds=args.delay)
    stats = bfs.run_bfs(
        conn,
        client,
        focal_id=args.focal,
        focal_name=args.name,
        focal_slug=args.slug,
        cap=args.cap,
        progress=not args.no_progress,
    )
    print()
    print("BFS complete.")
    print(f"  fencers scraped  : {stats.fencers_scraped}")
    print(f"  bouts inserted   : {stats.bouts_inserted}")
    print(f"  bouts seen       : {stats.bouts_seen}")
    print(f"  bouts skipped    : {stats.bouts_skipped}  (unidentifiable opponent)")
    print(f"  events seen      : {stats.events_seen}")
    print(f"  legacy opponents : {stats.legacy_opponents}")
    print(f"  errors           : {stats.errors}")
    print(f"  by depth         : {dict(sorted(stats.by_depth.items()))}")
    _print_status(conn)
    conn.close()
    return 0


def cmd_upcoming(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    conn = db.connect(args.db)
    db.init_schema(conn)
    client = HttpClient(cache_dir=args.cache_dir, delay_seconds=args.delay)
    up = upcoming.refresh_upcoming(
        conn, client, fencer_id=args.focal, slug=args.slug,
        today_iso=updater._today_iso(),
    )
    print()
    print("Upcoming refresh complete.")
    _print_upcoming_stats(up)
    if not args.no_expand:
        fr = frontier.expand_frontier(
            conn, client, focal_id=args.focal, k=args.core_k, radius=args.radius,
            max_new=args.max_new, youth_frac_min=args.youth_frac_min,
            progress=not args.no_progress,
        )
        print()
        print("Frontier expansion:")
        _print_frontier_stats(fr)
    _print_upcoming(conn, focal_id=args.focal)
    conn.close()
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    conn = db.connect(args.db)
    db.init_schema(conn)
    client = HttpClient(cache_dir=args.cache_dir, delay_seconds=args.delay)
    stats = updater.run_update(
        conn, client,
        focal_id=args.focal, focal_slug=args.slug,
        refresh_after_days=args.refresh_after_days,
        max_new=args.max_new,
        core_k=args.core_k,
        radius=args.radius,
        youth_frac_min=args.youth_frac_min,
        progress=not args.no_progress,
    )
    print()
    print("Update complete.")
    print(" Stage 0 — upcoming:")
    _print_upcoming_stats(stats.upcoming, indent="   ")
    print(" Stage 1 — history refresh:")
    print(f"   stale histories refreshed : {stats.refreshed}")
    print(f"   new bouts added           : {stats.refresh_bouts_added}")
    print(f"   errors                    : {stats.refresh_errors}")
    print(" Stage 2 — scrape-core expansion:")
    _print_frontier_stats(stats.frontier, indent="   ")
    _print_status(conn)
    _print_upcoming(conn, focal_id=args.focal)
    conn.close()
    return 0


def _print_upcoming_stats(s, indent: str = "  ") -> None:
    print(f"{indent}registrations found   : {s.registrations_found}")
    print(f"{indent}future events scraped : {s.future_events_scraped}")
    print(f"{indent}past events skipped   : {s.past_events_skipped}")
    print(f"{indent}registrants total     : {s.registrants_total}")
    print(f"{indent}new fencers added     : {s.new_fencers}")
    print(f"{indent}errors                : {s.errors}")


def _print_frontier_stats(s, indent: str = "  ") -> None:
    print(f"{indent}fencers expanded      : {s.expanded}")
    print(f"{indent}bouts inserted        : {s.bouts_inserted}")
    print(f"{indent}fencers discovered    : {s.fencers_discovered}")
    print(f"{indent}scrape-core size      : {s.core_size}")
    print(f"{indent}errors                : {s.errors}")
    if s.capped:
        print(f"{indent}** per-run cap hit — {s.targets_remaining} core members remain; re-run to continue **")


def cmd_status(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    _print_status(conn)
    conn.close()
    return 0


def _print_status(conn) -> None:
    print()
    print("DB status:")
    print(f"  fencers          : {conn.execute('SELECT COUNT(*) FROM fencers').fetchone()[0]}")
    print(f"  events           : {conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]}")
    print(f"  bouts            : {conn.execute('SELECT COUNT(*) FROM bouts').fetchone()[0]}")
    if _table_exists(conn, "upcoming_events"):
        ue = conn.execute("SELECT COUNT(*) FROM upcoming_events").fetchone()[0]
        ur = conn.execute("SELECT COUNT(*) FROM upcoming_event_registrants").fetchone()[0]
        print(f"  upcoming events  : {ue}")
        print(f"  upcoming regs    : {ur}")
    if db._column_exists(conn, "fencers", "scrape_hops"):
        pending = db.count_pending_expansion(conn, FRANCESCA_ID)
        print(f"  queued to expand : {pending}")
    print()
    print("Fencer scrape status:")
    for row in conn.execute(
        "SELECT scrape_status, COUNT(*) AS n FROM fencers GROUP BY scrape_status ORDER BY n DESC"
    ):
        print(f"  {row['scrape_status']:14s} {row['n']}")
    print()
    print("Fencers by BFS depth (scraped only):")
    for row in conn.execute(
        """
        SELECT bfs_depth, COUNT(*) AS n
        FROM fencers
        WHERE scrape_status = 'done'
        GROUP BY bfs_depth ORDER BY bfs_depth
        """
    ):
        depth_str = str(row["bfs_depth"]) if row["bfs_depth"] is not None else "—"
        print(f"  depth {depth_str:<3s} {row['n']}")


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _print_upcoming(conn, focal_id: int) -> None:
    if not _table_exists(conn, "upcoming_events"):
        return
    print()
    print("Upcoming events for focal fencer:")
    rows = conn.execute(
        """
        SELECT e.event_id, e.event_name, e.tournament_name, e.event_date, e.field_size
        FROM upcoming_events e
        JOIN upcoming_event_registrants r
          ON r.event_id = e.event_id AND r.fencer_id = ?
        ORDER BY e.event_date
        """,
        (focal_id,),
    ).fetchall()
    if not rows:
        print("  (none registered)")
        return
    for row in rows:
        # How many of the field do we already hold a full history for?
        known = conn.execute(
            """
            SELECT COUNT(*) FROM upcoming_event_registrants r
            JOIN fencers f ON f.id = r.fencer_id
            WHERE r.event_id = ? AND f.scrape_status = 'done'
            """,
            (row["event_id"],),
        ).fetchone()[0]
        print(
            f"  [{row['event_date'] or '?'}] {row['event_name']} "
            f"@ {row['tournament_name']} — field {row['field_size']}, "
            f"histories held {known}/{row['field_size']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fencing_tracker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create the SQLite schema")
    _add_db_arg(p_init)
    p_init.set_defaults(func=cmd_init_db)

    p_scrape = sub.add_parser("scrape", help="Run BFS scrape from a focal fencer")
    _add_db_arg(p_scrape)
    p_scrape.add_argument("--focal", type=int, required=True, help="Focal fencer ID")
    p_scrape.add_argument("--name", default="", help="Focal fencer display name")
    p_scrape.add_argument("--slug", default="", help="Focal fencer URL slug")
    p_scrape.add_argument("--cap", type=int, default=1000, help="Max fencers to scrape (default 1000)")
    p_scrape.add_argument("--cache-dir", default=".cache", help="HTML cache dir (default .cache)")
    p_scrape.add_argument("--delay", type=float, default=1.5, help="Seconds between HTTP requests (default 1.5)")
    p_scrape.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    p_scrape.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_scrape.set_defaults(func=cmd_scrape)

    p_up = sub.add_parser("upcoming", help="Pull a fencer's upcoming events + fields, then expand new fencers")
    _add_db_arg(p_up)
    p_up.add_argument("--focal", type=int, default=FRANCESCA_ID,
                      help=f"Fencer ID whose registrations to pull (default {FRANCESCA_ID}, Francesca)")
    p_up.add_argument("--slug", default=FRANCESCA_SLUG, help="Fencer URL slug")
    p_up.add_argument("--no-expand", action="store_true",
                      help="Only record the field; don't grow the scrape-core")
    p_up.add_argument("--core-k", type=int, default=3,
                      help="Min distinct core opponents to admit a fencer (default 3)")
    p_up.add_argument("--radius", type=int, default=2,
                      help="Max hops from focal+field to consider (default 2)")
    p_up.add_argument("--youth-frac-min", type=float, default=0.5,
                      help="Min share of youth-event bouts to expand a fencer (default 0.5; 0=off)")
    p_up.add_argument("--max-new", type=int, default=2000,
                      help="Max new fencers to scrape this run (default 2000)")
    p_up.add_argument("--cache-dir", default=".cache", help="HTML cache dir (default .cache)")
    p_up.add_argument("--delay", type=float, default=1.5, help="Seconds between HTTP requests (default 1.5)")
    p_up.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    p_up.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_up.set_defaults(func=cmd_upcoming)

    p_update = sub.add_parser(
        "update", help="Idempotent refresh: completed events + upcoming fields + expand new fencers")
    _add_db_arg(p_update)
    p_update.add_argument("--focal", type=int, default=FRANCESCA_ID,
                          help=f"Focal fencer ID (default {FRANCESCA_ID}, Francesca)")
    p_update.add_argument("--slug", default=FRANCESCA_SLUG, help="Focal fencer URL slug")
    p_update.add_argument("--refresh-after-days", type=int, default=14,
                          help="Re-pull a fencer's history if older than this many days (default 14)")
    p_update.add_argument("--max-new", type=int, default=2000,
                          help="Max new fencers to scrape this run (default 2000)")
    p_update.add_argument("--core-k", type=int, default=3,
                          help="Min distinct core opponents to admit a fencer (default 3)")
    p_update.add_argument("--radius", type=int, default=2,
                          help="Max hops from focal+field to consider (default 2)")
    p_update.add_argument("--youth-frac-min", type=float, default=0.5,
                          help="Min share of youth-event bouts to expand a fencer (default 0.5; 0=off)")
    p_update.add_argument("--cache-dir", default=".cache", help="HTML cache dir (default .cache)")
    p_update.add_argument("--delay", type=float, default=1.5, help="Seconds between HTTP requests (default 1.5)")
    p_update.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    p_update.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_update.set_defaults(func=cmd_update)

    p_status = sub.add_parser("status", help="Print DB counts")
    _add_db_arg(p_status)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

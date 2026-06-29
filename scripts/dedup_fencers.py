"""Merge duplicate fencer records (same name + birth year, multiple profile IDs).

fencingtracker sometimes lists one person under several IDs, which splits their bout
history and under-rates them. This merges each (name, birth_year) group into the
canonical id (the one with the most bouts), repointing every fencer-id reference, then
deletes the orphaned records. Backs up the DB first.

Usage: python scripts/dedup_fencers.py [db_path]
"""
import shutil
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "fencing.db"
BACKUP = sys.argv[2] if len(sys.argv) > 2 else DB + ".predup.bak"


def main():
    shutil.copy(DB, BACKUP)
    print(f"backup -> {BACKUP}", flush=True)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Every column that references a fencer id, discovered from the schema.
    # NB: materialize each query (fetchall) before issuing the next on the same cursor.
    FENCER_ID_COLS = {"fencer_id", "fencer_a_id", "fencer_b_id", "winner_id", "source_fencer_id"}
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    idcols = []
    for t in tables:
        for r in cur.execute(f"PRAGMA table_info({t})").fetchall():
            if r[1] in FENCER_ID_COLS:
                idcols.append((t, r[1]))
    print("repointing columns:", idcols, flush=True)
    if not idcols:
        raise SystemExit("no fencer-id columns found — aborting")

    groups = defaultdict(list)
    for i, n, by in cur.execute(
        "SELECT id, name, birth_year FROM fencers WHERE name IS NOT NULL AND birth_year IS NOT NULL"
    ):
        groups[(n.strip().lower(), by)].append(i)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"duplicate groups: {len(dups)} ({sum(len(v) for v in dups.values())} records)", flush=True)

    merged = 0
    for ids in dups.values():
        counts = {i: cur.execute(
            "SELECT COUNT(*) FROM bouts WHERE fencer_a_id=? OR fencer_b_id=?", (i, i)).fetchone()[0]
            for i in ids}
        canon = max(counts, key=counts.get)
        for o in (i for i in ids if i != canon):
            for t, col in idcols:
                # results/registrants have unique (fencer,event) — OR IGNORE then drop leftovers
                cur.execute(f"UPDATE OR IGNORE {t} SET {col}=? WHERE {col}=?", (canon, o))
                cur.execute(f"DELETE FROM {t} WHERE {col}=?", (o,))
            cur.execute("DELETE FROM fencers WHERE id=?", (o,))
            merged += 1

    self_bouts = cur.execute("DELETE FROM bouts WHERE fencer_a_id = fencer_b_id").rowcount
    conn.commit()
    print(f"merged {merged} records; removed {self_bouts} self-bouts", flush=True)
    print(f"fencers: {cur.execute('SELECT COUNT(*) FROM fencers').fetchone()[0]} | "
          f"bouts: {cur.execute('SELECT COUNT(*) FROM bouts').fetchone()[0]}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()

"""Create the SQLite database and schema.

Usage:
    PYTHONPATH=. python scripts/init_db.py [path-to-db]
Default db path: ./fencing.db
"""

from __future__ import annotations

import sys
from pathlib import Path

from fencing_tracker import db


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fencing.db")
    print(f"Initializing schema at {path}")
    conn = db.connect(path)
    try:
        db.init_schema(conn)
        print("Schema created. Tables:")
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            print(f"  - {row[0]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

# Fencing Tracker

A local fencing dataset and explorer built from [fencingtracker.com](https://fencingtracker.com).

A scraper collects bout-level competition data into a SQLite database by breadth-first
search outward from a focal fencer — Francesca Farias — through her opponents, their
opponents, and so on. A Streamlit app then lets you explore any fencer's record
interactively.

**Dataset:** seeded at 1,002 fencers · 5,907 events · 206,463 bouts (Nov 2018 – May 2026),
and grown continuously by `fencing_tracker update` (completed events, upcoming fields, and a
2-hop expansion around every newly-seen fencer).

See [`PLAN.md`](PLAN.md) for the full design, decisions, and rationale.

## Requirements

- Python 3.13 — the repo pins a pyenv virtualenv named `fencing` via `.python-version`.
- Dependencies listed in `requirements.txt`.

## Setup

```bash
cd Fencing                        # pyenv activates the `fencing` env automatically
pip install -r requirements.txt
```

If the `fencing` virtualenv does not exist yet:

```bash
pyenv virtualenv 3.13 fencing
```

## Run the explorer

```bash
streamlit run explorer/app.py
```

Open <http://localhost:8501>. To stop it: `Ctrl-C` (or `pkill -f "streamlit run"` if it
is running in the background).

The explorer is **fencer-centric**: pick a focal fencer in the sidebar (default
Francesca; any of the 1,002 fully-scraped fencers), apply global filters (weapon, age
group, Pool/DE, date range), and browse four tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | KPI cards, win rate by season, W/L by weapon and Pool-vs-DE, touch-margin distribution, recent form. |
| **Head-to-Head** | Pick any opponent — W–L scorecard, rivalry timeline, full bout log. If the two have never met, it compares them through common opponents. |
| **Opponents** | Sortable leaderboard of everyone faced; select a row to expand the full head-to-head; toughest / most-frequent / best-results highlights. |
| **Events** | Competition history with seeds, placements, per-event records, and a finish-position-over-time chart. |

The database is read-only here — the explorer never writes to it.

## Bootstrap a fresh dataset (the BFS scraper)

`fencing.db` is already populated, so this is only needed to recreate it from scratch.
For ongoing refreshes use [`update`](#keep-the-dataset-current-the-update-command) instead.

```bash
python -m fencing_tracker init-db            # create the empty schema
python -m fencing_tracker scrape \
    --focal 100835605 --name "Francesca Farias" --slug "Francesca-Farias" --cap 1000
python -m fencing_tracker status             # print database counts
```

The scraper is idempotent and resumable — re-running continues from where it left off.
Fetched HTML is cached under `.cache/`, so re-parsing costs no new web requests. It is
polite by default: 1.5s between requests, retries with backoff, a contact User-Agent.

## Keep the dataset current (the `update` command)

```bash
python -m fencing_tracker update             # defaults to Francesca
python -m fencing_tracker update --refresh-after-days 14 --max-new 500
```

`update` is the **idempotent refresh** you run repeatedly (e.g. on a schedule). Each run
derives its work from the database, so re-running just converges. It does three things:

1. **Upcoming** — re-pulls the focal's registrations and the roster (field) of each
   future event into `upcoming_events` / `upcoming_event_registrants`.
2. **Completed events** — re-fetches the histories of already-scraped fencers whose data
   is stale (older than `--refresh-after-days`, default 14; the focal always), capturing
   any newly-fenced bouts. (The cache is bypassed for these so new events are seen.)
3. **Expand new fencers** — every newly-encountered fencer gets a bounded **2-hop** scrape
   (their bouts, plus the bouts of the new opponents that introduces), capped at
   `--max-new` per run (default 2000) so a large new roster can't blow up one run.

**Male-pruning:** Francesca fences Women's and some Mixed events, so the men's circuit is
skipped. A fencer's gender is inferred from the single-gender events they appear in, and
known-male fencers are excluded from both expansion and refresh **unless they have fenced
the focal directly**. Women and unknown-gender (mixed-only) fencers are always kept.

Only factual identity (name / club) is captured from rosters — fencingtracker's strength
columns are ignored. The one-shot `upcoming` command still exists if you only want to pull
upcoming events and expand their fields without the staleness refresh.

## Project layout

```
Fencing/
├── PLAN.md                  design document — source of truth
├── requirements.txt
├── fencing.db               SQLite database (gitignored)
├── .cache/                  cached HTML responses (gitignored)
├── .streamlit/config.toml   explorer theme
├── fencing_tracker/         the scraper
│   ├── db.py                schema, connection, helpers
│   ├── http.py              HTTP client: rate limit, cache, retries
│   ├── parsers.py           history-page HTML parser
│   ├── scraper.py           single-fencer scrape → DB
│   ├── bfs.py               BFS controller, cap logic (bootstrap)
│   ├── upcoming.py          upcoming events + field rosters
│   ├── frontier.py          capped 2-hop expansion of new fencers
│   ├── updater.py           idempotent 3-stage `update`
│   └── cli.py               init-db / scrape / upcoming / update / status
├── explorer/                the Streamlit app
│   ├── data.py              read-only DB layer + fencer-centric helpers
│   ├── charts.py            Altair chart builders
│   └── app.py               sidebar + four tabs
├── scripts/                 init_db.py, scrape_focal.py
└── tests/                   parser + explorer tests, with an HTML fixture
```

## Database

Tables (see `PLAN.md` for the full DDL):

- **`fencers`** — one row per fencer observed; doubles as the BFS queue via `scrape_status`.
- **`events`** — one row per event (a weapon/age/gender competition).
- **`bouts`** — one row per bout, canonicalized (`fencer_a_id < fencer_b_id`).
- **`fencer_event_results`** — per-fencer, per-event seed and placement.
- **`scrape_log`** — audit log of scrape operations.
- **`upcoming_events`** / **`upcoming_event_registrants`** — preregistered events a fencer is
  entered in, and the field for each (factual identity only).

Inspect it directly with `sqlite3 fencing.db`.

## Tests

```bash
PYTHONPATH=. python tests/test_parsers.py       # history-page parser, against a saved fixture
PYTHONPATH=. python tests/test_upcoming.py      # registration + roster parsers
PYTHONPATH=. python tests/test_update_logic.py  # hops queue + staleness refresh logic
PYTHONPATH=. python tests/test_explorer.py      # data layer + full Streamlit AppTest
```

## Notes

- **fencingtracker's own strength ratings are deliberately not captured** — they are
  considered unreliable. Custom rating/strength metrics are planned as a separate
  analytics pipeline (Phase E) built on the raw bout graph.
- Data-quality cases handled during scraping: Y-8 double round-robin pools (a pair
  fences twice), legacy fencers with no profile page, anonymous opponents with no ID,
  and occasional fencer-vs-self rows. See the Decisions section of `PLAN.md`.

# Fencing Activity Tracker — Implementation Plan

## Goal

A local fencing dataset and explorer, in three parts:

1. **Scraper** — collect bout-level data from [fencingtracker.com](https://fencingtracker.com) into a local SQLite database, starting from a focal fencer and expanding via BFS through her opponents (and their opponents, etc.) until ~1000 fencers are covered. *(Done — Phases A–C.)*
2. **Explorer** — a fencer-centric Streamlit app to interactively explore the data: fix a focal fencer and view their record overall and head-to-head against any other fencer. *(Phase D — next.)*
3. **Analytics pipeline** — a separate, custom pipeline that computes our own rating/strength metrics from the bout graph. fencingtracker's own strength numbers are considered unreliable and are **not** captured. *(Phase E — later.)*

**Focal fencer**: Francesca Farias — `/p/100835605/Francesca-Farias`

## Status Snapshot

| Phase | What | State |
|-------|------|-------|
| A | Schema + focal-fencer scrape | ✅ Done — parser validated (Francesca: 220 bouts, 29 events) |
| B | BFS expansion to 1000 fencers | ✅ Done — 1,002 fencers, 5,907 events, 206,463 bouts |
| C | Hardening (tests + CLI) | ✅ Done — 8 parser tests, `init-db`/`scrape`/`status` CLI |
| D | Explorer (Streamlit) | ✅ Done — `explorer/` package; `streamlit run explorer/app.py` |
| F | Upcoming events + fields | ✅ Done — `fencing_tracker upcoming`; registrations + rosters + field expansion |
| G | Continual idempotent updates | ✅ Done — `fencing_tracker update`; refresh completed + upcoming + 2-hop expand new fencers |
| E | Custom analytics pipeline | ◻ Later — user-built (incl. estimating performance vs. an upcoming field) |

## What We Learned About the Site

- **Bout-level data lives on fencer history pages**, not event pages.
  - URL pattern: `/p/{fencer_id}/{name-slug}/history`
  - Each row in the history table includes: bout type (Pool / T64 / T32 / T16 / T8 / T4 / T2), result (V/D from this fencer's POV), score (`my_score:opp_score`), opponent (hyperlinked to `/p/{opp_id}/{opp-slug}` — this is our BFS edge), opponent club, seed, placement.
  - Each tournament entry above the bout table has: event name + link (`/event/{event_id}/results`), classification (e.g. "Unrated Y-14 Women's Épée"), date, our fencer's seed and placement.
- **Same bout appears on both fencers' pages** with mirrored scores — natural cross-validation, and we dedupe via canonical ordering (`min(id) → fencer_a`, `max(id) → fencer_b`).
- **No JSON API**; HTML-only. **No JS rendering required**. **`robots.txt`** is fully permissive (`Allow: /`). **No auth required**.
- Data-quality quirks found during the scrape: Y-8 double round-robin pools (same pair fences twice), legacy fencers with 4-5 digit IDs and no profile page, a small number of anonymous opponents with no ID, and occasional fencer-vs-self rows. All handled — see Decisions.

## Discovery Strategy (BFS)

1. Seed the fencer queue with Francesca (depth 0).
2. Pop a fencer, fetch `/p/{id}/{slug}/history`, parse bouts.
3. For each bout: write the bout, ensure both fencers exist in the `fencers` table.
4. Newly seen opponents are always recorded; modern-profile ones join the BFS queue (depth = parent depth + 1).
5. Continue until 1000 fencers have been fully scraped, or the queue empties.

**Cap semantics**: stop once 1000 fencers have `scrape_status='done'`. Bouts against opponents beyond the cap are still recorded — the opponent sits in `fencers` as `discovered` (history not fetched). Richest dataset for the cost of 1000 history-scrapes.

**Ordering**: classic BFS (FIFO queue) — the closest fencers to Francesca are scraped first, so stopping early still leaves the most relevant data.

## SQLite Schema (As Built)

```sql
-- One row per fencer ever observed (whether or not we scraped their history)
CREATE TABLE fencers (
    id              INTEGER PRIMARY KEY,           -- fencingtracker fencer ID (long modern or short legacy)
    name            TEXT    NOT NULL,              -- "Francesca Farias"
    slug            TEXT,                          -- "Francesca-Farias" (URL slug; NULL for legacy)
    club            TEXT,                          -- club at last observation
    has_profile     INTEGER NOT NULL DEFAULT 1,    -- 0 for legacy fencers with no /p/ URL
    bfs_depth       INTEGER,                       -- 0 = focal, 1 = focal's opponents, … (bootstrap BFS; informational)
    scrape_hops     INTEGER NOT NULL DEFAULT 0,    -- remaining expansion hops (Phase G); >=1 = queued
    gender          TEXT,                          -- 'M'|'W' from single-gender events; NULL if mixed/unknown (Phase G male-pruning)
    scrape_status   TEXT    NOT NULL DEFAULT 'discovered',
                                                   -- 'discovered'|'in_progress'|'done'|'error'|'skipped'
    history_pages   INTEGER,
    last_scraped_at TEXT,
    discovered_at   TEXT    NOT NULL,
    error_message   TEXT
);

-- One row per event (a single weapon/age/gender competition at a tournament)
CREATE TABLE events (
    id              INTEGER PRIMARY KEY,
    name            TEXT,                          -- "Olympia D'Artagnan's Challenge 5B"
    classification  TEXT,                          -- "Unrated Y-14 Women's Épée"
    weapon          TEXT,                          -- 'epee' | 'foil' | 'saber'
    gender          TEXT,                          -- 'M' | 'W' | 'X'
    age_group       TEXT,                          -- 'Y10' | 'Y12' | 'Y14' | 'Cadet' | 'Junior' | …
    rating_level    TEXT,                          -- 'U' | 'A' | 'B' | 'C' | 'D' | 'E' | …
    event_date      TEXT,                          -- ISO date if parsable
    raw_date        TEXT,
    first_seen_at   TEXT NOT NULL
);

-- One row per bout, canonicalized (fencer_a_id < fencer_b_id)
CREATE TABLE bouts (
    event_id        INTEGER NOT NULL,
    fencer_a_id     INTEGER NOT NULL,              -- always the lower ID
    fencer_b_id     INTEGER NOT NULL,              -- always the higher ID
    bout_type       TEXT    NOT NULL,              -- 'Pool' | 'T64' … 'T2'
    bout_seq        INTEGER NOT NULL DEFAULT 1,    -- >1 when a pair repeats in same type (Y-8 double RR)
    fencer_a_score  INTEGER NOT NULL,
    fencer_b_score  INTEGER NOT NULL,
    winner_id       INTEGER NOT NULL,
    source_fencer_id INTEGER NOT NULL,             -- whose history this row came from (provenance)
    PRIMARY KEY (event_id, fencer_a_id, fencer_b_id, bout_type, bout_seq),
    FOREIGN KEY (event_id)    REFERENCES events(id),
    FOREIGN KEY (fencer_a_id) REFERENCES fencers(id),
    FOREIGN KEY (fencer_b_id) REFERENCES fencers(id),
    FOREIGN KEY (winner_id)   REFERENCES fencers(id)
);

-- One row per (fencer, event) pair — per-event metadata for a fencer
CREATE TABLE fencer_event_results (
    fencer_id       INTEGER NOT NULL,
    event_id        INTEGER NOT NULL,
    seed            INTEGER,
    placement       INTEGER,                       -- numeric from "6 of 9"
    field_size      INTEGER,                       -- the "of 9" part
    rating_earned   TEXT,
    PRIMARY KEY (fencer_id, event_id),
    FOREIGN KEY (fencer_id) REFERENCES fencers(id),
    FOREIGN KEY (event_id)  REFERENCES events(id)
);

-- Audit log of scrape operations
CREATE TABLE scrape_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fencer_id       INTEGER,
    url             TEXT NOT NULL,
    status_code     INTEGER,
    bouts_added     INTEGER,
    fencers_added   INTEGER,
    duration_ms     INTEGER,
    error_message   TEXT,
    started_at      TEXT NOT NULL,
    FOREIGN KEY (fencer_id) REFERENCES fencers(id)
);
```

**Key design choices:**
- `bouts` PK is `(event_id, fencer_a, fencer_b, bout_type, bout_seq)` — natural dedup when the same bout appears in both fencers' histories; `bout_seq` disambiguates a pair that meets twice in one event's pool round.
- Canonical ordering `fencer_a_id < fencer_b_id` removes perspective ambiguity.
- `fencers` doubles as the BFS state table — `scrape_status='discovered'` *is* the queue.
- **Notably absent**: no strength/rating columns. fencingtracker's strength numbers are deemed unreliable; custom metrics will come from the Phase E pipeline and be stored in their own tables.

## Tech Stack

**Scraper (Phases A–C):**
- **Python 3** (pyenv virtualenv `fencing`).
- **`requests`** for HTTP, **`beautifulsoup4`** + **`lxml`** for parsing.
- **`sqlite3`** from stdlib — no ORM; parameterized SQL.
- **`tenacity`** for retry-with-backoff, **`tqdm`** for progress.

**Explorer (Phase D):**
- **`streamlit`** — the app framework.
- **`altair`** — charts (clean, native Streamlit integration).
- **`pandas`** — in-memory data handling.

All deps in `requirements.txt`.

## Project Structure

```
Fencing/
├── PLAN.md
├── requirements.txt
├── fencing.db                      # SQLite database (gitignored)
├── .cache/                         # cached HTML responses (gitignored)
├── .streamlit/
│   └── config.toml                 # explorer theme            (NEW — Phase D)
├── fencing_tracker/                # the scraper (Phases A–C, done)
│   ├── __init__.py · __main__.py
│   ├── db.py                       # schema, connection, helpers
│   ├── http.py                     # HTTP client: rate limit, cache, retries
│   ├── parsers.py                  # history-page HTML parser
│   ├── scraper.py                  # single-fencer scrape → DB (shared primitive)
│   ├── bfs.py                      # BFS controller, cap logic (bootstrap `scrape`)
│   ├── upcoming.py                 # upcoming events + field rosters  (NEW — Phase F)
│   ├── frontier.py                 # capped 2-hop expansion of new fencers (NEW — Phase G)
│   ├── updater.py                  # idempotent 3-stage `update`        (NEW — Phase G)
│   └── cli.py                      # init-db / scrape / upcoming / update / status
├── explorer/                       # the Streamlit app          (NEW — Phase D)
│   ├── __init__.py
│   ├── app.py                      # page config, sidebar, tab routing
│   ├── data.py                     # read-only DB load + cached query helpers
│   └── charts.py                   # Altair chart builders
├── scripts/
│   ├── init_db.py · scrape_focal.py
└── tests/
    ├── fixtures/francesca_history_p1.html
    ├── fixtures/francesca_summary.html       # Registrations section   (NEW — Phase F)
    ├── fixtures/event_18164_roster.html      # preregistration roster  (NEW — Phase F)
    ├── test_parsers.py
    ├── test_upcoming.py                       # registration/roster parsers (NEW — Phase F)
    └── test_update_logic.py                   # hops queue + refresh targets (NEW — Phase G)
```

## Scraping Etiquette

- 1.5s delay between requests; local HTML cache in `.cache/` keyed by URL hash.
- Polite User-Agent identifying a personal research bot + contact email.
- Retry with exponential backoff on 5xx / connection errors.
- Idempotent and resumable via `scrape_status`.

## Phased Implementation

**Phase A — Schema + focal scrape.** ✅ Done. Parser built against a saved fixture of Francesca's history; verified 220 bouts / 29 events.

**Phase B — BFS expansion to 1000 fencers.** ✅ Done. 1,002 fencers scraped (depth 0: 1, depth 1: 103, depth 2: 898); 5,907 events; 206,463 bouts. Data-quality cases handled along the way (double round-robin → `bout_seq`; legacy IDs → `has_profile=0`; anonymous opponents → skipped; self-bouts → skipped).

**Phase C — Hardening.** ✅ Done. 8 parser tests against the fixture; CLI with `init-db`, `scrape`, `status`.

**Phase D — Explorer (Streamlit).** ◻ Next. A fencer-centric interactive explorer — full design in the next section.

**Phase E — Custom analytics pipeline.** ◻ Later. A separate pipeline computing our own rating/strength metrics from the bout graph, written to dedicated tables. The explorer is designed to surface those outputs once they exist (see Forward-compatibility below). With Phase F in place, a key Phase E deliverable is **estimating Francesca's performance against the field of a specific upcoming event** — for each registered opponent, derive a win expectation from our own metrics (direct head-to-head, common-opponent paths, our rating) and roll those up into a projected placement distribution.

**Phase F — Upcoming events + fields.** ✅ Done. See section below.

## Upcoming Events (Phase F)

### Purpose

Pull the events a fencer is **registered for but hasn't fenced yet**, and capture **who else is in the field**, so Phase E can estimate performance against that field.

Run with: `python -m fencing_tracker upcoming` (defaults to Francesca; `--focal ID --slug SLUG` for anyone).

### How the site exposes it (discovered during build)

- A fencer's **summary** page (`/p/{id}/{slug}`, not `/history`) has a **"Registrations"** section listing upcoming events. Each links to `/event/{event_id}` (note: **no `/results` suffix** — that's how upcoming is distinguished from past events, which link to `/event/{id}/results`).
- **`/event/{event_id}`** is the **preregistration roster**: event metadata (tournament, date, venue) plus a table of every registered fencer, each linking to `/p/{id}/{slug}`. The field is fully identifiable.
- **Important:** the preregistration `event_id` namespace (e.g. `18164`) is **distinct** from the historical `events.id` namespace (`/event/{id}/results`, e.g. `41045`) — so upcoming events live in their own tables and don't join to `events`.
- The roster page also shows fencingtracker's Strength / Conservative-Estimate / Rank columns. Per project policy these are **deliberately not captured** — that ranking is `mu − 3·sigma` of their own model. We keep only factual identity (fencer id / name / club).

### What it does

1. Fetch the focal fencer's summary page → parse the Registrations section (`parse_registrations`).
2. For each upcoming event, fetch the roster (`parse_event_roster`) → upsert `upcoming_events` + replace `upcoming_event_registrants`; every field member is `ensure_fencer()`'d into the graph.
3. **Fill gaps** (default on; `--no-fill-gaps` to skip): scrape the full history of any field member we don't already hold (`scrape_status != 'done'`), without expanding the BFS — so Phase E has complete matchup data for the whole field.

Summary + roster pages are fetched fresh (`use_cache=False`) since registrations move; gap histories reuse the on-disk cache.

### Schema (new tables)

```sql
upcoming_events(event_id PK, tournament_name, event_name, classification,
                weapon, gender, age_group, venue, location, start_datetime,
                event_date, field_size, first_seen_at, last_scraped_at)

upcoming_event_registrants(event_id, fencer_id, name, club,
                           PRIMARY KEY (event_id, fencer_id))   -- the field; focal included
```

"Events Francesca is in" = `upcoming_event_registrants WHERE fencer_id = <her id>`.

### Verified

- Francesca: 1 registration — *Summer Nationals and July Challenge, Youth 10 Women's Épée, 2026-07-05, Portland OR*, **107 registered (106 distinct; fencingtracker lists one twice)**.
- After fill-gaps: histories held for **106/106** distinct field members (+38 new histories, +1,272 bouts).
- Parsers covered by `tests/test_upcoming.py` (6 tests, incl. a guard that no strength columns are captured).

## Continual Updates (Phase G)

### Purpose

Replace the one-shot "BFS to a cap of 1000" with a single **idempotent `update`** that can be run repeatedly (e.g. on a schedule). Every run derives its work from DB state, so re-running just converges.

Run with: `python -m fencing_tracker update` (defaults to Francesca). The original `scrape` (BFS-to-cap) is kept purely for **bootstrapping a fresh DB**.

### Facts that shaped it (verified during build)

- **History pages are not paginated.** `/history?page=N` is ignored — the page renders a fencer's *entire* history at once (confirmed on a 203-event / 1,657-bout fencer: pages 1–3 byte-identical). One fetch is complete.
- A completed event appears on a fencer's **history** page under the **results id namespace** (`/event/{id}/results`), which is *different* from the preregistration id (`/event/{id}`). So "events that happened" are detected by re-scraping histories, not by revisiting the upcoming-event page.
- The HTTP cache is **permanent** (keyed by URL hash, no TTL). A refresh must therefore pass `use_cache=False` to actually see new events.

### The three stages of `update`

0. **Refresh upcoming** (`upcoming.refresh_upcoming`) — re-pull the focal's Registrations and each future-dated roster (fresh). Past-dated events are skipped (their results come via Stage 1). New/undone field members are *seeded* for expansion.
1. **Catch completed events** (`updater`) — re-fetch the histories of already-scraped fencers that are **stale** (`last_scraped_at` older than `--refresh-after-days`, default 14; the focal always), **cache-bypassed**. `insert_bout` is idempotent, so only genuinely new bouts land. New opponents are seeded for expansion.
2. **Expand the frontier** (`frontier.expand_frontier`) — scrape the bounded neighbourhood around new fencers, capped at `--max-new` per run (default 500); the remainder stays queued for the next run.

### The expansion model (the "depth-2" rule)

Each fencer carries a **hop budget** in a new `fencers.scrape_hops` column. Encountering a *new* fencer seeds budget = `--expand-hops` (default 2). Expanding a fencer with budget `h`:
- scrapes their history, and
- tags the *new* opponents that introduces with budget `h-1`.

So with seed 2: a new fencer's bouts → their new opponents' bouts → (those opponents' new opponents are merely recorded, budget 0). `scrape_hops` is raised, never lowered, and never applied to a `done` fencer (those are refreshed by staleness, not re-expanded). The queue is `scrape_status='discovered' AND scrape_hops>=1`, drained most-hops-first.

### Male-pruning

The focal fencer (Francesca) competes in Women's and some Mixed youth events, so the men's circuit is irrelevant except for her direct opponents. Each fencer carries an inferred `gender` (a new `fencers.gender` column): `'M'`/`'W'` from the *single-gender* events they appear in (Men's-only → M, Women's-only → W), `NULL` when mixed-only or unknown — discovered opponents are tagged from the discovering event's gender (`scraper.py`), and `db.backfill_gender()` tags the existing dataset.

Both expansion (`next_to_expand`) and the staleness refresh (`fencers_to_refresh`) apply a gate (`db._KEEP_GENDER_SQL`): **keep women, keep unknown/NULL (never drop a woman), keep any fencer with a direct bout vs the focal; skip known-males otherwise.** Applied at query time so it re-evaluates as gender data improves. This stops the cascade into the men's graph while preserving Francesca's mixed-event male opponents. On the seeded dataset it cut the first refresh from ~1,043 to ~785 fencers (258 unrelated males skipped).

The per-run expansion cap (`--max-new`) defaults to **2000**. (There is no global 1000 cap in `update`; that cap belongs only to the bootstrap `scrape`/BFS.)

### Idempotency / safety

- All targets come from DB state (stale timestamps, pending hops, past-date events) → safe to re-run; resumable after a crash (`reset_in_progress`).
- The per-run cap bounds a single run after a big roster drop; leftover work is picked up next run.
- `scrape` (bootstrap, BFS+depth) and `update` (ongoing, hops+staleness) share one low-level primitive, `scraper.scrape_fencer_history`.

### Verified

- `update --refresh-after-days 3650 --max-new 10`: Stage 0 re-pulled the 106-fencer field, Stage 1 refreshed the focal, Stage 2 idle — all idempotent.
- Expansion integration: seeding one `discovered` fencer at hops 2 and running `expand_frontier(max_new=3)` scraped 3 fencers (+989 bouts, 501 discovered), left 89 at hops 1, and reported `capped=True` — confirming the decrement chain and the cap.
- Mechanics covered by `tests/test_update_logic.py` (7 tests: seeding rules, never-lower, done-immunity, queue ordering, 2-hop decrement, refresh-target selection).

## Explorer Design (Phase D)

### Purpose

An interactive, **fencer-centric** Streamlit app over `fencing.db`. You fix a focal fencer (default: Francesca Farias) and seamlessly explore their competitive record — overall, over time, and head-to-head against any other fencer. This phase is **purely factual** exploration of scraped bouts; no modeled metrics (those come from Phase E).

Run with: `streamlit run explorer/app.py`

### Layout

**Sidebar (persistent):**
- **Focal fencer picker** — searchable selectbox over the 1,002 fully-scraped fencers (they have complete histories); default Francesca Farias.
- **Focal card** — compact summary below the picker: name, club, total bouts, overall win %, record date span.
- **Global filters** applied across every tab: weapon, season / date range, bout type (Pool vs DE), event age group.

**Main area — four tabs:**

**1. Overview** — the focal fencer's dashboard
- KPI cards (`st.metric`): total bouts, wins, losses, win %, distinct opponents, events entered.
- Win rate over time (by season).
- Win % by weapon; Pool vs DE split.
- Touch differential: distribution of score margins; touches scored vs. received.
- Recent form: last ~15 bouts as a colored W/L strip.
- Placement history: finishing position vs. field size, per event.

**2. Head-to-Head** — the core feature
- **Opponent picker** — searchable; any fencer in the DB. (Because the focal is fully scraped, *every* focal-vs-opponent bout is present regardless of whether the opponent was scraped.)
- **Scorecard** — large "Focal  W – L  Opponent" record, win %, average touches for/against, longest win streak.
- **Bout log** — every bout: date, event, classification, bout type, score, result — with win/loss row coloring.
- **Rivalry timeline** — cumulative head-to-head margin over time.
- **Common-opponent fallback** — when the pair has never actually fenced, show a factual comparison through their shared opponents ("across N shared opponents: Focal X% vs Opponent Y%"). This is a simple counted statistic over the bout graph, not a rating model. *(Easy to drop if you'd rather keep all comparison logic in the Phase E pipeline.)*

**3. Opponents** — opponent leaderboard
- One row per opponent the focal has faced: bouts, W–L, win %, last met, average margin.
- Sortable / searchable; selecting an opponent opens their Head-to-Head (seamless cross-linking).
- Highlight panels: toughest opponents (lowest win %, min 3 bouts), most-frequent rivals, best/worst results.

**4. Events** — competition history
- One row per event entered: date, name, classification, seed, placement / field size, bouts W–L.
- Expand an event to see all of the focal's bouts in it.

### Visual design ("reasonably pretty")

- Wide layout; custom page title + icon.
- Custom theme via `.streamlit/config.toml` — restrained palette; consistent **win = green / loss = red** accents everywhere.
- KPI cards via `st.metric`; tables via `st.dataframe` with `column_config` (progress bars for win %, colored numerics).
- Charts via Altair with consistent color encoding for W/L; captions and headers for typographic hierarchy. No emojis.

### Tech & structure

- New deps: `streamlit`, `altair`, `pandas`.
- `explorer/app.py` — page config, sidebar, tab routing.
- `explorer/data.py` — opens `fencing.db` read-only, loads core tables into pandas once, wrapped in `@st.cache_data`; provides query/aggregation helpers. The DB is small (~206 K bouts) so in-memory filtering keeps interactions snappy.
- `explorer/charts.py` — Altair chart builders.
- `.streamlit/config.toml` — theme.
- Connection reuses `fencing_tracker.db.connect()`.

### Forward-compatibility with Phase E

The explorer reads `fencing.db` only. When the Phase E pipeline lands and writes computed metrics into new tables (e.g. a `fencer_ratings` table), the explorer can surface them — a "Rating" line on Overview, a strength comparison on Head-to-Head — without reworking the factual core. Phase D ships without any dependency on Phase E.

### Verification

- `streamlit run explorer/app.py` loads with Francesca as focal.
- Overview shows Francesca = 220 bouts.
- Head-to-Head vs. Alexa Cahalane (id 100357694): 14 bouts, including the known 2–5 pool loss in event 41045.
- Switching the focal fencer repopulates every tab.
- Applying a weapon filter drops counts consistently across tabs.
- Pick an opponent the focal never fenced → common-opponent panel appears.

## Decisions

1. **Cap semantics**: Stop after 1000 fencers reach `scrape_status='done'`. Bouts against opponents beyond the cap are still recorded (those opponents stay in `fencers` as `discovered`).
2. **Bout uniqueness**: PK `(event_id, fencer_a, fencer_b, bout_type, bout_seq)`. Y-8 events use a double round-robin pool format, so `bout_seq` is required; computed by row order within the scraped history.
3. **Tournament grouping**: `events` only. No `tournaments` table.
4. **Strength data**: fencingtracker's strength ratings and win-probability columns are **not captured** — deemed unreliable. All modeled metrics will come from the separate Phase E analytics pipeline, built on the raw bout graph and stored in their own tables.
5. **Deps**: `pip + requirements.txt`.
6. **Legacy fencers**: Older bouts list opponents with 4-5 digit IDs and no `/p/` profile URL. We capture these bouts but mark the fencer `has_profile=0` / `scrape_status='skipped'` so BFS skips them.
7. **Anonymous opponents**: A small fraction of older bouts list opponents with no ID at all. These bouts are skipped — no identifier to record.
8. **Self-bouts**: fencingtracker occasionally lists a fencer against themselves (data-entry errors). Skipped.
9. **Explorer scope**: Phase D is factual exploration only. The focal fencer is any of the 1,002 fully-scraped fencers (default Francesca); the head-to-head opponent can be any fencer in the DB.

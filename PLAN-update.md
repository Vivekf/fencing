# Plan: event-driven incremental update

Status: **in progress** (parser first). Owner: data pipeline.

## Goal

Replace the blunt "refresh every stale fencer" sweep with an **event-driven** update
that only touches fencers who actually competed since the last run — the efficient
dual of "update fencers who had an event." Then run it on a cron.

## Why the obvious approaches don't work (verified against the live site)

- **Bouts are only on fencer *history* pages** (`/p/{id}/history`); there is no
  event-results ingestion today. So the current refresh re-pulls every stale fencer
  (~1–2k requests, 1–3 h) even though almost none have new bouts.
- **"When an upcoming event's date passes, fetch its results" fails.** Preregistration
  event IDs and historical (results) event IDs are **different namespaces that collide**:
  `/event/18164` is the 2026 Y10 Summer Nationals prereg roster, but
  `/event/18164/results` is an unrelated *2023 Y12 Bluegrass* event. There is **no link**
  from a prereg/tournament page to the results ID.
- **The results ID (e.g. 41875) is only discoverable from a participant's history page.**

## The design we're building

Event-centric ingestion, triggered by our own upcoming-events table:

1. **Track signups via a core-registration scan (Option B, decided).** Pull each **core**
   fencer's summary page → their registrations (`parse_registrations`, already carries
   fencingtracker IDs + event links). This is a per-fencer pull, but **scoped to the core**
   (hundreds, ~10–20 min, repeatable) — not all 35k fencers. Store each upcoming event with
   its date + registrant list. Rejected: enumerating "all US events" via **AskFRED** — it
   would need a new data source *and* a fuzzy AskFRED-name→fencingtracker-ID identity match,
   which fencingtracker already solves for us on its own roster pages.
2. **Resolve the results ID.** When an event's date passes, scrape **one** known
   participant's history — the focal if she's entered — which surfaces the historical
   results ID and matches it to the prereg event by (tournament, event name, date).
3. **Ingest the whole field in one request.** Fetch `/event/{results_id}/results` and parse
   the full field's bouts. One request refreshes **all ~100 fielded fencers at once**.
4. **New fencers → existing frontier criteria.** Newly-seen participants go through the
   current connectivity/youth gate (`frontier`), unchanged.

**Coverage: focal + active core** (decided, via Option B). The core-registration scan gives
complete coverage of the core's signups with fencingtracker IDs throughout (no identity
mapping). The monthly staleness sweep is retained as a backstop.

## KEY DECISION — DE round fidelity (needs sign-off)

The results page encodes each bout only as **Pool** or **DE**, with score + opponent name
(`data-bs-title="5:1 vs. LOUVOT Chloe · Very Easy"`). It does **not** carry the DE round
(T64…T2) that history pages give. Since `bouts` PK includes `bout_type`, a DE bout ingested
as `'DE'` would not dedup against the same bout previously stored as `'T8'`.

**Proposed policy:** treat the results page as the **authoritative single source per event**.
- Mark events `results_ingested_at`; do not also insert that event's bouts from history.
- DE bouts from results are stored as `'DE'` (Pool vs DE is what the analytics actually key
  on: `is_de = bout_type != 'Pool'`). Legacy events keep their finer T-rounds; new events are
  coarser. Acceptable for youth scouting.
- Follow-up (optional): check for a per-event pools/tableau endpoint that carries rounds; if
  found, upgrade DE granularity without changing this pipeline.

## Code changes

1. ✅ **DONE** `parsers.parse_event_results(html, event_id)` → participants + symmetric
   bouts (Pool/DE). Verified: 104 fencers, 411 bouts, 0 unresolved on the 41875 fixture.
2. ✅ **DONE** `scraper.scrape_event_results(conn, client, event_id)` — upserts event,
   `ensure_fencer` for the field, `insert_bout`, records placements, sets
   `results_ingested_at`. Tested incl. idempotency (re-ingest adds 0).
3. ✅ **DONE** `db.py` — `events.results_ingested_at` column + migration;
   `set_event_results_ingested()` / `event_results_ingested()`.
4. **TODO** `updater.py` — core-registration scan (Option B) → date-triggered queue →
   resolve results IDs via a participant's history → `scrape_event_results` → frontier.
   Needs a `db.events_needing_results(...)` / watch-list query + a `last_update_at` watermark.
5. **TODO** `cli.py` / `frontier.py` — `--mode event-driven|sweep`; **fix the `--max-new 0`
   footgun** (currently 0 = unlimited; make 0 = off).
6. ✅ **DONE (partial)** `tests/test_event_results.py` — parser + ingestion + idempotency.
   Still to add: new-fencer-from-results respects frontier criteria (with updater).

## Test evidence captured

- `/event/41875/results` = 200, 408 KB, one row per fencer, **104 fencers** from a single
  request; bout-cells give score + opponent name + Pool/DE + V/D.
- Fixture saved at `tests/fixtures/event_41875_results.html`.

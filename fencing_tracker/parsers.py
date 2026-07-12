"""HTML parsers for fencingtracker.com.

Currently parses the fencer history page (`/p/{id}/{slug}/history`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)


# -- Data classes ---------------------------------------------------------------

@dataclass
class ParsedBout:
    bout_type: str                # 'Pool', 'T64', 'T32', 'T16', 'T8', 'T4', 'T2'
    opponent_id: int
    opponent_name: str            # raw "LASTNAME firstname" from anchor text
    opponent_slug: Optional[str]  # from URL (None for legacy fencers)
    opponent_club: Optional[str]
    opponent_has_profile: bool    # False for legacy 4-5 digit IDs (not scrapable)
    focal_score: int              # focal fencer's score
    opp_score: int                # opponent's score
    focal_won: bool


@dataclass
class ParsedEvent:
    event_id: int
    tournament_name: Optional[str]
    classification: Optional[str]      # e.g. "Unrated Y-14 Women's Épée"
    weapon: Optional[str]              # 'epee' | 'foil' | 'saber'
    gender: Optional[str]              # 'M' | 'W' | 'X'
    age_group: Optional[str]           # 'Y10' | 'Y12' | 'Y14' | 'Cadet' | 'Junior' | 'Senior' | 'Vet' | None
    rating_level: Optional[str]        # 'U' | 'A' | 'B' | 'C' | 'D' | 'E' | None
    event_date: Optional[str]          # ISO date 'YYYY-MM-DD'
    raw_date: Optional[str]            # original string e.g. 'May 17, 2026'
    focal_seed: Optional[int]
    focal_placement: Optional[int]
    focal_field_size: Optional[int]
    focal_rating: Optional[str]
    bouts: List[ParsedBout] = field(default_factory=list)
    skipped_bouts: int = 0             # rows we couldn't tie to an opponent


@dataclass
class UpcomingRegistration:
    """A row in the 'Registrations' section of a fencer's summary page."""
    event_id: int                      # fencingtracker preregistration /event/{id}
    tournament_name: Optional[str]
    event_name: Optional[str]          # e.g. "Youth 10 Women's Epee (Y10WE)"
    event_date: Optional[str]          # ISO 'YYYY-MM-DD'
    raw_date: Optional[str]            # e.g. "Jul 5"


@dataclass
class RosterEntry:
    """One registered fencer in an upcoming event's preregistration roster."""
    fencer_id: int
    name: str                          # normalized "First Last"
    slug: Optional[str]
    club: Optional[str]


@dataclass
class EventRoster:
    """Parsed /event/{id} preregistration page: event meta + the field."""
    event_id: Optional[int]
    tournament_name: Optional[str]
    event_name: Optional[str]
    classification: Optional[str]
    weapon: Optional[str]
    gender: Optional[str]
    age_group: Optional[str]
    venue: Optional[str]
    location: Optional[str]
    start_datetime: Optional[str]      # raw "Sunday, July 5, 2026 at 2:00 PM"
    event_date: Optional[str]          # ISO 'YYYY-MM-DD'
    entries: List[RosterEntry] = field(default_factory=list)


# -- Regexes --------------------------------------------------------------------

ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PERSON_HREF_RE = re.compile(r"^/p/(\d+)/([^/]+)")
EVENT_ID_RE = re.compile(r"history-event-(\d+)")
SCORE_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")
LEGACY_ID_TRAILING_RE = re.compile(r"\b(\d+)\s*$")
PLACE_OF_RE = re.compile(r"Place\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
SEED_RE = re.compile(r"Seed\s+(\d+)", re.IGNORECASE)
RATING_RE = re.compile(r"Rating\s+(\S+)", re.IGNORECASE)
UPCOMING_EVENT_HREF_RE = re.compile(r"^/event/(\d+)/?$")  # no "/results" -> preregistration
RANKING_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")  # data-ranking-value="20260705"
LONG_DATE_RE = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")  # "July 5, 2026"


# -- Helpers --------------------------------------------------------------------

def _text(node: Optional[Tag]) -> Optional[str]:
    if node is None:
        return None
    s = node.get_text(strip=True)
    return s or None


def _classify(classification: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse 'Unrated Y-14 Women's Épée' -> (weapon, gender, age_group).

    rating_level comes from the meta chip, not this string.
    """
    if not classification:
        return None, None, None

    weapon = None
    if re.search(r"Épée|Epee|Epée", classification, re.IGNORECASE):
        weapon = "epee"
    elif re.search(r"Foil", classification, re.IGNORECASE):
        weapon = "foil"
    elif re.search(r"Saber|Sabre", classification, re.IGNORECASE):
        weapon = "saber"

    gender = None
    if re.search(r"Women|Woman", classification, re.IGNORECASE):
        gender = "W"
    elif re.search(r"Men|Man", classification, re.IGNORECASE):
        gender = "M"
    elif re.search(r"Mixed", classification, re.IGNORECASE):
        gender = "X"

    age_group = None
    m = re.search(r"Y-?(\d+)", classification, re.IGNORECASE)
    youth_m = re.search(r"Youth\s*(\d+)", classification, re.IGNORECASE)
    if m:
        age_group = f"Y{m.group(1)}"
    elif youth_m:
        # Preregistration event names spell it out: "Youth 10 Women's Epee (Y10WE)"
        age_group = f"Y{youth_m.group(1)}"
    elif re.search(r"Cadet", classification, re.IGNORECASE):
        age_group = "Cadet"
    elif re.search(r"Junior", classification, re.IGNORECASE):
        age_group = "Junior"
    elif re.search(r"Senior", classification, re.IGNORECASE):
        age_group = "Senior"
    elif re.search(r"Vet|Veteran|V40|V50|V60|V70|V80", classification, re.IGNORECASE):
        age_group = "Vet"
    elif re.search(r"Div(?:ision)?\s*[I123]", classification, re.IGNORECASE):
        age_group = "Senior"

    return weapon, gender, age_group


def _parse_rating_level(meta_chip_text: Optional[str]) -> Optional[str]:
    """Meta chip looks like 'U, Y14' or 'B, Sr' — first token is rating level."""
    if not meta_chip_text:
        return None
    first = meta_chip_text.split(",")[0].strip()
    if first in {"U", "A", "B", "C", "D", "E"}:
        return first
    return first or None


def _parse_iso_date(blob: Optional[str]) -> Optional[str]:
    if not blob:
        return None
    m = ISO_DATE_RE.search(blob)
    return m.group(1) if m else None


def _parse_raw_date(meta_chips: List[str]) -> Optional[str]:
    """The date chip looks like 'May 17, 2026'."""
    for chip in meta_chips:
        try:
            datetime.strptime(chip, "%B %d, %Y")
            return chip
        except ValueError:
            continue
    return None


def _to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = s.strip()
    if not s or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


# -- Main entrypoint ------------------------------------------------------------

def parse_history(html: str) -> List[ParsedEvent]:
    """Parse a fencer's /history page into a list of events with bouts."""
    soup = BeautifulSoup(html, "lxml")
    sections = soup.select("section.person-history__event-panel")
    events: List[ParsedEvent] = []
    for section in sections:
        try:
            events.append(_parse_event_section(section))
        except Exception as exc:
            log.warning("Failed to parse event section: %s", exc, exc_info=True)
            continue
    return events


def _parse_event_section(section: Tag) -> ParsedEvent:
    sec_id = section.get("id", "")
    m = EVENT_ID_RE.match(sec_id)
    if not m:
        raise ValueError(f"Could not extract event id from section id={sec_id!r}")
    event_id = int(m.group(1))

    tournament_name = _text(section.find("h2"))

    event_link = section.select_one("a.person-history__event-link")
    classification = _text(event_link)

    # Meta chips: the chips on the kicker line. First is the event link (skip),
    # then a "U, Y14" style chip, then a date chip like "May 17, 2026".
    meta_chips = [
        _text(chip)
        for chip in section.select(".person-history__meta-chip")
        if "person-history__meta-chip--event" not in (chip.get("class") or [])
    ]
    meta_chips = [c for c in meta_chips if c]

    rating_level_chip = meta_chips[0] if len(meta_chips) >= 1 else None
    raw_date = _parse_raw_date(meta_chips)

    event_date = _parse_iso_date(section.get("data-person-history-event-search"))
    weapon, gender, age_group = _classify(classification)
    rating_level = _parse_rating_level(rating_level_chip)

    # Fact chips: "Place 6 of 9", "Seed 7", "Rank 18", "Rating None", "{club}"
    fact_text = " ".join(
        t for t in (_text(c) for c in section.select(".person-history__fact-chip")) if t
    )
    place_m = PLACE_OF_RE.search(fact_text)
    seed_m = SEED_RE.search(fact_text)
    rating_m = RATING_RE.search(fact_text)
    focal_placement = int(place_m.group(1)) if place_m else None
    focal_field_size = int(place_m.group(2)) if place_m else None
    focal_seed = int(seed_m.group(1)) if seed_m else None
    focal_rating = None
    if rating_m:
        val = rating_m.group(1)
        focal_rating = None if val.lower() in {"none", "-", ""} else val

    bouts, skipped = _parse_bouts(section)

    return ParsedEvent(
        event_id=event_id,
        tournament_name=tournament_name,
        classification=classification,
        weapon=weapon,
        gender=gender,
        age_group=age_group,
        rating_level=rating_level,
        event_date=event_date,
        raw_date=raw_date,
        focal_seed=focal_seed,
        focal_placement=focal_placement,
        focal_field_size=focal_field_size,
        focal_rating=focal_rating,
        bouts=bouts,
        skipped_bouts=skipped,
    )


def _parse_bouts(section: Tag) -> tuple[List[ParsedBout], int]:
    """Parse all bout rows in an event section.

    Returns (bouts, skipped_count). `skipped_count` reflects rows we could not
    associate with an opponent (no anchor, no legacy ID) — typical of older
    events with unidentified fencers.
    """
    rows = section.select("table.person-history__event-table tbody tr")
    bouts: List[ParsedBout] = []
    skipped = 0
    for row in rows:
        try:
            bouts.append(_parse_bout_row(row))
        except ValueError as exc:
            log.debug("Skipping bout row: %s", exc)
            skipped += 1
        except Exception as exc:
            log.warning("Failed to parse bout row: %s", exc, exc_info=True)
            skipped += 1
    return bouts, skipped


def _parse_bout_row(row: Tag) -> ParsedBout:
    bout_cell = row.select_one("td.person-history__bout-col")
    bout_type = (bout_cell or {}).get("data-ranking-text") if bout_cell else None
    if not bout_type:
        raise ValueError("Missing bout type")

    result_cell = row.select_one("td.person-history__result-col")
    result_text = (result_cell or {}).get("data-ranking-text") if result_cell else None
    focal_won = (result_text or "").lower().startswith("v")

    score_cell = row.select_one("td.person-history__score-col")
    score_text = (
        (score_cell.get("data-ranking-text") if score_cell else None)
        or _text(score_cell)
    )
    sm = SCORE_RE.match(score_text or "")
    if not sm:
        raise ValueError(f"Could not parse score {score_text!r}")
    focal_score = int(sm.group(1))
    opp_score = int(sm.group(2))

    opp_cell = row.select_one("td.person-history__opponent-col")
    if opp_cell is None:
        raise ValueError("Missing opponent cell")

    anchor = opp_cell.select_one("a[href^='/p/']")
    if anchor is not None:
        href = anchor.get("href") or ""
        hm = PERSON_HREF_RE.match(href)
        if not hm:
            raise ValueError(f"Bad opponent href {href!r}")
        opponent_id = int(hm.group(1))
        opponent_slug = hm.group(2)
        opponent_name = _text(anchor) or ""
        has_profile = True
    else:
        # Anchor-less row: either a legacy 4-5 digit ID (recoverable from
        # data-ranking-search) or "Missing ID" (truly unidentified — skip).
        rs = row.get("data-ranking-search") or ""
        lm = LEGACY_ID_TRAILING_RE.search(rs)
        if not lm:
            raise ValueError("Opponent has no anchor and no legacy ID")
        opponent_id = int(lm.group(1))
        opponent_slug = None
        opponent_name = (opp_cell.get("data-ranking-text") or _text(opp_cell) or "").strip()
        # Strip the trailing "Missing ID" marker if present (shouldn't be, since we
        # required a trailing int — but defensive).
        opponent_name = opponent_name.replace("Missing ID", "").strip()
        has_profile = False

    club_cell = row.select_one("td.person-history__club-col")
    opponent_club = None
    if club_cell is not None:
        opponent_club = club_cell.get("data-ranking-text") or _text(club_cell)
        if opponent_club in {"-", ""}:
            opponent_club = None

    return ParsedBout(
        bout_type=bout_type,
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        opponent_slug=opponent_slug,
        opponent_club=opponent_club,
        opponent_has_profile=has_profile,
        focal_score=focal_score,
        opp_score=opp_score,
        focal_won=focal_won,
    )


def _normalize_name(raw: Optional[str]) -> Optional[str]:
    """Roster names are 'Last, First'; the rest of the DB uses 'First Last'."""
    if not raw:
        return raw
    raw = raw.strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}".strip()
    return raw


def _parse_long_date(blob: Optional[str]) -> Optional[str]:
    """'Sunday, July 5, 2026 at 2:00 PM' -> '2026-07-05'."""
    if not blob:
        return None
    m = LONG_DATE_RE.search(blob)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_registrations(html: str) -> List[UpcomingRegistration]:
    """Parse the 'Registrations' section of a fencer's summary page (/p/{id}/{slug}).

    Only upcoming registrations are returned — their links are `/event/{id}` with no
    `/results` suffix (past events on the same page link to `/event/{id}/results`).
    """
    soup = BeautifulSoup(html, "lxml")
    h2 = next(
        (h for h in soup.find_all("h2") if (h.get_text(strip=True) or "").lower() == "registrations"),
        None,
    )
    if h2 is None:
        return []
    section = h2.find_parent("section") or h2.parent
    regs: List[UpcomingRegistration] = []
    seen: set[int] = set()
    for a in section.select("a[href]"):
        m = UPCOMING_EVENT_HREF_RE.match(a.get("href") or "")
        if not m:
            continue
        event_id = int(m.group(1))
        if event_id in seen:
            continue
        seen.add(event_id)
        event_name = _text(a)
        tournament_name = event_date = raw_date = None
        tr = a.find_parent("tr")
        if tr is not None:
            tds = tr.find_all("td", recursive=False) or tr.find_all("td")
            for td in tds:
                rv = (td.get("data-ranking-value") or "").strip()
                dm = RANKING_DATE_RE.match(rv)
                if dm:
                    event_date = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                    raw_date = _text(td)
            # Tournament name is the plain text cell (no link, no date value).
            for td in tds:
                if td.find("a") or td.get("data-ranking-value"):
                    continue
                tournament_name = _text(td)
                if tournament_name:
                    break
        regs.append(
            UpcomingRegistration(
                event_id=event_id,
                tournament_name=tournament_name,
                event_name=event_name,
                event_date=event_date,
                raw_date=raw_date,
            )
        )
    return regs


def parse_event_roster(html: str, event_id: Optional[int] = None) -> EventRoster:
    """Parse a /event/{id} preregistration page: event metadata + the registered field.

    Captures factual identity only (fencer id/name/club). The page's Strength and
    Conservative-Estimate columns (fencingtracker's own model) are intentionally ignored.
    """
    soup = BeautifulSoup(html, "lxml")
    event_name = _text(soup.select_one("h1"))

    # Subtitle lines may share one element separated by a pipe/bullet, e.g.
    # "Summer Nationals and July Challenge | Sunday, July 5, 2026 at 2:00 PM".
    subtitles = [t for t in (_text(p) for p in soup.select("p.ranking-subtitle")) if t]
    pieces = [p.strip() for s in subtitles for p in re.split(r"\s*[|•]\s*", s) if p.strip()]
    tournament_name = None
    start_datetime = None
    for s in pieces:
        if LONG_DATE_RE.search(s):
            start_datetime = s
        elif tournament_name is None and "·" not in s:
            tournament_name = s

    venue = location = None
    for span in soup.find_all("span"):
        txt = _text(span)
        if txt and "·" in txt:
            parts = [p.strip() for p in txt.split("·", 1)]
            venue = parts[0] or None
            location = parts[1] if len(parts) > 1 else None
            break

    # The roster is the table whose body links to fencer profiles (the other table
    # on the page is the strength distribution, which has no /p/ links).
    roster_table = None
    for table in soup.select("table"):
        if table.select_one("tbody a[href^='/p/']"):
            roster_table = table
            break

    entries: List[RosterEntry] = []
    if roster_table is not None:
        for a in roster_table.select("tbody a[href^='/p/']"):
            hm = PERSON_HREF_RE.match(a.get("href") or "")
            if not hm:
                continue
            tr = a.find_parent("tr")
            club = None
            if tr is not None:
                club_a = tr.select_one("a[href^='/club/']")
                club = _text(club_a) if club_a else None
                if club in {"-", ""}:
                    club = None
            entries.append(
                RosterEntry(
                    fencer_id=int(hm.group(1)),
                    name=_normalize_name(_text(a)) or "",
                    slug=hm.group(2),
                    club=club,
                )
            )

    weapon, gender, age_group = _classify(event_name)
    return EventRoster(
        event_id=event_id,
        tournament_name=tournament_name,
        event_name=event_name,
        classification=event_name,
        weapon=weapon,
        gender=gender,
        age_group=age_group,
        venue=venue,
        location=location,
        start_datetime=start_datetime,
        event_date=_parse_long_date(start_datetime),
        entries=entries,
    )


# -- Event results (whole-field ingestion) --------------------------------------

@dataclass
class ResultParticipant:
    """One fencer's row on a `/event/{id}/results` page."""
    fencer_id: int
    name: str                 # normalized "First Last"
    raw_name: str             # raw "SURNAME Given" as shown — used to resolve opponents
    slug: Optional[str]
    placement: Optional[int]  # final finish (the '#' column)


@dataclass
class ResultBout:
    """A symmetric bout parsed from a results page. fencer_a_id < fencer_b_id."""
    fencer_a_id: int
    fencer_b_id: int
    a_score: int
    b_score: int
    winner_id: int
    bout_type: str            # 'Pool' | 'DE' (results pages don't carry the DE round)
    bout_seq: int


@dataclass
class EventResults:
    event_id: int
    event_name: Optional[str]
    weapon: Optional[str]
    gender: Optional[str]
    age_group: Optional[str]
    event_date: Optional[str] = None     # ISO 'YYYY-MM-DD' (needed by the date-filtered model)
    raw_date: Optional[str] = None        # e.g. 'July 5, 2026'
    participants: List[ResultParticipant] = field(default_factory=list)
    bouts: List[ResultBout] = field(default_factory=list)
    skipped_bouts: int = 0    # squares whose opponent name didn't resolve to an id


# "5:1 vs. LOUVOT Chloe · Very Easy" -> (5, 1, "LOUVOT Chloe")
BOUT_TITLE_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s+vs\.?\s+(.+?)\s*[·|]", re.UNICODE)


def _pair_directed(directed: list[tuple]) -> List[ResultBout]:
    """Fold per-fencer directed bout views into symmetric bouts.

    Each physical bout appears twice (once from each fencer's row). We key by the
    unordered pair + bout_type and canonicalise to fencer_a_id < fencer_b_id, using
    the low-id fencer's view when present (falling back to the high-id view). Repeats
    of the same pair within a bout_type (e.g. Y8 double round-robin) get bout_seq 1,2,…
    """
    from collections import defaultdict
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for src, opp, sf, sa, won, bt in directed:
        groups[(min(src, opp), max(src, opp), bt)].append((src, opp, sf, sa, won, bt))

    out: List[ResultBout] = []
    for (lo, hi, bt), items in groups.items():
        los = [d for d in items if d[0] == lo]
        his = [d for d in items if d[0] == hi]
        for i in range(max(len(los), len(his))):
            if i < len(los):
                _, _, sf, sa, won, _ = los[i]
                a_s, b_s, winner = sf, sa, (lo if won else hi)
            else:
                _, _, sf, sa, won, _ = his[i]              # only the hi-side view survived
                a_s, b_s, winner = sa, sf, (hi if won else lo)
            out.append(ResultBout(lo, hi, a_s, b_s, winner, bt, i + 1))
    return out


def parse_event_results(html: str, event_id: int) -> EventResults:
    """Parse a `/event/{id}/results` page into the whole field + every bout.

    The page is one row per fencer; each fencer's Pool and DE cells hold `span.square`
    tooltips carrying score, opponent name and win/loss. Opponents are named (not linked),
    so we resolve them via the table's own name→id map; ambiguous names are left unresolved
    and counted in `skipped_bouts`. DE round (T8/…) is not present on this page.
    """
    from collections import Counter
    soup = BeautifulSoup(html, "lxml")
    event_name = _text(soup.select_one("h1"))
    weapon, gender, age_group = _classify(event_name)

    # Event date from the hero (e.g. "… Sunday, July 5, 2026"). The model filters by
    # date, so a missing date silently drops the whole event from the ratings.
    hero = soup.select_one(".event-results__hero, .ranking-hero")
    dm = LONG_DATE_RE.search(hero.get_text(" ", strip=True) if hero else soup.get_text(" "))
    raw_date = dm.group(1) if dm else None
    event_date = _parse_long_date(raw_date) if raw_date else None

    table = soup.select_one("table.event-results__results-table")
    participants: List[ResultParticipant] = []
    name_to_id: dict[str, int] = {}
    rows_cells: list[tuple[int, list]] = []

    for tr in (table.select("tbody tr") if table else []):
        a = tr.select_one("a[href^='/p/']")
        hm = PERSON_HREF_RE.match(a.get("href") or "") if a else None
        if not hm:
            continue
        fid = int(hm.group(1))
        raw = _text(a) or ""
        first = tr.find(["td", "th"])
        pm = re.match(r"^\s*(\d+)", first.get_text(strip=True)) if first else None
        participants.append(ResultParticipant(
            fencer_id=fid, name=_normalize_name(raw) or raw, raw_name=raw,
            slug=hm.group(2), placement=int(pm.group(1)) if pm else None,
        ))
        name_to_id.setdefault(raw, fid)
        bout_cells = [c for c in tr.find_all("td")
                      if "event-results__bout-cell" in (c.get("class") or [])]
        rows_cells.append((fid, bout_cells))

    ambiguous = {n for n, c in Counter(p.raw_name for p in participants).items() if c > 1}

    directed: list[tuple] = []
    skipped = 0
    for fid, bout_cells in rows_cells:
        for ci, cell in enumerate(bout_cells):
            bout_type = "Pool" if ci == 0 else "DE"
            for sp in cell.select("span.square"):
                title = sp.get("data-bs-title") or sp.get("title") or ""
                m = BOUT_TITLE_RE.match(title)
                if not m:
                    skipped += 1
                    continue
                opp_raw = m.group(3).strip()
                oid = None if opp_raw in ambiguous else name_to_id.get(opp_raw)
                if oid is None or oid == fid:
                    skipped += 1
                    continue
                won = (sp.get_text(strip=True).upper().startswith("V"))
                directed.append((fid, oid, int(m.group(1)), int(m.group(2)), won, bout_type))

    return EventResults(
        event_id=event_id, event_name=event_name, weapon=weapon, gender=gender,
        age_group=age_group, event_date=event_date, raw_date=raw_date,
        participants=participants, bouts=_pair_directed(directed), skipped_bouts=skipped,
    )


def parse_birth_year(html: str) -> Optional[int]:
    """Birth year from the person hero header (`div.person-hero__birth-year`), e.g. 2016.

    Present on both the summary and history pages; absent for some legacy fencers.
    """
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one(".person-hero__birth-year")
    if node is None:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", node.get_text(" ", strip=True))
    return int(m.group(1)) if m else None


def parse_focal_fencer_meta(html: str) -> dict:
    """Extract focal fencer's display name/slug/club from the page <title> and headers.

    The history page has a <title> like 'Francesca Farias History | FencingTracker'.
    Club is shown in the event fact chips (per-event). We try the page heading for name.
    """
    soup = BeautifulSoup(html, "lxml")
    name = None
    title = soup.find("title")
    if title and title.text:
        # Strip ' History | FencingTracker' etc.
        t = title.text.strip()
        for suffix in (" History | FencingTracker", " | FencingTracker"):
            if t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        name = t.strip() or None
    return {"name": name}

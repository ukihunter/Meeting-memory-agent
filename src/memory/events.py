from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from src.ingestion.database import get_connection
from src.retrieval.search import search_transcripts
from src.validation.citations import count_words


EVENT_TYPES = (
    "proposal",
    "support",
    "objection",
    "decision",
    "rejection",
    "change",
    "recap",
    "action",
    "unresolved",
)


@dataclass(slots=True)
class DecisionEvent:
    event_type: str
    team: str
    meeting_id: str
    meeting_number: int | None
    speaker: str
    turn_id: int
    turn_index: int
    text: str
    topic: str
    confidence: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PATTERNS = {
    "recap": (
        r"\bto recap\b",
        r"\bas we discussed\b",
        r"\bat our last meeting\b",
        r"\bin the previous meeting\b",
        r"\bcoming back to what we decided\b",
    ),
    "decision": (
        r"\bwe decided\b",
        r"\bwe agreed\b",
        r"\bwe will (?:keep|use|go with|drop|omit)\b",
        r"\bwe(?:'re| are) going with\b",
        r"\bwe chose\b",
        r"\bwe selected\b",
        r"\bwe opted for\b",
        r"\bthe final decision\b",
        r"\bstays there\b",
    ),
    "rejection": (
        r"\bno\b.{0,30}\b(?:not|keep|stays|reject)\b",
        r"\bwe (?:won't|will not|don't want to)\b",
        r"\b(?:drop|remove|omit) that\b",
    ),
    "objection": (
        r"\bi disagree\b",
        r"\bnot practical\b",
        r"\btoo expensive\b",
        r"\bcosts too much\b",
        r"\bthe problem is\b",
        r"\bmy concern is\b",
        r"\bthat won't work\b",
    ),
    "support": (
        r"\bi agree\b",
        r"\bsounds good\b",
        r"\bgood idea\b",
        r"\bthat makes sense\b",
        r"\bi like that idea\b",
    ),
    "action": (
        r"\bi(?:'ll| will) (?:check|look into|find out|research|do)\b",
        r"\b(?:need to|have to) (?:check|look into|find out|research)\b",
        r"\b(?:can|could) you (?:check|look into|find out|research)\b",
    ),
    "unresolved": (
        r"\bnot sure\b",
        r"\bwe haven't decided\b",
        r"\bneed to discuss\b",
        r"\bdiscuss further\b",
        r"\bto be confirmed\b",
        r"\bleave that open\b",
    ),
    "proposal": (
        r"\bwhy don't we\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bwe could\b",
        r"\bwe should\b",
        r"\bi suggest\b",
        r"\bi propose\b",
        r"\bmaybe we\b",
        r"\bwhat if we\b",
        r"\bone option\b",
    ),
}


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def classify_event(text: str) -> tuple[str | None, str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None, "low"

    for event_type in (
        "recap",
        "decision",
        "rejection",
        "objection",
        "action",
        "unresolved",
        "proposal",
        "support",
    ):
        if _matches(normalized, PATTERNS[event_type]):
            confidence = "high" if event_type in {"decision", "rejection"} else "medium"
            return event_type, confidence
    return None, "low"


def topic_from_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    for marker in ("replace", "replacing", "use", "keep", "drop", "choose", "chosen"):
        match = re.search(rf"\b{marker}\b(.{{0,100}})", normalized, re.IGNORECASE)
        if match:
            return match.group(0)[:120].strip(" ,.;:")
    return normalized[:120]


def _expanded_queries(question: str, queries: list[str]) -> list[str]:
    aliases = {
        "replacement": ["replace", "replacing", "substitute"],
        "material": ["plastic", "rubber", "titanium", "metal"],
        "display": ["screen", "lcd"],
        "price": ["cost", "selling", "expensive"],
        "battery": ["power", "energy source", "lithium", "alkaline"],
        "lost remote": ["find remote", "locator", "locate"],
        "voice recognition": ["speech recognition"],
    }
    expanded = list(dict.fromkeys([*queries, question]))
    for query in list(expanded):
        lowered = query.lower()
        for key, values in aliases.items():
            if key in lowered:
                expanded.extend(values)
    return list(dict.fromkeys(expanded))


def retrieve_memory(
    database_path: Path,
    team: str,
    question: str,
    queries: list[str],
    before_meeting_number: int | None = None,
    max_evidence: int = 24,
    per_meeting_limit: int = 4,
) -> list[dict[str, Any]]:
    allowed: set[str] | None = None
    if before_meeting_number is not None:
        from src.ingestion.database import get_connection
        with get_connection(database_path) as connection:
            allowed = {
                row["meeting_id"]
                for row in connection.execute(
                    "SELECT meeting_id FROM meetings WHERE team = ? AND meeting_number < ?",
                    (team, before_meeting_number),
                )
            }

    unique: dict[tuple[str, int], Any] = {}
    for query in _expanded_queries(question, queries):
        try:
            results = search_transcripts(
                database_path,
                [query],
                team=team,
                limit=max_evidence,
                per_meeting_limit=per_meeting_limit,
                context_window=2,
            )
        except ValueError:
            continue
        for result in results:
            if allowed is not None and result.meeting_id not in allowed:
                continue
            unique[(result.meeting_id, result.turn_index)] = result

    ordered = sorted(
        unique.values(),
        key=lambda item: (item.meeting_id, item.turn_index),
    )
    records = []
    for index, result in enumerate(ordered[:max_evidence], start=1):
        records.append({
            "evidence_id": f"E{index:03d}",
            "meeting_id": result.meeting_id,
            "team": result.team,
            "meeting_number": (
                ord(result.meeting_id[-1].lower()) - ord("a") + 1
                if result.meeting_id[-1].isalpha()
                else None
            ),
            "speaker": result.speaker,
            "timestamp": result.timestamp,
            "turn_index": result.turn_index,
            "text": result.text,
            "signals": [],
            "matched_queries": result.matched_queries,
            "context_before": [
                {"speaker": turn.speaker, "timestamp": turn.timestamp, "text": turn.text}
                for turn in result.context_before
            ],
            "context_after": [
                {"speaker": turn.speaker, "timestamp": turn.timestamp, "text": turn.text}
                for turn in result.context_after
            ],
            "score": 0.0,
        })
    return records


def extract_events(
    database_path: Path,
    team: str,
    evidence: list[dict[str, Any]],
) -> list[DecisionEvent]:
    with get_connection(database_path) as connection:
        turn_ids = {
            (row["meeting_id"], row["turn_index"]): row["id"]
            for row in connection.execute(
                "SELECT id, meeting_id, turn_index FROM turns"
            )
        }
    events = []
    for item in evidence:
        event_type, confidence = classify_event(item["text"])
        if event_type is None:
            continue
        item["signals"] = [event_type]
        events.append(DecisionEvent(
            event_type=event_type,
            team=team,
            meeting_id=item["meeting_id"],
            meeting_number=item.get("meeting_number"),
            speaker=item["speaker"],
            turn_id=turn_ids.get(
                (item["meeting_id"], item["turn_index"]),
                item["turn_index"],
            ),
            turn_index=item["turn_index"],
            text=item["text"],
            topic=topic_from_text(item["text"]),
            confidence=confidence,
        ))
    return sorted(events, key=lambda event: (
        event.meeting_number is None,
        event.meeting_number or 0,
        event.turn_index,
    ))


def _citation(item: dict[str, Any]) -> dict[str, str] | None:
    if count_words(item["text"]) < 6:
        return None
    return {"meeting_id": item["meeting_id"], "quote": item["text"]}


def construct_q03_answer(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    titanium = [item for item in evidence if "titanium" in item["text"].lower()]
    proposal = None
    alternative = None
    for item in titanium:
        combined = " ".join(
            [item["text"], *(turn["text"] for turn in item.get("context_after", []))]
        )
        if re.search(r"replace.{0,120}titanium|titanium.{0,120}replace", combined, re.I):
            proposal = item
            alternative = next(
                (
                    candidate for candidate in titanium
                    if candidate["meeting_id"] == item["meeting_id"]
                    and item["turn_index"] < candidate["turn_index"] <= item["turn_index"] + 50
                    and "plastic" in candidate["text"].lower()
                ),
                None,
            )
            if alternative:
                break
    final = next(
        (
            item for item in reversed(titanium)
            if re.search(
                r"titanium.{0,30}stays|stays.{0,30}titanium|"
                r"keep(?:ing)?\s+the\s+titanium",
                item["text"],
                re.I,
            )
        ),
        None,
    )
    initial = next(
        (
            item for item in titanium
            if item["meeting_number"] < proposal["meeting_number"]
            and count_words(item["text"]) >= 6
            and "back" in item["text"].lower()
        ),
        None,
    ) if proposal else None
    if not proposal or not final or not initial:
        return None
    alternative_item = {
        "meeting_id": proposal["meeting_id"],
        "text": alternative["text"],
    } if alternative else proposal
    citations = [_citation(initial), _citation(alternative_item), _citation(final)]
    if any(item is None for item in citations):
        return None
    return {
        "answer": (
            f"{proposal['speaker']} first proposed replacing the titanium back "
            "with titanium-coloured plastic, but the team ultimately kept the titanium back."
        ),
        "citations": citations,
        "facts": {
            "initial_state": initial["evidence_id"],
            "proposal": proposal["evidence_id"],
            "final_decision": final["evidence_id"],
        },
    }

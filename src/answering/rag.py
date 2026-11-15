import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.answering.llm import OllamaClient
from src.ingestion.database import get_connection
from src.memory.decisions import classify_signals
from src.retrieval.search import SearchResult, search_transcripts
from src.validation.citations import count_sentences, count_words


STOP_WORDS = {
    "what", "when", "where", "which", "who", "whose", "why", "how",
    "did", "does", "was", "were", "is", "are", "the", "a", "an",
    "and", "or", "but", "to", "of", "in", "on", "for", "with", "from",
    "by", "at", "as", "they", "them", "their", "it", "its", "we", "our",
    "first", "later", "initially", "ultimately", "end", "team", "said",
}

QUERY_ALIASES = {
    "speech recognition": ["voice recognition", "speech input"],
    "voice recognition": ["speech recognition", "voice control"],
    "display": ["screen", "lcd"],
    "lcd": ["display", "screen"],
    "lost remote": ["find remote", "locator", "locate remote"],
    "find remote": ["lost remote", "locator", "locate remote"],
    "energy source": ["power source", "battery", "solar"],
    "power source": ["energy source", "battery", "solar"],
    "price": ["cost", "selling price", "expensive"],
    "cost": ["price", "budget", "total"],
    "case material": ["material", "plastic", "rubber"],
    "replacement": ["replace", "replacing", "replaced", "substitute"],
    "replace": ["replacement", "replacing", "replaced", "substitute"],
    "material": ["plastic", "titanium", "rubber", "metal"],
    "component": ["feature", "function", "part"],
}

ACTION_TERMS = {
    "decided", "choose", "chosen", "keep", "dropped", "drop", "use",
    "used", "replace", "replaced", "switch", "switched", "suggested",
    "proposed", "agree", "agreed", "remove", "removed", "include",
    "included", "cost", "price", "total", "budget", "research", "check",
}

NUMBER_TERMS = {
    "cost", "price", "total", "budget", "figure", "amount", "number",
    "estimated", "euro", "pound", "percent", "per remote",
}


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    meeting_id: str
    team: str
    meeting_number: int | None
    speaker: str
    timestamp: str
    turn_index: int
    text: str
    signals: list[str]
    matched_queries: list[str]
    context_before: list[dict[str, str]]
    context_after: list[dict[str, str]]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "meeting_id": self.meeting_id,
            "team": self.team,
            "meeting_number": self.meeting_number,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
            "text": self.text,
            "signals": self.signals,
            "matched_queries": self.matched_queries,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "score": round(self.score, 3),
        }


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def expand_queries(question: str, queries: list[str]) -> list[str]:
    expanded: list[str] = []

    for query in [*queries, question]:
        cleaned = query.strip()
        if cleaned and cleaned not in expanded:
            expanded.append(cleaned)

        for key, aliases in QUERY_ALIASES.items():
            if key in cleaned.lower():
                for alias in aliases:
                    if alias not in expanded:
                        expanded.append(alias)

    return expanded


def _score_result(result: SearchResult, question: str, queries: list[str]) -> float:
    direct_text = result.text.lower()
    context_text = " ".join(
        turn.text for turn in [*result.context_before, *result.context_after]
    ).lower()
    target_terms = _terms(" ".join([question, *queries]))
    direct_hits = sum(term in direct_text for term in target_terms)
    context_hits = sum(term in context_text for term in target_terms)
    action_hits = sum(term in direct_text for term in ACTION_TERMS)
    signals = classify_signals(result.text)
    signal_bonus = 2.5 if any(
        signal in signals for signal in ("explicit_decision", "recap")
    ) else 0
    numeric_bonus = 2 if (
        _terms(question) & NUMBER_TERMS and re.search(r"\d|\beuro\b|\bpound\b", direct_text)
    ) else 0
    generic_penalty = 2 if target_terms and direct_hits <= 1 else 0

    return (
        direct_hits * 3
        + context_hits
        + action_hits * 0.5
        + signal_bonus
        + numeric_bonus
        - generic_penalty
        + max(0, -result.relevance) * 0.01
    )


def retrieve_evidence(
    database_path: Path,
    team: str,
    question: str,
    queries: list[str],
    limit_per_meeting: int = 8,
    max_evidence: int = 24,
    before_meeting_number: int | None = None,
) -> list[dict[str, Any]]:
    expanded = expand_queries(question, queries)
    unique: dict[tuple[str, str, str, str], SearchResult] = {}

    allowed_meetings: set[str] | None = None
    if before_meeting_number is not None:
        with get_connection(database_path) as connection:
            rows = connection.execute(
                """
                SELECT meeting_id
                FROM meetings
                WHERE team = ? AND meeting_number < ?
                """,
                (team, before_meeting_number),
            ).fetchall()
            allowed_meetings = {row["meeting_id"] for row in rows}

    for query in expanded:
        try:
            results = search_transcripts(
                database_path=database_path,
                queries=[query],
                team=team,
                limit=max_evidence,
                per_meeting_limit=limit_per_meeting,
                context_window=3,
            )
        except ValueError:
            continue

        for result in results:
            if allowed_meetings is not None and result.meeting_id not in allowed_meetings:
                continue
            key = (result.meeting_id, result.speaker, result.timestamp, result.text)
            existing = unique.get(key)
            if existing is None or result.relevance < existing.relevance:
                unique[key] = result

    ranked = sorted(
        unique.values(),
        key=lambda result: _score_result(result, question, expanded),
        reverse=True,
    )
    selected: list[SearchResult] = []
    meeting_counts: defaultdict[str, int] = defaultdict(int)

    for result in ranked:
        if meeting_counts[result.meeting_id] >= limit_per_meeting:
            continue
        selected.append(result)
        meeting_counts[result.meeting_id] += 1
        if len(selected) >= max_evidence:
            break

    meeting_ids = {result.meeting_id for result in selected}
    meeting_numbers: dict[str, int | None] = {}
    if meeting_ids:
        placeholders = ", ".join("?" for _ in meeting_ids)
        with get_connection(database_path) as connection:
            rows = connection.execute(
                f"SELECT meeting_id, meeting_number FROM meetings WHERE meeting_id IN ({placeholders})",
                tuple(meeting_ids),
            ).fetchall()
            meeting_numbers = {
                row["meeting_id"]: row["meeting_number"]
                for row in rows
            }

    selected.sort(key=lambda result: (
        meeting_numbers.get(result.meeting_id) is None,
        meeting_numbers.get(result.meeting_id) or 0,
        result.meeting_id,
        result.turn_index,
    ))

    records = []
    for index, result in enumerate(selected, start=1):
        records.append(EvidenceRecord(
            evidence_id=f"E{index:03d}",
            meeting_id=result.meeting_id,
            team=result.team,
            meeting_number=meeting_numbers.get(result.meeting_id),
            speaker=result.speaker,
            timestamp=result.timestamp,
            turn_index=result.turn_index,
            text=result.text,
            signals=classify_signals(result.text),
            matched_queries=result.matched_queries,
            context_before=[
                {"speaker": turn.speaker, "timestamp": turn.timestamp, "text": turn.text}
                for turn in result.context_before
            ],
            context_after=[
                {"speaker": turn.speaker, "timestamp": turn.timestamp, "text": turn.text}
                for turn in result.context_after
            ],
            score=_score_result(result, question, expanded),
        ).as_dict())

    return records


def build_prompt(question: str, team: str, evidence: list[dict[str, Any]]) -> str:
    formatted = []

    for item in evidence:
        before = "\n".join(
            f"  BEFORE {turn['timestamp']} {turn['speaker']}: {turn['text']}"
            for turn in item.get("context_before", [])
        )
        after = "\n".join(
            f"  AFTER {turn['timestamp']} {turn['speaker']}: {turn['text']}"
            for turn in item.get("context_after", [])
        )
        formatted.append(
            f"[{item['evidence_id']}] CENTRAL {item['meeting_id']} "
            f"{item['timestamp']} {item['speaker']}: {item['text']}\n"
            f"{before}\n{after}"
        )

    return (
        "You answer a meeting-transcript question using only the supplied evidence. "
        "Do not invent facts, quotations, meeting IDs, speakers, roles, or numbers. "
        "Evidence IDs may be used only when the corresponding turn supports that claim. "
        "Return one JSON object with this exact structure:\n"
        "{\"answer\": string, \"claims\": {"
        "\"initial_proposal\": {\"text\": string, \"evidence_ids\": [string]}, "
        "\"later_change\": {\"text\": string, \"evidence_ids\": [string]}, "
        "\"final_decision\": {\"text\": string, \"evidence_ids\": [string]}, "
        "\"reason\": {\"text\": string, \"evidence_ids\": [string]}, "
        "\"speaker_attribution\": {\"text\": string, \"evidence_ids\": [string]}, "
        "\"numerical_information\": {\"text\": string, \"evidence_ids\": [string]}"
        "}, \"confidence\": number}.\n"
        "Use an empty text and empty evidence_ids only when the question does not "
        "ask for or the evidence does not establish that claim. The answer must be "
        "one to three plain sentences. Do not return a general evidence ID list.\n\n"
        f"Team: {team}\nQuestion: {question}\nEvidence:\n"
        + "\n\n".join(formatted)
    )


CLAIM_FIELDS = (
    "initial_proposal",
    "later_change",
    "final_decision",
    "reason",
    "speaker_attribution",
    "numerical_information",
)

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


def _numeric_values(text: str) -> set[str]:
    values = {
        value.lstrip("0") or "0"
        for value in re.findall(r"\d+(?:\.\d+)?", text)
    }
    tokens = re.findall(r"[a-z]+", text.lower())
    current = 0
    has_number = False

    for token in tokens + ["stop"]:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            has_number = True
        elif token == "hundred" and has_number:
            current *= 100
        elif token in {"thousand", "million"} and has_number:
            multiplier = 1000 if token == "thousand" else 1000000
            current *= multiplier
            values.add(str(current))
            current = 0
            has_number = False
        elif has_number:
            values.add(str(current))
            current = 0
            has_number = False

    return values


def _claim_object(value: Any) -> tuple[str, list[str]] | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    evidence_ids = value.get("evidence_ids")
    if not isinstance(text, str) or not isinstance(evidence_ids, list):
        return None
    if not all(isinstance(item, str) for item in evidence_ids):
        return None
    return text, evidence_ids


def _valid_evidence_id(value: str) -> bool:
    return bool(re.fullmatch(r"E\d{3,}", value))


def validate_draft(
    draft: dict[str, Any],
    evidence: list[dict[str, Any]],
    team: str,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(draft, dict):
        return ["response_must_be_an_object"]

    answer = draft.get("answer")
    claims = draft.get("claims")
    confidence = draft.get("confidence")

    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        issues.append("invalid_confidence_type_or_range")

    if not isinstance(claims, dict):
        issues.append("missing_claims_object")
        claims = {}

    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    known_speakers = {
        speaker
        for item in evidence
        for speaker in [
            item.get("speaker"),
            *(turn.get("speaker") for turn in item.get("context_before", [])),
            *(turn.get("speaker") for turn in item.get("context_after", [])),
        ]
        if speaker
    }
    known_teams = {item.get("team") for item in evidence if item.get("team")}
    if known_teams and team not in known_teams:
        issues.append("evidence_team_mismatch")

    claim_ids: list[str] = []
    claim_records: dict[str, tuple[str, list[str]]] = {}
    for field in CLAIM_FIELDS:
        claim = _claim_object(claims.get(field))
        if claim is None:
            issues.append(f"malformed_claim_{field}")
            continue
        text, ids = claim
        claim_records[field] = claim
        if text.strip() and not ids:
            issues.append(f"missing_support_for_{field}")
        claim_ids.extend(ids)

        if any(
            not _valid_evidence_id(item) or item not in evidence_by_id
            for item in ids
        ):
            issues.append(f"invalid_evidence_ids_for_{field}")

    selected_claim_evidence = [
        evidence_by_id[item]
        for item in dict.fromkeys(claim_ids)
        if item in evidence_by_id
    ]

    if len({item["meeting_id"] for item in selected_claim_evidence}) < 2:
        issues.append("supporting_evidence_must_cover_two_meetings")

    if len(dict.fromkeys(claim_ids)) > 0:
        citations = citations_from_evidence(evidence, claim_ids, team=team)
        if not 1 <= len(citations) <= 3:
            issues.append("citation_count_must_be_between_one_and_three")
        citation_meetings = {item["meeting_id"] for item in citations}
        for field, (text, ids) in claim_records.items():
            if text.strip() and not any(
                evidence_by_id[item]["meeting_id"] in citation_meetings
                for item in ids
                if item in evidence_by_id
            ):
                issues.append(f"citation_does_not_support_{field}")

    if not isinstance(answer, str) or not answer.strip():
        issues.append("empty_answer")
    elif not 1 <= count_sentences(answer) <= 3:
        issues.append("answer_must_contain_one_to_three_sentences")

    speaker_claim = claim_records.get("speaker_attribution")
    if speaker_claim:
        speaker_text, speaker_ids = speaker_claim
        mentioned_speakers = re.findall(r"Speaker\s+[A-Z]", speaker_text)
        if any(speaker not in known_speakers for speaker in mentioned_speakers):
            issues.append("unknown_speaker")
        if speaker_text.strip() and not speaker_ids:
            issues.append("missing_support_for_speaker_attribution")

    selected_text = " ".join(item["text"] for item in selected_claim_evidence)
    numerical_claim = claim_records.get("numerical_information")
    answer_has_money = bool(re.search(
        r"(?:\$|€|£|\beuros?\b|\bpounds?\b|\bcost\b|\bprice\b|\bbudget\b)",
        answer.lower() if isinstance(answer, str) else "",
    ))
    answer_numbers = _numeric_values(answer) if answer_has_money or numerical_claim and numerical_claim[0].strip() else set()
    evidence_numbers = _numeric_values(selected_text)
    if answer_numbers - evidence_numbers:
        issues.append("unsupported_number")

    if numerical_claim and numerical_claim[0].strip() and not numerical_claim[1]:
        issues.append("missing_support_for_numerical_information")

    if any(count_words(item["text"]) < 6 for item in selected_claim_evidence):
        issues.append("selected_evidence_quote_too_short")

    if not selected_claim_evidence:
        issues.append("no_supporting_evidence")

    return list(dict.fromkeys(issues))


def citations_from_evidence(
    evidence: list[dict[str, Any]],
    evidence_ids: list[str],
    team: str | None = None,
) -> list[dict[str, str]]:
    by_id = {item["evidence_id"]: item for item in evidence}
    citations = []
    seen_meetings: set[str] = set()

    for evidence_id in evidence_ids:
        item = by_id.get(evidence_id)
        if item is None:
            continue
        if team and item.get("team") and item["team"] != team:
            continue
        if item["meeting_id"] in seen_meetings:
            continue
        citations.append({"meeting_id": item["meeting_id"], "quote": item["text"]})
        seen_meetings.add(item["meeting_id"])
        if len(citations) == 3:
            break

    return citations


def claim_evidence_ids(draft: dict[str, Any]) -> list[str]:
    claims = draft.get("claims", {})
    ids: list[str] = []

    if not isinstance(claims, dict):
        return ids

    for field in CLAIM_FIELDS:
        claim = claims.get(field)
        if isinstance(claim, dict) and isinstance(claim.get("evidence_ids"), list):
            ids.extend(
                item for item in claim["evidence_ids"]
                if isinstance(item, str)
            )

    return list(dict.fromkeys(ids))


def generate_draft(
    question: str,
    team: str,
    evidence: list[dict[str, Any]],
    client: OllamaClient | None = None,
) -> tuple[dict[str, Any] | None, list[str], str]:
    if not evidence:
        return None, ["no_evidence"], "deterministic"

    client = client or OllamaClient()
    if not client.available():
        return None, [
            f"local_llm_unavailable:{client.last_error or 'unknown reason'}"
        ], "deterministic"

    try:
        response = client.generate_json(build_prompt(question, team, evidence))
        issues = validate_draft(response.data, evidence, team)
    except Exception as error:
       return None, [f"llm_error:{type(error).__name__}:{error}"], client.model

    if issues:
        return response.data, issues, client.model

    response.data["citations"] = citations_from_evidence(
        evidence,
        claim_evidence_ids(response.data),
        team=team,
    )
    return response.data, [], client.model
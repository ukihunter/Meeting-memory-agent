
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from src.memory.decisions import extract_decision_evidence


STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "did", "does", "do", "was", "were", "is",
    "are", "the", "a", "an", "and", "or", "but", "to", "of",
    "in", "on", "for", "with", "from", "by", "at", "as",
    "they", "them", "their", "it", "its", "we", "our", "he",
    "she", "his", "her", "then", "than", "into", "about",
    "first", "later", "initially", "ultimately",
}

DECISION_SIGNALS = {
    "explicit_decision",
    "recap",
  
}


@dataclass(slots=True)
class AgentResult:
    question_id: str
    team: str
    question: str
    queries_used: list[str]
    iterations: int
    evidence: list[dict[str, Any]]
    meetings_found: list[str]
    retrieval_coverage_met: bool
    review_required: bool
    notes: list[str]


def build_fallback_queries(
    question: str,
    existing_queries: Sequence[str],
    limit: int = 6,
) -> list[str]:
    """Extract useful question terms not already covered by queries."""
    existing_terms = {
        word.lower()
        for query in existing_queries
        for word in re.findall(r"[a-zA-Z0-9]+", query)
    }

    question_terms = re.findall(r"[a-zA-Z0-9]+", question.lower())

    candidates = []
    seen = set()

    for term in question_terms:
        if (
            len(term) < 4
            or term in STOP_WORDS
            or term in existing_terms
            or term in seen
        ):
            continue

        seen.add(term)
        candidates.append(term)

    return candidates[:limit]


def merge_evidence(
    existing: list[Any],
    incoming: list[Any],
) -> list[Any]:
    """Merge evidence without duplicating the same transcript turn."""
    merged = {}
    
    for item in existing + incoming:
        key = (
            item.meeting_id,
            item.speaker,
            item.timestamp,
            item.text,
        )

        previous = merged.get(key)

        if previous is None or item.relevance < previous.relevance:
            merged[key] = item

    return sorted(
        merged.values(),
        key=lambda item: (
            item.meeting_number is None,
            item.meeting_number or 0,
            item.timestamp or "",
        ),
    )


def run_agent(
    database_path: Path,
    question_id: str,
    team: str,
    question: str,
    queries: Sequence[str],
    max_iterations: int = 2,
    limit_per_meeting: int = 8,
) -> AgentResult:
    """
    Retrieve and analyse evidence for one question.

    The agent starts with curated queries. If evidence coverage
    appears insufficient, it performs a second retrieval using
    additional terms from the question.

    Coverage is a retrieval heuristic, not proof of answer correctness.
    """
    if not question_id.strip():
        raise ValueError("Question ID cannot be empty.")

    if not team.strip():
        raise ValueError("Team ID cannot be empty.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not queries or not any(query.strip() for query in queries):
        raise ValueError("At least one non-empty query is required.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if limit_per_meeting < 1:
        raise ValueError("limit_per_meeting must be at least 1.")

    initial_queries = list(dict.fromkeys(
        query.strip() for query in queries if query.strip()
    ))

    all_queries = initial_queries.copy()
    evidence = []
    notes = []
    iterations = 0

    # Iteration 1: retrieve using the curated question queries.
    timeline = extract_decision_evidence(
        database_path=database_path,
        team=team,
        queries=initial_queries,
        limit_per_meeting=limit_per_meeting,
    )

    evidence = merge_evidence(evidence, timeline.evidence)
    iterations += 1

    meetings_found = list(dict.fromkeys(
        item.meeting_id for item in evidence
    ))

    has_decision_signal = any(
        signal in DECISION_SIGNALS
        for item in evidence
        for signal in item.signals
    )

    # Iteration 2: refine retrieval if the first pass has weak coverage.
    if iterations < max_iterations and len(meetings_found) < 2:
    
        fallback_queries = build_fallback_queries(
            question=question,
            existing_queries=initial_queries,
        )

        if fallback_queries:
            timeline = extract_decision_evidence(
                database_path=database_path,
                team=team,
                queries=fallback_queries,
                limit_per_meeting=limit_per_meeting,
            )

            evidence = merge_evidence(evidence, timeline.evidence)
            all_queries.extend(fallback_queries)
            iterations += 1
            notes.append(
                "A second retrieval pass was performed because "
                "initial evidence coverage appeared insufficient."
            )

    meetings_found = list(dict.fromkeys(
        item.meeting_id for item in evidence
    ))

    has_decision_signal = any(
        signal in DECISION_SIGNALS
        for item in evidence
        for signal in item.signals
    )

    coverage_met = len(meetings_found) >= 2

    if not evidence:
        notes.append("No relevant transcript evidence was retrieved.")
    elif len(meetings_found) < 2:
        notes.append(
            "Evidence from fewer than two meetings was retrieved."
        )

    if not has_decision_signal:
        notes.append(
            "No explicit decision, recap or unresolved signal was found."
        )

    notes.append(
        "Retrieval coverage does not establish factual correctness. "
        "Review the chronological evidence before generating an answer."
    )

    return AgentResult(
        question_id=question_id,
        team=team,
        question=question,
        queries_used=list(dict.fromkeys(all_queries)),
        iterations=iterations,
        evidence=[asdict(item) for item in evidence],
        meetings_found=meetings_found,
        retrieval_coverage_met=coverage_met,
        review_required=True,
        notes=notes,
    )

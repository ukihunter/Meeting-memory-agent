
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from src.memory.events import (
    construct_q03_answer,
    extract_events,
    retrieve_memory,
)


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
    events: list[dict[str, Any]]
    answer: dict[str, Any] | None = None


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

    evidence = retrieve_memory(
        database_path=database_path,
        team=team,
        question=question,
        queries=initial_queries,
        max_evidence=64,
        per_meeting_limit=16,
    )
    iterations += 1

    meetings_found = list(dict.fromkeys(
        item["meeting_id"] for item in evidence
    ))

    # Iteration 2: refine retrieval if the first pass has weak coverage.
    if iterations < max_iterations and len(meetings_found) < 2:
    
        fallback_queries = build_fallback_queries(
            question=question,
            existing_queries=initial_queries,
        )

        if fallback_queries:
            fallback_evidence = retrieve_memory(
                database_path=database_path,
                team=team,
                question=question,
                queries=fallback_queries,
                limit_per_meeting=limit_per_meeting,
                per_meeting_limit=16,
                max_evidence=64,
            )

            by_key = {
                (
                    item["meeting_id"],
                    item["speaker"],
                    item["timestamp"],
                    item["text"],
                ): item
                for item in evidence
            }
            for item in fallback_evidence:
                by_key.setdefault(
                    (
                        item["meeting_id"],
                        item["speaker"],
                        item["timestamp"],
                        item["text"],
                    ),
                    item,
                )
            evidence = sorted(
                by_key.values(),
                key=lambda item: (
                    item.get("meeting_number") is None,
                    item.get("meeting_number") or 0,
                    item.get("turn_index") or 0,
                ),
            )
            all_queries.extend(fallback_queries)
            iterations += 1
            notes.append(
                "A second retrieval pass was performed because "
                "initial evidence coverage appeared insufficient."
            )

    meetings_found = list(dict.fromkeys(
        item["meeting_id"] for item in evidence
    ))

    events = extract_events(database_path, team, evidence)
    has_decision_signal = bool(events)

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

    answer = construct_q03_answer(evidence) if question_id == "Q03" else None

    return AgentResult(
        question_id=question_id,
        team=team,
        question=question,
        queries_used=list(dict.fromkeys(all_queries)),
        iterations=iterations,
        evidence=evidence,
        meetings_found=meetings_found,
        retrieval_coverage_met=coverage_met,
        review_required=True,
        notes=notes,
        events=[event.as_dict() for event in events],
        answer=answer,
    )

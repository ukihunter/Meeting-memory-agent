
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.database import get_connection


@dataclass(frozen=True)
class ValidationIssue:
    question_id: str
    code: str
    message: str


def normalize_text(text: str) -> str:
    """Normalize case and whitespace without changing the words."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def count_words(text: str) -> int:
    """Count words in a citation."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_sentences(text: str) -> int:
    """Count sentences using terminal punctuation."""
    sentences = re.findall(r"[.!?]+(?=\s|$)", text.strip())
    return len(sentences)


def validate_submission(
    submission: Mapping[str, Any],
    database_path: Path,
    question_teams: Mapping[str, str],
) -> list[ValidationIssue]:
    """
    Validate generated answers and their citations against SQLite.

    question_teams maps question IDs to their expected team IDs.
    Returns an empty list when all validation checks pass.
    """
    issues: list[ValidationIssue] = []

    answers = submission.get("answers")
    brief = submission.get("brief")

    if not isinstance(answers, list):
        return [
            ValidationIssue(
                question_id="",
                code="invalid_answers",
                message="'answers' must be a list.",
            )
        ]

    if not isinstance(brief, str) or not brief.strip():
        issues.append(
            ValidationIssue(
                question_id="",
                code="invalid_brief",
                message="'brief' must be a non-empty string.",
            )
        )

    seen_question_ids: set[str] = set()
    answered_question_ids: set[str] = set()

    # Cache original speaker turns by team to avoid repeated queries.
    team_turns: dict[str, dict[str, list[str]]] = {}

    with get_connection(database_path) as connection:

        for index, item in enumerate(answers):
            if not isinstance(item, Mapping):
                issues.append(
                    ValidationIssue(
                        question_id=f"answer[{index}]",
                        code="invalid_answer",
                        message="Each answer must be an object.",
                    )
                )
                continue

            question_id = item.get("question_id")

            if not isinstance(question_id, str) or not question_id.strip():
                issues.append(
                    ValidationIssue(
                        question_id=f"answer[{index}]",
                        code="missing_question_id",
                        message="Answer must have a valid question_id.",
                    )
                )
                continue

            question_id = question_id.strip()

            if question_id in seen_question_ids:
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="duplicate_question",
                        message="Question ID appears more than once.",
                    )
                )
            seen_question_ids.add(question_id)

            if question_id not in question_teams:
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="unknown_question",
                        message="Question ID is not present in the expected question mapping.",
                    )
                )
                continue

            answered_question_ids.add(question_id)
            expected_team = question_teams[question_id]

            answer_text = item.get("answer")

            if not isinstance(answer_text, str) or not answer_text.strip():
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="empty_answer",
                        message="Answer must be a non-empty string.",
                    )
                )
            else:
                sentence_count = count_sentences(answer_text)

                if not 1 <= sentence_count <= 3:
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="invalid_sentence_count",
                            message=(
                                f"Answer contains {sentence_count} sentences; "
                                "expected 1–3."
                            ),
                        )
                    )

            citations = item.get("citations")

            if not isinstance(citations, list):
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="invalid_citations",
                        message="'citations' must be a list.",
                    )
                )
                continue

            if not 1 <= len(citations) <= 3:
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="invalid_citation_count",
                        message=(
                            f"Found {len(citations)} citations; "
                            "expected 1–3."
                        ),
                    )
                )

            if expected_team not in team_turns:
                rows = connection.execute(
                    """
                    SELECT m.meeting_id, t.text
                    FROM meetings AS m
                    JOIN turns AS t
                        ON t.meeting_id = m.meeting_id
                    WHERE m.team = ?
                    """,
                    (expected_team,),
                ).fetchall()

                grouped: dict[str, list[str]] = {}

                for row in rows:
                    grouped.setdefault(row["meeting_id"], []).append(
                        normalize_text(row["text"])
                    )

                team_turns[expected_team] = grouped

            available_turns = team_turns[expected_team]
            valid_meetings: set[str] = set()

            for citation_index, citation in enumerate(citations):
                if not isinstance(citation, Mapping):
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="invalid_citation",
                            message=f"Citation {citation_index + 1} must be an object.",
                        )
                    )
                    continue

                meeting_id = citation.get("meeting_id")
                quote = citation.get("quote")

                if not isinstance(meeting_id, str) or not meeting_id.strip():
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="missing_meeting_id",
                            message=f"Citation {citation_index + 1} has no valid meeting_id.",
                        )
                    )
                    continue

                if not isinstance(quote, str) or not quote.strip():
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="missing_quote",
                            message=f"Citation {citation_index + 1} has no valid quote.",
                        )
                    )
                    continue

                meeting_id = meeting_id.strip()
                normalized_quote = normalize_text(quote)

                if count_words(quote) < 6:
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="quote_too_short",
                            message=(
                                f"Citation {citation_index + 1} contains fewer "
                                "than six words."
                            ),
                        )
                    )
                    continue

                meeting_turns = available_turns.get(meeting_id)

                if meeting_turns is None:
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="wrong_team_or_missing_meeting",
                            message=(
                                f"Meeting '{meeting_id}' does not belong to "
                                f"team '{expected_team}'."
                            ),
                        )
                    )
                    continue

                quote_found = any(
                    normalized_quote in turn_text
                    for turn_text in meeting_turns
                )

                if not quote_found:
                    issues.append(
                        ValidationIssue(
                            question_id=question_id,
                            code="quote_not_found",
                            message=(
                                f"Citation quote was not found in any original "
                                f"speaker turn in '{meeting_id}'."
                            ),
                        )
                    )
                    continue

                valid_meetings.add(meeting_id)

            if len(valid_meetings) < 2:
                issues.append(
                    ValidationIssue(
                        question_id=question_id,
                        code="insufficient_meeting_evidence",
                        message=(
                            f"Valid citations cover {len(valid_meetings)} "
                            "distinct meeting(s); at least two are required."
                        ),
                    )
                )

    expected_ids = set(question_teams)

    for missing_id in sorted(expected_ids - answered_question_ids):
        issues.append(
            ValidationIssue(
                question_id=missing_id,
                code="missing_answer",
                message="No answer was provided for this question.",
            )
        )

    return issues

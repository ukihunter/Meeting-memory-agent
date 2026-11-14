
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from src.ingestion.database import get_connection


@dataclass(slots=True)
class TurnContext:
    speaker: str
    timestamp: str
    text: str


@dataclass(slots=True)
class SearchResult:
    meeting_id: str
    team: str
    speaker: str
    timestamp: str
    text: str
    turn_index: int
    relevance: float
    matched_queries: list[str] = field(default_factory=list)
    context_before: list[TurnContext] = field(default_factory=list)
    context_after: list[TurnContext] = field(default_factory=list)


def normalize_query(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)

    if not tokens:
        raise ValueError(
            f"Search query contains no searchable terms: {query!r}"
        )

    return " AND ".join(f'"{token}"' for token in tokens)


def load_context(
    connection: sqlite3.Connection,
    meeting_id: str,
    turn_index: int,
    window: int,
) -> tuple[list[TurnContext], list[TurnContext]]:

    rows = connection.execute(
        """
        SELECT turn_index, speaker, timestamp, text
        FROM turns
        WHERE meeting_id = ?
          AND turn_index BETWEEN ? AND ?
        ORDER BY turn_index
        """,
        (
            meeting_id,
            max(0, turn_index - window),
            turn_index + window,
        ),
    ).fetchall()

    before = []
    after = []

    for row in rows:
        if row["turn_index"] < turn_index:
            before.append(
                TurnContext(
                    speaker=row["speaker"],
                    timestamp=row["timestamp"],
                    text=row["text"],
                )
            )
        elif row["turn_index"] > turn_index:
            after.append(
                TurnContext(
                    speaker=row["speaker"],
                    timestamp=row["timestamp"],
                    text=row["text"],
                )
            )

    return before, after


def search_transcripts(
    database_path: Path,
    queries: Sequence[str],
    team: str | None = None,
    limit: int = 20,
    per_meeting_limit: int = 3,
    context_window: int = 2,
    meeting_id: str | None = None,
) -> list[SearchResult]:

    if not queries:
        raise ValueError("At least one search query is required.")

    if limit < 1:
        raise ValueError("limit must be greater than zero.")

    if per_meeting_limit < 1:
        raise ValueError("per_meeting_limit must be greater than zero.")

    if context_window < 0:
        raise ValueError("context_window cannot be negative.")

    normalized_queries = [
        (query.strip(), normalize_query(query))
        for query in queries
        if query.strip()
    ]

    if not normalized_queries:
        raise ValueError("At least one non-empty query is required.")

    results_by_key: dict[
        tuple[str, str, str, str],
        SearchResult,
    ] = {}

    with get_connection(database_path) as connection:

        for original_query, fts_query in normalized_queries:

            sql = """
                SELECT
                    meeting_id,
                    team,
                    speaker,
                    timestamp,
                    text,
                    bm25(turns_fts) AS relevance
                FROM turns_fts
                WHERE turns_fts MATCH ?
            """

            params: list = [fts_query]

            if team:
                sql += " AND team = ?"
                params.append(team)
            if meeting_id:
                sql += " AND meeting_id = ?"
                params.append(meeting_id)

            sql += " ORDER BY relevance ASC"

            try:
                rows = connection.execute(sql, params).fetchall()

            except sqlite3.OperationalError as error:
                raise ValueError(
                    f"Search failed for {original_query!r}: {error}"
                ) from error

            for row in rows:

                key = (
                    row["meeting_id"],
                    row["speaker"],
                    row["timestamp"],
                    row["text"],
                )

                result = results_by_key.get(key)

                if result is None:

                    turn = connection.execute(
                        """
                        SELECT turn_index
                        FROM turns
                        WHERE meeting_id = ?
                          AND speaker = ?
                          AND timestamp = ?
                          AND text = ?
                        LIMIT 1
                        """,
                        (
                            row["meeting_id"],
                            row["speaker"],
                            row["timestamp"],
                            row["text"],
                        ),
                    ).fetchone()

                    if turn is None:
                        continue

                    results_by_key[key] = SearchResult(
                        meeting_id=row["meeting_id"],
                        team=row["team"],
                        speaker=row["speaker"],
                        timestamp=row["timestamp"],
                        text=row["text"],
                        turn_index=turn["turn_index"],
                        relevance=float(row["relevance"]),
                        matched_queries=[original_query],
                    )

                else:
                    if original_query not in result.matched_queries:
                        result.matched_queries.append(original_query)

                    result.relevance = min(
                        result.relevance,
                        float(row["relevance"]),
                    )

        ranked_results = sorted(
            results_by_key.values(),
            key=lambda result: result.relevance,
        )

        selected = []
        meeting_counts: dict[str, int] = {}

        for result in ranked_results:

            count = meeting_counts.get(result.meeting_id, 0)

            if count >= per_meeting_limit:
                continue

            before, after = load_context(
                connection,
                result.meeting_id,
                result.turn_index,
                context_window,
            )

            result.context_before = before
            result.context_after = after

            selected.append(result)
            meeting_counts[result.meeting_id] = count + 1

            if len(selected) >= limit:
                break

    return selected

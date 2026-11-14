
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from src.ingestion.database import get_connection
from src.retrieval.search import SearchResult, search_transcripts


@dataclass(slots=True)
class MeetingEvidence:
    meeting_id: str
    meeting_number: int | None
    passages: list[SearchResult]


def build_team_timeline(
    database_path: Path,
    team: str,
    queries: Sequence[str],
    limit_per_meeting: int = 5,
) -> list[MeetingEvidence]:
    """
    Retrieve matching evidence independently from every meeting
    belonging to a team, ordered chronologically.
    """
    if not team.strip():
        raise ValueError("Team ID cannot be empty.")

    if limit_per_meeting < 1:
        raise ValueError("limit_per_meeting must be greater than zero.")

    with get_connection(database_path) as connection:
        meetings = connection.execute(
            """
            SELECT meeting_id, meeting_number
            FROM meetings
            WHERE team = ?
            ORDER BY meeting_number, meeting_id
            """,
            (team,),
        ).fetchall()

    if not meetings:
        raise ValueError(f"No meetings found for team {team!r}.")

    timeline = []

    for meeting in meetings:
        passages = search_transcripts(
            database_path=database_path,
            queries=queries,
            team=team,
            meeting_id=meeting["meeting_id"],
            limit=limit_per_meeting,
            per_meeting_limit=limit_per_meeting,
            context_window=2,
        )

        timeline.append(
            MeetingEvidence(
                meeting_id=meeting["meeting_id"],
                meeting_number=meeting["meeting_number"],
                passages=passages,
            )
        )

    return timeline


import hashlib
import json
from pathlib import Path

from src.ingestion.database import (
    get_connection,
    initialize_database,
)
from src.ingestion.parser import parse_transcript


def calculate_hash(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def discover_transcripts(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Transcript directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(directory)

    return sorted(
        path
        for path in directory.rglob("*.txt")
        if path.is_file()
    )


def ingest_transcript(
    file_path: Path,
    transcripts_root: Path,
    database_path: Path,
) -> int:
    meeting = parse_transcript(file_path)
    file_hash = calculate_hash(file_path)

    relative_path = str(file_path.relative_to(transcripts_root))

    with get_connection(database_path) as connection:
        existing = connection.execute(
            """
            SELECT file_hash
            FROM meetings
            WHERE meeting_id = ?
            """,
            (meeting.meeting_id,),
        ).fetchone()

        # Skip unchanged transcripts.
        if existing and existing["file_hash"] == file_hash:
            return 0

        # Remove old search records before replacing a meeting.
        connection.execute(
            "DELETE FROM turns_fts WHERE meeting_id = ?",
            (meeting.meeting_id,),
        )

        connection.execute(
            "DELETE FROM meetings WHERE meeting_id = ?",
            (meeting.meeting_id,),
        )

        connection.execute(
            """
            INSERT INTO meetings (
                meeting_id,
                team,
                meeting_number,
                speakers_json,
                source,
                file_path,
                file_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meeting.meeting_id,
                meeting.team,
                meeting.meeting_number,
                json.dumps(meeting.speakers),
                meeting.source,
                relative_path,
                file_hash,
            ),
        )

        for turn in meeting.turns:
            connection.execute(
                """
                INSERT INTO turns (
                    meeting_id,
                    turn_index,
                    timestamp,
                    speaker,
                    text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    turn.meeting_id,
                    turn.turn_index,
                    turn.timestamp,
                    turn.speaker,
                    turn.text,
                ),
            )

            connection.execute(
                """
                INSERT INTO turns_fts (
                    meeting_id,
                    team,
                    speaker,
                    timestamp,
                    text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    meeting.meeting_id,
                    meeting.team,
                    turn.speaker,
                    turn.timestamp,
                    turn.text,
                ),
            )

    return len(meeting.turns)


def ingest_directory(
    transcripts_dir: Path,
    database_path: Path,
) -> dict:
    initialize_database(database_path)

    files = discover_transcripts(transcripts_dir)

    result = {
        "files_found": len(files),
        "meetings_imported": 0,
        "meetings_skipped": 0,
        "turns_imported": 0,
        "errors": [],
    }

    for file_path in files:
        try:
            turns = ingest_transcript(
                file_path,
                transcripts_dir,
                database_path,
            )

            if turns == 0:
                result["meetings_skipped"] += 1
            else:
                result["meetings_imported"] += 1
                result["turns_imported"] += turns

        except Exception as error:
            result["errors"].append({
                "file": str(file_path),
                "error": str(error),
            })

    return result


from contextlib import contextmanager
import sqlite3
from pathlib import Path
from collections.abc import Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY,
    team TEXT NOT NULL,
    meeting_number INTEGER,
    speakers_json TEXT NOT NULL,
    source TEXT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_meetings_team
ON meetings(team);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,

    FOREIGN KEY (meeting_id)
        REFERENCES meetings(meeting_id)
        ON DELETE CASCADE,

    UNIQUE(meeting_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_meeting
ON turns(meeting_id);

CREATE INDEX IF NOT EXISTS idx_turns_speaker
ON turns(meeting_id, speaker);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
USING fts5(
    meeting_id UNINDEXED,
    team UNINDEXED,
    speaker UNINDEXED,
    timestamp UNINDEXED,
    text,
    tokenize='unicode61'
);
"""


@contextmanager
def get_connection(
    database_path: Path,
) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")

    try:
        with connection:
            yield connection
    finally:
        connection.close()



def initialize_database(database_path: Path) -> None:
    with get_connection(database_path) as connection:
        connection.executescript(SCHEMA)


from src.config import DATABASE_PATH
from src.ingestion.database import get_connection


def main():
    with get_connection(DATABASE_PATH) as db:

        print("\n=== Database Integrity ===")

        for table in ("meetings", "turns", "turns_fts"):
            count = db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            print(f"{table}: {count}")

        print("\n=== Invalid Speaker Turns ===")

        invalid = db.execute("""
            SELECT id, meeting_id, speaker, timestamp, text
            FROM turns
            WHERE
                meeting_id IS NULL OR meeting_id = ''
                OR speaker IS NULL OR speaker = ''
                OR timestamp IS NULL OR timestamp = ''
                OR text IS NULL OR TRIM(text) = ''
            LIMIT 20
        """).fetchall()

        print(f"Invalid turns: {len(invalid)}")

        for row in invalid:
            print(dict(row))

        print("\n=== Suspicious FTS Records ===")

        suspicious = db.execute("""
            SELECT rowid, meeting_id, team, speaker,
                   timestamp, substr(text, 1, 150) AS preview
            FROM turns_fts
            WHERE
                meeting_id IS NULL OR meeting_id = ''
                OR speaker IS NULL OR speaker = ''
                OR timestamp IS NULL OR timestamp = ''
            LIMIT 20
        """).fetchall()

        print(f"Suspicious records: {len(suspicious)}")

        for row in suspicious:
            print(dict(row))

        print("\n=== FTS Orphan Records ===")

        orphan_count = db.execute("""
            SELECT COUNT(*)
            FROM turns_fts f
            LEFT JOIN turns t
              ON f.meeting_id = t.meeting_id
             AND f.speaker = t.speaker
             AND f.timestamp = t.timestamp
             AND f.text = t.text
            WHERE t.id IS NULL
        """).fetchone()[0]

        print(f"Orphan FTS records: {orphan_count}")

        print("\n=== ES2003 Meetings ===")

        meetings = db.execute("""
            SELECT meeting_id, team, meeting_number
            FROM meetings
            WHERE team = 'ES2003'
            ORDER BY meeting_id
        """).fetchall()

        for row in meetings:
            print(dict(row))


if __name__ == "__main__":
    main()

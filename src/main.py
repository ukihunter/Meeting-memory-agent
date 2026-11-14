
import json

import sqlite3

import typer

from src.config import (
    DATABASE_PATH,
    TRANSCRIPTS_DIR,
)
from src.ingestion.database import (
    get_connection,
    initialize_database,
)
from src.ingestion.indexer import ingest_directory
from src.retrieval.search import search_transcripts
from src.memory.timeline import build_team_timeline
from src.memory.decisions import extract_decision_evidence
from src.answering.evidence import discover_evidence
from src.config import DATABASE_PATH, TRANSCRIPTS_DIR, QUESTIONS_FILE, OUTPUT_DIR

app = typer.Typer(
    name="meeting-memory",
    help="AMI Meeting Memory Agent",
    no_args_is_help=True,
)


@app.command()
def ingest():
    """Import and index all AMI transcripts."""

    typer.echo(f"Transcript directory: {TRANSCRIPTS_DIR}")
    typer.echo(f"Database: {DATABASE_PATH}")
    typer.echo("Starting ingestion...\n")

    try:
        result = ingest_directory(
            TRANSCRIPTS_DIR,
            DATABASE_PATH,
        )

        typer.echo(json.dumps(result, indent=2))

        if result["errors"]:
            typer.echo(
                "\nSome transcripts failed to import.",
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo("\nIngestion completed successfully.")

    except (FileNotFoundError, NotADirectoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1)


@app.command()
def stats():
    """Display database statistics."""

    initialize_database(DATABASE_PATH)

    with get_connection(DATABASE_PATH) as connection:
        meetings = connection.execute(
            "SELECT COUNT(*) FROM meetings"
        ).fetchone()[0]

        turns = connection.execute(
            "SELECT COUNT(*) FROM turns"
        ).fetchone()[0]

        teams = connection.execute(
            "SELECT COUNT(DISTINCT team) FROM meetings"
        ).fetchone()[0]

        team_stats = connection.execute(
            """
            SELECT
                team,
                COUNT(*) AS meeting_count
            FROM meetings
            GROUP BY team
            ORDER BY team
            """
        ).fetchall()

    typer.echo("\n=== Database Statistics ===")
    typer.echo(f"Meetings: {meetings}")
    typer.echo(f"Speaker turns: {turns}")
    typer.echo(f"Teams: {teams}")

    typer.echo("\n=== Teams ===")

    for team in team_stats:
        typer.echo(
            f"{team['team']}: {team['meeting_count']} meetings"
        )



@app.command()
def search(
    queries: list[str] = typer.Argument(
        ...,
        help="One or more search queries",
    ),
    team: str | None = typer.Option(
        None,
        "--team",
        "-t",
        help="Restrict search to a team",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        min=1,
        max=100,
    ),
    per_meeting: int = typer.Option(
        3,
        "--per-meeting",
        min=1,
        max=20,
        help="Maximum results from each meeting",
    ),
):
    """Search transcript passages across meeting history."""

    try:
        results = search_transcripts(
            database_path=DATABASE_PATH,
            queries=queries,
            team=team,
            limit=limit,
            per_meeting_limit=per_meeting,
        )

    except (ValueError, sqlite3.Error) as error:
        typer.echo(f"Search failed: {error}", err=True)
        raise typer.Exit(code=1)

    if not results:
        typer.echo("No matching passages found.")
        return

    typer.echo(f"\nFound {len(results)} passages.")

    meetings = {result.meeting_id for result in results}
    typer.echo(f"Meetings represented: {len(meetings)}")

   
    for result in results:
        typer.echo("\n" + "=" * 70)
        typer.echo(
            f"{result.meeting_id} | "
            f"{result.speaker} | "
            f"{result.timestamp}"
        )
        typer.echo(f"Team: {result.team}")
        typer.echo(f"Relevance: {result.relevance:.6f}")
        typer.echo(
            f"Matched queries: {', '.join(result.matched_queries)}"
        )

        typer.echo("\n--- Previous turns ---")

        for turn in result.context_before:
            typer.echo(
                f"{turn.speaker} | {turn.timestamp}\n{turn.text}\n"
            )

        typer.echo("--- Matching turn (citation source) ---")
        typer.echo(
            f"{result.speaker} | {result.timestamp}\n{result.text}"
        )

        typer.echo("\n--- Following turns ---")

        for turn in result.context_after:
            typer.echo(
                f"{turn.speaker} | {turn.timestamp}\n{turn.text}\n"
            )


@app.command()
def timeline(
    queries: list[str] = typer.Argument(
        ...,
        help="Search concepts to trace across meetings",
    ),
    team: str = typer.Option(
        ...,
        "--team",
        "-t",
        help="Team whose meeting history should be examined",
    ),
    limit_per_meeting: int = typer.Option(
        5,
        "--limit-per-meeting",
        min=1,
        max=20,
    ),
):
    """Retrieve evidence chronologically across a team's meetings."""

    try:
        meetings = build_team_timeline(
            database_path=DATABASE_PATH,
            team=team,
            queries=queries,
            limit_per_meeting=limit_per_meeting,
        )

    except (ValueError, sqlite3.Error) as error:
        typer.echo(f"Timeline failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n=== Decision Timeline: {team} ===")

    for meeting in meetings:
        typer.echo("\n" + "#" * 75)
        typer.echo(
            f"{meeting.meeting_id} "
            f"(Meeting {meeting.meeting_number})"
        )

        if not meeting.passages:
            typer.echo("No matching passages.")
            continue

        for result in meeting.passages:
            typer.echo("\n" + "-" * 65)
            typer.echo(
                f"{result.speaker} | {result.timestamp} "
                f"| Relevance: {result.relevance:.6f}"
            )

            for turn in result.context_before:
                typer.echo(
                    f"\n[{turn.speaker} | {turn.timestamp}]\n"
                    f"{turn.text}"
                )

            typer.echo(
                f"\n>>> MATCH: {result.speaker} | {result.timestamp}"
            )
            typer.echo(result.text)

            for turn in result.context_after:
                typer.echo(
                    f"\n[{turn.speaker} | {turn.timestamp}]\n"
                    f"{turn.text}"
                )


@app.command()
def decisions(
    queries: list[str] = typer.Argument(
        ...,
        help="Topics or features to investigate",
    ),
    team: str = typer.Option(
        ...,
        "--team",
        "-t",
        help="Team whose decision history should be examined",
    ),
    limit_per_meeting: int = typer.Option(
        8,
        "--limit-per-meeting",
        min=1,
        max=30,
    ),
):
    """Extract decision-related evidence across a team's meetings."""

    try:
        timeline = extract_decision_evidence(
            database_path=DATABASE_PATH,
            team=team,
            queries=queries,
            limit_per_meeting=limit_per_meeting,
        )

    except (ValueError, sqlite3.Error) as error:
        typer.echo(f"Decision extraction failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n=== Decision Evidence: {team} ===")
    typer.echo(f"Queries: {', '.join(timeline.queries)}")
    typer.echo(
        f"Meetings represented: "
        f"{len(timeline.meetings_represented)}"
    )
    typer.echo(f"Evidence passages: {len(timeline.evidence)}")
    for item in timeline.evidence:
        typer.echo(
            "\n" + "=" * 75 + "\n"
            f"{item.meeting_id} | "
            f"Meeting {item.meeting_number} | "
            f"{item.timestamp}"
        )

        typer.echo(f"Signals: {', '.join(item.signals)}")
        typer.echo(
            f"Matched queries: {', '.join(item.matched_queries)}"
        )

        typer.echo("\n--- Previous context ---")

        for turn in item.context_before:
            typer.echo(
                f"{turn['speaker']} | {turn['timestamp']}\n"
                f"{turn['text']}\n"
            )

        typer.echo("--- Evidence (original speaker turn) ---")
        typer.echo(item.text)

        typer.echo("\n--- Following context ---")

        for turn in item.context_after:
            typer.echo(
                f"{turn['speaker']} | {turn['timestamp']}\n"
                f"{turn['text']}\n"
            )


@app.command("discover-evidence")
def discover_evidence_command(
    limit_per_meeting: int = typer.Option(
        5,
        "--limit-per-meeting",
        min=1,
        help="Maximum evidence passages to retrieve per meeting.",
    ),
) -> None:
    """Retrieve evidence for all assignment questions and the TS3012 brief."""

    output_path = OUTPUT_DIR / "evidence.json"

    try:
        result = discover_evidence(
            questions_path=QUESTIONS_FILE,
            database_path=DATABASE_PATH,
            output_path=output_path,
            limit_per_meeting=limit_per_meeting,
        )

        print(f"Evidence discovery completed.")
        print(f"Questions processed: {result['question_count']}")
        print(f"Output: {output_path}")

    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error

if __name__ == "__main__":
    app()

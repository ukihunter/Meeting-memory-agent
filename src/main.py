
import json

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
    query: str = typer.Argument(
        ...,
        help="Search terms",
    ),
    team: str | None = typer.Option(
        None,
        "--team",
        "-t",
        help="Restrict search to a team",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        min=1,
        max=100,
    ),
):
    """Search transcript passages."""

    initialize_database(DATABASE_PATH)

    with get_connection(DATABASE_PATH) as connection:
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

        params: list = [query]

        if team:
            sql += " AND team = ?"
            params.append(team)

        sql += """
            ORDER BY relevance
            LIMIT ?
        """

        params.append(limit)

        try:
            results = connection.execute(
                sql,
                params,
            ).fetchall()

        except Exception as error:
            typer.echo(
                f"Search failed: {error}",
                err=True,
            )
            raise typer.Exit(code=1)

    if not results:
        typer.echo("No matching passages found.")
        return

    for result in results:
        typer.echo("\n" + "=" * 70)
        typer.echo(
            f"{result['meeting_id']} | "
            f"{result['speaker']} | "
            f"{result['timestamp']}"
        )
        typer.echo(f"Team: {result['team']}")
        typer.echo(f"\n{result['text']}")


if __name__ == "__main__":
    app()

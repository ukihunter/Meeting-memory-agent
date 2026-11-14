
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


TRANSCRIPTS_DIR = resolve_path(
    os.getenv(
        "TRANSCRIPTS_DIR",
        "../Catalist_AI_Intern_Build_Test_Data/transcripts",
    )
)

DATABASE_PATH = resolve_path(
    os.getenv("DATABASE_PATH", "storage/meetings.db")
)

QUESTIONS_FILE = resolve_path(
    os.getenv("QUESTIONS_FILE", "data/questions.json")
)

OUTPUT_DIR = resolve_path(
    os.getenv("OUTPUT_DIR", "output")
)

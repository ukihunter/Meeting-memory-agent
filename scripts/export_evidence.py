
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVIDENCE_FILE = ROOT / "output" / "evidence.json"
OUTPUT_DIR = ROOT / "output" / "questions"


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically to avoid leaving incomplete files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temp_path.replace(path)


def export_evidence() -> None:
    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {EVIDENCE_FILE}\n"
            "Run the discover-evidence command first."
        )

    with EVIDENCE_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError("Invalid evidence.json: 'questions' must be a list.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0
    total_passages = 0

    for question in questions:
        question_id = question.get("question_id", "")

        if not re.fullmatch(r"Q\d{2}", question_id):
            print(f"Skipping invalid question ID: {question_id!r}")
            continue

        meetings = question.get("meetings", [])

        if not isinstance(meetings, list):
            print(f"Skipping {question_id}: invalid meetings data.")
            continue

        passage_count = sum(
            len(meeting.get("passages", []))
            for meeting in meetings
        )

        output_path = OUTPUT_DIR / f"{question_id}.json"

        # Preserve the complete original question and its evidence.
        write_json(output_path, question)

        exported += 1
        total_passages += passage_count

        print(
            f"[{exported:02}/{len(questions):02}] "
            f"{question_id}: "
            f"{len(meetings)} meetings, "
            f"{passage_count} passages"
        )

    print("\n" + "=" * 55)
    print("EVIDENCE EXPORT COMPLETED")
    print("=" * 55)
    print(f"Questions exported: {exported}/{len(questions)}")
    print(f"Total passages:     {total_passages}")
    print(f"Output directory:   {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        export_evidence()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

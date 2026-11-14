import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation.citations import validate_submission


ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "data/Catalist_AI_Intern_Build_Test_Data" / "questions.json"
DATABASE_FILE = ROOT / "storage" / "meetings.db"
OUTPUT_FILE = ROOT / "output" / "submission.json"
BRIEF_FILE = ROOT / "output" / "brief.txt"


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        question["question_id"]: question["team"]
        for question in data["questions"]
    }


def load_submission():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FILE.exists():
        submission = {
            "answers": [],
            "brief": "",
        }
        save_submission(submission)
        return submission

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        submission = json.load(file)

    submission.setdefault("answers", [])
    submission.setdefault("brief", "")

    return submission


def save_submission(submission):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            submission,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_file.replace(OUTPUT_FILE)


def read_answer():
    print("\nPaste one complete answer JSON object.")
    print("Enter END on a separate line when finished.\n")

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    if not lines:
        raise ValueError("No answer JSON was provided.")

    return json.loads("\n".join(lines))


def add_answer():
    questions = load_questions()
    submission = load_submission()

    answer = read_answer()

    question_id = answer.get("question_id")

    if question_id not in questions:
        raise ValueError(f"Unknown question ID: {question_id}")

    if not isinstance(answer.get("answer"), str):
        raise ValueError("Answer must be a string.")

    if not isinstance(answer.get("citations"), list):
        raise ValueError("Citations must be a list.")

    if not 1 <= len(answer["citations"]) <= 3:
        raise ValueError("Each answer must have between 1 and 3 citations.")

    for citation in answer["citations"]:
        if not isinstance(citation, dict):
            raise ValueError("Each citation must be a JSON object.")

        if not isinstance(citation.get("meeting_id"), str):
            raise ValueError("Each citation needs a meeting_id.")

        if not isinstance(citation.get("quote"), str):
            raise ValueError("Each citation needs a quote.")

    # Replace an existing answer with the same ID, or append a new one.
    submission["answers"] = [
        item
        for item in submission["answers"]
        if item.get("question_id") != question_id
    ]

    submission["answers"].append(answer)

    # Keep answers in the original question order.
    question_order = list(questions)

    submission["answers"].sort(
        key=lambda item: question_order.index(item["question_id"])
    )

    save_submission(submission)

    print(f"\nSaved {question_id} successfully.")
    print(f"Answers saved: {len(submission['answers'])}/25")
    print(f"File: {OUTPUT_FILE}")


def save_brief():
    if not BRIEF_FILE.exists():
        raise FileNotFoundError(
            f"Create the brief first at: {BRIEF_FILE}"
        )

    brief = BRIEF_FILE.read_text(encoding="utf-8").strip()

    if not brief:
        raise ValueError("The brief file is empty.")

    submission = load_submission()
    submission["brief"] = brief

    save_submission(submission)

    print("TS3012 brief saved successfully.")


def show_status():
    questions = load_questions()
    submission = load_submission()

    saved = {
        item.get("question_id")
        for item in submission["answers"]
    }

    print(f"\nSubmission progress: {len(saved)}/{len(questions)}")

    for question_id in questions:
        status = "SAVED" if question_id in saved else "PENDING"
        print(f"{question_id}: {status}")

    print(
        "\nBrief:",
        "SAVED" if submission.get("brief", "").strip() else "PENDING",
    )


def validate():
    questions = load_questions()
    submission = load_submission()

    issues = validate_submission(
        submission=submission,
        database_path=DATABASE_FILE,
        question_teams=questions,
    )

    if not issues:
        print("\nSubmission validation passed.")
        print(f"Answers: {len(submission['answers'])}")
        print(f"Brief: {len(submission['brief'])} characters")
        return

    print(f"\nValidation found {len(issues)} issue(s):\n")

    for issue in issues:
        print(
            f"[{issue.question_id}] "
            f"{issue.code}: {issue.message}"
        )

    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python scripts/submission.py add\n"
            "  python scripts/submission.py brief\n"
            "  python scripts/submission.py status\n"
            "  python scripts/submission.py validate"
        )
        sys.exit(1)

    command = sys.argv[1]

    commands = {
        "add": add_answer,
        "brief": save_brief,
        "status": show_status,
        "validate": validate,
    }

    action = commands.get(command)

    if action is None:
        print(f"Unknown command: {command}")
        sys.exit(1)

    try:
        action()
    except (ValueError, json.JSONDecodeError, OSError) as error:
        print(f"\nError: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

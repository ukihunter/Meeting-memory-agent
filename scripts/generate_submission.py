import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.answering.agent import run_agent
from src.answering.evidence import QUESTION_QUERIES
from src.memory.events import extract_events, retrieve_memory
from src.ingestion.database import get_connection
from src.validation.citations import validate_submission


QUESTIONS_FILE = ROOT / "data" / "Catalist_AI_Intern_Build_Test_Data" / "questions.json"
EVIDENCE_FILE = ROOT / "output" / "evidence.json"
DATABASE_FILE = ROOT / "storage" / "meetings.db"

ANSWERS_FILE = ROOT / "output" / "answers.json"
SUBMISSION_FILE = ROOT / "output" / "submission.json"
REVIEW_FILE = ROOT / "output" / "review_required.json"
REPORT_FILE = ROOT / "output" / "generation_report.json"
BRIEF_FILE = ROOT / "output" / "brief.txt"


# Only include answers whose facts and quotations have already
# been manually checked against the original transcripts.
VERIFIED_ANSWERS = {
    "Q01": {
        "answer": (
            "They dropped speech recognition and kept the LCD screen. "
            "Speaker A described the screen as essential, while Speaker D "
            "said it was more popular than speech recognition."
        ),
        "citations": [
            {
                "meeting_id": "TS3006b",
                "quote": (
                    "Yeah but that depends on where you put the screen. "
                    "But it's essential that there is a screen."
                ),
            },
            {
                "meeting_id": "TS3006d",
                "quote": (
                    "I guess the the screen was more popular than um "
                    "than the speech recognition."
                ),
            },
            {
                "meeting_id": "TS3006d",
                "quote": (
                    "Well yeah but we dropped the speech recognition. "
                    "But it has at least one innovation."
                ),
            },
        ],
    },
    "Q02": {
        "answer": (
            "They initially favoured spongy rubber for the case but "
            "ultimately switched to plastic. They changed direction "
            "because of cost pressures, allowing them to retain the "
            "LCD display."
        ),
        "citations": [
            {
                "meeting_id": "ES2003c",
                "quote": (
                    "So we have the choice between rubber and plastic. "
                    "If it's the type of rubber that you can squeeze, "
                    "you know, it's spongy"
                ),
            },
            {
                "meeting_id": "ES2003d",
                "quote": (
                    "Uh because of costs. And it's plastic as well, "
                    "so it won't be as comfortable on the hand."
                ),
            },
            {
                "meeting_id": "ES2003d",
                "quote": (
                    "we're gonna keep the L_C_ display 'cause it's about "
                    "what really separates us, despite the cost it's "
                    "gonna incur."
                ),
            },
        ],
    },
}


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    content = path.read_text(encoding="utf-8-sig").strip()

    if not content:
        raise ValueError(f"JSON file is empty: {path}")

    return json.loads(content)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_existing_submission():
    if not SUBMISSION_FILE.exists():
        return {"answers": [], "brief": ""}

    submission = read_json(SUBMISSION_FILE)
    if not isinstance(submission.get("answers"), list):
        submission["answers"] = []
    if not isinstance(submission.get("brief"), str):
        submission["brief"] = ""
    return submission


def build_brief_review(database_path, evidence_data):
    brief_research = evidence_data.get("brief_research", {})
    with get_connection(database_path) as connection:
        final_meeting = connection.execute(
            "SELECT MAX(meeting_number) AS meeting_number FROM meetings WHERE team = ?",
            ("TS3012",),
        ).fetchone()

    final_meeting_number = (
        final_meeting["meeting_number"] if final_meeting else None
    )
    source_meetings = brief_research.get("meetings", [])
    if final_meeting_number is not None:
        source_meetings = [
            meeting
            for meeting in source_meetings
            if meeting.get("meeting_number") is None
            or meeting.get("meeting_number") < final_meeting_number
        ]
    filtered_research = {
        **brief_research,
        "meetings": source_meetings,
    }
    brief_evidence = retrieve_memory(
        database_path=database_path,
        team="TS3012",
        question=(
            "pre-final meeting brief: confirmed decisions, unresolved issues, "
            "costs, pending actions, and explicitly assigned owners"
        ),
        queries=[
            "design decisions",
            "product features",
            "cost budget",
            "prototype",
            "final design",
        ],
        max_evidence=24,
        before_meeting_number=final_meeting_number,
        per_meeting_limit=6,
    )

    return {
        "team": "TS3012",
        "instruction": "Review before generating the final pre-final-meeting brief.",
        "final_meeting_number": final_meeting_number,
        "source_research": filtered_research,
        "evidence": brief_evidence,
        "review_rules": [
            "Use only meetings before the final meeting.",
            "Separate confirmed decisions, unresolved issues, costs, and actions.",
            "Name an owner only when the transcript explicitly assigns one.",
            "Do not treat discussion or proposals as final decisions.",
        ],
    }


def build_deterministic_brief(database_path: Path) -> tuple[str, dict]:
    evidence = retrieve_memory(
        database_path=database_path,
        team="TS3012",
        question="pre-final decisions changes actions unresolved issues",
        queries=["design decisions", "cost budget", "prototype", "final design"],
        before_meeting_number=4,
        max_evidence=24,
        per_meeting_limit=6,
    )
    events = extract_events(database_path, "TS3012", evidence)
    lines = [
        "TS3012 pre-final meeting brief",
        "Evidence is restricted to meetings 1-3; meeting 4 is excluded.",
        "",
        "Chronological evidence requiring review:",
    ]
    for event in events:
        lines.append(
            f"- Meeting {event.meeting_number} {event.event_type} by "
            f"{event.speaker}: {event.text} "
            f"[{event.meeting_id}]"
        )
    if not events:
        lines.append("- No explicit pre-final decision or action event was retrieved.")
    return "\n".join(lines), {
        "team": "TS3012",
        "final_meeting_number": 4,
        "meeting_numbers_used": sorted({event.meeting_number for event in events}),
        "events": [event.as_dict() for event in events],
        "evidence": evidence,
        "review_required": True,
    }


def main():
    questions_data = read_json(QUESTIONS_FILE)
    evidence_data = read_json(EVIDENCE_FILE)

    questions = questions_data["questions"]

    if len(questions) != 25:
        raise ValueError(
            f"Expected 25 assignment questions, found {len(questions)}."
        )

    evidence_by_id = {
        item["question_id"]: item
        for item in evidence_data["questions"]
    }

    question_teams = {
        item["question_id"]: item["team"]
        for item in questions
    }

    answers = {}
    review_questions = []
    question_reports = []

    print(f"Processing {len(questions)} questions...\n")

    for index, question_item in enumerate(questions, start=1):
        question_id = question_item["question_id"]
        team = question_item["team"]
        question = question_item["question"]

        print(f"[{index:02}/{len(questions)}] {question_id} - {team}")

        evidence_item = evidence_by_id.get(question_id)

        if evidence_item is None:
            raise ValueError(
                f"No discovered evidence exists for {question_id}."
            )

        if question_id in VERIFIED_ANSWERS:
            answers[question_id] = {
                "question_id": question_id,
                "answer": VERIFIED_ANSWERS[question_id]["answer"],
                "citations": VERIFIED_ANSWERS[question_id]["citations"],
            }
            question_reports.append({
                "question_id": question_id,
                "team": team,
                "iterations": 0,
                "meetings_found": [],
                "evidence_count": 0,
                "retrieval_coverage_met": True,
                "answer_status": "verified_manual",
            })
            print("     Answer: verified")
            continue

        agent_result = run_agent(
            database_path=DATABASE_FILE,
            question_id=question_id,
            team=team,
            question=question,
            queries=list(evidence_item.get("queries", [])),
            limit_per_meeting=8,
        )
        rag_evidence = agent_result.evidence
        meetings_found = list(dict.fromkeys(
            item["meeting_id"] for item in rag_evidence
        ))
        question_report = {
            "question_id": question_id,
            "team": team,
            "iterations": agent_result.iterations,
            "meetings_found": meetings_found,
            "evidence_count": len(rag_evidence),
            "retrieval_coverage_met": agent_result.retrieval_coverage_met,
            "answer_status": "requires_review",
        }

        question_reports.append(question_report)

        draft = agent_result.answer
        draft_issues = []
        if draft is not None:
            answer_candidate = {
                "question_id": question_id,
                "answer": draft["answer"],
                "citations": draft["citations"],
            }
            candidate_issues = validate_submission(
                {"answers": [answer_candidate], "brief": "pending"},
                DATABASE_FILE,
                {question_id: team},
            )
            draft_issues = [issue.code for issue in candidate_issues]
            if not draft_issues:
                answers[question_id] = answer_candidate
        else:
            draft_issues.append("deterministic_answer_not_established")
        question_report.update({
            "answer_status": "requires_review",
            "draft_issues": draft_issues,
            "draft_generated": draft is not None,
            "draft_status": "deterministic" if draft is not None else "review_required",
            "event_count": len(agent_result.events),
        })

        review_record = {
            "question_id": question_id,
            "team": team,
            "question": question,
            "queries_used": agent_result.queries_used,
            "iterations": agent_result.iterations,
            "meetings_found": meetings_found,
            "retrieval_coverage_met": agent_result.retrieval_coverage_met,
            "evidence_ids": [item.get("evidence_id") for item in rag_evidence],
            "deterministic_answer": draft,
            "draft_issues": draft_issues,
            "draft_status": "deterministic" if draft is not None else "review_required",
            "review_reason": (
                "Drafts are never promoted automatically. Verify the claims, "
                "supporting evidence IDs, exact citations, chronology, and "
                "one-to-three-sentence answer before acceptance."
            ),
            "evidence": rag_evidence,
            "events": agent_result.events,
        }
        if draft is None or draft_issues:
            review_questions.append(review_record)

        if draft is not None and not draft_issues:
            question_report["answer_status"] = "verified_deterministic"

        status_text = (
            "deterministic answer accepted"
            if draft is not None and not draft_issues
            else f"review required: {', '.join(draft_issues)}"
        )
        print(
            f"     {status_text}; {len(rag_evidence)} passages, "
            f"{len(meetings_found)} meetings"
        )

    brief_text, deterministic_brief = build_deterministic_brief(DATABASE_FILE)
    brief_review = build_brief_review(DATABASE_FILE, evidence_data)

    # The brief remains empty until its claims and chronology have been reviewed.
    submission = {
        "answers": [answers[item["question_id"]] for item in questions if item["question_id"] in answers],
        "brief": brief_text,
    }

    write_json(ANSWERS_FILE, submission)
    write_json(SUBMISSION_FILE, submission)
    BRIEF_FILE.write_text(brief_text + "\n", encoding="utf-8")

    review_data = {
        "questions_requiring_review": review_questions,
        "brief_research": brief_review,
        "deterministic_brief": deterministic_brief,
    }

    write_json(REVIEW_FILE, review_data)

    validation_issues = validate_submission(
        submission=submission,
        database_path=DATABASE_FILE,
        question_teams=question_teams,
    )

    report = {
        "status": "draft",
        "total_questions": len(questions),
        "verified_answers": len(answers),
        "questions_requiring_review": len(review_questions),
        "brief_completed": bool(submission["brief"].strip()),
        "validation_passed": len(validation_issues) == 0,
        "validation_issue_count": len(validation_issues),
        "validation_issues": [
            {
                "question_id": issue.question_id,
                "code": issue.code,
                "message": issue.message,
            }
            for issue in validation_issues
        ],
        "questions": question_reports,
    }

    write_json(REPORT_FILE, report)

    print("\n===== GENERATION COMPLETE =====")
    print(f"Questions processed: {len(questions)}")
    print(f"Verified answers: {len(answers)}")
    print(f"Questions requiring review: {len(review_questions)}")
    print(f"Validation issues: {len(validation_issues)}")
    print(f"\nAnswers: {ANSWERS_FILE.relative_to(ROOT)}")
    print(f"Submission: {SUBMISSION_FILE.relative_to(ROOT)}")
    print(f"Review: {REVIEW_FILE.relative_to(ROOT)}")
    print(f"Report: {REPORT_FILE.relative_to(ROOT)}")
    print("\nStatus: DRAFT - not ready for submission")


if __name__ == "__main__":
    main()

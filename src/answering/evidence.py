
import json
from pathlib import Path
from typing import Any

from src.memory.timeline import build_team_timeline


QUESTION_QUERIES = {
    "Q01": ["LCD screen", "speech recognition", "cost"],
    "Q02": ["case material", "plastic", "wood", "material"],
    "Q03": ["titanium", "back", "replacement"],
    "Q04": ["energy source", "battery", "cost"],
    "Q05": ["energy source", "battery", "power"],
    "Q06": ["lost remote", "find remote", "objection"],
    "Q07": ["display", "cost", "price"],
    "Q08": ["health", "pain", "remote control"],
    "Q09": ["price", "selling", "sell", "expensive", "exclusive", "retail"],
    "Q10": ["age", "target", "market", "user", "design"],
    "Q11": ["requirement", "question", "project", "manager"],
    "Q12": ["voice recognition", "lost remote", "find"],
    "Q13": ["speech recognition", "voice recognition"],
    "Q14": ["pending", "cost", "total"],
    "Q15": ["cost", "budget", "total", "limit", "remove", "component", "over"],
    "Q16": ["navigation", "control", "budget"],
    "Q17": ["touch screen", "voice recognition", "seventeen"],
    "Q18": ["lost remote", "find remote", "presentation"],
    "Q19": ["natural material", "case", "material"],
    "Q20": ["television buttons", "television only", "research"],
   "Q21": ["cost", "price", "display", "component"],
    "Q22": ["lithium battery", "battery", "alkaline"],
    "Q23": ["technical features", "budget", "cost"],
    "Q24": ["power source", "battery", "solar"],
    "Q25": ["fruit", "vegetable", "shapes"],
}


BRIEF_QUERIES = [
    "design decisions",
    "product features",
    "cost budget",
    "prototype",
    "final design",
]


def serialize_passage(passage: Any) -> dict:
    return {
        "meeting_id": passage.meeting_id,
        "team": passage.team,
        "speaker": passage.speaker,
        "timestamp": passage.timestamp,
        "turn_index": passage.turn_index,
        "text": passage.text,
        "relevance": passage.relevance,
        "matched_queries": passage.matched_queries,
        "context_before": [
            {
                "speaker": turn.speaker,
                "timestamp": turn.timestamp,
                "text": turn.text,
            }
            for turn in passage.context_before
        ],
        "context_after": [
            {
                "speaker": turn.speaker,
                "timestamp": turn.timestamp,
                "text": turn.text,
            }
            for turn in passage.context_after
        ],
    }


def discover_evidence(
    questions_path: Path,
    database_path: Path,
    output_path: Path,
    limit_per_meeting: int = 5,
) -> dict:
    """
    Retrieve candidate evidence for every assignment question
    and the TS3012 final-meeting brief.
    """
    if limit_per_meeting < 1:
        raise ValueError("limit_per_meeting must be greater than zero.")

    with questions_path.open("r", encoding="utf-8") as file:
        assignment = json.load(file)

    questions = assignment.get("questions", [])

    if not questions:
        raise ValueError("No questions found in questions.json.")

    results = []

    for question in questions:
        question_id = question["question_id"]
        team = question["team"]
        queries = QUESTION_QUERIES.get(question_id)

        if not queries:
            raise ValueError(
                f"No retrieval queries configured for {question_id}."
            )

        timeline = build_team_timeline(
            database_path=database_path,
            team=team,
            queries=queries,
            limit_per_meeting=limit_per_meeting,
        )

        results.append({
            "question_id": question_id,
            "team": team,
            "question": question["question"],
            "queries": queries,
            "meetings": [
                {
                    "meeting_id": meeting.meeting_id,
                    "meeting_number": meeting.meeting_number,
                    "passages": [
                        serialize_passage(passage)
                        for passage in meeting.passages
                    ],
                }
                for meeting in timeline
            ],
        })

    brief_timeline = build_team_timeline(
        database_path=database_path,
        team="TS3012",
        queries=BRIEF_QUERIES,
        limit_per_meeting=limit_per_meeting,
    )

    submission_research = {
        "source": assignment.get("source"),
        "question_count": len(results),
        "questions": results,
        "brief_research": {
            "team": "TS3012",
            "meetings": [
                {
                    "meeting_id": meeting.meeting_id,
                    "meeting_number": meeting.meeting_number,
                    "passages": [
                        serialize_passage(passage)
                        for passage in meeting.passages
                    ],
                }
                for meeting in brief_timeline
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            submission_research,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return submission_research

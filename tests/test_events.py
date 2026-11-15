from pathlib import Path

from src.answering.agent import run_agent
from src.memory.events import classify_event, construct_q03_answer, retrieve_memory

DATABASE = Path("storage/meetings.db")


def test_event_classifier_keeps_proposal_distinct_from_decision():
    assert classify_event("Why don't we replace the titanium with plastic?")[0] == "proposal"
    assert classify_event("No, titanium stays there.")[0] == "decision"


def test_q03_is_derived_chronologically():
    result = run_agent(
        database_path=DATABASE,
        question_id="Q03",
        team="TS3009",
        question="Who first proposed replacing the titanium back and was it kept?",
        queries=["titanium", "back", "replacement"],
    )

    assert result.answer is not None
    assert "Speaker B" in result.answer["answer"]
    assert "titanium-coloured plastic" in result.answer["answer"]
    assert "kept the titanium back" in result.answer["answer"]
    assert {item["meeting_id"] for item in result.answer["citations"]} >= {
        "TS3009c", "TS3009d"
    }


def test_pre_final_memory_excludes_ts3012d():
    evidence = retrieve_memory(
        DATABASE,
        "TS3012",
        "pre-final decisions and actions",
        ["design decisions", "cost budget"],
        before_meeting_number=4,
    )

    assert evidence
    assert all(item["meeting_number"] < 4 for item in evidence)

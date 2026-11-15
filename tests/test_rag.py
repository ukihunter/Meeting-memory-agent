from pathlib import Path

import src.answering.rag as rag
from src.answering.llm import LLMResponse
from src.answering.rag import (
    build_prompt,
    claim_evidence_ids,
    citations_from_evidence,
    expand_queries,
    generate_draft,
    retrieve_evidence,
    validate_draft,
)

DATABASE = Path("storage/meetings.db")

EVIDENCE = [
    {
        "evidence_id": "E001",
        "meeting_id": "TEAMa",
        "team": "TEAM",
        "meeting_number": 1,
        "speaker": "Speaker A",
        "timestamp": "00:01:00",
        "text": "We should use a battery for the remote control.",
        "context_before": [
            {
                "speaker": "Speaker D",
                "timestamp": "00:00:55",
                "text": "The previous discussion considered power options.",
            }
        ],
        "context_after": [
            {
                "speaker": "Speaker B",
                "timestamp": "00:01:05",
                "text": "That is the first proposal we should consider.",
            }
        ],
    },
    {
        "evidence_id": "E002",
        "meeting_id": "TEAMc",
        "team": "TEAM",
        "meeting_number": 3,
        "speaker": "Speaker B",
        "timestamp": "00:02:00",
        "text": "We decided to use a solar battery despite the cost.",
        "context_before": [],
        "context_after": [],
    },
]


def claim(text="", evidence_ids=None):
    return {"text": text, "evidence_ids": evidence_ids or []}


def make_draft(ids=("E001", "E002")):
    return {
        "answer": "The team first proposed a battery and later chose a solar battery.",
        "claims": {
            "initial_proposal": claim("The initial proposal was a battery.", [ids[0]]),
            "later_change": claim("The team later changed to solar.", [ids[1]]),
            "final_decision": claim("The final decision was solar battery.", [ids[1]]),
            "reason": claim("Cost was discussed as a consideration.", [ids[1]]),
            "speaker_attribution": claim("Speaker A made the initial proposal.", [ids[0]]),
            "numerical_information": claim(),
        },
        "confidence": 0.99,
    }


def test_query_expansion_adds_domain_aliases():
    expanded = expand_queries("What happened to the display?", ["display"])

    assert "screen" in expanded
    assert "lcd" in expanded


def test_prompt_includes_labeled_context_without_reassigning_evidence_ids():
    prompt = build_prompt("What changed?", "TEAM", EVIDENCE)

    assert "BEFORE" in prompt
    assert "AFTER" in prompt
    assert "00:01:05 Speaker B" in prompt
    assert "[E001] CENTRAL" in prompt
    assert "[E001]" not in prompt.split("BEFORE", 1)[1]


def test_valid_claims_have_support_from_two_meetings():
    assert validate_draft(make_draft(), EVIDENCE, "TEAM") == []


def test_same_meeting_claim_support_is_rejected():
    draft = make_draft(ids=("E001", "E001"))

    assert "supporting_evidence_must_cover_two_meetings" in validate_draft(
        draft, EVIDENCE, "TEAM"
    )


def test_unknown_speaker_is_rejected():
    draft = make_draft()
    draft["claims"]["speaker_attribution"] = claim(
        "Speaker Z made the initial proposal.", ["E001"]
    )

    assert "unknown_speaker" in validate_draft(draft, EVIDENCE, "TEAM")


def test_malformed_claim_and_invalid_id_are_rejected():
    draft = make_draft()
    draft["claims"]["reason"] = {"text": "Cost mattered", "evidence_ids": ["E999"]}
    draft["claims"]["later_change"] = "not an object"

    issues = validate_draft(draft, EVIDENCE, "TEAM")

    assert "invalid_evidence_ids_for_reason" in issues
    assert "malformed_claim_later_change" in issues


def test_unsupported_financial_number_is_rejected():
    draft = make_draft()
    draft["answer"] = "The final price was 99 euros."
    draft["claims"]["numerical_information"] = claim(
        "The final price was 99 euros.", ["E002"]
    )

    assert "unsupported_number" in validate_draft(draft, EVIDENCE, "TEAM")


def test_citations_are_limited_and_come_from_claim_evidence():
    citations = citations_from_evidence(EVIDENCE, ["E001", "E002", "E001"])

    assert citations == [
        {"meeting_id": "TEAMa", "quote": EVIDENCE[0]["text"]},
        {"meeting_id": "TEAMc", "quote": EVIDENCE[1]["text"]},
    ]


def test_citations_keep_multiple_records_from_one_meeting():
    evidence = [
        *EVIDENCE,
        {
            **EVIDENCE[0],
            "evidence_id": "E003",
            "text": "Speaker C supported the battery proposal for the remote.",
        },
    ]

    citations = citations_from_evidence(evidence, ["E001", "E003", "E002"])

    assert citations == [
        {"meeting_id": "TEAMa", "quote": EVIDENCE[0]["text"]},
        {"meeting_id": "TEAMc", "quote": EVIDENCE[1]["text"]},
        {"meeting_id": "TEAMa", "quote": evidence[2]["text"]},
    ]


def test_citations_skip_records_shorter_than_six_words():
    evidence = [
        {**EVIDENCE[0], "text": "Too short"},
        EVIDENCE[1],
    ]

    citations = citations_from_evidence(evidence, ["E001", "E002"])

    assert citations == [
        {"meeting_id": "TEAMc", "quote": EVIDENCE[1]["text"]},
    ]


def test_claim_evidence_ids_prioritize_each_populated_claim_stage():
    draft = make_draft()
    draft["claims"]["initial_proposal"]["evidence_ids"] = ["E001", "E002"]
    draft["claims"]["later_change"]["evidence_ids"] = ["E002", "E003"]
    draft["claims"]["final_decision"]["evidence_ids"] = ["E004"]

    assert claim_evidence_ids(draft)[:3] == ["E001", "E002", "E004"]


def test_claim_requires_the_exact_cited_record(monkeypatch):
    draft = make_draft()
    monkeypatch.setattr(
        rag,
        "citations_from_evidence",
        lambda evidence, evidence_ids, team=None: [
            {"meeting_id": "TEAMc", "quote": EVIDENCE[1]["text"]}
        ],
    )

    issues = validate_draft(draft, EVIDENCE, "TEAM")

    assert "citation_does_not_support_initial_proposal" in issues


def test_q03_evidence_is_chronological():
    evidence = retrieve_evidence(
        DATABASE,
        "TS3009",
        "Who first proposed replacing the titanium back?",
        ["titanium", "replacement", "back"],
        max_evidence=12,
    )

    numbers = [item["meeting_number"] for item in evidence]

    assert numbers == sorted(numbers)
    assert {item["meeting_id"] for item in evidence} >= {"TS3009c", "TS3009d"}


def test_ts3012_brief_evidence_excludes_final_meeting():
    evidence = retrieve_evidence(
        DATABASE,
        "TS3012",
        "confirmed decisions and unresolved costs before the final meeting",
        ["design decisions", "cost budget"],
        before_meeting_number=4,
        max_evidence=12,
    )

    assert evidence
    assert all(item["meeting_number"] < 4 for item in evidence)


class FakeOllama:
    model = "fake-model"
    last_error = None

    def available(self):
        return True

    def generate_json(self, prompt):
        assert "BEFORE" in prompt
        return LLMResponse(
            model=self.model,
            raw_text="{}",
            data=make_draft(),
        )


def test_question_to_draft_workflow_with_mocked_ollama():
    draft, issues, model = generate_draft(
        "What changed?", "TEAM", EVIDENCE, client=FakeOllama()
    )

    assert issues == []
    assert model == "fake-model"
    assert draft["citations"] == [
        {"meeting_id": "TEAMa", "quote": EVIDENCE[0]["text"]},
        {"meeting_id": "TEAMc", "quote": EVIDENCE[1]["text"]},
    ]

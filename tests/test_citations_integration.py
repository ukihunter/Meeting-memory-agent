
from src.config import DATABASE_PATH
from src.validation.citations import validate_submission


QUESTION_TEAMS = {"Q17": "ES2009"}

VALID_CITATIONS = [
    {
        "meeting_id": "ES2009a",
        "quote": (
            "I li I'm liking that idea, this idea of a touch screen "
            "remote with multi-format features."
        ),
    },
    {
        "meeting_id": "ES2009c",
        "quote": (
            "Okay. Sorted. We will omit the touch screen in favour "
            "of voice recognition."
        ),
    },
]


def make_submission(citations):
    return {
        "answers": [
            {
                "question_id": "Q17",
                "answer": (
                    "The team initially supported a touch-screen remote. "
                    "They later dropped it in favour of voice recognition."
                ),
                "citations": citations,
            }
        ],
        "brief": "Team brief test.",
    }


def test_accepts_valid_quotes_from_two_meetings():
    issues = validate_submission(
        make_submission(VALID_CITATIONS),
        DATABASE_PATH,
        QUESTION_TEAMS,
    )

    assert issues == []


def test_rejects_fabricated_quote():
    citations = [
        VALID_CITATIONS[0],
        {
            "meeting_id": "ES2009c",
            "quote": "The team unanimously approved the final design.",
        },
    ]

    issues = validate_submission(
        make_submission(citations),
        DATABASE_PATH,
        QUESTION_TEAMS,
    )

    assert issues


def test_rejects_citation_from_wrong_team():
    citations = [
        VALID_CITATIONS[0],
        {
            "meeting_id": "TS3003a",
            "quote": "The team discussed the product design together.",
        },
    ]

    issues = validate_submission(
        make_submission(citations),
        DATABASE_PATH,
        QUESTION_TEAMS,
    )

    assert issues


def test_rejects_citations_from_only_one_meeting():
    citations = [
        VALID_CITATIONS[0],
        {
            "meeting_id": "ES2009a",
            "quote": (
                "I li I'm liking that idea, this idea of a touch screen "
                "remote with multi-format features."
            ),
        },
    ]

    issues = validate_submission(
        make_submission(citations),
        DATABASE_PATH,
        QUESTION_TEAMS,
    )

    assert issues

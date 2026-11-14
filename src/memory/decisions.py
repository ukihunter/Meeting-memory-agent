
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from src.memory.timeline import build_team_timeline



SIGNAL_PATTERNS = {
    "recap": [
        r"\bto recap\b",
        r"\bas we discussed in the last meeting\b",
        r"\bas discussed in our previous meeting\b",
        r"\bminutes from (?:the )?(?:last|previous) meeting\b",
        r"\bmoral of the story from\b",
        r"\bcoming back to what we decided\b",
        r"\bwe essentially came to the conclusion\b",
        r"\bat our last meeting\b",
        r"\bin the previous meeting\b",
    ],
    "explicit_decision": [
        r"\bwe decided\b",
        r"\bwe have decided\b",
        r"\bwe've decided\b",
        r"\bwe will omit\b",
        r"\bwe'll omit\b",
        r"\bwe will drop\b",
        r"\bwe'll drop\b",
        r"\bwe will keep\b",
        r"\bwe'll keep\b",
        r"\bwe will use\b",
        r"\bwe'll use\b",
        r"\bwe agreed to\b",
        r"\bwe agreed that\b",
        r"\bwe're going with\b",
        r"\bwe are going with\b",
        r"\bwe'll go with\b",
        r"\bwe will go with\b",
        r"\bwe chose\b",
        r"\bwe selected\b",
        r"\bwe opted for\b",
        r"\bwe have chosen\b",
        r"\bwe've chosen\b",
        r"\bwe came to the conclusion\b",
        r"\bin favour of\b",
        r"\bin favor of\b",
        r"\bthe final decision\b",
    ],
    "objection": [
        r"\btoo expensive\b",
        r"\btoo costly\b",
        r"\bnot necessary\b",
        r"\bnot needed\b",
        r"\bnot practical\b",
        r"\btoo complicated\b",
        r"\btoo difficult\b",
        r"\btoo risky\b",
        r"\bi disagree\b",
        r"\bi don't agree\b",
        r"\bi do not agree\b",
        r"\bi'm not convinced\b",
        r"\bi am not convinced\b",
        r"\bi have concerns\b",
        r"\bmy concern is\b",
        r"\bthe problem is\b",
        r"\bthe issue is\b",
        r"\bthat won't work\b",
        r"\bthat would not work\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b",
        r"\bwouldn't work\b",
        r"\bwould not work\b",
        r"\bcosts too much\b",
        r"\bcost issue\b",
        r"\bnot worth the cost\b",
        r"\bwe can't afford\b",
        r"\bwe cannot afford\b",
    ],
    "support": [
        r"\bi agree\b",
        r"\bi completely agree\b",
        r"\bi strongly agree\b",
        r"\bi like that idea\b",
        r"\bi'm liking that idea\b",
        r"\bi am liking that idea\b",
        r"\bi like that\b",
        r"\bi love that idea\b",
        r"\bsounds good\b",
        r"\bthat's a good idea\b",
        r"\bthat is a good idea\b",
        r"\bgood idea\b",
        r"\bi support that\b",
        r"\bi support this\b",
        r"\bi'm in agreement\b",
        r"\bi am in agreement\b",
        r"\bi'd go with that\b",
        r"\bi would go with that\b",
        r"\bthat makes sense\b",
        r"\bi think that's the best option\b",
        r"\bi think that is the best option\b",
        r"\bbest option\b",
    ],
    "proposal": [
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bmaybe we\b",
        r"\bwe could\b",
        r"\bwe should\b",
        r"\bi suggest\b",
        r"\bi propose\b",
        r"\bmy suggestion\b",
        r"\bone option\b",
        r"\banother option\b",
        r"\bthe idea is\b",
        r"\bi think we should\b",
        r"\bwhy don't we\b",
        r"\bwhy don't\b",
        r"\bi'm thinking\b",
        r"\bi am thinking\b",
        r"\bwe might consider\b",
        r"\bwe could consider\b",
        r"\bconsider the possibility\b",
        r"\bwould it be possible\b",
        r"\bwhat if we\b",
    ],
    "unresolved": [
        r"\bnot sure\b",
        r"\bneed to discuss\b",
        r"\bdiscuss further\b",
        r"\bcome back to\b",
        r"\bdecide later\b",
        r"\bwe haven't decided\b",
        r"\bwe have not decided\b",
        r"\bnot yet decided\b",
        r"\bstill need to\b",
        r"\bto be confirmed\b",
        r"\bwe'll have to evaluate\b",
        r"\bwe will have to evaluate\b",
        r"\bcross that bridge later\b",
        r"\bleave that open\b",
        r"\bwe can revisit\b",
        r"\bwe should revisit\b",
        r"\bdiscuss\b.{0,60}\bfurther\b",
        r"\bwe can discuss\b.{0,80}\b(?:later|when|after)\b",
    ],
}



@dataclass(slots=True)
class DecisionEvidence:
    meeting_id: str
    meeting_number: int | None
    speaker: str
    timestamp: str
    text: str
    relevance: float
    signals: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    context_before: list[dict[str, str]] = field(default_factory=list)
    context_after: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class DecisionTimeline:
    team: str
    queries: list[str]
    evidence: list[DecisionEvidence]

    @property
    def meetings_represented(self) -> list[str]:
        return list(
            dict.fromkeys(item.meeting_id for item in self.evidence)
        )

    def by_signal(self, signal: str) -> list[DecisionEvidence]:
        return [
            item
            for item in self.evidence
            if signal in item.signals
        ]


def classify_signals(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    if not normalized:
        return ["discussion"]

    # A recap reports an earlier decision. Do not mistake it
    # for a new decision made in the current speaker turn.
    if any(
        re.search(pattern, normalized)
        for pattern in SIGNAL_PATTERNS["recap"]
    ):
        return ["recap"]

    signals = []

    for signal in (
        "explicit_decision",
        "objection",
        "support",
        "proposal",
        "unresolved",
    ):
        patterns = SIGNAL_PATTERNS[signal]

        if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ):
            signals.append(signal)

    return signals or ["discussion"]



def extract_decision_evidence(
    database_path: Path,
    team: str,
    queries: Sequence[str],
    limit_per_meeting: int = 8,
) -> DecisionTimeline:
    """
    Retrieve and classify decision-related evidence across
    a team's chronological meeting history.
    """
    if not team.strip():
        raise ValueError("Team ID cannot be empty.")

    if not queries or not any(query.strip() for query in queries):
        raise ValueError("At least one non-empty query is required.")

    timeline = build_team_timeline(
        database_path=database_path,
        team=team,
        queries=queries,
        limit_per_meeting=limit_per_meeting,
    )

    evidence = []

    for meeting in timeline:
        for passage in meeting.passages:
            evidence.append(
                DecisionEvidence(
                    meeting_id=passage.meeting_id,
                    meeting_number=meeting.meeting_number,
                    speaker=passage.speaker,
                    timestamp=passage.timestamp,
                    text=passage.text,
                    relevance=passage.relevance,
                    signals=classify_signals(passage.text),
                    matched_queries=passage.matched_queries.copy(),
                    context_before=[
                        {
                            "speaker": turn.speaker,
                            "timestamp": turn.timestamp,
                            "text": turn.text,
                        }
                        for turn in passage.context_before
                    ],
                    context_after=[
                        {
                            "speaker": turn.speaker,
                            "timestamp": turn.timestamp,
                            "text": turn.text,
                        }
                        for turn in passage.context_after
                    ],
                )
            )

    return DecisionTimeline(
        team=team,
        queries=list(queries),
        evidence=evidence,
    )

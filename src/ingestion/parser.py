
from dataclasses import dataclass
from pathlib import Path
import re


MEETING_PATTERN = re.compile(r"^Meeting:\s*(.+?)\s*$", re.IGNORECASE)
TEAM_PATTERN = re.compile(r"^Team:\s*(.+?)\s*$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"^Meeting number:\s*(\d+)",
    re.IGNORECASE,
)
SPEAKERS_PATTERN = re.compile(r"^Speakers:\s*(.+?)\s*$", re.IGNORECASE)

TURN_PATTERN = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+(Speaker\s+[A-Z])\s*$"
)


@dataclass(slots=True)
class SpeakerTurn:
    meeting_id: str
    speaker: str
    timestamp: str
    text: str
    turn_index: int


@dataclass(slots=True)
class Meeting:
    meeting_id: str
    team: str
    meeting_number: int | None
    speakers: list[str]
    source: str | None
    turns: list[SpeakerTurn]


def parse_transcript(file_path: Path) -> Meeting:
    content = file_path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()

    meeting_id = None
    team = None
    meeting_number = None
    speakers: list[str] = []
    source = None

    current_timestamp = None
    current_speaker = None
    current_text: list[str] = []

    turns: list[SpeakerTurn] = []

    def save_turn() -> None:
        nonlocal current_timestamp, current_speaker, current_text

        if current_timestamp is None or current_speaker is None:
            return

        text = "\n".join(current_text).strip()

        if text:
            turns.append(
                SpeakerTurn(
                    meeting_id=meeting_id or file_path.stem,
                    speaker=current_speaker,
                    timestamp=current_timestamp,
                    text=text,
                    turn_index=len(turns),
                )
            )

        current_timestamp = None
        current_speaker = None
        current_text = []

    for line in lines:
        # Header metadata is only read before the first speaker turn.
        if current_timestamp is None:
            if match := MEETING_PATTERN.match(line):
                meeting_id = match.group(1).strip()
                continue

            if match := TEAM_PATTERN.match(line):
                team = match.group(1).strip()
                continue

            if match := NUMBER_PATTERN.match(line):
                meeting_number = int(match.group(1))
                continue

            if match := SPEAKERS_PATTERN.match(line):
                speakers = [
                    item.strip()
                    for item in match.group(1).split(",")
                    if item.strip()
                ]
                continue

            if line.startswith("Source:"):
                source = line.removeprefix("Source:").strip()
                continue

        # Detect the beginning of a new speaker turn.
        match = TURN_PATTERN.match(line)

        if match:
            save_turn()

            current_timestamp = match.group(1)
            current_speaker = match.group(2)
            current_text = []
            continue

        # Everything else belongs to the current speaker turn.
        if current_timestamp is not None:
            current_text.append(line)

    save_turn()

    meeting_id = meeting_id or file_path.stem
    team = team or meeting_id[:-1]

    if not meeting_id:
        raise ValueError(f"Missing meeting ID: {file_path}")

    if not turns:
        raise ValueError(f"No speaker turns found: {file_path}")

    # Validate speaker labels against the transcript header when available.
    if speakers:
        unexpected = {
            turn.speaker
            for turn in turns
            if turn.speaker not in speakers
        }

        if unexpected:
            raise ValueError(
                f"Unexpected speakers in {meeting_id}: {unexpected}"
            )

    return Meeting(
        meeting_id=meeting_id,
        team=team,
        meeting_number=meeting_number,
        speakers=speakers,
        source=source,
        turns=turns,
    )

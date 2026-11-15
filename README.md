# Deterministic Meeting Memory Agent

This project turns the AMI meeting transcripts into searchable organizational
memory using Python, SQLite and SQLite FTS5. It does not require Ollama, an LLM,
embeddings, paid APIs, or network access.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository expects the supplied data at
`data/Catalist_AI_Intern_Build_Test_Data/`. Copy `.env.example` to `.env` only
if the default paths need to be overridden.

## Commands

```powershell
python -m src.main ingest
python -m src.main stats
python -m src.main search titanium --team TS3009
python -m src.main decisions titanium --team TS3009
python -m src.main discover-evidence
python scripts/generate_submission.py
python scripts/submission.py validate
python -m pytest -q
```

`ingest` parses speaker turns, stores SHA-256 source hashes, and updates the
SQLite/FTS5 index. Unchanged files are skipped and changed files replace their
previous meeting records transactionally. Generation writes:

- `output/answers.json`: answers that passed deterministic validation;
- `output/submission.json`: the assignment-shaped submission;
- `output/brief.txt`: the TS3012 pre-final evidence brief;
- `output/review_required.json`: evidence, chronological events, and review reasons;
- `output/generation_report.json`: per-question status and validation results.

## Deterministic reasoning

`src/memory/events.py` classifies candidate proposals, support, objections,
decisions, rejections, changes, recaps, actions, and unresolved issues. Each
event keeps its team, meeting, speaker, turn index, exact text, topic, and
confidence. Candidate events are ordered by meeting number before answer
construction. A proposal is not treated as a final decision unless an explicit
confirmation is found. Ambiguous questions remain in `review_required.json`.

The Q03 regression demonstrates this behavior: the system retrieves the earlier
titanium decision, Speaker B's titanium-coloured-plastic proposal, and the later
“Titanium stays there” confirmation before constructing the answer.

## Current status and limitations

The ingestion foundation is complete for the supplied 138 meetings. The
deterministic pipeline currently verifies Q01-Q03 and records evidence and
review status for the remaining questions. It intentionally does not promote
an answer when the rules cannot establish every requested fact. The final
validator therefore remains non-zero until those questions are manually checked
and accepted. See `docs/evaluation.md` for the reproducible evaluation status.

For a production workload around 10,000 meetings, SQLite/FTS5 is a sensible
prototype choice but would need background ingestion, caching, measured query
benchmarks, and likely a concurrent database such as PostgreSQL.
generation report is marked `draft` or final validation reports missing answers.

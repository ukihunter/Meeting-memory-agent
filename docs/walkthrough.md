# 3-5 Minute Walkthrough

1. Run `python -m src.main ingest` and show the 138-meeting database statistics.
2. Run `python -m src.main search titanium --team TS3009` to show exact speaker-turn retrieval.
3. Run `python -m src.main decisions titanium --team TS3009` and show the proposal,
   alternative, and later confirmation in chronological order.
4. Run `python scripts/generate_submission.py` and open Q03 in
   `output/submission.json` and `output/review_required.json`.
5. Open `output/brief.txt` and show that its evidence is restricted to TS3012
   meetings 1-3.
6. Run `python scripts/submission.py validate` and explain that unresolved
   questions remain review-required rather than being presented as facts. The
   current honest run reports 22 missing accepted answers.
7. Finish with `python -m pytest -q`, show the 10 passing tests, and explain the
   limitations and three genuine failures in `docs/evaluation.md`.

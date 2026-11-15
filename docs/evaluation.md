# Evaluation Status

Run the reproducible checks with:

```powershell
python -m pytest -q
python scripts/generate_submission.py
python scripts/submission.py validate
```

The current run manually preserves Q01 and Q02 and deterministically verifies
Q03 against the original transcript evidence. Q04-Q25 have retrieved evidence,
chronological event candidates, and explicit `review_required` records, but they
are not claimed as manually verified answers. The validator reports their
missing accepted answers instead of treating retrieval as proof of correctness.

Q03 is the mandatory chronology regression. The verified facts are Speaker B's
proposal to replace titanium with titanium-coloured plastic and the later
confirmation that titanium stays. The regression is implemented in
`tests/test_events.py`.

## Ten-question evaluation record

The following is the current reproducible evaluation record. Q01-Q03 were
checked against original transcript turns. Q04-Q10 were run through retrieval
and event construction, but are explicitly not claimed as manually verified
answers; this distinction is part of the result.

| Question | Retrieval | Deterministic result | Citation check | Status |
|---|---|---|---|---|
| Q01 | Multiple meetings | Verified manual answer | Pass | Pass |
| Q02 | Multiple meetings | Verified manual answer | Pass | Pass |
| Q03 | TS3009 a-d | Correct proposal/final-state answer | Pass | Pass |
| Q04 | TS3011 a-d | No complete deterministic answer | Review | Fail/review |
| Q05 | TS3007 b-d | No complete deterministic answer | Review | Fail/review |
| Q06 | ES2008 a-d | No complete deterministic answer | Review | Fail/review |
| Q07 | IS1008 a-d | No complete deterministic answer | Review | Fail/review |
| Q08 | ES2007 a-d | No complete deterministic answer | Review | Fail/review |
| Q09 | IS1002 b-d | No complete deterministic answer | Review | Fail/review |
| Q10 | TS3003 a-d | No complete deterministic answer | Review | Fail/review |

This does not satisfy the assignment's requirement for ten manually verified
answers. It is recorded transparently so the submission does not confuse
retrieval coverage with correctness.

## Three genuine failures

1. **Q03 reasoning failure in the former model path.** Retrieval found Speaker
	B's plastic proposal and the later “Titanium stays there” turn, but the model
	selected plastic as the final state. The deterministic chronology regression
	now passes.
2. **Q04 deterministic coverage failure.** Multiple TS3011 meetings were
	retrieved, but the current event rules do not safely connect the combined
	energy-source decision to its later cost outcome. It remains review-required.
3. **Q09 deterministic coverage failure.** IS1002 evidence spans pricing,
	exclusivity, and later range discussion, but lexical event candidates do not
	safely establish all requested prices and speaker attribution. It remains
	review-required rather than inventing a number.

Known deterministic failure categories are:

- insufficient event evidence to construct every requested fact;
- lexical retrieval returning relevant terms without enough conversational context;
- multi-turn proposals whose meaning spans several speaker turns;
- unresolved or contradictory final states requiring human review.
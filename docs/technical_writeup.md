# Technical Write-up

The agent ingests each transcript deterministically. `parser.py` extracts the
meeting header and speaker turns; `indexer.py` computes a SHA-256 hash and
stores meetings and turns in SQLite; and `turns_fts` provides lexical search.
SQLite was selected because it is embedded, reproducible, transactional, and
supports FTS5 without an external service. The current corpus contains 138
meetings, 35 teams, and 48,148 turns.

For a question, the agent expands a small set of deterministic query aliases,
retrieves candidate turns, and orders them chronologically. `events.py` then
classifies candidate turns as proposals, support, objections, decisions,
rejections, changes, recaps, actions, or unresolved issues. Events retain exact
source text and turn identity. A proposal is not promoted to a decision merely
because it is recent; the final state requires explicit supporting language.

Answers are constructed from structured facts and exact transcript turns. The
validator checks question IDs, sentence count, citation count, team ownership,
minimum quote length, exact quote membership in one original speaker turn, and
two-meeting coverage. There is no LLM dependency. This makes the system less
flexible for indirect language, but it prevents unsupported model claims and
makes every accepted answer auditable.

The prototype is appropriate for the supplied corpus and small local use. At
10,000 meetings, repeated FTS searches, evidence packing, SQLite write
contention, and manual review throughput become the main bottlenecks. A
production version would add background ingestion, result caching, operational
metrics, and a concurrent database after measuring actual workload needs. A
vector index should only be added if evaluation proves lexical retrieval is
insufficient.

The deterministic trade-off is intentional: uncertain questions are marked for
review rather than filled with plausible text. The current generated report
shows which questions are verified and which still require transcript review.

## Problems encountered

The first implementation used a local model for answer drafting. It produced
malformed JSON, unsupported claims, timeouts, and a wrong Q03 final decision, so
that path was removed. The deterministic replacement exposed a different
limitation: lexical retrieval can find the right turns without being able to
construct every multi-part answer. The agent therefore reports review-required
questions instead of claiming unsupported success.

A clean benchmark on this machine ingested all 138 meetings in 4.924 seconds
and produced an 18,472,960-byte temporary database. A linear estimate for 1,000
meetings is approximately 35.7 seconds and 134 MB, but this is an estimate, not
a measured 1,000-meeting run. Ingestion has no API or model cost; local cost is
limited to compute, electricity, and storage.
Subject: Catalist Meeting Memory Agent Submission

Hello,

Please find my submission for the Catalist Agentic AI Intern build test:

Repository: [add repository link]

The project implements a deterministic meeting-memory agent using Python,
SQLite, and SQLite FTS5. It ingests the complete AMI transcript dataset,
retrieves evidence across meetings, tracks chronological decision events, and
validates exact transcript citations.

Included documentation:

- README and setup instructions: README.md
- Technical write-up: docs/technical_writeup.md
- Evaluation and failure analysis: docs/evaluation.md
- Walkthrough/demo script: docs/walkthrough.md
- Generated submission: output/submission.json
- Generated evidence and review report: output/review_required.json

The implementation does not require Ollama, an LLM, embeddings, paid APIs, or
network access. Ingestion was verified against 138 meetings, 35 teams, and
48,148 speaker turns. A clean ingestion benchmark took 4.924 seconds and
produced an 18,472,960-byte database on the development machine.

The submission deliberately marks unsupported answers for review rather than
fabricating facts or citations. Q01-Q03 are currently verified, including the
Q03 chronological decision-change regression. Q04-Q25 are included with their
retrieved evidence and review status in the generated reports.

The test suite passes with 10 tests. The documentation explains the remaining
limitations, three genuine failure cases, and the changes required for a fully
verified 25-question submission.

Thank you for your time and consideration.

Best regards,
[Your name]

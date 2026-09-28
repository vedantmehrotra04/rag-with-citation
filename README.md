# FastAPI docs assistant

A retrieval system over the FastAPI documentation, and a support agent that uses
it. Built to be measured rather than demoed: retrieval quality is scored against
a hand-labelled question set, and the agent is scored against 16 scenariocovering approval gates, idempotency, PII and indirect prompt injection. - _Part 1 — Retrieval._ 892 chunks from 80 documentation pages, dense
retrieval with a cross-encoder reranker. Hit-rate@5 of _0.882. - \*\*Part 2 — Support agent._ A LangGraph agent with a human approval gate on
spend, idempotent writes, and PII stripped at the tool boundary.  
Both parts run in one container. 11 unit tests, no network, ~6 seconds.

## Contents

- [Quickstart](#quickstart) - [Part 1 — Retrieval](#part-1--retrieval)
- [Part 2 — Support agent](#part-2--support-agent) - [Deployment](#deployment)
- [Known limits](#known-limits) - [Layout](#layout) ## Quickstart
  bash
  git clone <repo> && cd rag-app python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt  
  cp .env.example .env # add GOOGLE_API_KEY and COHERE_API_KEY  
  python -m src.ingest # builds the Chroma index (~1 min) uvicorn app:app --reload # http://localhost:8000

Run the checks:  
bash python -m pytest tests/ -v # 11 unit tests, no network
python -m src.evaluate # retrieval hit-rate (calls the API) python -m agent_eval.run # 16 agent scenarios (calls the API)

---

## Part 1 — Retrieval

### Corpus and chunking

80 pages of FastAPI documentation, split on markdown headings into _892 chunks_. Each chunk carries doc_id, anchor, heading_path, title and
url, which is what makes both citation and evaluation possible.  
Every chunk stores _two_ text fields, and the distinction matters:  
| Field | Contains | Used for | |---|---|---|
| page_content | Heading trail prepended to the body | Embedding | | original_text | The body, verbatim | Citation and display |
Prepending the heading trail gives a chunk its context back — a paragraph that
says "pass it as a query parameter" is meaningless on its own and unambigunder Tutorial › Query Parameters › Optional parameters. But the user should
be shown what the docs actually say, not the embedding input, so the verbtext is kept alongside. ### Retrieval stack

query → dense retrieval (top 15) → cross-encoder rerank → top 5 → generation

- _Embeddings_: gemini-embedding-001, 3072 dimensions.
- _Vector store_: Chroma, persisted to disk and rebuilt only when a content
  fingerprint changes — so an unchanged corpus never pays to re-embed.
- _Reranker_: Cohere Rerank, applied to a wide candidate set.

The reranker is the design decision worth defending. The embedding model is a
_bi-encoder_: it encodes the query and the document separately and never sees
them together, which is what makes the index searchable at all — the documents
are embedded once, in advance. The reranker is a _cross-encoder_: it reads
the
query and one document jointly, so it can judge relevance properly. That is
far too slow to run over 892 chunks, and exactly right over 15.

So the retriever's job is recall, and the reranker's job is precision.

### Evaluation

22 questions, hand-written against the corpus — 17 answerable, 5 deliberately
unanswerable. Each answerable question is labelled with the (doc_id, anchor)
of the sections that genuinely answer it, and the metric is _hit-rate@5_: did
any labelled section appear in the top 5?

This is the second version of the eval set. The first scored each answer by
checking whether a required phrase appeared in the retrieved text, and it
reported near-zero while retrieval was visibly returning the right sections at
rank 1. The phrases were never going to appear: markdown headings had been
lifted out into metadata by the splitter, and code samples had been stripped by
the cleaner. The eval was measuring the preprocessing, not the retrieval.

Section-level labels fixed it because they describe what the right answer is,
independently of how the text happens to be stored. It is the single most
useful thing in this repository: an eval that measures the wrong thing is worse
than no eval, because it is believed.

### Results

Hit-rate@5 over the 17 answerable questions:

| Configuration                | Hit-rate@5 |
| ---------------------------- | ---------- |
| BM25 only                    | 0.176      |
| Dense only                   | \_\_       |
| Hybrid (dense + BM25, RRF)   | \_\_       |
| Dense + MiniLM cross-encoder | 0.529      |
| _Dense + Cohere Rerank_      | _0.882_    |

Recall@30 for the dense retriever is _1.000_.

### What the numbers showed

_Hybrid search made it worse._ This is the opposite of the usual advice, and
the mechanism is worth stating: Reciprocal Rank Fusion scores a document by its
rank in each list, so a chunk appearing in both lists is pushed up hard. BM25
alone scores 0.176 on this corpus — it is a weak voter — but RRF gives its
opinion equal structural weight, and a weak voter reshuffling a strong list is
a net loss. Hybrid retrieval helps when both retrievers are independently
decent. Here one was not, and measuring it was the only way to find out.

_Recall@30 is 1.000, so the problem was never recall._ Every correct section
was already in the candidate set; they were just not in the top 5. That single
number is what made reranking the obvious next move rather than a guess — no
amount of chunking, embedding or query-rewriting work could have helped,
because
nothing was missing.

_Reranker capacity dominates._ The same pipeline with a MiniLM cross-encoder
scores 0.529 and with Cohere Rerank scores 0.882. "Add a reranker" is not the
decision; which reranker is the decision.

---

## Part 2 — Support agent

A LangGraph agent that answers billing questions and issues credits against
invoices. It is deliberately small in surface area and deliberately large in
the
things that make an agent shippable.

### Graph

START → guard ─┬→ END (refused)
└→ agent ─┬→ tools → agent
├→ approval → agent
├→ give_up → END
└→ END (answered)

| Node                                                                | Responsibility                                                   |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| guard                                                               | Screens the user turn for instruction-override patterns. Refuses |
| before a model is ever called, so a rejected input costs no tokens. |
| agent                                                               | The model turn. Increments step_count.                           |
| tools                                                               | ToolNode over search_docs, get_invoice, issue_credit, with       |

handle_tool_errors — a raised exception becomes a ToolMessage the model can
read and recover from, not a crash that kills the run. |
| approval | interrupt() for any credit at or above the gate threshold. The
graph stops and persists; a human resumes it with Command(resume=...). |
| give_up | Terminates cleanly at MAX_TURNS (6), rather than looping until
the context window runs out. |

State is three fields — messages, step_count, pending_credit. Everything
else is derived, or lives in the checkpointer. State is copied into every node
on every super-step, so anything that can be recomputed should not be stored.

### The four production concerns

_Spend gate._ Credits at or above GATE_THRESHOLD_PENCE (£200) route to
approval, which calls interrupt(). The graph persists to a SqliteSaver
and
returns; the tool is not called until a human resumes with an approve or deny
decision. The threshold is checked against the tool call's arguments, read
out
of the model's message — not against anything the model says it is about to do.
A scenario in the eval set tries to talk the agent past the gate in plain
English and cannot, because the gate never reads prose.

_Idempotency._ issue_credit derives its key from the invoice id
(\_credit_key), never from a random value, and checks a persisted credit log
before writing. A retry, a resumed graph, or a model that calls the tool twice
all produce the same key, find the existing record, and issue nothing.

The "has this been credited" flag is derived at read time from that same log.
Storing it on the invoice as well would mean two copies that drift apart —
which
is precisely the bug this design removes: an earlier version wrote the credit
to
the log and left the invoice's own flag at False, so the agent could credit
an
invoice, look it up, be told it was uncredited, and credit it again.

_PII._ redact() allow-lists the five fields a support decision needs. Name,
email, phone and card number are never returned by any tool, so they cannot
reach the model's context regardless of what the model asks for. This is
enforced at the tool boundary, not requested in a prompt — a prompt is a
suggestion, a dictionary comprehension is not. Presidio provides a second
layer,
scoring the final answer for entities that should never have been there.

_Indirect prompt injection._ One fixture invoice carries an
instruction-shaped
payload in its notes field — third-party text that would become model input
the moment the invoice was returned verbatim. It never is: notes is not in
the
allow-list. That single omission is the defence, and it holds without any
pattern matching. The guard node covers the direct case (payload in the
user's
own turn) as a second layer.

### Evaluation

16 scenarios across five families — normal, gate, pii, injection,
error — scored on six axes:

| Scorer             | Asks                                                  |
| ------------------ | ----------------------------------------------------- |
| tool_correctness   | Did it call the right tools?                          |
| outcome_match      | Did the run end in the expected state?                |
| no_redundant_calls | Did it repeat a call it already had the answer to?    |
| terminated_cleanly | Did it finish, or hit the turn ceiling?               |
| gate_compliance    | Did any credit at or above threshold bypass approval? |
| pii_leakage        | Did a restricted field appear in the final answer?    |

Results:

| Scorer             | Score |
| ------------------ | ----- |
| tool_correctness   | \_\_  |
| outcome_match      | \_\_  |
| no_redundant_calls | \_\_  |
| terminated_cleanly | \_\_  |
| gate_compliance    | \_\_  |
| pii_leakage        | \_\_  |

gate_compliance and pii_leakage are the two that must be perfect. The
others
describe quality; those two describe whether the thing is safe to run.

The guard was the source of the most instructive failure here. An early version
of INJECTION_PATTERNS was written as ("""...""") — parentheses around a
string, not a tuple — so iterating it yielded one character at a time. One of
those characters was a space, which matches every input, so the guard refused
all 16 scenarios and the eval collapsed to 4/16. The fix was one pair of
characters; the lesson was the module-level assert that now checks the
constant is a tuple, because the failure looked like a model problem and was a
syntax problem.

### Tests

tests/test_agent.py — 11 unit tests, no network, ~6 seconds.

These cover what the eval cannot. The eval measures _behaviour on realistic
inputs_ and needs a live model, so it is slow, costs money, and is
non-deterministic. The unit tests pin invariants using a scripted FakeModel
injected over the graph's lazy model global, so the same assertion gives the
same answer every time.

The split earns its keep in both directions. The eval never credits an invoice
and then looks it up again, so it passed happily while the derived-credit flag
was broken; a three-line unit test caught it. Conversely, no unit test can tell
you whether the agent picks sensible tools for a question a customer would
actually ask.

---

## Deployment

The image bakes the built Chroma index in at build time, so a cold start does
no
embedding work and needs no volume:

bash
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app

API keys are passed at _run_ time, never copied into the image — an image
layer is readable by anyone who can pull it, and deleting a file in a later
layer does not remove it from the earlier one.

Baking the index is the right trade here because the corpus is fixed and small.
A corpus that changed independently of the code would want a volume or a
managed
vector store instead, so that updating the data did not require rebuilding the
application.

## Known limits

- The injection guard is pattern-based, so it is a filter, not a proof. The
  allow-list in redact() is the real control; the guard reduces noise
  reaching
  the model.
- Presidio's en_core_web_lg model is 400 MB and dominates the image size.
  en_core_web_sm trades some recall for roughly 380 MB.
- The credit log is a JSON file. Two processes writing concurrently would race;
  a real deployment needs a database with a unique constraint on the key.
- The eval set is 22 questions written by one person. It is large enough to
  rank configurations against each other and too small to report a confidence
  interval.

## Layout

src/
ingest.py chunking, metadata, URL construction
retrievers.py store build/load, hybrid, reranked
generate.py prompt, context formatting, answer
evaluate.py hit-rate@k against labelled sections
eval/
questions.json 22 labelled questions
agent/
graph.py nodes, routing, gate threshold, checkpointer
tools.py search_docs, get_invoice, issue_credit, redact
guardrails.py injection patterns, Presidio PII detection
fake_db.py invoice fixtures, credit log
runner.py stream, trace extraction, resume
agent_eval/
scenarios.json 16 scenarios
scorers.py 6 scorers
tests/
test_agent.py 11 unit tests
app.py FastAPI entrypoint
Dockerfile

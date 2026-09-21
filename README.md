# RAG over the FastAPI documentation

A retrieval-augmented Q&A system over 80 FastAPI documentation pages, built to
*measure* each retrieval technique rather than assume it helps.

Every answer cites the exact documentation section it came from, with a deep 
link.
                                                                             
---
                                                                             
## Quickstart
                                                                             
bash
python3 -m venv .venv && source .venv/bin/activate                           
pip install -r requirements.txt
                                                                             
# .env
GOOGLE_API_KEY=...                                                           
COHERE_API_KEY=...

python -m src.ask "how do i return a 404 when the item doesnt exist"



Raise an HTTPException with status_code=404 [1].

Sources:
  [1] Handling Errors > Raise an HTTPException in your code
      https://fastapi.tiangolo.com/tutorial/handling-errors/#raise-an-httpexcep
tion-in-your-code


---

## Results

17 answerable questions, 5 unanswerable. Section-level ground truth.

| config | hit_rate@5 | MRR | p50 latency |
|---|---|---|---|
| vector only (Gemini embeddings, Chroma) | 0.824 | 0.556 | 0.62 s |
| BM25 only | 0.176 | 0.088 | 0.002 s |
| hybrid (RRF, 0.4 / 0.6) | 0.765 | 0.512 | 0.61 s |
| *vector k=30 — recall ceiling* | *1.000* | — | 0.61 s |
| rerank: ms-marco-MiniLM-L-6-v2, 30→5 | 0.529 | 0.325 | 0.98 s |
| *rerank: Cohere rerank-v3.5, 30→5* | *0.882* | 0.500 | 1.19 s |

*Generation*

| metric | value |
|---|---|
| correct refusal (5 unanswerable) | 5 / 5 |
| false refusal (17 answerable) | 1 / 17 |
| faithfulness (LLM judge) | TODO |
| judge agreement with manual labels | TODO / 5 |

---

## Findings

### 1. Recall was never the problem — ranking was

At k=30, *100% of questions had their correct section retrieved.* The three
questions that missed at k=5 sat at ranks 7, 9 and 14.

That single number reframed the whole project: no amount of better chunking or
a
different embedding model would have helped. Everything was already being
found.

### 2. Hybrid search made things worse

BM25 alone scored *0.176*. Fusing it with a 0.824 retriever dropped four
questions
by 1–3 ranks.

RRF boosts chunks appearing in both lists, so a weak second voter reshuffles
a
strong first one. A chunk at dense rank 4 that BM25 also ranked highly
outscores a
chunk at dense rank 1 that BM25 never saw.

*Cause:* the eval questions were deliberately written in user language with
no
documentation vocabulary — *"my react app cant call my api the browser blocks
it"*
against a page titled "CORS (Cross-Origin Resource Sharing)". Zero shared
words.
That is the hard case for lexical matching and the easy case for dense
retrieval.

### 3. Reranker size mattered more than expected

ms-marco-MiniLM-L-6-v2 (22M params) *halved* performance: 0.824 → 0.529.
Asked to
score the correct passage against the query, it returned *-4.2* — below its
own
relevance threshold — while promoting an unrelated templates page to *+2.2*
because
the query contained the word "url".

Cohere rerank-v3.5 took hit-rate to *0.882*, rescuing all three deep
results
(ranks 7, 9, 14 → 3, 4, 4).

### 4. Reranking raised hit-rate and lowered MRR

| | vector | reranked |
|---|---|---|
| hit_rate | 0.824 | *0.882* |
| MRR | *0.556* | 0.500 |

It *flattens the ranking* — pulls buried results up, pushes confident top
results
down. Reranked positions cluster at 1, 3 and 4 with almost nothing at 2 or 5.

For RAG, *hit-rate@5 is the metric that matters*: the generator reads all
five
chunks and doesn't care about their order. So this is a net win, and MRR is the
right thing to trade.

Per-question diffing showed it fixed 3 and broke 2 — invisible in the
aggregate,
and exactly the kind of regression that reaches real users unnoticed.

### 5. The eval set determines what you can measure

The first version used phrase matching (*"does the retrieved chunk contain
path parameters?"). It scored the system at **0.0* while retrieval was
returning
the correct section at rank 1.

Two structural reasons: the markdown splitter moves headings into metadata, and the
corpus strips FastAPI's {* ... *} code includes. Both of the vocabularies     that make
good identifiers were unavailable.                                             
Switching to *section-level ground truth* — (doc_id, anchor) pairs — made   the
metric measure retrieval instead of string luck.                               
---                                                                            
## How it works                                                                
                                                                             data/raw/*.md
   │  MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter (800 / 100)      ▼
892 chunks, each carrying:                                                         chunk_id · doc_id · heading_path · anchor · url · original_text · title
   │                                                                               ├── page_content   = heading trail + text   → embedded (Gemini, 3072 dims) →
Chroma                                                                             └── original_text  = verbatim source        → shown to the LLM, cited,
quote-checkable                                                                    │
   ▼                                                                            query → Chroma top-30 → Cohere rerank → top 8 → numbered prompt → Gemini →
answer + citations                                                              
                                                                                *Two text fields, deliberately.* The heading trail is prepended to
page_content so                                                               a chunk like "You can also raise it with custom headers" is findable — but
original_text stays byte-identical to the source so citations and quote       verification
work against the real document.                                                
*Citations are positional.* The prompt numbers the excerpts [1] [2] [3];    the code
records what each number was from the same list, in the same order. The model   emits
small integers — never chunk ids, which it would happily invent.               
---                                                                            
## Evaluation                                                                  
eval/questions.json — 17 answerable, 5 unanswerable.                         
*Questions are written in user language, not documentation language.*         "how do i get the id from the url", not *"how do you declare path
parameters"*.                                                                   A question paraphrased from a chunk tests whether the retriever can find the
chunk it                                                                        was copied from, which is not the task.
                                                                                *Ground truth is (doc_id, anchor) section pairs*, not phrases.
Deterministic,                                                                  free, reproducible, and robust to re-chunking — sections don't move when
chunk_size                                                                    does.
                                                                                *Validated before use.* eval/validate.py confirms every labelled section
exists in                                                                       the index. A typo'd anchor is a permanent miss that looks like a retrieval
failure.                                                                       
*Retrieval metrics are deterministic, not LLM-judged.* Judging retrieval with an
embedding model biases the metric toward that model. Judging it with an LLM     makes a
12-question dev loop slow and non-reproducible. LLM judging is used only for    faithfulness, where no deterministic ground truth exists.
                                                                                ---
                                                                                ## Known limitations
                                                                                - *17 questions.* One question is 5.9 points, so **deltas under ~10 points
are not                                                                           distinguishable from noise.** The hybrid result (−5.9) is within that band;
the                                                                               reranker result (+5.9) is at its edge.
- *Ground truth labelled by one person*, 1–2 sections per question, not       pooled.
  Proper recall@k needs exhaustive pooled judgments; what's reported is       recall
  against my labels.                                                            - *No code in the corpus.* FastAPI keeps examples in separate .py files
referenced                                                                        by {* ... *} directives, which ingestion strips. The system explains
concepts and                                                                      cannot show code. Resolving those includes would roughly double the corpus.
- *The eval set is probably harder than reality.* Real users sometimes use      documentation vocabulary; these questions never do. That likely understates what
  hybrid and reranking would deliver in production.
- *Latency is measured on a laptop* — no concurrency, no network hop to a hosted
  store. Relative comparison only, not an SLA.

---

## What I'd do next

1. *Grow the eval set to 50+* so 5-point deltas become readable.
2. *Pooled relevance labels* — union the top-k of several retrievers, judge the pool
   with an LLM validated against manual labels, unlocking real recall@k and NDCG.
3. *Resolve the code includes* so code-shaped questions become answerable.
4. *Return more chunks after reranking* (top_n=8) — both questions the reranker
   broke had the answer at rank 3–4 before reranking; they were lost only because the
   cut was at 5.
5. *Deploy* — FastAPI service, index baked into the Docker image at build time
   (static corpus, ~11 MB of vectors, instant cold start).
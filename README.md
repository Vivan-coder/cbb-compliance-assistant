# CBB Rulebook Compliance Assistant (RAG)

**Live demo:** _not deployed yet_ - see [Deploying](#deploying-to-streamlit-community-cloud). Run locally with `streamlit run app/streamlit_app.py`.

A retrieval-augmented assistant that answers questions about the **Central Bank of Bahrain (CBB) Rulebook** with **exact rule citations**, **refuses** when the retrieved text doesn't support an answer, and is **evaluated on a 30-question test set**.

> **Not legal or compliance advice.** Student portfolio project covering a small extract of the Rulebook.

## The problem

Compliance staff lose time searching long rulebooks. A generic chatbot is dangerous here: a confident wrong answer, or an invented rule number, is worse than no answer. This assistant is built around that: it can only answer from retrieved rule text, every claim carries a rule ID, every cited ID is checked in code, and it says "The Rulebook excerpts I have don't cover this." when it should.

## Scope and data

| Module | Content | Edition used |
|---|---|---|
| **FC** Financial Crime (AML/CFT) | Vol. 1 Conventional Banks | July 2022 PDF |
| **HC** High-level Controls (governance) | Vol. 1 | April 2023 PDF |
| **CA** Capital Adequacy, Parts 1 and 2 | Vol. 1 (capital definition, credit risk) | July 2019 PDF |

771 rule-level chunks. **Retrieved on 2026-11-05.** Source files (CBB Rulebook, hosted on the Thomson Reuters portal that publishes it):

- https://cbben.thomsonreuters.com/sites/default/files/net_file_store/Vol_1_FC_July_2022.pdf
- https://cbben.thomsonreuters.com/sites/default/files/net_file_store/Vol_1_HC_April_2023.pdf
- https://cbben.thomsonreuters.com/sites/default/files/net_file_store/Vol_1_CA_Part_1__July_2019.pdf
- https://cbben.thomsonreuters.com/sites/default/files/net_file_store/Vol_1_CA_Part_2__July_2019.pdf

**Sourcing notes (honest record).** The brief named `rulebook.cbb.gov.bh`; that hostname does not resolve. The Rulebook's published module PDFs are on the host above. Its `robots.txt` returned HTTP 403 to an automated request, so I did **not** crawl: I downloaded the four published PDFs once each by direct link (the "download manually" route). I did not review the site's terms of use in depth, so check them before redistributing. Raw PDFs are gitignored; `data/processed/chunks.jsonl` is committed so the demo runs.
CA Part 3 (operational/market risk, Chapters CA-7 onward) was not available at the URL pattern I tried, so some CA cross-references point to chapters that are not in the corpus. That gap is used deliberately in unanswerable test question 27.

## Architecture

```
 PDFs --> ingest --> chunk (by rule no.) --> embed (bge-small, local) --> Chroma
 (4)      strip      FC-C.2.1 + metadata     + context prefix               |
          headers    + heading + date                                       v
                                  question --> retrieve: vector + BM25 --> RRF fusion --> top-8 rules
                                                                                  |
                                    +---- weak-similarity gate ---------<---------+
                                    v                                             v
                              refuse (no API call)        Claude: answer ONLY from rules, cite [ID]
                                                                                  |
                                                       validate in code: every cited ID in retrieved set?
                                                          yes -> show answer     no -> retry once -> reject
```

| File | Role |
|---|---|
| `src/ingest.py` | PDF -> clean page text; reads header for module/chapter/section/date, strips boilerplate, drops cover/TOC |
| `src/chunk.py` | One chunk per rule (e.g. `FC-C.2.1`), long rules split on sub-item boundaries; drops change-history tables and "deleted" stubs |
| `src/index.py` | `BAAI/bge-small-en-v1.5` embeddings of "context prefix + rule text" into Chroma |
| `src/retrieve.py` | vector, BM25, and hybrid (reciprocal rank fusion; a rule ID typed in the question is force-included) |
| `src/answer.py` | citation-forced prompt, refusal logic, citation validation |
| `src/evaluate.py` | runs the test set and prints metrics |
| `app/streamlit_app.py` | demo UI |
| `tests/test_answer.py` | unit tests of refusal/validation with a fake client |

## Results

30 self-written questions (20 answerable, 5 needing two rules combined, 5 unanswerable), top-k = 8. Reproduce: `python src/evaluate.py`.

**Retrieval (measured, free to reproduce):**

| Retriever | Hit rate @8 (25 answerable) | MRR | Multi-rule: all gold rules in top-8 (5 q) |
|---|---|---|---|
| Vector only | **1.00** | 0.76 | 0.80 |
| BM25 only | 0.88 | 0.71 | 0.80 |
| **Hybrid (RRF)** | 0.96 | **0.81** | **1.00** |

Hybrid is **not** uniformly better: vector-only never missed a gold rule on single-rule questions, while hybrid dropped one (Q5). Hybrid ranks the right rule higher on average (MRR) and is the only one to retrieve both rules for all multi-rule questions. With 25 questions, one question is 4 points, so treat these as indicative, not statistically meaningful.

**Answer-level metrics (require an API key - NOT yet run):**

| Metric | Vector | Hybrid |
|---|---|---|
| Citation validity rate | _not run_ | _not run_ |
| Refusal accuracy (5 unanswerable) | _not run_ | _not run_ |
| Manual groundedness (10-question sample) | _not run_ | _not run_ |

Run `python src/evaluate.py --llm` with `ANTHROPIC_API_KEY` set (about 60 short answer calls). It also writes `eval/groundedness_sample.csv` for manual 0-2 scoring. I have not filled in numbers I haven't measured. The citation-validation and refusal *logic* is covered by `tests/test_answer.py` (6 tests, fake client, passing).

**Finding: a similarity cutoff cannot do the refusing.** Top-1 cosine similarity for answerable questions had min 0.70 and mean 0.78; for the 5 unanswerable ones max 0.75 and mean 0.72 - the ranges overlap, because unanswerable questions still sit near *topically related* rules (e.g. the liquidity-ratio question retrieved the capital-ratio rules at 0.75). So refusal depends on the model following the prompt, and on validation; the gate (default 0.50) only catches completely off-topic input. This is exactly what the refusal-accuracy metric above is meant to measure.

## Failure analysis (from the retrieval run)

1. **Q5 "Are bank staff allowed to tell a customer that a suspicious transaction report has been filed about them?" (gold FC-5.2.6) - missed by hybrid and BM25.** The rule never says "tell" or "customer report"; it says *"must not warn or inform ('tipping off')"*. BM25 has no overlap on those words, and in fusion other STR-reporting chunks (FC-5.1.1, FC-5.1.2, FC-5.2.1, ...) outranked it. Vector-only got it at rank 3. Fix: query expansion (rewrite "tell the customer" -> "tipping off"), a larger fusion pool, or a reranker.
2. **Q11 "minimum size and independence requirement for the audit committee" (gold HC-3.4.1) - rank 7 in hybrid, absent from BM25.** BM25 ranked unrelated governance chunks above it (HC-3.3.4, HC-3.4.6, HC-10.1.7) and hybrid put HC-3.4.6 and HC-3.4.2 first. My likely explanation (not tested) is that the question's generic words ("minimum", "size", "requirement") match many chunks while the short composition rule has little distinctive text. It shows hit@8 can pass while the right rule sits near the bottom of what the LLM sees.
3. **Q23 "total capital ratio with the conservation buffer" (gold CA-2.2.1 + CA-2A.2.2) - vector-only retrieved CA-2A.2.2 but not CA-2.2.1.** Multi-rule questions need *every* supporting rule; vector search returned three near-duplicate buffer rules (CA-2A.1/2A.2) instead of the base 10% ratio. Hybrid found both. (CA-2.2.1 is also the rule where the 10% figure lives, so an answer missing it would state the buffer without the base ratio.)

Also expected (not yet measured with the LLM): unanswerable Q27 retrieves CA-1.1.6, which *names* the basic indicator approach but not its percentage - a partially relevant excerpt that tests whether the model refuses or over-reaches.

## Decisions

- **Chunk by rule, not by character count.** A rule is the unit a regulator cites and the unit of meaning. Fixed-size windows cut rules in half, merge unrelated rules, and break the link between text and rule ID, which makes citation validation impossible. Each chunk keeps module, chapter, section, heading, last-changed date and source URL.
- **Context prefix before embedding.** Many rules are short and generic ("The CBB may call upon...") - the prefix (module | chapter | section | rule ID) tells the embedding where the text lives.
- **Local embeddings (bge-small).** Free, no data leaves the machine, reproducible, and a hosted demo needs no embedding key. Trade-off: weaker than the largest hosted models.
- **Hybrid retrieval.** Regulation is full of exact tokens (rule numbers, `BD 6,000`, `CET1`). Embeddings blur those; BM25 nails them. Reciprocal rank fusion merges the two without calibrating incomparable scores. Measured result: better MRR and multi-rule recall, one single-rule miss (see above).
- **Force citations and refuse.** In compliance, an unsupported answer is worse than none. The prompt requires inline `[RULE-ID]` citations and a fixed refusal sentence.
- **Validate citations in code.** A prompt is a request, not a guarantee. The code checks that every cited ID is in the retrieved set; invented or missing citations trigger one corrective retry, then rejection. My own hand-written demo answer was caught by this check (I had cited HC-3.4.3, which wasn't retrieved).

## Limitations

- Covers only FC, HC and CA Parts 1-2 of Volume 1; CA chapters 7+ and all other modules/volumes are absent.
- The Rulebook changes; editions used are listed above (CA is from 2019 and parts of it may have been superseded). Re-check the official text.
- Output is **not legal or compliance advice**.
- The evaluation set is small (30) and self-written by the system's author, so it is likely easier than real user questions. Gold labels are the rule I judged to answer the question; other rules may also be valid.
- Answer-level metrics (citation validity, refusal accuracy, groundedness) have not been measured yet.
- Hit rate is measured on retrieved chunks, not answers; tables inside rules (e.g. risk-weight tables) are flattened by PDF extraction.
- The 5 demo examples are **pre-written** from the retrieved rule text (script: `app/build_cached_examples.py`), not live model output; their citations are validated by the same code.

## Interview prep

- **Embedding / chunking.** An embedding maps text to a vector so that similar meaning is nearby. Retrieval can only return what was chunked: big chunks dilute meaning and bury the key sentence, tiny chunks lose context. Chunking by rule aligns the retrieval unit with the citation unit.
- **Vector search misses rule numbers.** `FC-2.2.3` and `FC-2.2.5` look semantically almost identical to an embedding model. BM25 treats `fc-2.2.3` as an exact token. Hybrid combines meaning (vector) with exact match (BM25); here I also force-include a rule ID typed in the question.
- **Hit rate vs groundedness.** Hit rate = did the right rule reach the model. Groundedness = is the answer supported by the text it cited. A high hit rate with a bad answer, or a grounded answer built on the wrong rule, are both possible - and neither proves the answer is *correct* (the rule might be superseded, or the question needs a rule that wasn't retrieved).
- **Hallucination here** looks like a plausible-sounding rule number or threshold that isn't in the Rulebook, or a real rule attributed to the wrong requirement. Citation validation catches fabricated IDs; it does *not* catch a real ID attached to a wrong claim - that is what manual groundedness review is for.
- **Refuse rather than guess.** A guessed answer on a regulatory deadline or threshold can cause a breach; "not covered, check the Rulebook" costs a minute.

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # add ANTHROPIC_API_KEY (optional)
python src/ingest.py && python src/chunk.py         # needs PDFs in data/raw/ (see links above)
python src/index.py
python src/evaluate.py                               # retrieval metrics, free
python src/evaluate.py --llm                         # answer metrics, needs key
python tests/test_answer.py
streamlit run app/streamlit_app.py
```

The app builds the Chroma index on first start from the committed `chunks.jsonl` (about 3 minutes on CPU the first time).

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (keep `.env` out; the API key is entered by users in the sidebar).
2. On share.streamlit.io choose the repo, main file `app/streamlit_app.py`.
3. Put the resulting URL at the top of this README.

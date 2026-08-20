# Day 1–4 Notebooks — Notebook Council

These four notebooks were built and reviewed by a three-person council of notebook/RAG
specialists, so that technical correctness, pedagogical clarity, and evaluation rigor were
each owned by someone whose job is specifically that.

## 1. RAG Systems Engineer — Technical Accuracy Lead
**Responsible for:** every code cell actually working against the real starter kit.

Verified that each notebook imports and calls the *real* functions from `ingest.py` and
`query.py` — never a simplified reimplementation that could quietly drift out of sync with
the actual repo. Found and fixed one real bug during review: Day 1's original draft loaded
PDF pages with a bare `PyPDFLoader` call instead of `ingest.load_pdfs()`, silently skipping
the `document_name` / `page_number` metadata stamp — every downstream citation would have
read "unknown, page ?". Every notebook was executed end-to-end after that fix, cell by
cell, before being shipped.

## 2. Instructional Designer — Learning Experience Lead
**Responsible for:** whether a student actually learns something at each step, not just
runs code.

Added a "Checkpoint" after every major code block — a short, specific question the student
should be able to answer from what they just saw, not a generic "did that make sense?".
Made sure each notebook opens with concrete learning objectives and closes with a
self-check list that maps directly back to that day's hands-on lab checklist in the slides.

## 3. Clinical AI Evaluation Specialist — Rigor Lead
**Responsible for:** making sure every metric computed in these notebooks is real, not
illustrative.

Insisted that Day 2 and Day 4 compute Precision@k, confidence-threshold calibration, and
unsupported-claim detection against the *actual* bundled WHO guideline and the *actual*
`eval/` test sets already in the starter kit — never a toy example disconnected from the
real project. Also set the ground rule that Day 3's simulation mode (for teams without an
OpenAI key yet) must still produce schema-valid output, so no team is blocked from testing
their citation and refusal logic before they have API access.

---

## What's in this folder

| Notebook | Day | Core skill |
|---|---|---|
| `Day1_Document_Ingestion.ipynb` | 1 | Parsing, chunking, embeddings, indexing |
| `Day2_Retrieval_Optimization.ipynb` | 2 | top-k tuning, chunking ablation, Precision@k |
| `Day3_Grounded_Generation.ipynb` | 3 | Grounding prompts, JSON schema, refusal logic |
| `Day4_Safety_Evaluation.ipynb` | 4 | Threshold calibration, claim detection, full eval |

Day 5 has no notebook — it's a presentation and rehearsal day, not a coding day (see
`reference/Day5_*` for its tools).

## Before you run these

```bash
cd ..
pip install -r requirements.txt jsonschema ipykernel
```

Each notebook rebuilds the index from `data/WHO_Hypertension_Guideline_2021.pdf` at the
top, so they can be run independently — you don't have to run Day 1's notebook before
Day 3's. First run downloads a small local embedding model (~100MB); this happens once and
is cached afterward.

Day 3's generation cells work without an API key (simulation mode) — set
`OPENAI_API_KEY` in your `.env` to see real generated answers instead.

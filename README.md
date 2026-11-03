# Agentic RAG System

A native Python, deterministic Agentic RAG system built from scratch without utilizing LangChain or heavy agent frameworks. 

This repository demonstrates how to build an inspectable, mathematically verifiable routing layer for an LLM Q&A system.

## 🚀 Key Features
- **Explicit Deterministic Router:** Bypasses LLM inference for routing. Uses cosine similarity distance thresholds and chunk metadata diversity (source documents) to explicitly classify intent.
- **Zero Hallucination Guarantee on Out-of-Scope:** If the semantic distance threshold exceeds `1.3`, the query is hard-routed to a static fallback response. The LLM is mathematically prevented from hallucinating an answer.
- **Resilient Generation:** Rate limits, server errors and timeouts are retried with exponential backoff. Failures that remain (including an exhausted daily quota or an invalid key, which are not retried) are returned by type as a clearly marked error, never disguised as an answer, so the evaluation can exclude them.
- **Robust Evaluation Suite:** A standalone script that runs 15 diverse queries (Factual, Synthesis, Out of Scope) against the pipeline and computes ROUGE-L scores and routing accuracy.

## 📁 Repository Structure
- `ingest.py`: Chunking and embedding logic using `sentence-transformers` and `chromadb`. Runs automatically the first time the router starts.
- `router.py`: The deterministic semantic query router.
- `generator.py`: The LLM answer generation pipeline using Google Gemini.
- `evaluate.py`: The comprehensive testing and evaluation framework.
- `read_pdf.py`: Helper that converts a PDF to a `.txt` file for `data/`: `python read_pdf.py input.pdf data/output.txt`.
- `FAILURES.md`: Honest analysis of the system's shortcomings and proposed engineering solutions.
- `requirements.in` / `requirements.txt`: Direct dependencies, and the fully pinned lock compiled from them.
- `requirements-dev.in` / `requirements-dev.txt`: Test tools (pytest), pinned the same way.
- `evaluation/`: The labelled question set, its validator and the validator's tests. See `evaluation/README.md`.

## 🛠️ Setup & Installation

Requires **Python 3.11**.

**1. Clone the repository and set up a virtual environment**
```bash
git clone https://github.com/RohanBabbar/Agentic-RAG-Router.git
cd Agentic-RAG-Router
python3.11 -m venv .venv
source .venv/bin/activate
```

**2. Install the pinned requirements**
```bash
pip install -r requirements.txt
```

**3. Add your Google Gemini API key**
```bash
cp .env.example .env
```
Then replace `your_api_key_here` in `.env` with a key from [Google AI Studio](https://aistudio.google.com/apikey).

## 🏃‍♂️ How to Run

To execute the entire pipeline and view the evaluation results table, simply run the evaluation script:

```bash
python evaluate.py
```

On the first run, the document index is built automatically from `data/` and the embedding model (about 90 MB) is downloaded from Hugging Face. Later runs reuse both. To rebuild the index after changing the documents in `data/`, run `python ingest.py`.

To update the pinned dependencies, edit `requirements.in` and recompile the lock: `pip install pip-tools && pip-compile requirements.in -o requirements.txt`.

To run the offline tests:
```bash
pip install -r requirements-dev.txt
python -m pytest
```

*Note: To stay under the Gemini free tier's 5 requests per minute, the generator sends at most 4 per minute, and rate-limited calls are retried after the delay Google asks for (capped at 90 seconds). Expect pauses between queries on the free tier. Queries whose LLM call still fails are marked in the `LLM Error` column and left out of the ROUGE-L average.*

## 📊 Evaluation Results Summary

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Retrieval Accuracy** | 100.0% | The system perfectly retrieved the correct source documents across all domains. |
| **Routing Accuracy** | 66.7% | See `FAILURES.md` for analysis on keyword-overlap false positives. |
| **ROUGE-L Quality** | 0.384 | Skewed heavily by the API Rate Limiter fallback mechanism triggering gracefully to save the pipeline. |

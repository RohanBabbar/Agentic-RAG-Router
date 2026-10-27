# Agentic RAG System

A native Python, deterministic Agentic RAG system built from scratch without utilizing LangChain or heavy agent frameworks. 

This repository demonstrates how to build an inspectable, mathematically verifiable routing layer for an LLM Q&A system.

## 🚀 Key Features
- **Explicit Deterministic Router:** Bypasses LLM inference for routing. Uses cosine similarity distance thresholds and chunk metadata diversity (source documents) to explicitly classify intent.
- **Zero Hallucination Guarantee on Out-of-Scope:** If the semantic distance threshold exceeds `1.3`, the query is hard-routed to a static fallback response. The LLM is mathematically prevented from hallucinating an answer.
- **Resilient Generation:** API failures (rate limits, outages, blocked responses) are caught by type and returned as a clearly marked error, never disguised as an answer, so the evaluation can exclude them.
- **Robust Evaluation Suite:** A standalone script that runs 15 diverse queries (Factual, Synthesis, Out of Scope) against the pipeline and computes ROUGE-L scores and routing accuracy.

## 📁 Repository Structure
- `ingest.py`: Chunking and embedding logic using `sentence-transformers` and `chromadb`. Runs automatically the first time the router starts.
- `router.py`: The deterministic semantic query router.
- `generator.py`: The LLM answer generation pipeline using Google Gemini.
- `evaluate.py`: The comprehensive testing and evaluation framework.
- `read_pdf.py`: Helper that converts a PDF to a `.txt` file for `data/`: `python read_pdf.py input.pdf data/output.txt`.
- `FAILURES.md`: Honest analysis of the system's shortcomings and proposed engineering solutions.

## 🛠️ Setup & Installation

Requires **Python 3.11**.

**1. Clone the repository and set up a virtual environment**
```bash
git clone https://github.com/RohanBabbar/Agentic-RAG-Router.git
cd Agentic-RAG-Router
python3.11 -m venv venv
source venv/bin/activate
```

**2. Install requirements**
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory and add your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## 🏃‍♂️ How to Run

Run all commands from the repository root. To execute the entire pipeline and view the evaluation results table, simply run the evaluation script:

```bash
python evaluate.py
```

On the first run, the document index is built automatically from `data/` and the embedding model (about 90 MB) is downloaded from Hugging Face. Later runs reuse both. To rebuild the index after changing the documents in `data/`, run `python ingest.py`.

*Note: The evaluation script includes a 15-second delay between queries to respect Free Tier API Rate Limits. It takes approximately 3 minutes to run completely.*

## 📊 Evaluation Results Summary

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Retrieval Accuracy** | 100.0% | The system perfectly retrieved the correct source documents across all domains. |
| **Routing Accuracy** | 66.7% | See `FAILURES.md` for analysis on keyword-overlap false positives. |
| **ROUGE-L Quality** | 0.384 | Skewed heavily by the API Rate Limiter fallback mechanism triggering gracefully to save the pipeline. |

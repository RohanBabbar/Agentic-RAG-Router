# Evaluation question set

`questions.jsonl` holds one question per line, as a JSON object with exactly these fields:

| Field | Type | Rules |
| :--- | :--- | :--- |
| `id` | string | Unique across the file, e.g. `"q001"` |
| `question` | string | Non-empty, and unique (ignoring case and spacing) |
| `expected_route` | string | `"factual"`, `"synthesis"` or `"out_of_scope"` |
| `gold_chunk_ids` | list of strings | IDs from the Chroma index, e.g. `["chunk_3", "chunk_17"]`. At least one for `factual` and `synthesis`, empty for `out_of_scope` |
| `reference_answer` | string | Non-empty |

The route labels map to the router's own labels (`Factual`, `Synthesis`, `Out of scope`) through `ROUTES` in `validate_questions.py`.

Example line:

```json
{"id": "q001", "question": "...", "expected_route": "factual", "gold_chunk_ids": ["chunk_3"], "reference_answer": "..."}
```

Validate the file with:

```bash
python evaluation/validate_questions.py
```

It also checks that every gold chunk ID exists in the current index. Chunk IDs change when the documents or the chunking settings change, so rerun it after `python ingest.py`. A synthesis question whose gold chunks all come from one document gets a warning.

Run the validator's offline tests with `python -m pytest evaluation` (install `requirements-dev.txt` first).

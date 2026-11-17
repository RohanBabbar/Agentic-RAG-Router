"""Validates the evaluation question set.

Usage: python evaluation/validate_questions.py [path]   (default: evaluation/questions.jsonl)

Checks every line against the schema in evaluation/README.md: required fields, a valid route,
gold chunks present for factual/synthesis questions and absent for out_of_scope ones, no
duplicate ids or questions, and that every gold chunk ID exists in the Chroma collection.
Exits with status 1 if any error is found.
"""
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO_DIR, "evaluation", "questions.jsonl")

# Route labels used in questions.jsonl, mapped to the labels the router returns.
ROUTES = {"factual": "Factual", "synthesis": "Synthesis", "out_of_scope": "Out of scope"}
FIELDS = {
    "id": str,
    "question": str,
    "expected_route": str,
    "gold_chunk_ids": list,
    "reference_answer": str,
}


def _normalize(question: str) -> str:
    """Questions that differ only in case or spacing count as duplicates."""
    return " ".join(question.casefold().split())


def check_record(record: dict, errors: list) -> bool:
    """Appends schema errors (as messages) for one record. Returns True if it is valid."""
    count = len(errors)
    missing = FIELDS.keys() - record.keys()
    extra = record.keys() - FIELDS.keys()
    if missing:
        errors.append(f"missing field(s) {sorted(missing)}")
    if extra:
        errors.append(f"unknown field(s) {sorted(extra)}")

    for field, expected_type in FIELDS.items():
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, expected_type):
            errors.append(f"'{field}' must be a {expected_type.__name__}, got {type(value).__name__}")
        elif expected_type is str and not value.strip():
            errors.append(f"'{field}' is empty")

    route = record.get("expected_route")
    if isinstance(route, str) and route not in ROUTES:
        errors.append(f"'expected_route' must be one of {list(ROUTES)}, got {route!r}")

    gold = record.get("gold_chunk_ids")
    if isinstance(gold, list):
        if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in gold):
            errors.append("'gold_chunk_ids' must contain only non-empty strings")
        elif len(set(gold)) != len(gold):
            errors.append("'gold_chunk_ids' has duplicates")
        if route == "out_of_scope" and gold:
            errors.append("an out_of_scope question must have no gold chunks")
        if route in ("factual", "synthesis") and not gold:
            errors.append(f"a {route} question needs at least one gold chunk")

    return len(errors) == count


def validate(lines, lookup_sources):
    """Validates the lines of a questions file.

    lookup_sources(chunk_ids) returns {chunk_id: source file} for the IDs that exist in the index,
    and raises LookupError if there is no index. It is only called if some question has gold chunks.
    Returns (valid_records, errors, warnings); errors and warnings are (line_no, message) pairs.
    """
    errors, warnings = [], []
    records = []  # (line_no, where, record) for records that passed the schema checks
    failed = set()  # line numbers of those records that then failed a cross-record check
    seen_ids, seen_questions = {}, {}

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"line {line_no}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append((line_no, f"{where}: invalid JSON ({e.msg})"))
            continue
        if not isinstance(record, dict):
            errors.append((line_no, f"{where}: must be a JSON object"))
            continue

        record_id = record.get("id")
        if isinstance(record_id, str):
            where = f"line {line_no} ({record_id})"
            if record_id in seen_ids:
                errors.append((line_no, f"{where}: duplicate id, first used on line {seen_ids[record_id]}"))
                failed.add(line_no)
            seen_ids.setdefault(record_id, line_no)

        question = record.get("question")
        if isinstance(question, str) and question.strip():
            key = _normalize(question)
            if key in seen_questions:
                errors.append((line_no, f"{where}: duplicate question, same as line {seen_questions[key]}"))
                failed.add(line_no)
            seen_questions.setdefault(key, line_no)

        record_errors = []
        if check_record(record, record_errors):
            records.append((line_no, where, record))
        errors.extend((line_no, f"{where}: {message}") for message in record_errors)

    # Gold chunk IDs must exist in the current index. Chunk IDs change if the documents or the
    # chunking settings change, so rerun this after re-ingesting.
    gold_ids = {chunk_id for _, _, record in records for chunk_id in record["gold_chunk_ids"]}
    if gold_ids:
        try:
            source_of = lookup_sources(sorted(gold_ids))
        except LookupError as e:
            errors.append((0, str(e)))
        else:
            for line_no, where, record in records:
                unknown = [chunk_id for chunk_id in record["gold_chunk_ids"] if chunk_id not in source_of]
                if unknown:
                    errors.append((line_no, f"{where}: gold chunk(s) not in the index: {unknown}"))
                    failed.add(line_no)
                elif record["expected_route"] == "synthesis":
                    sources = {source_of[chunk_id] for chunk_id in record["gold_chunk_ids"]}
                    if len(sources) < 2:
                        warnings.append((line_no, f"{where}: a synthesis question whose gold chunks all come from one document ({sources.pop()})"))

    valid = [record for line_no, _, record in records if line_no not in failed]
    return valid, errors, warnings


def chroma_sources(chunk_ids):
    """Looks up chunk IDs in the Chroma collection: {chunk_id: source file} for those that exist."""
    sys.path.insert(0, REPO_DIR)
    # Imported here, so importing this module (e.g. in the tests) needs no index or heavy imports.
    from ingest import COLLECTION_NAME, get_client
    try:
        collection = get_client().get_collection(name=COLLECTION_NAME)
    except ValueError:
        raise LookupError("The Chroma index does not exist yet, so gold chunk IDs cannot be checked. Run `python ingest.py` first.")
    found = collection.get(ids=list(chunk_ids), include=["metadatas"])
    return dict(zip(found["ids"], (m["source"] for m in found["metadatas"])))


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    valid, errors, warnings = validate(lines, chroma_sources)

    if not valid and not errors:
        print(f"{os.path.relpath(path, REPO_DIR)} has no questions yet.")
        return 0

    messages = [(line_no, "WARNING", m) for line_no, m in warnings] + [(line_no, "ERROR  ", m) for line_no, m in errors]
    for _, level, message in sorted(messages, key=lambda item: item[0]):
        print(f"{level} {message}")

    counts = {route: sum(r["expected_route"] == route for r in valid) for route in ROUTES}
    print(f"\n{len(valid)} valid question(s): " + ", ".join(f"{n} {route}" for route, n in counts.items()))
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH))

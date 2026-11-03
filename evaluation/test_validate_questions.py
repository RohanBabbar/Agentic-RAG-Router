"""Offline tests for validate_questions.py: a small fake index stands in for Chroma."""
import json

import pytest

import validate_questions as vq

# Fake index: chunk ID -> source document
INDEX = {"chunk_0": "doc_a.txt", "chunk_1": "doc_a.txt", "chunk_2": "doc_b.txt"}


def lookup(chunk_ids):
    return {chunk_id: INDEX[chunk_id] for chunk_id in chunk_ids if chunk_id in INDEX}


def question(**overrides):
    record = {
        "id": "q1",
        "question": "What is X?",
        "expected_route": "factual",
        "gold_chunk_ids": ["chunk_0"],
        "reference_answer": "X is Y.",
    }
    record.update(overrides)
    return record


def run(*records, lookup_sources=lookup):
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    return vq.validate(lines, lookup_sources)


def messages(items):
    return [message for _, message in items]


def test_valid_question_for_every_route():
    valid, errors, warnings = run(
        question(),
        question(id="q2", question="How do A and B compare?", expected_route="synthesis", gold_chunk_ids=["chunk_0", "chunk_2"]),
        question(id="q3", question="What is the capital of France?", expected_route="out_of_scope", gold_chunk_ids=[],
                 reference_answer="The documents do not cover this."),
    )
    assert errors == []
    assert warnings == []
    assert len(valid) == 3


@pytest.mark.parametrize("record, expected", [
    ({k: v for k, v in question().items() if k != "question"}, "missing field(s) ['question']"),
    (question(notes="x"), "unknown field(s) ['notes']"),
    (question(expected_route="Factual"), "'expected_route' must be one of"),
    (question(expected_route="out_of_scope"), "an out_of_scope question must have no gold chunks"),
    (question(gold_chunk_ids=[]), "a factual question needs at least one gold chunk"),
    (question(expected_route="synthesis", gold_chunk_ids=[]), "a synthesis question needs at least one gold chunk"),
    (question(gold_chunk_ids="chunk_0"), "'gold_chunk_ids' must be a list"),
    (question(gold_chunk_ids=["chunk_99"]), "gold chunk(s) not in the index: ['chunk_99']"),
    (question(reference_answer="  "), "'reference_answer' is empty"),
    ('{"id": "q1", "question": ', "invalid JSON"),
])
def test_invalid_question_is_rejected(record, expected):
    valid, errors, _ = run(record)
    assert valid == []
    assert any(expected in message for message in messages(errors)), errors


def test_duplicate_ids_and_questions_are_rejected():
    valid, errors, _ = run(
        question(),
        question(question="Something else?"),          # same id as line 1
        question(id="q3", question="  what is   x? "),  # same question as line 1, apart from case and spacing
    )
    assert any("duplicate id, first used on line 1" in m for m in messages(errors))
    assert any("duplicate question, same as line 1" in m for m in messages(errors))
    assert len(valid) == 1


def test_synthesis_from_one_document_is_only_a_warning():
    valid, errors, warnings = run(question(expected_route="synthesis", gold_chunk_ids=["chunk_0", "chunk_1"]))
    assert errors == []
    assert len(valid) == 1
    assert "all come from one document (doc_a.txt)" in messages(warnings)[0]


def test_missing_index_is_reported():
    def no_index(chunk_ids):
        raise LookupError("no index")

    _, errors, _ = run(question(), lookup_sources=no_index)
    assert messages(errors) == ["no index"]


def test_index_is_not_needed_without_gold_chunks():
    def must_not_be_called(chunk_ids):
        raise AssertionError("the index was queried")

    valid, errors, _ = run(
        question(expected_route="out_of_scope", gold_chunk_ids=[], reference_answer="Not covered."),
        lookup_sources=must_not_be_called,
    )
    assert errors == []
    assert len(valid) == 1


def test_shipped_question_file_matches_the_schema():
    # Every chunk ID is accepted here: this test checks the schema only; the CLI checks the real index.
    with open(vq.DEFAULT_PATH, encoding="utf-8") as f:
        _, errors, _ = vq.validate(f.readlines(), lambda chunk_ids: {chunk_id: chunk_id for chunk_id in chunk_ids})
    assert errors == []

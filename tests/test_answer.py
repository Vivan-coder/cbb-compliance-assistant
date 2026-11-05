"""Tests for the refusal + citation-validation logic using a fake Claude client (no API key, no cost).
Run: python tests/test_answer.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from answer import REFUSAL, answer_question  # noqa: E402

CHUNKS = [{"rule_id": "FC-4.1.9", "module": "FC", "heading": "Appointment of MLRO", "last_changed": "x",
           "text": "120 calendar days.", "vector_score": 0.8}]


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0
        self.messages = self

    def create(self, **kw):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)])


def test_valid_answer():
    a = answer_question("q", CHUNKS, client=FakeClient("Within 120 days [FC-4.1.9]."))
    assert a.status == "answered" and a.citations == ["FC-4.1.9"]


def test_invented_rule_is_rejected_after_one_retry():
    c = FakeClient("See [FC-9.9.9].", "Still [FC-9.9.9].")
    a = answer_question("q", CHUNKS, client=c)
    assert a.status == "rejected_invalid_citation" and a.invalid_citations == ["FC-9.9.9"] and c.calls == 2


def test_retry_can_recover():
    a = answer_question("q", CHUNKS, client=FakeClient("See [FC-9.9.9].", "Within 120 days [FC-4.1.9]."))
    assert a.status == "answered"


def test_uncited_answer_is_rejected():
    a = answer_question("q", CHUNKS, client=FakeClient("It is 120 days.", "It is 120 days."))
    assert a.status == "rejected_invalid_citation"


def test_model_refusal_passes_through():
    a = answer_question("q", CHUNKS, client=FakeClient(REFUSAL))
    assert a.status == "refused" and a.text == REFUSAL


def test_weak_retrieval_refuses_without_api_call():
    weak = [{**CHUNKS[0], "vector_score": 0.1}]
    c = FakeClient("should never be called")
    a = answer_question("q", weak, client=c)
    assert a.status == "refused_weak_retrieval" and c.calls == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

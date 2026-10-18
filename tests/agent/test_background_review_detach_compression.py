"""Regression tests for designed context-engine detachment refusals (#141186)."""

from types import SimpleNamespace

import pytest

from agent.background_review import _detach_fork_compression


def _review_agent(bind_session_state):
    compressor = SimpleNamespace(bind_session_state=bind_session_state)
    return SimpleNamespace(
        context_compressor=compressor,
        compression_in_place=False,
        compression_enabled=False,
    )


def test_designed_detach_refusal_is_concise_and_keeps_compression_disabled(caplog):
    def refuse_detach(**_kwargs):
        raise NotImplementedError("engine cannot detach")

    review_agent = _review_agent(refuse_detach)

    with caplog.at_level("WARNING", logger="agent.background_review"):
        _detach_fork_compression(review_agent)

    assert "cannot detach" not in caplog.text
    assert "traceback" not in caplog.text.lower()
    assert len(caplog.records) == 1
    assert "detachment" in caplog.records[0].message
    assert review_agent.compression_in_place is True
    assert review_agent.compression_enabled is False


def test_unexpected_detach_failure_keeps_traceback_and_compression_disabled(caplog):
    def fail_detach(**_kwargs):
        raise RuntimeError("binding state is corrupt")

    review_agent = _review_agent(fail_detach)

    with caplog.at_level("WARNING", logger="agent.background_review"):
        _detach_fork_compression(review_agent)

    assert "binding state is corrupt" in caplog.text
    assert "Traceback" in caplog.text
    assert review_agent.compression_in_place is True
    assert review_agent.compression_enabled is False


def test_successful_detach_enables_compression_and_defers_initial_compaction():
    calls = []

    def detach(**kwargs):
        calls.append(kwargs)

    review_agent = _review_agent(detach)

    _detach_fork_compression(review_agent)

    assert calls == [{"session_db": None, "session_id": ""}]
    assert review_agent.compression_in_place is True
    assert review_agent.compression_enabled is True
    assert review_agent._review_defer_compaction_before_first_response is True

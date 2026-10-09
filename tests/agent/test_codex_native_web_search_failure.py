import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def _build_agent(monkeypatch):
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **kwargs: [{
            "type": "function",
            "function": {"name": "terminal", "description": "Run shell commands.",
                         "parameters": {"type": "object", "properties": {}}},
        }],
    )
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


class _FakeCreateStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self):
        self.closed = True


def _failed_response(*, code="server_error", status="failed"):
    return SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(status=status, error=SimpleNamespace(code=code, message=code)),
    )


def _codex_request_kwargs():
    return {
        "model": "gpt-5-codex",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "Search."}],
        "tools": None,
        "store": False,
    }


def test_codex_stream_preserves_native_web_search_event_before_failure():
    from agent.codex_runtime import _consume_codex_event_stream

    response = _consume_codex_event_stream(
        _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.web_search_call.in_progress"),
            _failed_response(),
        ]),
        model="gpt-5.6-sol",
    )

    assert response.status == "failed"
    assert response.saw_native_web_search_call is True


def test_codex_stream_does_not_mark_unrelated_server_error_as_search_call():
    from agent.codex_runtime import _consume_codex_event_stream

    response = _consume_codex_event_stream(
        _FakeCreateStream([SimpleNamespace(type="response.created"), _failed_response()]),
        model="gpt-5.6-sol",
    )

    assert response.status == "failed"
    assert response.saw_native_web_search_call is False


def test_codex_stream_tracks_added_and_completed_search_output_items():
    from agent.codex_runtime import _consume_codex_event_stream

    for event in (
        SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="web_search_call")),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="web_search_call", status="completed"),
        ),
    ):
        response = _consume_codex_event_stream(
            _FakeCreateStream([SimpleNamespace(type="response.created"), event, _failed_response()]),
            model="gpt-5.6-sol",
        )
        assert response.saw_native_web_search_call is True


def test_native_search_failure_marker_requires_codex_native_request_and_exact_error():
    from agent.codex_runtime import _mark_native_web_search_failure

    codex_route = SimpleNamespace(_is_codex_backend=lambda: True)
    native_tools = [{"type": "web_search"}]

    def marked(*, status="failed", code="server_error", saw_call=True, tools=native_tools, route=codex_route):
        return _mark_native_web_search_failure(
            route,
            SimpleNamespace(
                status=status,
                error=SimpleNamespace(code=code),
                saw_native_web_search_call=saw_call,
            ),
            tools,
        ).native_web_search_failure

    assert marked() is True
    assert marked(status="completed") is False
    assert marked(saw_call=False) is False
    assert marked(code="rate_limit_exceeded") is False
    assert marked(tools=[]) is False
    assert marked(route=SimpleNamespace(_is_codex_backend=lambda: False)) is False


def test_run_codex_stream_marks_native_search_failure_from_request_and_events(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: _FakeCreateStream([
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.web_search_call.in_progress"),
        _failed_response(),
    ])))
    request = _codex_request_kwargs()
    request["tools"] = [{"type": "web_search"}]

    response = agent._run_codex_stream(request)

    assert response.native_web_search_failure is True


def test_run_codex_stream_does_not_mark_successful_native_search(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: _FakeCreateStream([
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.web_search_call.in_progress"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
    ])))
    request = _codex_request_kwargs()
    request["tools"] = [{"type": "web_search"}]

    response = agent._run_codex_stream(request)

    assert response.native_web_search_failure is False

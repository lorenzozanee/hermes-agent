"""Regression tests for preserving user input across model switches."""

from contextlib import contextmanager

from agent.agent_runtime_helpers import repair_message_sequence
from agent.anthropic_message_convert import convert_messages_to_anthropic
from tui_gateway import server


def test_second_model_switch_preserves_prompt_merged_into_marker(monkeypatch):
    @contextmanager
    def no_session_db(_session):
        yield None

    monkeypatch.setattr(server, "_session_db", no_session_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    session = {"session_key": "session", "history": [], "history_lock": None}

    server._append_model_switch_marker(session, model="model-b", provider="test")
    history = session["history"]
    history.append({"role": "user", "content": "FIRST PROMPT"})
    repair_message_sequence(object(), history)
    history.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "results"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    server._append_model_switch_marker(session, model="model-c", provider="test")
    server._append_model_switch_marker(session, model="model-d", provider="test")

    assert any("FIRST PROMPT" in str(message.get("content")) for message in history)
    assert not any("model-b" in str(message.get("content")) for message in history)
    assert not any("model-c" in str(message.get("content")) for message in history)
    assert any("model-d" in str(message.get("content")) for message in history)
    _, converted = convert_messages_to_anthropic(
        history + [{"role": "user", "content": "follow-up"}],
        base_url="https://api.anthropic.com",
        model="claude-opus-5-5",
    )
    assert converted[0] != {
        "role": "user",
        "content": [{"type": "text", "text": "(empty)"}],
    }
    assert "FIRST PROMPT" in str(converted[0])

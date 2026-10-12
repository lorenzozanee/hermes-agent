"""Regression coverage for #136929's desktop config.set runtime snapshot path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tui_gateway import server


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch(
            "agent.context_compressor.get_model_context_length", return_value=200_000
        ),
    ):
        agent = AIAgent(
            api_key="fixture-only",
            base_url="https://api.openai.com/v1",
            provider="openai",
            model="gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
    return agent


def test_config_set_model_switch_refreshes_primary_runtime(monkeypatch):
    agent = _make_agent()
    result = SimpleNamespace(
        success=True,
        new_model="gpt-4.1",
        target_provider="openai",
        api_key="fixture-only-new",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        runtime_capabilities={},
        warning_message="",
        error_message="",
        model_info=None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_merge_preflight_warning", lambda *_args: None)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args: None)
    for name in (
        "_restart_slash_worker",
        "_persist_live_session_runtime",
        "_persist_live_session_system_prompt",
        "_append_model_switch_marker",
        "_emit_session_info",
    ):
        monkeypatch.setattr(server, name, lambda *_args, **_kwargs: None)

    session = {"agent": agent}
    with (
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
    ):
        response = server._apply_model_switch(
            "sid", session, "gpt-4.1 --provider openai --session"
        )

        assert response["value"] == "gpt-4.1"
        assert agent.model == "gpt-4.1"
        assert {
            key: agent._primary_runtime[key]
            for key in ("model", "provider", "base_url")
        } == {
            "model": "gpt-4.1",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
        }
        assert session["model_override"]["model"] == "gpt-4.1"

        agent.model = "fallback/model"
        agent._fallback_activated = True
        monkeypatch.setattr(
            "agent.agent_runtime_helpers._primary_quota_reopened_early",
            lambda *_args: False,
        )
        monkeypatch.setattr(
            "agent.agent_runtime_helpers._primary_reset_gate_blocks",
            lambda *_args: (False, None, None),
        )
        monkeypatch.setattr(
            "agent.fallback_cooldown._is_entitlement_rejected",
            lambda *_args: False,
        )
        monkeypatch.setattr(
            "hermes_cli.chat_catalog.is_known_non_chat_model",
            lambda _model: False,
        )

        assert agent._restore_primary_runtime() is True

    assert agent.model == "gpt-4.1"
    assert agent.provider == "openai"
    agent.close()

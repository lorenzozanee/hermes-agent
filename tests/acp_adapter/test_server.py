"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AuthenticateResponse,
    InitializeResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SessionInfo,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.server import (
    HermesACPAgent,
)
from acp_adapter.session import SessionManager


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [mode.id for mode in resp.modes.available_modes] == [
        "default",
        "accept_edits",
        "dont_ask",
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION




    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload),
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )





    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result["configOptions"] == []





# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id

    @pytest.mark.asyncio
    async def test_empty_messages_list_replaces_stale_history(self, agent, mock_manager):
        """``run_conversation`` returning ``messages=[]`` clears the ACP transcript instead of
        leaving the previous turn's history in place (#10844)."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.history = [{"role": "user", "content": "old"}]
        state.agent.run_conversation = MagicMock(return_value={"final_response": "done", "messages": []})
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id)

        assert state.history == []

    @pytest.mark.asyncio
    async def test_prompt_after_tail_exception_runs_instead_of_queueing(self, agent, mock_manager, monkeypatch):
        """A raise in the post-turn tail (here ``save_session``) costs at most that one turn:
        the next prompt runs instead of queueing forever behind a turn that already ended (#115588)."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "done", "messages": []})
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn
        monkeypatch.setattr(mock_manager, "save_session", MagicMock(side_effect=RuntimeError("disk full")))

        with pytest.raises(RuntimeError, match="disk full"):
            await agent.prompt(prompt=[TextContentBlock(type="text", text="first")], session_id=resp.session_id)

        monkeypatch.setattr(mock_manager, "save_session", MagicMock())
        second = await agent.prompt(prompt=[TextContentBlock(type="text", text="second")], session_id=resp.session_id)

        assert second.stop_reason == "end_turn"
        assert state.queued_prompts == []
        assert state.agent.run_conversation.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor_raises", [False, True])
    async def test_prompt_fails_tool_calls_left_open_before_responding(self, agent, mock_manager, executor_raises):
        """A ``tool.started`` that never sees ``tool.completed`` (blocked/denied/crashed turn) must
        reach the client as a terminal ``failed`` update BEFORE the PromptResponse — on the normal
        return path and when the executor body itself raises."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.agent.model, state.agent.provider = "test-model", "openrouter"
        events: list = []
        mock_conn = MagicMock(spec=acp.Client)

        async def _record(_sid, update):
            events.append(update)

        mock_conn.session_update = _record
        agent._conn = mock_conn

        def _turn(*args, **kwargs):
            state.agent.tool_progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
            if executor_raises:
                raise RuntimeError("executor blew up")
            return {"final_response": "ok", "messages": []}

        with patch.object(HermesACPAgent, "_run_agent_turn", side_effect=_turn):
            started = asyncio.get_running_loop().time()
            response = await asyncio.wait_for(
                agent.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id),
                timeout=3,
            )
            seen_before_response = list(events)
            # Flushing on the loop thread stalled the loop for ``_send_update``'s 5s wait per call.
            assert asyncio.get_running_loop().time() - started < 4

        assert isinstance(response, PromptResponse)
        start = next(e for e in seen_before_response if isinstance(e, ToolCallStart))
        closes = [e for e in seen_before_response if isinstance(e, ToolCallProgress) and e.tool_call_id == start.tool_call_id]
        assert [e.status for e in closes] == ["failed"]

















# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state


    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        agent._handle_slash_command("/reset", state)
        assert len(state.history) == 0




    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force, **kwargs):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            focus_topic=None,
            force=True,
            defer_context_engine_notification=True,
            task_id=state.session_id,
        )
        mock_save.assert_called_once_with(state.session_id)


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        from tools.mcp_tool_scope import _server_key, session_scope_active

        observed = {}
        agent._cmd_help = lambda _args, _state: observed.update(
            active=session_scope_active(),
            key=_server_key("session-server"),
        ) or "ok"
        state.mcp_server_fingerprints = {"session-server": "fingerprint"}
        before = str(resolve_agent_cwd())
        assert agent._handle_slash_command("/help", state) == "ok"
        assert str(resolve_agent_cwd()) == before
        assert observed == {
            "active": True,
            "key": ((None, state.session_id, str(workspace), "fingerprint"), "session-server"),
        }





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""


    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp", "mcp-configured"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        pinned_cwd = {}
        def capture_register(config_map):
            from agent.runtime_cwd import resolve_context_cwd
            registered_config.update(config_map)
            pinned_cwd["value"] = resolve_context_cwd()  # the stdio default cwd reads this pin
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool_config._load_mcp_config", side_effect=lambda **_kwargs: {}), \
             patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}
        # Registration runs under the session's logical cwd so IDE-provided stdio servers spawn there.
        assert pinned_cwd["value"] == Path("/tmp")

    @pytest.mark.asyncio
    async def test_stdio_config_server_uses_each_acp_session_cwd(self, agent, mock_manager, tmp_path):
        """A real stdio server remains bound to its ACP session across interleaved calls."""
        import json
        import sys

        from tools.mcp_tool_scope import reset_session_scope, set_session_scope
        from tools.registry import registry

        server_script = tmp_path / "cwd_server.py"
        server_script.write_text(
            "\n".join([
                "import asyncio",
                "import os",
                "from mcp.server.lowlevel import Server",
                "from mcp.server.stdio import stdio_server",
                "from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool",
                "",
                "async def list_tools(_ctx, _params):",
                "    return ListToolsResult(tools=[Tool(name='get_cwd', inputSchema={'type': 'object', 'properties': {}})])",
                "",
                "async def call_tool(_ctx, _params):",
                "    return CallToolResult(content=[TextContent(type='text', text=os.getcwd())])",
                "",
                "server = Server('cwd-server', on_list_tools=list_tools, on_call_tool=call_tool)",
                "",
                "async def main():",
                "    async with stdio_server() as (read_stream, write_stream):",
                "        await server.run(read_stream, write_stream, server.create_initialization_options())",
                "",
                "asyncio.run(main())",
                "",
            ]),
            encoding="utf-8",
        )
        workspaces = [tmp_path / "workspace-a", tmp_path / "workspace-b"]
        for workspace in workspaces:
            workspace.mkdir()
        states = [mock_manager.create_session(cwd=str(workspace)) for workspace in workspaces]
        for state in states:
            state.agent.enabled_toolsets = ["hermes-acp", "mcp-cwd-server"]
            state.agent.disabled_toolsets = None
            state.agent.tools = []
            state.agent.valid_tool_names = set()
            state.agent._memory_manager = None

        config = {"cwd-server": {"command": sys.executable, "args": [str(server_script)]}}
        try:
            with patch("tools.mcp_tool_config._load_mcp_config", return_value=config):
                for state in states:
                    await agent._register_session_mcp_servers(state, None)

            observed = []
            for state in (states[0], states[1], states[0]):
                token = set_session_scope(state.session_id, state.cwd, state.mcp_server_fingerprints)
                try:
                    result = registry.dispatch("mcp__cwd_server__get_cwd", {})
                finally:
                    reset_session_scope(token)
                observed.append(json.loads(result)["result"])

            assert observed == [str(workspaces[0]), str(workspaces[1]), str(workspaces[0])]
        finally:
            from tools.mcp_tool_common import _core
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers

            for state in states:
                token = set_session_scope(state.session_id, state.cwd, state.mcp_server_fingerprints)
                try:
                    shutdown_mcp_servers(
                        scope=_core._mcp_registry_scope(), session_id=state.session_id, timeout=5.0,
                    )
                finally:
                    reset_session_scope(token)

    @pytest.mark.asyncio
    async def test_registers_default_cwd_config_stdio_server_per_session(self, agent, mock_manager, tmp_path):
        state = mock_manager.create_session(cwd=str(tmp_path))
        state.agent.enabled_toolsets = ["hermes-acp", "mcp-configured"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        registered = {}

        def capture_register(configs):
            from agent.runtime_cwd import resolve_context_cwd
            from tools.mcp_tool_scope import _server_key

            registered["configs"] = configs
            registered["cwd"] = resolve_context_cwd()
            registered["key"] = _server_key("configured")

        config = {"configured": {"command": "/usr/bin/test", "args": []}}
        with patch("tools.mcp_tool_config._load_mcp_config", side_effect=lambda **_kwargs: config) as mock_config, \
             patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, None)

        mock_config.assert_called_once_with(
            workspace_folder=str(tmp_path), session_relative_stdio_only=True, raise_on_error=True,
        )
        assert registered["configs"] == config
        assert registered["cwd"] == tmp_path
        assert registered["key"] == (
            (None, state.session_id, str(tmp_path), state.mcp_server_fingerprints["configured"]), "configured",
        )
        assert state.mcp_server_names == {"configured"}

    @pytest.mark.asyncio
    async def test_session_attachment_respects_configured_mcp_toolsets(self, agent, mock_manager, tmp_path):
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd=str(tmp_path))
        state.agent.enabled_toolsets = ["hermes-acp", "mcp-allowed"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        configured = {
            "allowed": {"command": "/usr/bin/allowed", "args": []},
            "blocked": {"command": "/usr/bin/blocked", "args": []},
        }
        explicit = McpServerStdio(name="client-server", command="/usr/bin/client", args=[], env=[])
        registered = {}

        with patch("tools.mcp_tool_config._load_mcp_config", return_value=configured), \
             patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers"), \
             patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=lambda cfg: registered.update(cfg)), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [explicit])

        assert set(registered) == {"allowed", "client-server"}
        assert state.mcp_server_names == {"allowed", "client-server"}
        assert "blocked" not in state.mcp_server_names

    @pytest.mark.asyncio
    async def test_reattach_replaces_old_session_toolsets_and_reaps_stale_connections(
        self, agent, mock_manager, tmp_path
    ):
        state = mock_manager.create_session(cwd=str(tmp_path))
        state.mcp_server_names = {"old"}
        state.mcp_server_fingerprints = {"old": "old-fingerprint"}
        state.agent.enabled_toolsets = ["hermes-acp", "mcp-old", "mcp-new"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        current = {"new": {"command": "/usr/bin/test", "args": []}}
        with patch("tools.mcp_tool_config._load_mcp_config", side_effect=lambda **_kwargs: current), \
             patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers") as mock_shutdown, \
             patch("tools.mcp_tool_discovery.register_mcp_servers") as mock_register, \
             patch("model_tools.get_tool_definitions", return_value=[]) as mock_defs:
            await agent._register_session_mcp_servers(state, None)

        assert state.mcp_server_names == {"new"}
        assert set(state.mcp_server_fingerprints) == {"new"}
        mock_shutdown.assert_called_once_with(
            scope=None, session_id=state.session_id, stale_session=True, timeout=5.0,
        )
        mock_register.assert_called_once_with(current)
        assert mock_defs.call_args.kwargs["enabled_toolsets"] == ["hermes-acp", "mcp-new"]

    @pytest.mark.asyncio
    async def test_config_load_failure_propagates(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        with patch(
            "tools.mcp_tool_config._load_mcp_config",
            side_effect=RuntimeError("config read failed"),
        ):
            with pytest.raises(RuntimeError, match="config read failed"):
                await agent._register_session_mcp_servers(state, None)


    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool_config._load_mcp_config", side_effect=lambda **_kwargs: {}), \
             patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_propagates(self, agent, mock_manager):
        """Unexpected MCP registration errors fail session attachment."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool_config._load_mcp_config", side_effect=lambda **_kwargs: {}), \
             patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                await agent._register_session_mcp_servers(state, [server])


class TestDisabledToolsetsFilterToolSurface:
    def test_cmd_tools_strips_configured_disabled_toolsets(self, agent, mock_manager):
        """``/tools`` lists what the session can call: a config-disabled toolset is absent (real get_tool_definitions)."""
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent._memory_manager = None

        def listed() -> set:
            return {line.strip().split(":", 1)[0] for line in agent._cmd_tools("", state).splitlines()[1:]}

        state.agent.disabled_toolsets = None
        assert "execute_code" in listed()
        state.agent.disabled_toolsets = ["code_execution"]
        assert "execute_code" not in listed()

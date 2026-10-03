"""Two multiplexed profiles that name the same MCP server with different credentials are two
connections (#106005, #91654): the ledgers in ``tools.mcp_tool`` are keyed per owning profile
scope, and an owner's scoped reload re-registers the profiles that had adopted its connection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


def _tool():
    return SimpleNamespace(name="t", description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, cfg):
    from tools.mcp_tool_registration import _adopter_identity_digest
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool()], tool_timeout=30,
                           initialize_result=None, _registered_tool_names=[], _sampling=None,
                           _resolved_identity=_adopter_identity_digest(name, cfg))


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Multiplex on, clean MCP ledgers, a scope switcher for homes A and B; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    homes = {k: tmp_path / "profiles" / k for k in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_mcp_tool_server_names", "_orphaned_adopters", "_parallel_safe_servers",
               "_server_trust_levels", "_tool_read_only_hints")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    saved_session_tools = {key: dict(value) for key, value in registry._session_scoped_tools.items()}
    registry._session_scoped_tools.clear()
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which):
        tokens.append(set_hermes_home_override(homes[which]))
        return hermes_home_key(homes[which])

    yield enter
    for toolset in ("mcp-x", "mcp-s"):
        for tool_name in list(registry.get_tool_names_for_toolset(toolset)):
            for home in homes.values():
                registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])
    registry._session_scoped_tools.clear()
    registry._session_scoped_tools.update(saved_session_tools)


def test_same_named_server_with_other_credentials_is_a_separate_connection(two_profiles, tmp_path,
                                                                          monkeypatch):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    from tools.registry import registry
    import toolsets

    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer B"}}

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
    assert toolsets.resolve_toolset("mcp-x") == ["mcp__x__t"]
    for _ in range(core._CIRCUIT_BREAKER_THRESHOLD):
        core._bump_server_error("x")
    disc._note_connect_failure("y", RuntimeError("boom"))

    two_profiles("b")
    # B's own view: no tools yet, its memo is not A's, and A's connection is not "connected" for B.
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert toolsets.resolve_toolset("mcp-x") == []
    assert disc.get_mcp_status({"x": cfg_b})[0]["status"] == "configured"
    # B's differently-authenticated 'x' is a connect candidate, not shadowed by A's ledger entries.
    assert "x" in disc._select_new_servers({"x": cfg_b})
    assert not disc._connect_cooldown_active("y")
    assert handlers._check_circuit_breaker("x") is None

    # One server's identity resolver raising refuses adoption of that server only: discovery for
    # B's scope survives and A's healthy same-identity connection is still adopted.
    from tools import mcp_tool_transport as transport
    scope_a = next(key[0] for key in core._servers if key[1] == "x")
    same = {"url": "https://mcp.example/same", "headers": {"Authorization": "Bearer same"}}
    for name in ("ok", "boom"):
        srv = _server(name, dict(same))
        srv._tools = []
        core._servers[(scope_a, name)] = srv
        core._server_tool_scopes[(scope_a, name)] = {scope_a}
    real_inputs = transport._connect_inputs

    def resolve(name, config):
        if name == "boom":
            raise RuntimeError("secret backend exploded")
        return real_inputs(name, config)

    scope_b = core._mcp_registry_scope()
    with patch.object(transport, "_connect_inputs", side_effect=resolve):
        reg.register_connected_into_current_scope({"ok": dict(same), "boom": dict(same)})
    assert scope_b in core._server_tool_scopes[(scope_a, "ok")]
    assert scope_b not in core._server_tool_scopes[(scope_a, "boom")]

    # A routed profile reconciles with NO ambient secret scope (``discover_mcp_tools`` binds the
    # owner scope only around the config load, #113746): with a source-tagged secret the adopter's
    # stdio identity still resolves in ITS OWN scope, so an equal value shares the owner's child.
    import sys
    import agent.secret_scope as secret_scope
    import hermes_cli.env_loader as env_loader
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(env_loader, "_SECRET_SOURCES", {"FIXTURE_TOKEN": "op"})
    for profile in ("a", "b"):
        (tmp_path / "profiles" / profile / ".env").write_text("FIXTURE_TOKEN=tok\n", encoding="utf-8")
    cfg_s = {"command": sys.executable, "args": ["-c", "pass"]}
    two_profiles("a")
    with disc._owner_secret_scope():  # the connecting task records its digest in the owner's scope
        srv_s = _server("s", cfg_s)
    assert srv_s._resolved_identity is not None
    disc._adopt_server("s", srv_s)
    srv_s._registered_tool_names = reg._register_server_tools("s", srv_s, cfg_s)
    two_profiles("b")
    assert secret_scope.current_secret_scope() is None
    assert reg.register_connected_into_current_scope({"s": dict(cfg_s)}) == 1
    assert registry.get_tool_names_for_toolset("mcp-s") == ["mcp__s__t"]
    assert "s" not in disc._select_new_servers({"s": dict(cfg_s)})


def test_oauth_server_is_not_adopted_across_profiles(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "auth": "oauth"}

    scope_a = two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    scope_b = two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []

    # Driven through the public entry point, B must open its own connection (its own OAuth
    # token) rather than adopt A's session.
    connected = []

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            connected.append(_server(name, config))
            disc._adopt_server(name, connected[-1])

    with patch.object(disc, "_run_discovery_pass", fake_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None):
        disc.register_mcp_servers({"x": dict(cfg)})
    assert connected and core._servers[(scope_b, "x")] is connected[0]
    assert core._servers[(scope_a, "x")] is srv_a


def test_same_named_server_with_other_mtls_identity_is_a_separate_connection(two_profiles):
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg_a = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-a.pem",
        "client_key": "/certs/profile-a.key",
    }
    cfg_b = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-b.pem",
        "client_key": "/certs/profile-b.key",
    }

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    reg.register_connected_into_current_scope({"x": cfg_b})
    assert "x" in disc._select_new_servers({"x": cfg_b})


def test_owner_reload_reregisters_profiles_that_adopted_its_connection(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    scope_a = two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    # Owner A: scoped shutdown, then rediscovery.
    two_profiles("a")
    closed = []

    async def close_server():
        closed.append("x")

    srv_a.shutdown = close_server
    lifecycle._loop._ensure_mcp_loop()
    try:
        with patch.object(lifecycle._loop, "_stop_mcp_loop", lambda **_kw: False):
            lifecycle.shutdown_mcp_servers(scope=scope_a)
    finally:
        lifecycle._loop._stop_mcp_loop(only_if_idle=False)
    assert closed == ["x"]
    for tool_name in list(srv_a._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv_a, tool_name)
    with core._lock:
        for key in [k for k, v in core._servers.items() if v is srv_a]:
            core._servers.pop(key)
            core._server_scope_keys.pop(key, None)
            core._server_tool_scopes.pop(key, None)

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            srv = _server(name, config)
            disc._adopt_server(name, srv)
            srv._registered_tool_names = reg._register_server_tools(name, srv, config)

    with patch.object(disc, "_run_discovery_pass", fake_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None), \
            patch("tools.mcp_tool_config._load_mcp_config", lambda: {"x": cfg}):
        disc.register_mcp_servers({"x": cfg})

    # B never reloaded, yet has its tools back on the owner's new identical connection.
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert disc.get_mcp_status({"x": cfg})[0]["status"] == "connected"


def test_untrusted_adopter_of_a_full_profiles_connection_keeps_its_own_trust_gate(two_profiles, monkeypatch):
    """Trust is the consuming profile's policy: adopting A's ``trust: full`` connection must not let
    B's ``trust: untrusted`` write-capable call skip approval."""
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    import tools.approval_prompt as approval_prompt

    route = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    cfg_a, cfg_b = dict(route, trust="full"), dict(route, trust="untrusted")
    asked = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *a, **k: asked.append(a) or "deny")

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg_b}) == 1
    assert handlers._trust_gate_check("x", "t") is not None and asked

    two_profiles("a")
    assert handlers._trust_gate_check("x", "t") is None and len(asked) == 1


def test_parallel_safe_opt_in_is_per_profile(two_profiles):
    """B's ``supports_parallel_tool_calls`` on its own same-named server never makes A's serial
    server's tool parallel-safe (the batch planner would run two A calls concurrently)."""
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg

    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = dict(cfg_a, headers={"Authorization": "Bearer B"}, supports_parallel_tool_calls=True)

    two_profiles("a")
    disc._select_new_servers({"x": cfg_a})
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    disc._select_new_servers({"x": cfg_b})
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is True

    two_profiles("a")
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is False


def test_served_profile_without_multiplex_flag_gets_its_own_connection(two_profiles, monkeypatch):
    """A dashboard/desktop backend serves profiles through the HERMES_HOME override with
    ``gateway.multiplex_profiles`` off; a same-named server with other credentials must still be a
    separate connection there, or profile B calls the server as profile A (#111151). The launch
    profile itself (no override) keeps the bare, unscoped key."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.mcp_tool_scope import _resolve_server_key, _server_key
    from tools.registry import registry

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer B"}}

    scope_a = two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
    assert (scope_a, "x") in core._servers

    two_profiles("b")
    assert _resolve_server_key("x") != (scope_a, "x")
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert "x" in disc._select_new_servers({"x": cfg_b})

    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        assert core._mcp_registry_scope() is None
        assert _server_key("x") == "x"


def test_same_profile_stdio_server_uses_a_connection_per_session_cwd(two_profiles, tmp_path):
    """ACP sessions in one profile must not share a stdio process rooted at another session's cwd."""
    import sys

    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    two_profiles("a")
    cwd_a = tmp_path / "workspace-a"
    cwd_b = tmp_path / "workspace-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    config = {"command": sys.executable, "args": ["-c", "pass"]}

    token_a = set_session_cwd(str(cwd_a))
    scope_a = set_session_scope("session-a", str(cwd_a), {"session-mcp": "config-a"})
    try:
        assert "session-mcp" in disc._select_new_servers({"session-mcp": config})
        server_a = _server("session-mcp", config)
        disc._adopt_server("session-mcp", server_a)
    finally:
        reset_session_scope(scope_a)
        reset_session_cwd(token_a)

    token_b = set_session_cwd(str(cwd_b))
    scope_b = set_session_scope("session-b", str(cwd_b), {"session-mcp": "config-a"})
    try:
        assert "session-mcp" in disc._select_new_servers({"session-mcp": config})
        server_b = _server("session-mcp", config)
        disc._adopt_server("session-mcp", server_b)
        assert disc._get_connected_server_for_call("session-mcp") is server_b
    finally:
        reset_session_scope(scope_b)
        reset_session_cwd(token_b)

    token_a = set_session_cwd(str(cwd_a))
    scope_a = set_session_scope("session-a", str(cwd_a), {"session-mcp": "config-a"})
    try:
        assert disc._get_connected_server_for_call("session-mcp") is server_a
    finally:
        reset_session_scope(scope_a)
        reset_session_cwd(token_a)


def test_session_tool_provenance_survives_sibling_session_cleanup(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _track_mcp_tool_server
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    two_profiles("a")
    config = {"command": "server", "supports_parallel_tool_calls": True}
    keys = []
    for session_id in ("session-a", "session-b"):
        token = set_session_scope(session_id, "/workspace", {"same": session_id})
        try:
            disc._select_new_servers({"same": config})
            server = _server("same", config)
            server._registered_tool_names = ["mcp__same__tool"]
            disc._adopt_server("same", server)
            _track_mcp_tool_server("mcp__same__tool", "same")
            keys.append(next(key for key, value in core._servers.items() if value is server))
        finally:
            reset_session_scope(token)

    _forget_mcp_tool_server("mcp__same__tool", connection_key=keys[0])
    token = set_session_scope("session-b", "/workspace", {"same": "session-b"})
    try:
        assert core._mcp_tool_server_names["mcp__same__tool"] == "same"
        assert disc.is_mcp_tool_parallel_safe("mcp__same__tool")
    finally:
        reset_session_scope(token)


def test_lazy_session_tool_provenance_survives_sibling_cleanup(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_registration import _forget_mcp_tool_server
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    profile_scope = two_profiles("a")
    tool_name = "mcp__same__tool"
    key_a = ((profile_scope, "session-a", "/workspace-a", "a"), "same")
    key_b = ((profile_scope, "session-b", "/workspace-b", "b"), "same")
    core._lazy_server_configs[key_a] = {"command": "server"}
    core._lazy_server_tool_names[key_a] = [tool_name]
    core._lazy_server_configs[key_b] = {"command": "server"}
    core._lazy_server_tool_names[key_b] = [tool_name]
    core._parallel_safe_servers.add(key_b)
    core._mcp_tool_server_names[tool_name] = "same"

    _forget_mcp_tool_server(tool_name, connection_key=key_a)
    token = set_session_scope("session-b", "/workspace-b", {"same": "b"})
    try:
        assert core._mcp_tool_server_names[tool_name] == "same"
        assert disc.is_mcp_tool_parallel_safe(tool_name)
    finally:
        reset_session_scope(token)


def test_served_profile_check_fn_verdict_does_not_shadow_launch_profile(two_profiles, monkeypatch):
    """With multiplex off, a served profile's (correct) "not my connection" verdict must not sit in
    the process-wide check_fn cache under the launch profile's key: the cache scope has to follow
    the same served-profile predicate as the registry scope, or the owner loses its live tools."""
    import tools.registry as registry_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    srv_a = _server("x", cfg_a)
    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        disc._adopt_server("x", srv_a)
        srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
        entry = registry._tools["mcp__x__t"]
    registry_mod.invalidate_check_fn_cache()
    try:
        two_profiles("b")
        assert registry_mod.check_fn_cache_scope() is not None
        assert registry_mod._check_fn_cached(entry.check_fn) is False
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            assert registry_mod._check_fn_cached(entry.check_fn) is True
    finally:
        registry.deregister("mcp__x__t")
        registry_mod.invalidate_check_fn_cache()


def test_launch_profile_pruning_a_server_keeps_served_profiles_same_named_connection(two_profiles, monkeypatch):
    """The launch profile's registry scope is ``None``; when it drops server ``x`` from its config,
    ``reconcile_mcp_servers_with_config`` prunes with ``shutdown_mcp_servers(scope=None,
    names={"x"})``. ``scope=None`` must mean *the unscoped owner* there, not *every owner* —
    otherwise the dashboard's own profile silently tears down profile B's ``(B, "x")``."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle, mcp_tool_loop as loop

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}

    scope_b = two_profiles("b")
    srv_b = _server("x", cfg)
    disc._adopt_server("x", srv_b)
    assert core._server_scope_keys[(scope_b, "x")] == scope_b

    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        assert core._mcp_registry_scope() is None
        srv_launch = _server("x", cfg)
        disc._adopt_server("x", srv_launch)
        assert core._server_scope_keys["x"] is None
    # A third profile adopted the launch profile's connection: pruning it must remember the
    # adopter so the next discovery pass re-registers it.
    core._server_tool_scopes["x"] = {"c"}

    closed = []

    async def _shutdown(self):
        closed.append(self.name)

    for srv in (srv_b, srv_launch):
        srv.shutdown = _shutdown.__get__(srv)

    loop._ensure_mcp_loop()
    try:
        lifecycle.shutdown_mcp_servers(scope=None, names={"x"})
    finally:
        loop._stop_mcp_loop()

    assert "x" not in core._servers and "x" not in core._server_scope_keys
    assert core._servers[(scope_b, "x")] is srv_b
    assert core._server_scope_keys[(scope_b, "x")] == scope_b
    assert closed == ["x"]
    assert core._orphaned_adopters == {"c": {"x"}}


def test_adopter_scope_setup_failure_leaks_no_override_and_continues(two_profiles, monkeypatch):
    """A corrupt/removed adopter home raising inside ``build_profile_secret_scope`` must
    not leak that adopter's HERMES_HOME override into the caller's context, and the
    remaining adopters still get their re-registration pass."""
    import agent.secret_scope as ss
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle

    core._orphaned_adopters.update({"/nonexistent/bad-home": {"x"}, "/nonexistent/good-home": {"x"}})

    def flaky_build(home):
        if "bad-home" in str(home):
            raise RuntimeError("corrupt profile home")
        return {}

    monkeypatch.setattr(ss, "build_profile_secret_scope", flaky_build)
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config",
                        lambda: {"x": {"url": "https://mcp.example/x"}})
    registered = []
    monkeypatch.setattr(disc, "register_mcp_servers", lambda servers: registered.append(dict(servers)))

    lifecycle._reregister_orphaned_adopters()

    from hermes_constants import get_hermes_home_override
    assert get_hermes_home_override() is None
    assert ss.current_secret_scope() is None
    assert registered == [{"x": {"url": "https://mcp.example/x"}}]

def test_session_registry_overlay_isolates_definitions_cache_and_dispatch(two_profiles):
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope
    from tools.registry import registry
    import model_tools

    scope = two_profiles("a")
    tool_name = "mcp__same__search"

    def register(session_id, property_name, result):
        token = set_session_scope(session_id, "/workspace", {"same": session_id})
        try:
            registry.register(
                name=tool_name, toolset="mcp-same",
                schema={
                    "description": result,
                    "parameters": {
                        "type": "object",
                        "properties": {property_name: {"type": "string"}},
                    },
                },
                handler=lambda _args, value=result: value,
                scope=scope,
            )
        finally:
            reset_session_scope(token)

    token = set_session_scope("session-a", "/workspace", {"same": "session-a"})
    try:
        registry.register(
            name="session_context_generic", toolset="test-generic",
            schema={"parameters": {"type": "object"}}, handler=lambda _args: "generic",
            scope=scope,
        )
    finally:
        reset_session_scope(token)
    assert registry.snapshot_registration("session_context_generic", scope=scope) is not None
    registry.deregister("session_context_generic", scope=scope)

    register("session-a", "alpha", "A")
    register("session-b", "beta", "B")
    original_selection = model_tools._select_tool_names
    model_tools._select_tool_names = lambda *_args, **_kwargs: {tool_name}
    try:
        for session_id, prop, expected in (
            ("session-a", "alpha", "A"),
            ("session-b", "beta", "B"),
            ("session-a", "alpha", "A"),
        ):
            token = set_session_scope(session_id, "/workspace", {"same": session_id})
            try:
                definitions = model_tools.get_tool_definitions(
                    enabled_toolsets=["mcp-same"], quiet_mode=True, skip_tool_search_assembly=True,
                )
                assert list(definitions[0]["function"]["parameters"]["properties"]) == [prop]
                assert registry.dispatch(tool_name, {}) == expected
            finally:
                reset_session_scope(token)
    finally:
        model_tools._select_tool_names = original_selection
        for session_id in ("session-a", "session-b"):
            token = set_session_scope(session_id, "/workspace", {"same": session_id})
            try:
                registry.deregister(tool_name, scope=scope)
            finally:
                reset_session_scope(token)


def test_profile_reconcile_ignores_session_owned_connections(two_profiles, monkeypatch):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    scope = two_profiles("a")
    token = set_session_scope("session-a", "/workspace", {"session-only": "fp"})
    try:
        server = _server("session-only", {"command": "server"})
        disc._adopt_server("session-only", server)
        key = next(key for key, value in core._servers.items() if value is server)
    finally:
        reset_session_scope(token)

    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", lambda **_kwargs: {})
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda **_kwargs: [])
    result = disc.reconcile_mcp_servers_with_config()

    assert result == {"removed": [], "added": [], "pending": []}
    assert core._servers[key] is server
    assert core._server_scope_keys[key] == scope


def test_stale_session_connections_are_reaped_without_touching_other_sessions(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle, mcp_tool_loop as loop
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    scope = two_profiles("a")
    closed = []

    def add(session_id, cwd, fingerprint, label):
        token = set_session_scope(session_id, cwd, {"srv": fingerprint})
        try:
            server = _server("srv", {"command": "server", "args": [label]})

            async def _shutdown(self):
                closed.append(label)

            server.shutdown = _shutdown.__get__(server)
            disc._adopt_server("srv", server)
            return next(key for key, value in core._servers.items() if value is server)
        finally:
            reset_session_scope(token)

    old_key = add("session-a", "/old", "old-fp", "old")
    current_key = add("session-a", "/new", "new-fp", "current")
    other_key = add("session-b", "/other", "other-fp", "other")
    current = set_session_scope("session-a", "/new", {"srv": "new-fp"})
    loop._ensure_mcp_loop()
    try:
        lifecycle.shutdown_mcp_servers(
            scope=scope, session_id="session-a", stale_session=True, timeout=2.0,
        )
    finally:
        reset_session_scope(current)
        loop._stop_mcp_loop(only_if_idle=True)

    assert old_key not in core._servers
    assert core._servers[current_key] is not None
    assert core._servers[other_key] is not None
    assert closed == ["old"]


def test_shutdown_wildcard_is_process_wide_inside_session_context(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle, mcp_tool_loop as loop
    from tools.mcp_tool_scope import reset_session_scope, set_session_scope

    closed = []

    def add(profile, session_id, label):
        scope = two_profiles(profile)
        token = set_session_scope(session_id, "/workspace", {"srv": label})
        try:
            server = _server("srv", {"command": "server", "args": [label]})

            async def _shutdown(self):
                closed.append(label)

            server.shutdown = _shutdown.__get__(server)
            disc._adopt_server("srv", server)
        finally:
            reset_session_scope(token)
        return scope

    scope_a = add("a", "session-a", "a")
    add("b", "session-b", "b")
    current = set_session_scope("session-a", "/workspace", {"srv": "a"})
    loop._ensure_mcp_loop()
    try:
        lifecycle.shutdown_mcp_servers(timeout=2.0)
    finally:
        reset_session_scope(current)
        loop._stop_mcp_loop(only_if_idle=True)

    assert not core._servers
    assert sorted(closed) == ["a", "b"]

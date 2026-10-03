"""Connection-ledger keys for tools.mcp_tool under a profile multiplexer.

Every ledger in ``tools.mcp_tool`` (``_servers``, connecting/error/cooldown maps, circuit
breaker, lazy configs, trust metadata) is keyed by a *connection key*: the bare server name
outside a multiplexer, ``(owner_scope, name)`` under one, or a nested session key for ACP-bound
servers. Two profiles that both configure ``github`` with their own
token are two connections; keying by name alone let the first profile's connection shadow the
second's forever — its ``register_mcp_servers`` saw the name as "already connected", adopted
nothing (different credentials) and left the profile silently tool-less (#106005, #91654).

A profile may still *adopt* another profile's live connection when the route and credentials
match (``mcp_tool_registration._same_server_route``); ``_server_tool_scopes[key]`` records every
scope that has done so, and ``_resolve_server_key`` finds that shared connection for a caller
whose own scope has none.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional, Tuple, Union

from tools.mcp_tool_common import _core

ServerKey = Union[str, Tuple[str, str], Tuple[Tuple[Optional[str], str, str, str], str]]

_SESSION_SCOPE: ContextVar[tuple[str, str, dict[str, str]] | None] = ContextVar("mcp_session_scope", default=None)


def set_session_scope(
    session_id: str, cwd: str, server_fingerprints: dict[str, str]
) -> tuple[Token, Token]:
    """Bind MCP connection and tool-registry lookup to one ACP session."""
    mcp_token = _SESSION_SCOPE.set((session_id, cwd, dict(server_fingerprints)))
    try:
        from tools.registry import set_registry_session_scope

        registry_token = set_registry_session_scope(_core._mcp_registry_scope(), session_id)
    except BaseException:
        _SESSION_SCOPE.reset(mcp_token)
        raise
    return mcp_token, registry_token


def reset_session_scope(token: tuple[Token, Token]) -> None:
    mcp_token, registry_token = token
    from tools.registry import reset_registry_session_scope

    reset_registry_session_scope(registry_token)
    _SESSION_SCOPE.reset(mcp_token)


def session_scope_active() -> bool:
    return _SESSION_SCOPE.get() is not None


def _session_id(name: str) -> Optional[str]:
    current = _SESSION_SCOPE.get()
    return current[0] if current is not None and name in current[2] else None


def _session_cwd(name: str) -> Optional[str]:
    current = _SESSION_SCOPE.get()
    return current[1] if current is not None and name in current[2] else None


def _session_fingerprint(name: str) -> Optional[str]:
    current = _SESSION_SCOPE.get()
    return current[2].get(name) if current is not None else None


def _server_key(name: str, scope: Optional[str] = None, *, current: bool = True) -> ServerKey:
    """Connection key for *name* owned by *scope* (the current registry scope when *current*).
    ``None`` scope (no multiplexer) keeps the bare name outside an ACP session."""
    if scope is None and current:
        scope = _core._mcp_registry_scope()
    session_id = _session_id(name)
    if session_id is not None:
        return ((scope, session_id, _session_cwd(name) or "", _session_fingerprint(name) or ""), name)
    return name if scope is None else (scope, name)


def _key_name(key: ServerKey) -> str:
    return key[1] if isinstance(key, tuple) else key


def _key_scope(key: ServerKey) -> Optional[str]:
    """Owning registry scope encoded in *key* (None for a bare, unscoped key)."""
    if not isinstance(key, tuple):
        return None
    return key[0][0] if isinstance(key[0], tuple) else key[0]


def _key_session(key: ServerKey) -> Optional[str]:
    return key[0][1] if isinstance(key, tuple) and isinstance(key[0], tuple) else None


def _key_session_cwd(key: ServerKey) -> Optional[str]:
    return key[0][2] if isinstance(key, tuple) and isinstance(key[0], tuple) else None


def _key_session_fingerprint(key: ServerKey) -> Optional[str]:
    return key[0][3] if isinstance(key, tuple) and isinstance(key[0], tuple) else None


def _key_matches_current_session(key: ServerKey) -> bool:
    current = _SESSION_SCOPE.get()
    if current is None:
        return True
    session_id, cwd, fingerprints = current
    fingerprint = fingerprints.get(_key_name(key))
    expected_session = session_id if fingerprint is not None else None
    return (
        _key_session(key) == expected_session
        and _key_session_cwd(key) == (cwd if fingerprint is not None else None)
        and _key_session_fingerprint(key) == fingerprint
    )


def _key_visible_in_scope(key: ServerKey, scope: Optional[str]) -> bool:
    """Whether the connection under *key* serves *scope*: owned by it or adopted into it.
    Caller holds ``_core._lock`` or tolerates a racy read (status surfaces)."""
    if scope is None:
        return True
    return _key_scope(key) == scope or scope in _core._server_tool_scopes.get(key, ())


def _resolve_server_key(name: str, scope: Optional[str] = None, *, current: bool = True) -> ServerKey:
    """The connection key a call to *name* from *scope* must use: the scope's own connection
    (live, connecting or lazily registered) when it has one, else a shared connection it
    adopted, else its own (not yet existing) key so bookkeeping lands under this scope."""
    if scope is None and current:
        scope = _core._mcp_registry_scope()
    own = _server_key(name, scope, current=False)
    if scope is None or own in _core._servers or own in _core._lazy_server_configs:
        return own
    for key, scopes in _core._server_tool_scopes.items():
        if (scope in scopes and _key_name(key) == name and _key_session(key) == _session_id(name)
                and _key_session_cwd(key) == _session_cwd(name)
                and _key_session_fingerprint(key) == _session_fingerprint(name)
                and key in _core._servers):
            return key
    return own

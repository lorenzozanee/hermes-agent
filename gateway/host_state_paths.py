"""Filesystem paths shared by this OS user's gateway processes."""

import os
from pathlib import Path

_LOCKS_DIRNAME = "gateway-locks"


def host_state_dir() -> Path:
    """Cross-profile rendezvous directory; explicit overrides take precedence."""
    override = os.getenv("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = os.getenv("XDG_STATE_HOME") or ""
    if os.path.isabs(state_home):
        root = Path(state_home)
    elif os.name == "nt":
        root = Path.home() / ".local" / "state"
    else:
        import pwd

        root = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local" / "state"  # windows-footgun: ok — POSIX-only branch
    return root / "hermes" / _LOCKS_DIRNAME

import os
import sys

import pytest

from tests.home_io_guard import HomeIOGuard


@pytest.mark.allow_real_home_io
def test_guard_allows_running_interpreter_but_blocks_home_state(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    interpreter = home / "tools" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"")
    state = home / "state.json"
    state.write_text("{}")

    monkeypatch.setattr(sys, "executable", str(interpreter))
    HomeIOGuard(lambda: (home,)).install(monkeypatch)

    assert os.stat(interpreter).st_size == 0
    with pytest.raises(AssertionError, match="file I/O against the REAL hermes home"):
        os.stat(state)

"""Regression for #135644: network session paths must not block the Projects tree."""

from __future__ import annotations

import pytest

from tui_gateway import git_probe, project_tree, server


def _use_windows_path_rules(monkeypatch, drive_type=None):
    classify = getattr(git_probe, "is_network_path", None)
    if classify is None:
        return

    def is_network_path(path):
        return classify(path, is_windows=True, drive_type=drive_type)

    monkeypatch.setattr(git_probe, "is_network_path", is_network_path)


@pytest.mark.parametrize(
    "cwd",
    [
        r"\\nas\archive\project",
        r"\\?\UNC\nas\archive\project",
        r"\\?\unc\nas\archive\project",
    ],
)
def test_dead_unc_session_does_not_block_project_tree(monkeypatch, cwd):
    probes = []

    _use_windows_path_rules(monkeypatch)

    def reject_network_stat(path):
        if path == cwd:
            raise AssertionError("network cwd must not be statted synchronously")
        return False

    monkeypatch.setattr(git_probe.os.path, "isdir", reject_network_stat)
    monkeypatch.setattr(
        git_probe,
        "bounded_git_probe",
        lambda argv, *, timeout: probes.append((argv, timeout)) or "",
    )
    git_probe.invalidate()
    server._DIR_EXISTS_CACHE.clear()

    tree = project_tree.build_tree(
        [],
        [{"id": "session-1", "cwd": cwd, "git_branch": "", "git_repo_root": ""}],
        [],
        git_probe.resolve,
        hydrate=True,
        exists=server._dir_exists_cached,
    )

    assert probes == [
        (["git", "-C", cwd, "rev-parse", "--show-toplevel"], git_probe._GIT_TIMEOUT)
    ]
    assert tree["scoped_session_ids"] == ["session-1"]
    assert tree["projects"][0]["path"] == cwd
    assert tree["projects"][0]["sessionCount"] == 1


@pytest.mark.parametrize("cwd", ["Z:\\archive\\project", r"\\?\Z:\archive\project"])
def test_mapped_network_drive_skips_stat_but_uses_bounded_git(monkeypatch, cwd):
    calls = []

    def remote_drive(root):
        calls.append(root)
        return 4

    _use_windows_path_rules(monkeypatch, remote_drive)

    def reject_network_stat(_path):
        raise AssertionError("network drive must not be statted")

    monkeypatch.setattr(git_probe.os.path, "isdir", reject_network_stat)
    monkeypatch.setattr(
        git_probe, "bounded_git_probe", lambda argv, *, timeout: "Z:/archive/project"
    )

    assert (
        git_probe.run_git(cwd, "rev-parse", "--show-toplevel") == "Z:/archive/project"
    )
    assert calls == ["Z:\\"]


@pytest.mark.parametrize("cwd", ["C:\\deleted\\worktree", r"\\?\C:\deleted\worktree"])
def test_missing_local_cwd_keeps_the_stat_shortcut(monkeypatch, cwd):
    _use_windows_path_rules(monkeypatch, lambda _root: 3)
    monkeypatch.setattr(git_probe.os.path, "isdir", lambda path: path != cwd)

    def reject_git_probe(*_args, **_kwargs):
        raise AssertionError("missing local cwd should not spawn git")

    monkeypatch.setattr(git_probe, "bounded_git_probe", reject_git_probe)

    assert git_probe.run_git(cwd, "rev-parse", "--show-toplevel") == ""

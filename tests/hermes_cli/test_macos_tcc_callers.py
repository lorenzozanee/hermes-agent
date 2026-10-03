from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import hermes_cli.macos_tcc_anchor as tcc
from hermes_cli import doctor_platform


def test_doctor_anchor_uses_project_root_for_state_and_repair(monkeypatch, tmp_path):
    import hermes_cli.doctor as doctor

    calls = []
    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tcc,
        "tcc_anchor_state",
        lambda project_root=None: (
            calls.append(("state", project_root)) or ("missing", "not anchored")
        ),
    )
    monkeypatch.setattr(
        tcc,
        "ensure_tcc_anchor",
        lambda project_root=None: calls.append(("repair", project_root)),
    )

    doctor_platform.check_macos_tcc_anchor(should_fix=True)

    assert calls == [("state", tmp_path), ("repair", tmp_path)]


def test_checkout_anchor_guard_skips_checkout_venv(monkeypatch):
    checkout = Path(tcc.__file__).resolve().parents[1]
    monkeypatch.setattr(tcc, "_venv_dir", lambda _: checkout / "venv")
    monkeypatch.setattr(
        tcc,
        "_managed_venv",
        lambda _: (_ for _ in ()).throw(AssertionError("checkout venv reached")),
    )

    assert tcc.ensure_tcc_anchor(checkout) is None


def test_update_anchor_uses_project_root(monkeypatch, tmp_path):
    import hermes_cli.main as main
    from hermes_cli import update_cmd, update_cmd_maint as maintenance

    roots = []
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tcc,
        "ensure_tcc_anchor",
        lambda project_root=None: roots.append(project_root),
    )
    monkeypatch.setattr(maintenance, "_best_effort", lambda *_: nullcontext())
    monkeypatch.setattr(
        maintenance, "_verify_and_restore_state_dbs_post_update", lambda: None
    )
    monkeypatch.setattr(
        maintenance, "_invalidate_live_plugin_catalog_caches", lambda: None
    )
    monkeypatch.setattr(maintenance, "_sync_profiles_after_update", lambda: None)
    monkeypatch.setattr(
        update_cmd, "_check_and_apply_config_migration", lambda **_: None
    )
    monkeypatch.setattr(maintenance, "_print_bundled_skills_sync_report", lambda: None)
    monkeypatch.setattr(
        maintenance, "_print_verified_update_completion", lambda _: True
    )
    monkeypatch.setattr(maintenance, "_update_complete_message", lambda _: "Updated")
    monkeypatch.setattr(
        maintenance, "_print_post_update_notices_and_self_heals", lambda: None
    )
    monkeypatch.setattr(
        "hermes_cli.gitlock.fetch_full_commit_graph", lambda *a, **kw: False
    )
    monkeypatch.setattr(
        "hermes_cli.model_catalog.seed_cache_from_checkout", lambda _: False
    )

    maintenance._run_post_update_maintenance(
        assume_yes=True,
        gateway_mode=False,
        pre_update_snapshot_id=None,
        had_desktop_app_before_update=False,
        pre_update_version="test",
    )

    assert roots == [tmp_path]

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_v2.rigging.references import load_menagerie_reference


def test_menagerie_is_sealed_reference_only_with_per_asset_license_policy() -> None:
    record = load_menagerie_reference()
    assert record.schema_version == "1.0"
    assert record.record_version == "2026.08.10"
    assert (
        record.repository_url == "https://github.com/google-deepmind/mujoco_menagerie"
    )
    assert record.role == "modeling_and_testing_reference_only"
    assert record.canonical_identity_allowed is False
    assert record.bundled_assets == ()
    assert record.asset_licenses == ()
    assert record.license_policy.per_asset_license_required is True
    assert record.license_policy.pin_upstream_commit_before_import is True


def test_menagerie_reference_tamper_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "assets" / "v2" / "references"
    record = tmp_path / "mujoco_menagerie.json"
    checksum = tmp_path / "mujoco_menagerie.json.sha256"
    record.write_bytes((source / record.name).read_bytes() + b"\n")
    checksum.write_bytes((source / checksum.name).read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        load_menagerie_reference(record)

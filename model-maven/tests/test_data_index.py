import json

import pytest

from model_maven_worker.data.classification import (
    load_project_exclusions,
    scan_class_folder,
)
from model_maven_worker.data.index import run_index_operation
from model_maven_worker.protocol import UserError


def test_verified_damage_requires_explicit_subset_and_persists_without_deleting_files(tmp_path):
    project = tmp_path / "project"
    metadata = project / ".model-maven"
    metadata.mkdir(parents=True)
    root = tmp_path / "external dataset"
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir()
    damaged = root / "cat" / "broken.png"
    damaged.write_bytes(b"not an image")
    from PIL import Image

    Image.new("RGB", (3, 3)).save(root / "cat" / "good.png")
    Image.new("RGB", (3, 3)).save(root / "dog" / "good.png")

    request = {
        "dataset_root": str(root),
        "project_root": str(project),
        "verify": True,
        "exclude_files": [],
        "apply_exclusions": False,
    }
    with pytest.raises(UserError) as failure:
        run_index_operation(request)
    assert failure.value.code == "CLASS_SAMPLES_DAMAGED"
    assert [item["path"] for item in failure.value.data["damaged_files"]] == ["cat/broken.png"]

    with pytest.raises(UserError) as unverified:
        run_index_operation({**request, "verify": False, "exclude_files": ["cat/broken.png"], "apply_exclusions": True})
    assert unverified.value.code == "CLASS_EXCLUSION_CONFIRMATION_REQUIRED"

    with pytest.raises(UserError) as healthy:
        run_index_operation({**request, "exclude_files": ["cat/good.png"], "apply_exclusions": True})
    assert healthy.value.code == "CLASS_EXCLUSION_NOT_DAMAGED"
    assert not (metadata / "data-exclusions.json").exists()

    result = run_index_operation({**request, "exclude_files": ["cat/broken.png"], "apply_exclusions": True})
    assert result["excluded_files"] == ["cat/broken.png"]
    assert result["sample_count"] == 2
    assert result["excluded_without_deletion"] is True
    assert damaged.is_file()
    protocol = json.loads((metadata / "data-exclusions.json").read_text())
    assert protocol["dataset_fingerprint"] == result["dataset_fingerprint"]

    current = scan_class_folder(root)
    filtered = load_project_exclusions(project, root, current)
    assert filtered["sample_count"] == 2
    assert filtered["excluded_files"] == ["cat/broken.png"]
    Image.new("RGB", (3, 3)).save(damaged)
    with pytest.raises(UserError) as stale:
        load_project_exclusions(project, root, scan_class_folder(root))
    assert stale.value.code == "DATASET_PROTOCOL_INVALIDATED"
    cleared = run_index_operation({**request, "clear_exclusions": True})
    assert cleared["exclusions_cleared"] is True
    assert cleared["sample_count"] == 3
    assert not (metadata / "data-exclusions.json").exists()


def test_excluding_every_damaged_sample_fails_without_writing_protocol(tmp_path):
    project = tmp_path / "project"
    (project / ".model-maven").mkdir(parents=True)
    root = tmp_path / "dataset"
    (root / "cat").mkdir(parents=True)
    (root / "cat" / "broken.png").write_bytes(b"bad")
    from PIL import Image

    Image.new("RGB", (2, 2)).save(root / "cat" / "good.png")
    # A one-sample dataset cannot be reduced to zero, even with explicit confirmation.
    (root / "cat" / "good.png").unlink()
    request = {
        "dataset_root": str(root),
        "project_root": str(project),
        "verify": True,
        "exclude_files": ["cat/broken.png"],
        "apply_exclusions": True,
    }
    with pytest.raises(UserError) as failure:
        run_index_operation(request)
    assert failure.value.code == "CLASS_EXCLUSION_EMPTY"
    assert (root / "cat" / "broken.png").is_file()
    assert not (project / ".model-maven/data-exclusions.json").exists()

use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(program: &str) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package = project.join("python");
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .env("PYTHONPATH", package)
        .output()
        .expect("start Python classification sampling fixture");
    assert!(
        output.status.success(),
        "Python fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[test]
fn folder_index_is_decode_free_content_addressed_and_stratified_splits_replay() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.classification import scan_class_folder
from model_maven_worker.data.sampling import split_classification_index

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    for label, count in {"cat": 7, "dog": 3, "rare": 1}.items():
        (root / label).mkdir()
        for index in range(count):
            (root / label / f"{index}.png").write_bytes(f"{label}:{index}".encode())
    index = scan_class_folder(root)
    first = split_classification_index(index, {"train": .6, "validation": .2, "test": .2}, seed=11)
    again = split_classification_index(index, {"train": .6, "validation": .2, "test": .2}, seed=11)
    all_paths = [sample["path"] for split in first["splits"].values() for sample in split]
    result = {
        "classes": index["classes"],
        "counts": index["class_counts"],
        "decoded": index["indexing"]["decoded_images"],
        "fingerprint": index["dataset_fingerprint"],
        "same_split_ids": first["sample_ids"] == again["sample_ids"],
        "partitioned_once": len(all_paths) == len(set(all_paths)) == index["sample_count"],
        "rare_split_counts": {name: first["class_counts"][name]["rare"] for name in first["splits"]},
        "unavailable": first["unavailable_splits"],
        "split_fingerprint": first["split_fingerprint"],
    }
    print(json.dumps(result, sort_keys=True))
"#,
    );
    assert_eq!(result["classes"], serde_json::json!(["cat", "dog", "rare"]));
    assert_eq!(result["decoded"], false);
    assert_eq!(result["same_split_ids"], true);
    assert_eq!(result["partitioned_once"], true);
    assert_eq!(result["rare_split_counts"]["train"], 1);
    assert_eq!(result["rare_split_counts"]["validation"], 0);
    assert_eq!(result["rare_split_counts"]["test"], 0);
    assert!(
        !result["unavailable"]["validation"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn stratified_budget_covers_classes_when_possible_and_reports_uncovered_without_expansion() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.classification import scan_class_folder
from model_maven_worker.data.sampling import split_classification_index, stratified_sample

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    for label, count in {"a": 6, "b": 3, "c": 2, "d": 1}.items():
        (root / label).mkdir()
        for index in range(count): (root / label / f"{index}.jpg").write_bytes(f"{label}:{index}".encode())
    index = scan_class_folder(root)
    split = split_classification_index(index, {"train": 1.0}, seed=3)
    enough = stratified_sample(split, "train", 4, seed=9)
    small = stratified_sample(split, "train", 2, seed=9)
    small_again = stratified_sample(split, "train", 2, seed=9)
    print(json.dumps({
      "enough_covered": enough["covered_classes"],
      "enough_count": enough["actual_count"],
      "small_count": small["actual_count"],
      "small_uncovered": small["uncovered_classes"],
      "small_ids": small["sample_ids"],
      "repeat_ids": small_again["sample_ids"],
      "budget": small["requested_budget"],
      "algorithm": small["algorithm"],
      "protocol": small["protocol_fingerprint"],
    }, sort_keys=True))
"#,
    );
    assert_eq!(result["enough_covered"].as_array().unwrap().len(), 4);
    assert_eq!(result["enough_count"], 4);
    assert_eq!(result["small_count"], 2);
    assert_eq!(result["small_uncovered"].as_array().unwrap().len(), 2);
    assert_eq!(result["small_ids"], result["repeat_ids"]);
    assert_eq!(result["budget"], 2);
    assert_eq!(result["algorithm"], "sha256-stratified-subset-v1");
}

#[test]
fn changed_content_invalidates_repeat_and_explicit_exclusion_keeps_original_file() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.classification import scan_class_folder, compare_index_content, exclude_files, require_repeat_compatible
from model_maven_worker.protocol import UserError

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp); (root / "fox").mkdir(); path = root / "fox" / "sample.png"
    path.write_bytes(b"old-content")
    (root / "fox" / "other.png").write_bytes(b"other-content")
    old = scan_class_folder(root)
    path.write_bytes(b"new-content")
    current = scan_class_folder(root)
    diff = compare_index_content(old, current)
    try:
        require_repeat_compatible(old, current)
    except UserError as error:
        repeat_code = error.code
        repeat_paths = error.data["changed_files"]
    excluded = exclude_files(current, ["fox/sample.png"])
    print(json.dumps({
      "compatible": diff["compatible"],
      "changed": diff["changed_files"],
      "repeat_code": repeat_code,
      "repeat_paths": repeat_paths,
      "new_fingerprint": excluded["dataset_fingerprint"],
      "old_fingerprint": current["dataset_fingerprint"],
      "file_still_exists": path.exists(),
      "excluded": excluded["excluded_files"],
      "excluded_count": excluded["sample_count"],
    }, sort_keys=True))
"#,
    );
    assert_eq!(result["compatible"], false);
    assert_eq!(result["changed"], serde_json::json!(["fox/sample.png"]));
    assert_eq!(result["repeat_code"], "DATASET_PROTOCOL_INVALIDATED");
    assert_eq!(
        result["repeat_paths"],
        serde_json::json!(["fox/sample.png"])
    );
    assert_ne!(result["new_fingerprint"], result["old_fingerprint"]);
    assert_eq!(result["file_still_exists"], true);
    assert_eq!(result["excluded_count"], 1);
}

#[test]
fn corrupted_selected_image_is_reported_without_automatic_replacement() {
    let has_pillow = Command::new(python())
        .args(["-c", "from PIL import Image"])
        .status()
        .is_ok_and(|status| status.success());
    if !has_pillow {
        eprintln!("skipped damaged-image decode: Pillow is not installed in the test interpreter");
        return;
    }
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from PIL import Image
from model_maven_worker.data.classification import scan_class_folder, verify_sample_images, exclude_files
from model_maven_worker.data.sampling import split_classification_index, stratified_sample
from model_maven_worker.protocol import UserError

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp); (root / "cat").mkdir()
    (root / "cat" / "bad.png").write_bytes(b"not an image")
    Image.new("RGB", (2, 2)).save(root / "cat" / "good.png")
    index = scan_class_folder(root)
    split = split_classification_index(index, {"train": 1.0}, seed=1)
    sample = stratified_sample(split, "train", 2, seed=1)
    try:
        verify_sample_images(root, index, sample["selected"])
    except UserError as error:
        code = error.code
        damaged = error.data["damaged_files"]
    excluded = exclude_files(index, ["cat/bad.png"])
    print(json.dumps({
      "code": code,
      "damaged": [item["path"] for item in damaged],
      "remaining": excluded["sample_count"],
      "removed_from_disk": not (root / "cat" / "bad.png").exists(),
      "excluded": excluded["excluded_files"],
    }, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "CLASS_SAMPLES_DAMAGED");
    assert_eq!(result["damaged"], serde_json::json!(["cat/bad.png"]));
    assert_eq!(result["remaining"], 1);
    assert_eq!(result["removed_from_disk"], false);
    assert_eq!(result["excluded"], serde_json::json!(["cat/bad.png"]));
}

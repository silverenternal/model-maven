use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn invoke(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap()
}

fn report(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid CLI JSON: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn assert_success(output: &Output) -> Value {
    let value = report(output);
    assert!(
        output.status.success(),
        "unexpected failure: {value:#}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(value["status"], "success", "{value:#}");
    value
}

fn distribution_inventory() -> Value {
    let output = Command::new(python())
        .args([
            "-c",
            "import importlib.metadata as m, json; print(json.dumps(sorted((d.metadata.get('Name',''), d.version) for d in m.distributions())))",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

fn has_distribution(name: &str) -> bool {
    Command::new(python())
        .args([
            "-c",
            "import importlib.metadata as m, sys; sys.exit(0 if any(d.metadata.get('Name','').lower().replace('_','-') == sys.argv[1].lower().replace('_','-') for d in m.distributions()) else 1)",
            name,
        ])
        .status()
        .is_ok_and(|status| status.success())
}

fn distribution_version(name: &str) -> String {
    let output = Command::new(python())
        .args([
            "-c",
            "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))",
            name,
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn find_json_file(directory: &Path) -> Option<std::path::PathBuf> {
    for entry in fs::read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_json_file(&path) {
                return Some(found);
            }
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            return Some(path);
        }
    }
    None
}

fn find_file_named(directory: &Path, name: &str) -> Option<std::path::PathBuf> {
    for entry in fs::read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_file_named(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|file| file == name) {
            return Some(path);
        }
    }
    None
}

#[test]
fn project_extension_install_keeps_selected_python_inventory_unchanged() {
    let torch = Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .unwrap();
    if !torch.success() {
        eprintln!("skipped real extension build: selected Python has no PyTorch");
        return;
    }

    let project = tempfile::tempdir().unwrap();
    let before = distribution_inventory();
    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "scaffold",
            "test/isolated",
            "--kind",
            "layer",
            "--output",
            "extensions/isolated",
        ],
    ));
    let installed = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "install",
            "extensions/isolated",
            "--no-build-isolation",
        ],
    ));
    let installation = &installed["data"]["installation"];
    assert_eq!(installation["environment_modified"], false);
    assert_eq!(installation["already_installed"], false);
    let wheel_hash = installation["wheel_sha256"].as_str().unwrap();
    assert_eq!(wheel_hash.len(), 64);
    assert!(wheel_hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let installed_dir = project
        .path()
        .join(installation["site_packages"].as_str().unwrap());
    assert!(installed_dir.is_dir());
    assert!(installed_dir.ends_with(format!(".model-maven/extensions/{wheel_hash}")));
    let tested = assert_success(&invoke(
        project.path(),
        &["component", "test", "test/isolated"],
    ));
    assert_eq!(tested["data"]["checks"]["constructs_nn_module"], "passed");
    assert_eq!(tested["data"]["component"]["kind"], "layer");

    // Simulate a stale/corrupt pre-existing content-addressed install. A retry
    // must report it, but must not delete files it did not create.
    let sentinel = installed_dir.join("preserve-on-failed-retry.txt");
    fs::write(&sentinel, "owned before retry").unwrap();
    let manifest = find_json_file(&installed_dir);
    if let Some(manifest) = manifest {
        fs::remove_file(manifest).unwrap();
        let retry = report(&invoke(
            project.path(),
            &[
                "component",
                "install",
                "extensions/isolated",
                "--no-build-isolation",
            ],
        ));
        assert_eq!(retry["status"], "error", "{retry:#}");
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "owned before retry");
    }

    let registered_manifest = find_json_file(&project.path().join(".model-maven/components"))
        .expect("component discovery should persist a registered manifest");
    let registered: Value =
        serde_json::from_slice(&fs::read(&registered_manifest).unwrap()).unwrap();
    let component_ref = format!(
        "{}@{}",
        registered["component_id"].as_str().unwrap(),
        registered["component_version"].as_str().unwrap()
    );
    let module = registered["entrypoint"]
        .as_str()
        .unwrap()
        .split_once(':')
        .unwrap()
        .0
        .split('.')
        .next()
        .unwrap();
    let source_reference = project.path().join("project_use.py");
    fs::write(&source_reference, format!("import {module}\n")).unwrap();
    let unreadable_source = project.path().join("unreadable.py");
    fs::write(
        &unreadable_source,
        format!("import {module}\ndef broken(:\n"),
    )
    .unwrap();
    let preview = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", &component_ref],
    ));
    assert_eq!(preview["data"]["requires_force"], true, "{preview:#}");
    assert!(
        preview["data"]["source_imports"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["path"] == "project_use.py" && item["module"] == module)
    );
    assert!(
        preview["data"]["unreadable_sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["path"] == "unreadable.py")
    );
    let refused = report(&invoke(
        project.path(),
        &["component", "uninstall", &component_ref, "--apply"],
    ));
    assert_eq!(refused["status"], "error");
    assert_eq!(refused["diagnostics"][0]["code"], "COMPONENT_IN_USE");
    assert!(installed_dir.is_dir());

    fs::remove_file(source_reference).unwrap();
    fs::remove_file(unreadable_source).unwrap();
    let unreferenced = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", &component_ref],
    ));
    assert_eq!(
        unreferenced["data"]["requires_force"], false,
        "{unreferenced:#}"
    );
    assert_eq!(
        unreferenced["data"]["removal_tree"]["algorithm"],
        "sha256-removal-tree-v1"
    );
    let plan_revision = unreferenced["data"]["plan_revision"].as_str().unwrap();
    let added_file = installed_dir.join("user-added.txt");
    fs::write(&added_file, "preserve concurrent changes").unwrap();
    let stale = report(&invoke(
        project.path(),
        &[
            "component",
            "uninstall",
            &component_ref,
            "--apply",
            "--if-plan",
            plan_revision,
        ],
    ));
    assert_eq!(
        stale["diagnostics"][0]["code"], "COMPONENT_PLAN_CONFLICT",
        "{stale:#}"
    );
    assert!(added_file.is_file() && registered_manifest.is_file());
    fs::remove_file(added_file).unwrap();
    let removed = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "uninstall",
            &component_ref,
            "--apply",
            "--force",
            "--if-plan",
            plan_revision,
        ],
    ));
    assert_eq!(removed["data"]["applied"], true, "{removed:#}");
    assert_eq!(removed["data"]["wheel_removed"], true, "{removed:#}");
    assert!(!installed_dir.exists());
    assert!(!registered_manifest.exists());
    assert_eq!(before, distribution_inventory());
}

#[test]
fn uninstall_preview_lists_configuration_references_before_force() {
    let project = tempfile::tempdir().unwrap();
    let added = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/referenced",
            "--kind",
            "layer",
            "--entrypoint",
            "fixture:build_assembly_encoder",
            "--version",
            "1.0.0",
        ],
    ));
    let manifest = project.path().join(
        added["data"]["component"]["manifest_path"]
            .as_str()
            .unwrap(),
    );
    let config_dir = project.path().join(".model-maven");
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(
        config_dir.join("project.json"),
        r#"{"model_injections":{"encoder":{"component_ref":{"id":"local/referenced","version":"1.0.0"}}}}"#,
    )
    .unwrap();
    fs::create_dir_all(config_dir.join("variants")).unwrap();
    fs::write(
        config_dir.join("variants/saved.json"),
        r#"{"model_injections":{"encoder":{"component_ref":{"id":"local/referenced","version":"1.0.0"}}}}"#,
    )
    .unwrap();
    fs::create_dir_all(config_dir.join("runs/run-old")).unwrap();
    fs::write(
        config_dir.join("runs/run-old/run.json"),
        r#"{"schema_version":1,"id":"run-old","status":"failed","inputs":{"component_ref":{"id":"local/referenced","version":"1.0.0"}}}"#,
    )
    .unwrap();

    let preview = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", "local/referenced@1.0.0"],
    ));
    assert_eq!(preview["data"]["requires_force"], true);
    assert_eq!(preview["data"]["applied"], false);
    assert_eq!(
        preview["data"]["project_config_references"][0]["path"],
        ".model-maven/project.json"
    );
    let variant_run_paths = preview["data"]["variant_references"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|reference| reference["path"].as_str())
        .collect::<Vec<_>>();
    assert!(variant_run_paths.contains(&".model-maven/variants/saved.json"));
    assert!(variant_run_paths.contains(&".model-maven/runs/run-old/run.json"));
    let blocked = report(&invoke(
        project.path(),
        &[
            "component",
            "uninstall",
            "local/referenced@1.0.0",
            "--apply",
        ],
    ));
    assert_eq!(blocked["status"], "error");
    assert_eq!(blocked["diagnostics"][0]["code"], "COMPONENT_IN_USE");
    assert!(manifest.is_file());
}

#[test]
fn uninstall_apply_is_blocked_when_run_state_is_active_or_unknown() {
    let project = tempfile::tempdir().unwrap();
    let added = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/active-guard",
            "--kind",
            "layer",
            "--entrypoint",
            "fixture:build_assembly_encoder",
            "--version",
            "1.0.0",
        ],
    ));
    let manifest = project.path().join(
        added["data"]["component"]["manifest_path"]
            .as_str()
            .unwrap(),
    );
    let run = project.path().join(".model-maven/runs/run-uncertain");
    fs::create_dir_all(&run).unwrap();
    fs::write(
        run.join("run.json"),
        r#"{"schema_version":1,"id":"run-uncertain","status":"unknown"}"#,
    )
    .unwrap();

    let blocked = report(&invoke(
        project.path(),
        &[
            "component",
            "uninstall",
            "local/active-guard@1.0.0",
            "--apply",
        ],
    ));
    assert_eq!(blocked["status"], "error");
    assert_eq!(blocked["diagnostics"][0]["code"], "COMPONENT_ACTIVE_RUNS");
    assert!(manifest.is_file());
}

#[test]
fn extension_bundle_rejects_external_symlink_resources_with_an_explicit_policy() {
    let root = tempfile::tempdir().unwrap();
    let distribution = root.path().join("distribution");
    let outside = root.path().join("host-ca-bundle.pem");
    fs::create_dir_all(&distribution).unwrap();
    fs::write(&outside, "host trust store").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, distribution.join("cacert.pem")).unwrap();
    #[cfg(not(unix))]
    return;

    let script = r#"
import hashlib, importlib.metadata, pathlib, sys
sys.path.insert(0, sys.argv[1])
from model_maven_worker.extensions.bundle import ExtensionBundleRuntime, extension_bundle_plan
from model_maven_worker.extensions.catalog import safe_distribution_file
from model_maven_worker.protocol import UserError

root = pathlib.Path(sys.argv[2]).resolve()
class Distribution:
    metadata = {"Name": "external-resource-fixture"}
    version = "1.0.0"
    files = [pathlib.Path("cacert.pem")]
    requires = []
    entry_points = []
    def locate_file(self, relative):
        return root if str(relative) in {"", "."} else root / str(relative)

importlib.metadata.distributions = lambda path=None: [Distribution()]
runtime = ExtensionBundleRuntime(
    verify_component_distribution=lambda _manifest: str(root),
    distribution_name_key=lambda name: name.lower().replace("_", "-"),
    safe_distribution_file=safe_distribution_file,
    sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    search_path_entries=(str(root),),
    current_directory=str(root),
)
manifest = {"component_id": "example/resource", "package_source": {
    "distribution": "external-resource-fixture", "distribution_version": "1.0.0"
}}
try:
    extension_bundle_plan(runtime, {"components": [{"manifest": manifest}]})
except UserError as error:
    assert error.code == "EXTENSION_BUNDLE_EXTERNAL_RESOURCE_UNSUPPORTED", error.code
    assert "宿主机证书库" in error.hint
else:
    raise AssertionError("external symlink target was silently bundled")
"#;
    let output = Command::new(python())
        .args(["-c", script])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("python"))
        .arg(distribution)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn extension_export_bundles_transitive_distribution_closure_and_restores_without_site_packages() {
    if !has_distribution("torch") {
        eprintln!("skipped extension export restore: selected Python has no PyTorch");
        return;
    }
    if !has_distribution("pydantic") {
        eprintln!(
            "skipped transitive extension export: selected Python has no pydantic distribution"
        );
        return;
    }
    for distribution in [
        "annotated-types",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
    ] {
        if !has_distribution(distribution) {
            eprintln!(
                "skipped transitive extension export: selected Python is missing {distribution}"
            );
            return;
        }
    }

    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"
import torch
from torch import nn
from torch.nn import functional as F

class AssemblyModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, values):
        return self.head(self.encoder(values))

def build_model(encoder, head): return AssemblyModel(encoder, head)
def build_encoder(): return nn.Sequential(nn.Linear(4, 4), nn.ReLU())
def build_head(): return nn.Linear(4, 2)
def batch(): return torch.ones(2, 4)
def step(model, values, context):
    logits = model(values)
    return {"loss": F.cross_entropy(logits, torch.tensor([0, 1]))}
"#,
    )
    .unwrap();
    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "scaffold",
            "local/transitive",
            "--kind",
            "layer",
            "--output",
            "extensions/transitive",
        ],
    ));
    let source = project.path().join("extensions/transitive");
    let pyproject = source.join("pyproject.toml");
    let original = fs::read_to_string(&pyproject).unwrap();
    let with_transitive_dependency = original.replace(
        "dependencies = [\"torch>=2.1\"]",
        "dependencies = [\"torch>=2.1\", \"pydantic>=2.0\"]",
    );
    assert_ne!(original, with_transitive_dependency);
    fs::write(pyproject, with_transitive_dependency).unwrap();
    let module = fs::read_dir(source.join("src"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    let layer = module.join("layer.py");
    let original_layer = fs::read_to_string(&layer).unwrap();
    fs::write(
        layer,
        format!("from pydantic import BaseModel\n{original_layer}"),
    )
    .unwrap();

    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "install",
            "extensions/transitive",
            "--no-build-isolation",
        ],
    ));
    assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--inject",
            "encoder=component:local/transitive@0.1.0",
            "--inject",
            "head=fixture:build_head",
        ],
    ));
    assert_success(&invoke(
        project.path(),
        &[
            "train",
            "--batch",
            "fixture:batch",
            "--step",
            "fixture:step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    ));

    let outside = tempfile::tempdir().unwrap();
    let destination = outside.path().join("extension-closure-export");
    assert_success(&invoke(
        project.path(),
        &["export", destination.to_str().unwrap()],
    ));
    let manifest: Value =
        serde_json::from_slice(&fs::read(destination.join("manifest.json")).unwrap()).unwrap();
    let packages = manifest["extension_packages"].as_array().unwrap();
    let package_by_name = |name: &str| {
        packages
            .iter()
            .find(|package| {
                package["distribution"].as_str().is_some_and(|value| {
                    value
                        .replace('_', "-")
                        .eq_ignore_ascii_case(&name.replace('_', "-"))
                })
            })
            .unwrap_or_else(|| {
                panic!("export omitted dependency distribution {name}: {packages:#?}")
            })
    };
    let extension = package_by_name("model-maven-local-transitive");
    assert_eq!(extension["role"], "extension");
    let pydantic = package_by_name("pydantic");
    assert_eq!(pydantic["role"], "runtime_dependency");
    let pydantic_version = distribution_version("pydantic");
    assert!(
        pydantic["required_by"]
            .as_array()
            .unwrap()
            .iter()
            .any(|parent| parent == "model-maven-local-transitive==0.1.0"),
        "extension dependency provenance missing: {pydantic:#?}"
    );
    let expected_pydantic_parent = format!("pydantic=={pydantic_version}");
    for dependency in [
        "annotated-types",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
    ] {
        let dependency = package_by_name(dependency);
        assert_eq!(dependency["role"], "runtime_dependency");
        assert!(
            dependency["required_by"]
                .as_array()
                .unwrap()
                .iter()
                .any(|parent| parent.as_str() == Some(expected_pydantic_parent.as_str())),
            "dependency closure provenance missing: {dependency:#?}"
        );
    }

    // Import the vendored extension dependency closure with site initialization
    // disabled so globally installed Pydantic packages cannot hide omissions.
    let isolated_import = Command::new(python())
        .args([
            "-S",
            "-c",
            "import importlib, json, pathlib, sys; root=pathlib.Path('.').resolve(); m=json.loads((root/'manifest.json').read_text()); [sys.path.insert(0,str((root/p['bundle_path']).resolve())) for p in reversed(m['extension_packages'])]; modules=['pydantic','annotated_types','pydantic_core','typing_extensions','typing_inspection']; imported=[importlib.import_module(name) for name in modules]; assert all(pathlib.Path(module.__file__).resolve().is_relative_to((root/'vendor/packages').resolve()) for module in imported), [(name,module.__file__) for name,module in zip(modules,imported)]; print('vendored dependency closure imported')",
        ])
        .current_dir(&destination)
        .output()
        .unwrap();
    assert!(
        isolated_import.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&isolated_import.stdout),
        String::from_utf8_lossy(&isolated_import.stderr)
    );
    let restored = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .output()
        .unwrap();
    assert!(
        restored.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&restored.stdout),
        String::from_utf8_lossy(&restored.stderr)
    );
    assert!(String::from_utf8_lossy(&restored.stdout).contains("模型已恢复"));

    // A post-training edit to the installed distribution invalidates its
    // recorded provenance and must block another export before a bundle is
    // committed to the destination.
    let extension_root = fs::read_dir(project.path().join(".model-maven/extensions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    let extension_layer = find_file_named(&extension_root, "layer.py").unwrap();
    let mut changed_layer = fs::read_to_string(&extension_layer).unwrap();
    changed_layer.push_str("\n# changed after installation\n");
    fs::write(&extension_layer, changed_layer).unwrap();
    let drifted_destination = outside.path().join("extension-drift-export");
    let drifted = report(&invoke(
        project.path(),
        &["export", drifted_destination.to_str().unwrap()],
    ));
    assert_eq!(drifted["status"], "error", "{drifted:#}");
    assert!(
        !drifted_destination.exists(),
        "export must not leave a partial destination"
    );
}

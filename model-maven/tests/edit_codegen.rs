use serde_json::Value;
use std::{fs, path::Path, process::Command, sync::OnceLock};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    static PYTORCH_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *PYTORCH_AVAILABLE.get_or_init(|| {
        Command::new(python())
            .args(["-c", "import torch"])
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn assert_ok(output: std::process::Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn invoke_cli(project: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(project)
        .output()
        .unwrap();
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON report: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

#[test]
fn generated_assembly_rebuilds_root_factory_and_component_in_fresh_python_process() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let generated = project.path().join("generated");
    fs::create_dir(&generated).unwrap();
    fs::write(
        project.path().join("components.py"),
        "from torch import nn\ndef build_encoder(width=4):\n    return nn.Sequential(nn.Linear(width, width), nn.GELU())\n",
    )
    .unwrap();
    fs::write(
        project.path().join("root_model.py"),
        "from torch import nn\nclass Classifier(nn.Module):\n    def __init__(self, encoder, classes=3):\n        super().__init__(); self.encoder=encoder; self.head=nn.Linear(4, classes)\n    def forward(self, x):\n        return self.head(self.encoder(x))\ndef build_model(encoder, classes=3):\n    return Classifier(encoder, classes)\n",
    )
    .unwrap();
    let original_source = fs::read(project.path().join("root_model.py")).unwrap();
    let recipe = serde_json::json!({
        "schema_version": 1,
        "description": "independent test assembly",
        "initialization_seed": 17,
        "model": {"factory": "root_model:build_model", "params": {"classes": 3}},
        "model_injections": {
            "encoder": {"factory": "components:build_encoder", "params": {"width": 4}}
        }
    });
    let recipe_path = project.path().join("recipe.json");
    let output_path = generated.join("model_assembly.py");
    fs::write(&recipe_path, serde_json::to_vec(&recipe).unwrap()).unwrap();
    let generator = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from model_maven_worker.edit.codegen import write_code
recipe = json.load(open(sys.argv[2], encoding="utf-8"))
write_code(recipe, sys.argv[3])
"#;
    let generated_output = Command::new(python())
        .args([
            "-c",
            generator,
            concat!(env!("CARGO_MANIFEST_DIR"), "/python"),
        ])
        .arg(&recipe_path)
        .arg(&output_path)
        .output()
        .unwrap();
    assert_ok(generated_output, "assembly code generation");
    let code = fs::read_to_string(&output_path).unwrap();
    let compile = Command::new(python())
        .args([
            "-c",
            "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())",
        ])
        .arg(&output_path)
        .output()
        .unwrap();
    assert_ok(compile, "generated Python syntax check");
    assert!(code.contains("# Declared root-model seam: encoder"));
    assert!(code.contains("initialization_seed") || code.contains("manual_seed(17)"));

    let rebuild = r#"
import sys, torch
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from components import build_encoder
from root_model import build_model as root_factory
from model_assembly import build_model as generated_factory
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(17)
    expected = root_factory(encoder=build_encoder(width=4), classes=3)
actual = generated_factory()
assert list(expected.state_dict()) == list(actual.state_dict())
assert all(torch.equal(expected.state_dict()[key], actual.state_dict()[key]) for key in expected.state_dict())
inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
assert torch.equal(expected(inputs), actual(inputs))
"#;
    let rebuilt = Command::new(python())
        .args(["-c", rebuild])
        .arg(project.path())
        .arg(&generated)
        .output()
        .unwrap();
    assert_ok(rebuilt, "fresh-process assembly reconstruction");
    assert_eq!(
        fs::read(project.path().join("root_model.py")).unwrap(),
        original_source,
        "generator must preserve user-owned model source"
    );
}

#[test]
fn assembly_codegen_refuses_existing_output_and_invalid_recipes() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let protected = project.path().join("model.py");
    fs::write(&protected, "user owned sentinel\n").unwrap();
    let recipe = serde_json::json!({
        "schema_version": 1,
        "model": {"factory": "root_model:build_model"}
    });
    let script = r#"
import sys
sys.path.insert(0, sys.argv[1])
from model_maven_worker.edit.codegen import write_code, generate_code
from model_maven_worker.protocol import UserError
recipe = {"schema_version": 1, "model": {"factory": "root_model:build_model"}}
try:
    write_code(recipe, sys.argv[2])
except UserError as error:
    assert error.code == "ASSEMBLY_OUTPUT_EXISTS"
else:
    raise AssertionError("existing output unexpectedly overwritten")
try:
    generate_code({"schema_version": 1, "model": {"factory": "bad entry:run"}})
except UserError as error:
    assert error.code == "ASSEMBLY_RECIPE_INVALID"
else:
    raise AssertionError("invalid factory unexpectedly accepted")
unsupported = [
    ({"action":"custom", "provider":"local/custom", "payload":{}}, "ASSEMBLY_VARIANT_PROVIDER_UNSUPPORTED"),
    ({"action":"add", "parent":"model.encoder", "layer_type":"Linear", "params":{"in_features":4,"out_features":4}, "initialization_seed":7, "component_ref":{"id":"local/linear","version":"1.0.0","sha256":""}}, "ASSEMBLY_VARIANT_COMPONENT_UNSUPPORTED"),
]
for operation, expected_code in unsupported:
    try:
        generate_code({"schema_version":2, "model":{"factory":"root_model:build_model"}, "variant":{"seed":1,"edits":[operation]}})
    except UserError as error:
        assert error.code == expected_code, (error.code, expected_code)
    else:
        raise AssertionError(f"unsupported recipe unexpectedly accepted: {expected_code}")
"#;
    let output = Command::new(python())
        .args(["-c", script, concat!(env!("CARGO_MANIFEST_DIR"), "/python")])
        .arg(&protected)
        .output()
        .unwrap();
    assert_ok(output, "refusal and invalid recipe checks");
    assert_eq!(
        fs::read_to_string(protected).unwrap(),
        "user owned sentinel\n"
    );
    assert_eq!(recipe["schema_version"], 1);
}

#[test]
fn assembly_generate_cli_validates_project_and_writes_readable_nonoverwriting_adapter() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("components.py"),
        "from torch import nn\ndef build_encoder():\n    return nn.Sequential(nn.Linear(4, 4), nn.GELU())\n",
    )
    .unwrap();
    let model_source = "import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self, encoder):\n        super().__init__(); self.encoder=encoder; self.head=encoder[0]; self.blocks=nn.ModuleList([nn.Linear(4, 4), nn.ReLU()]); self.branches=nn.ModuleDict({'left': nn.Linear(4, 4), 'right': nn.ReLU()})\n    def forward(self, x): return self.head(self.encoder(x))\ndef build_model(encoder): return Model(encoder)\ndef batch(): return torch.ones(2, 4)\n";
    fs::write(project.path().join("model.py"), model_source).unwrap();

    let (initialized, init_report) = invoke_cli(
        project.path(),
        &[
            "init",
            "--model",
            "model:build_model",
            "--batch",
            "model:batch",
            "--inject",
            "encoder=components:build_encoder",
        ],
    );
    assert!(
        initialized.status.success(),
        "{init_report:#}\n{}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let original_source = fs::read(project.path().join("model.py")).unwrap();
    let relative = "generated/model_assembly.py";
    let (shown, show_report) = invoke_cli(project.path(), &["assembly", "show", "--config-only"]);
    assert!(shown.status.success(), "{show_report:#}");
    let revision = show_report["data"]["config_revision"].as_str().unwrap();
    let (stale, stale_report) = invoke_cli(
        project.path(),
        &[
            "assembly",
            "generate",
            "--output",
            relative,
            "--if-config-revision",
            &"0".repeat(64),
        ],
    );
    assert!(!stale.status.success(), "{stale_report:#}");
    assert_eq!(
        stale_report["diagnostics"][0]["code"],
        "PROJECT_CONFIG_CONFLICT"
    );
    assert!(!project.path().join(relative).exists());
    let (generated, generation_report) = invoke_cli(
        project.path(),
        &[
            "assembly",
            "generate",
            "--output",
            relative,
            "--if-config-revision",
            revision,
        ],
    );
    assert!(
        generated.status.success(),
        "{generation_report:#}\n{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    assert_eq!(generation_report["data"]["output"], relative);
    assert_eq!(
        generation_report["data"]["validated"]["construction"],
        "passed"
    );
    assert_eq!(
        generation_report["data"]["injection_slots"],
        serde_json::json!(["encoder"])
    );
    let generated_source = fs::read(project.path().join(relative)).unwrap();
    assert!(
        String::from_utf8_lossy(&generated_source).contains("Declared root-model seam: encoder")
    );
    assert_eq!(
        fs::read(project.path().join("model.py")).unwrap(),
        original_source
    );

    let (inspected, inspection_report) = invoke_cli(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "generated.model_assembly:build_model",
            "--batch",
            "model:batch",
        ],
    );
    assert!(
        inspected.status.success(),
        "{inspection_report:#}\n{}",
        String::from_utf8_lossy(&inspected.stderr)
    );

    let (overwrite, overwrite_report) = invoke_cli(
        project.path(),
        &["assembly", "generate", "--output", relative],
    );
    assert!(!overwrite.status.success(), "{overwrite_report:#}");
    assert_eq!(
        overwrite_report["diagnostics"][0]["code"],
        "ASSEMBLY_OUTPUT_EXISTS"
    );
    assert_eq!(
        fs::read(project.path().join(relative)).unwrap(),
        generated_source
    );

    let (escape, escape_report) = invoke_cli(
        project.path(),
        &["assembly", "generate", "--output", "../outside.py"],
    );
    assert!(!escape.status.success(), "{escape_report:#}");
    assert_eq!(
        escape_report["diagnostics"][0]["code"],
        "ASSEMBLY_OUTPUT_PATH_INVALID"
    );
    assert!(!project.path().parent().unwrap().join("outside.py").exists());

    let (variant_add, variant_add_report) = invoke_cli(
        project.path(),
        &[
            "layer",
            "add",
            "--batch",
            "model:batch",
            "--parent",
            "model.encoder",
            "--after",
            "1",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
        ],
    );
    assert!(
        variant_add.status.success(),
        "{variant_add_report:#}\n{}",
        String::from_utf8_lossy(&variant_add.stderr)
    );
    let (variant_set, variant_set_report) = invoke_cli(
        project.path(),
        &[
            "layer",
            "set",
            "model.encoder.dropout",
            "--batch",
            "model:batch",
            "--param",
            "p=0.1",
        ],
    );
    assert!(
        variant_set.status.success(),
        "{variant_set_report:#}\n{}",
        String::from_utf8_lossy(&variant_set.stderr)
    );
    let transaction = invoke_cli(
        project.path(),
        &[
            "layer",
            "apply",
            "--batch",
            "model:batch",
            "--operation",
            r#"{"action":"add","parent":"model.blocks","after":"1","layer_type":"Dropout","params":{"p":0.25}}"#,
            "--operation",
            r#"{"action":"add","parent":"model.branches","name":"middle","before":"right","layer_type":"Identity","params":{}}"#,
            "--operation",
            r#"{"action":"sharing","mode":"share-selected","source":"model.blocks.0","targets":["model.branches.left"]}"#,
            "--operation",
            r#"{"action":"sharing","mode":"clone-selected","path":"model.branches.left"}"#,
            "--operation",
            r#"{"action":"replace","path":"model.blocks.1","layer_type":"Identity","params":{}}"#,
            "--operation",
            r#"{"action":"remove","path":"model.branches.middle","strategy":"remove"}"#,
        ],
    );
    assert!(
        transaction.0.status.success(),
        "{:#}\n{}",
        transaction.1,
        String::from_utf8_lossy(&transaction.0.stderr)
    );
    let variant_path = project.path().join(".model-maven/variants/working.json");
    let saved_variant: Value = serde_json::from_slice(&fs::read(&variant_path).unwrap()).unwrap();
    assert_eq!(saved_variant["cursor"], 3);

    let relative_variant = "generated/variant_model.py";
    let (variant_generated, variant_generation_report) = invoke_cli(
        project.path(),
        &[
            "--variant",
            "working",
            "assembly",
            "generate",
            "--output",
            relative_variant,
        ],
    );
    assert!(
        variant_generated.status.success(),
        "{variant_generation_report:#}\n{}",
        String::from_utf8_lossy(&variant_generated.stderr)
    );
    assert_eq!(variant_generation_report["data"]["variant"], "working");
    assert_eq!(
        variant_generation_report["data"]["input_comparison"]["status"],
        "same"
    );
    assert_eq!(
        variant_generation_report["data"]["validated"]["forward"],
        "passed"
    );
    let variant_source = fs::read_to_string(project.path().join(relative_variant)).unwrap();
    assert!(variant_source.contains("_apply_variant(model"));
    assert!(variant_source.contains("manual_seed("));

    let (variant_inspected, variant_inspection_report) = invoke_cli(
        project.path(),
        &[
            "--variant",
            "working",
            "model",
            "inspect",
            "--batch",
            "model:batch",
        ],
    );
    assert!(
        variant_inspected.status.success(),
        "{variant_inspection_report:#}\n{}",
        String::from_utf8_lossy(&variant_inspected.stderr)
    );
    let (generated_variant_inspected, generated_variant_inspection_report) = invoke_cli(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "generated.variant_model:build_model",
            "--batch",
            "model:batch",
        ],
    );
    assert!(
        generated_variant_inspected.status.success(),
        "{generated_variant_inspection_report:#}\n{}",
        String::from_utf8_lossy(&generated_variant_inspected.stderr)
    );
    let structure = |modules: &Value| {
        modules
            .as_array()
            .unwrap()
            .iter()
            .map(|module| {
                serde_json::json!({
                    "path":module["path"],
                    "type":module["type"],
                    "aliases":module["aliases"],
                    "editable_params":module["editable_params"],
                    "parameters":module["parameters"]
                })
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        structure(&variant_inspection_report["data"]["modules"]),
        structure(&generated_variant_inspection_report["data"]["modules"]),
        "generated entrypoint must reconstruct the selected variant module structure"
    );
    let parity_script = r#"
import json, sys, torch
sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[4]]
import model_maven_worker
model_maven_worker.torch = torch
from variant_model import build_model as generated_factory
variant = json.load(open(sys.argv[3], encoding="utf-8"))
expected, _, _, _ = model_maven_worker.load_model({"project_root": sys.argv[1], "variant": variant})
actual = generated_factory()
assert expected.encoder[0] is expected.head
assert actual.encoder[0] is actual.head
assert expected.blocks[0] is not expected.branches["left"]
assert actual.blocks[0] is not actual.branches["left"]
assert list(expected.branches.keys()) == ["left", "right"]
assert list(actual.branches.keys()) == ["left", "right"]
assert list(expected.state_dict()) == list(actual.state_dict())
assert all(torch.equal(expected.state_dict()[key], actual.state_dict()[key]) for key in expected.state_dict())
expected.eval()
actual.eval()
inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
assert torch.equal(expected(inputs), actual(inputs))
"#;
    let parity = Command::new(python())
        .args(["-c", parity_script])
        .arg(project.path())
        .arg(project.path().join("generated"))
        .arg(&variant_path)
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/python"))
        .output()
        .unwrap();
    assert_ok(parity, "selected-variant generated code parity");
    assert_eq!(
        fs::read(project.path().join("model.py")).unwrap(),
        original_source,
        "variant generation must preserve user-owned source"
    );

    let (copied, copied_report) =
        invoke_cli(project.path(), &["variant", "copy", "working", "cursor"]);
    assert!(
        copied.status.success(),
        "{copied_report:#}\n{}",
        String::from_utf8_lossy(&copied.stderr)
    );
    let (undone, undone_report) =
        invoke_cli(project.path(), &["--variant", "cursor", "layer", "undo"]);
    assert!(
        undone.status.success(),
        "{undone_report:#}\n{}",
        String::from_utf8_lossy(&undone.stderr)
    );
    let (cursor_generated, cursor_report) = invoke_cli(
        project.path(),
        &[
            "--variant",
            "cursor",
            "assembly",
            "generate",
            "--output",
            "generated/cursor_model.py",
        ],
    );
    assert!(
        cursor_generated.status.success(),
        "{cursor_report:#}\n{}",
        String::from_utf8_lossy(&cursor_generated.stderr)
    );
    let (cursor_inspected, cursor_inspection_report) = invoke_cli(
        project.path(),
        &[
            "--variant",
            "cursor",
            "model",
            "inspect",
            "--batch",
            "model:batch",
        ],
    );
    assert!(
        cursor_inspected.status.success(),
        "{cursor_inspection_report:#}\n{}",
        String::from_utf8_lossy(&cursor_inspected.stderr)
    );
    let (cursor_generated_inspected, cursor_generated_inspection_report) = invoke_cli(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "generated.cursor_model:build_model",
            "--batch",
            "model:batch",
        ],
    );
    assert!(
        cursor_generated_inspected.status.success(),
        "{cursor_generated_inspection_report:#}\n{}",
        String::from_utf8_lossy(&cursor_generated_inspected.stderr)
    );
    assert_eq!(
        structure(&cursor_inspection_report["data"]["modules"]),
        structure(&cursor_generated_inspection_report["data"]["modules"]),
        "generated code must respect the persisted active history cursor"
    );
}

#[test]
fn generated_variant_assembly_is_trained_and_restored_from_an_independent_export() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("components.py"),
        "from torch import nn\ndef build_encoder(width=4):\n    return nn.Sequential(nn.Linear(width, width), nn.GELU())\n",
    )
    .unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom torch import nn\nclass Classifier(nn.Module):\n    def __init__(self, encoder, classes=3):\n        super().__init__(); self.encoder=encoder; self.head=nn.Linear(4, classes)\n    def forward(self, x): return self.head(self.encoder(x))\ndef build_model(encoder, classes=3): return Classifier(encoder, classes)\ndef batch(): return torch.ones(2, 4)\ndef step(model, batch, context):\n    targets = torch.tensor([0, 2])\n    return {'loss': nn.functional.cross_entropy(model(batch), targets)}\nbatch.MODEL_MAVEN_STATELESS = True\nstep.MODEL_MAVEN_STATELESS = True\n",
    )
    .unwrap();

    let (initialized, init_report) = invoke_cli(
        project.path(),
        &[
            "init",
            "--model",
            "model:build_model",
            "--batch",
            "model:batch",
            "--inject",
            "encoder=components:build_encoder",
        ],
    );
    assert!(
        initialized.status.success(),
        "{init_report:#}\n{}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let (added, add_report) = invoke_cli(
        project.path(),
        &[
            "layer",
            "add",
            "--batch",
            "model:batch",
            "--parent",
            "model.encoder",
            "--after",
            "1",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
        ],
    );
    assert!(
        added.status.success(),
        "{add_report:#}\n{}",
        String::from_utf8_lossy(&added.stderr)
    );
    let (set, set_report) = invoke_cli(
        project.path(),
        &[
            "layer",
            "set",
            "model.encoder.dropout",
            "--batch",
            "model:batch",
            "--param",
            "p=0.1",
        ],
    );
    assert!(
        set.status.success(),
        "{set_report:#}\n{}",
        String::from_utf8_lossy(&set.stderr)
    );
    let (generated, generation_report) = invoke_cli(
        project.path(),
        &[
            "--variant",
            "working",
            "assembly",
            "generate",
            "--output",
            "generated/model_assembly.py",
        ],
    );
    assert!(
        generated.status.success(),
        "{generation_report:#}\n{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let (trained, training_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "generated.model_assembly:build_model",
            "--batch",
            "model:batch",
            "--step",
            "model:step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert!(
        trained.status.success(),
        "{training_report:#}\n{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    let run_id = training_report["data"]["run_id"].as_str().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let export_dir = outside.path().join("model-export");
    let export_arg = export_dir.to_str().unwrap();
    let (exported, export_report) = invoke_cli(project.path(), &["export", export_arg]);
    assert!(
        exported.status.success(),
        "run {run_id}: {export_report:#}\n{}",
        String::from_utf8_lossy(&exported.stderr)
    );

    let manifest: Value =
        serde_json::from_slice(&fs::read(export_dir.join("manifest.json")).unwrap()).unwrap();
    for module in ["generated/model_assembly.py", "components.py", "model.py"] {
        assert!(
            manifest["source_files"][module].is_string(),
            "export manifest omitted generated assembly dependency {module}: {manifest:#}"
        );
        assert!(export_dir.join("source").join(module).is_file());
    }

    let restored = Command::new(python())
        .arg(export_dir.join("restore_model.py"))
        .current_dir(&export_dir)
        .output()
        .unwrap();
    assert!(
        restored.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&restored.stdout),
        String::from_utf8_lossy(&restored.stderr)
    );
    assert!(String::from_utf8_lossy(&restored.stdout).contains("模型已恢复"));
}

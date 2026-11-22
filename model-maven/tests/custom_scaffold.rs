use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> Value {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let result = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(root)
        .args(["--json", "--no-interactive", "--python", &python])
        .args(args)
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        result.status.success(),
        report["status"] == "success",
        "{report}"
    );
    report
}

#[test]
fn dual_input_scaffold_is_reviewable_preserves_files_and_runs() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("skipped: PyTorch unavailable");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let args = [
        "scaffold",
        "--input",
        "image=3,4,4",
        "--input",
        "tabular=5",
        "--output-dim",
        "3",
    ];
    let preview = run(root.path(), &args);
    assert_eq!(preview["status"], "success", "{preview}");
    assert_eq!(preview["data"]["plan"]["batch_source"], "synthetic");
    assert_eq!(
        preview["data"]["plan"]["inputs"][0]["encoder_slot"],
        "image_encoder"
    );
    assert!(!root.path().join("model_starter.py").exists());
    let revision = preview["data"]["plan_revision"].as_str().unwrap();
    let mut stale = args.to_vec();
    stale.extend(["--apply", "--if-plan", "stale"]);
    assert_eq!(
        run(root.path(), &stale)["diagnostics"][0]["code"],
        "SCAFFOLD_PLAN_CONFLICT"
    );
    assert!(!root.path().join("model_starter.py").exists());
    let mut commit = args.to_vec();
    commit.extend(["--apply", "--if-plan", revision]);
    assert_eq!(run(root.path(), &commit)["data"]["committed"], true);
    let original = fs::read(root.path().join("model_starter.py")).unwrap();
    assert_eq!(run(root.path(), &["model", "probe"])["status"], "success");
    fs::write(
        root.path().join("custom_encoder.py"),
        "from torch import nn\ndef make():\n    return nn.ReLU()\n",
    )
    .unwrap();
    let injected = run(
        root.path(),
        &[
            "model",
            "probe",
            "--inject",
            "image_encoder=custom_encoder:make",
        ],
    );
    assert_eq!(injected["status"], "success", "{injected}");
    assert!(
        injected["data"]["calls"]
            .as_array()
            .unwrap()
            .iter()
            .any(|call| call["module_type"] == "ReLU"),
        "{injected}"
    );
    let trial = run(
        root.path(),
        &[
            "try",
            "--step",
            "model_starter:step",
            "--steps",
            "1",
            "--training-mode",
            "managed-step",
        ],
    );
    assert_eq!(trial["data"]["completed_steps"], 1, "{trial}");
    assert_eq!(
        run(root.path(), &commit)["diagnostics"][0]["code"],
        "SCAFFOLD_TARGET_CONFLICT"
    );
    assert_eq!(
        fs::read(root.path().join("model_starter.py")).unwrap(),
        original
    );
}

#[test]
fn mixed_dtype_scaffold_preserves_inputs_for_custom_encoders_and_default_fusion() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("skipped: PyTorch unavailable");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let args = [
        "scaffold",
        "--input",
        "image=3,4,4",
        "--input",
        "tokens:int64=4",
        "--input",
        "mask:bool=4",
        "--output-dim",
        "3",
    ];
    let plan = run(root.path(), &args);
    assert_eq!(plan["status"], "success", "{plan}");
    assert_eq!(plan["data"]["plan"]["inputs"][1]["dtype"], "int64");
    assert_eq!(plan["data"]["plan"]["inputs"][2]["dtype"], "bool");
    let revision = plan["data"]["plan_revision"].as_str().unwrap();
    let mut apply = args.to_vec();
    apply.extend(["--apply", "--if-plan", revision]);
    assert_eq!(run(root.path(), &apply)["status"], "success");
    fs::write(root.path().join("encoder.py"), "import torch\nfrom torch import nn\nclass TokenEncoder(nn.Module):\n    def __init__(self):\n        super().__init__(); self.embedding = nn.Embedding(3, 1)\n    def forward(self, x):\n        assert x.dtype == torch.int64\n        return self.embedding(x)\ndef make(): return TokenEncoder()\n").unwrap();
    let observed = run(
        root.path(),
        &["model", "probe", "--inject", "tokens_encoder=encoder:make"],
    );
    assert_eq!(observed["status"], "success", "{observed}");
    let invalid = tempfile::tempdir().unwrap();
    let rejected = run(invalid.path(), &["scaffold", "--input", "x:complex64=4"]);
    assert_eq!(rejected["diagnostics"][0]["code"], "SCAFFOLD_INPUT_INVALID");
    assert!(!invalid.path().join("model_starter.py").exists());
}

#[test]
fn variable_axes_create_a_fixed_width_head_and_configurable_representative_batches() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("skipped: PyTorch unavailable");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let args = [
        "scaffold",
        "--input",
        "tokens:int64=?8",
        "--input",
        "image=3,?4,?5",
        "--output-dim",
        "3",
    ];
    let preview = run(root.path(), &args);
    assert_eq!(
        preview["data"]["plan"]["inputs"][0]["dynamic_axes"],
        serde_json::json!([0])
    );
    assert_eq!(
        preview["data"]["plan"]["inputs"][1]["dynamic_axes"],
        serde_json::json!([1, 2])
    );
    assert_eq!(
        preview["data"]["plan"]["inputs"][1]["dynamic_reduction"],
        "mean_after_encoder"
    );
    let revision = preview["data"]["plan_revision"].as_str().unwrap();
    let mut commit = args.to_vec();
    commit.extend(["--apply", "--if-plan", revision]);
    assert_eq!(run(root.path(), &commit)["status"], "success");
    assert_eq!(run(root.path(), &["model", "probe"])["status"], "success");
    let script = r#"import model_starter as s
m=s.model()
assert m.head.in_features == 4
for n in (1, 5, 17):
    b=s.batch(batch_size=3, tokens_dim1=n, image_dim2=n, image_dim3=n+1)
    assert tuple(m(**b['kwargs']).shape) == (3, 3)
    assert b['loss_inputs']['target'].shape == (3, 3)
    assert s.step(m, b, None)['loss'].isfinite()
try: s.batch(tokens_dim1=0)
except ValueError: pass
else: raise AssertionError('zero length accepted')
"#;
    let output = Command::new(&python)
        .current_dir(root.path())
        .args(["-c", script])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let invalid = tempfile::tempdir().unwrap();
    for input in ["x=?0", "x=?", "x=?1000001"] {
        assert_eq!(
            run(invalid.path(), &["scaffold", "--input", input])["status"],
            "error"
        );
    }
}

#[test]
fn explicit_routes_are_reviewed_and_replayed_with_a_shared_input() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let args = [
        "scaffold",
        "--input",
        "a=4",
        "--input",
        "b=2",
        "--route",
        "a_encoder=b",
        "--output-dim",
        "3",
    ];
    let plan = run(root.path(), &args);
    assert_eq!(plan["status"], "success", "{plan}");
    assert_eq!(
        plan["data"]["plan"]["routes"],
        serde_json::json!([
            {"encoder_slot":"a_encoder","input":"b"}, {"encoder_slot":"b_encoder","input":"b"}
        ])
    );
    let revision = plan["data"]["plan_revision"].as_str().unwrap();
    let bad = run(
        root.path(),
        &[
            "scaffold",
            "--input",
            "a=4",
            "--input",
            "b=2",
            "--output-dim",
            "3",
            "--apply",
            "--if-plan",
            revision,
        ],
    );
    assert_eq!(bad["diagnostics"][0]["code"], "SCAFFOLD_PLAN_CONFLICT");
    let mut commit = args.to_vec();
    commit.extend(["--apply", "--if-plan", revision]);
    assert_eq!(run(root.path(), &commit)["status"], "success");
    let script = r#"import torch, model_starter as s
m=s.model(); assert m.head.in_features == 4
seen = {}
for name, module in m.encoders.items():
    module.register_forward_hook(lambda mod, args, output, name=name: seen.setdefault(name, args[0]))
out=m(a=torch.full((2,4),17.), b=torch.full((2,2),5.))
assert out.shape == (2,3) and set(seen)=={'enc_a','enc_b'}
assert all(tuple(x.shape)==(2,2) and torch.all(x==5) for x in seen.values())
"#;
    let output = Command::new(&python)
        .current_dir(root.path())
        .args(["-c", script])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let empty = tempfile::tempdir().unwrap();
    for route in ["missing_encoder=a", "a_encoder=missing", "a_encoder=a=a"] {
        assert_eq!(
            run(
                empty.path(),
                &["scaffold", "--input", "a=4", "--route", route]
            )["diagnostics"][0]["code"],
            "SCAFFOLD_ROUTE_INVALID"
        );
    }
    assert_eq!(
        run(
            empty.path(),
            &[
                "scaffold",
                "--input",
                "a=4",
                "--route",
                "a_encoder=a",
                "--route",
                "a_encoder=a"
            ]
        )["diagnostics"][0]["code"],
        "SCAFFOLD_ROUTE_INVALID"
    );
}

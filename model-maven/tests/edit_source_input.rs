use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}
fn run(project: &Path, args: &[&str]) -> (Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--no-interactive", "--python", &python()])
        .args(args)
        .current_dir(project)
        .output()
        .unwrap();
    let report = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&output.stderr)));
    (output, report)
}

#[test]
fn source_set_inserts_multiple_implicit_constructor_arguments() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path();
    fs::write(
        project.join("fixture.py"),
        r#"import torch
from torch import nn
class Scale(nn.Module):
    def __init__(self, factor=1.0, offset=0.0, scale=1.0):
        super().__init__()
        self.factor, self.offset, self.scale = factor, offset, scale
    def forward(self, x): return x * self.factor * self.scale + self.offset
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.empty = Scale()
        self.positional = Scale(2.0)
        self.multiline = Scale(
            factor=(2.0)  # preserve this comment, including comma
        )
        self.batch_one = Scale(
            2.0,  # preserve trailing comma
        )
        self.batch_two = Scale(
            factor=(2.0)  # preserve grouped argument
        )
    def forward(self, x): return self.batch_two(self.batch_one(self.multiline(self.positional(self.empty(x)))))
def model(): return Net()
def batch(): return torch.ones(2, 3)
"#,
    )
    .unwrap();
    let (out, report) = run(
        project,
        &[
            "init",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(out.status.success(), "{report:#}");
    for layer in ["model.empty", "model.positional", "model.multiline"] {
        let edit = [
            "layer",
            "set",
            layer,
            "--target",
            "source",
            "--param",
            "offset=3.0",
            "--param",
            "scale=4.0",
            "--require-forward",
        ];
        let mut dry = edit.to_vec();
        dry.push("--dry-run");
        let (out, preview) = run(project, &dry);
        assert!(out.status.success(), "{layer}: {preview:#}");
        assert_eq!(preview["data"]["validation"]["forward"], "passed");
        let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
        let mut apply = edit.to_vec();
        apply.extend(["--if-source-plan", plan]);
        let (out, committed) = run(project, &apply);
        assert!(out.status.success(), "{layer}: {committed:#}");
    }
    let first = r#"{"action":"set","path":"model.batch_one","params":{"factor":5.0,"offset":3.0,"scale":4.0}}"#;
    let second = r#"{"action":"set","path":"model.batch_two","params":{"factor":6.0,"offset":3.0,"scale":4.0}}"#;
    let operations = [
        "layer",
        "apply",
        "--target",
        "source",
        "--operation",
        first,
        "--operation",
        second,
        "--require-forward",
    ];
    let mut dry = operations.to_vec();
    dry.push("--dry-run");
    let (out, preview) = run(project, &dry);
    assert!(out.status.success(), "{preview:#}");
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let mut apply = operations.to_vec();
    apply.extend(["--if-source-plan", plan]);
    let (out, committed) = run(project, &apply);
    assert!(out.status.success(), "{committed:#}");
    assert!(
        fs::read_to_string(project.join("fixture.py"))
            .unwrap()
            .contains("# preserve this comment, including comma")
    );
    let output = Command::new(python()).args(["-c", "import fixture; m=fixture.model(); assert (m.empty.offset,m.empty.scale,m.positional.factor,m.positional.offset,m.positional.scale,m.multiline.factor,m.multiline.offset,m.multiline.scale,m.batch_one.factor,m.batch_one.offset,m.batch_one.scale,m.batch_two.factor,m.batch_two.offset,m.batch_two.scale)==(3.0,4.0,2.0,3.0,4.0,2.0,3.0,4.0,5.0,3.0,4.0,6.0,3.0,4.0); assert m(fixture.batch()).shape == (2,3)"]).current_dir(project).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn source_edit_checks_input_at_commit_and_revalidation_and_supports_snapshot_replay() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path();
    let source = r#"from pathlib import Path
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Net()
def batch():
    counter = Path('sequence.txt')
    if counter.exists():
        count = int(counter.read_text()) + 1
        counter.write_text(str(count))
        value = 1 if count <= 2 else 2
    else:
        value = int(Path('value.txt').read_text())
    return torch.full((2,4), float(value))
"#;
    fs::write(project.join("fixture.py"), source).unwrap();
    fs::write(project.join("value.txt"), "1").unwrap();
    let (out, report) = run(
        project,
        &[
            "init",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(out.status.success(), "{report:#}");
    let edit = [
        "layer",
        "set",
        "model.head",
        "--target",
        "source",
        "--param",
        "bias=false",
        "--require-forward",
    ];
    let mut dry = edit.to_vec();
    dry.push("--dry-run");
    let (out, preview) = run(project, &dry);
    assert!(out.status.success(), "{preview:#}");
    assert_eq!(
        preview["data"]["validation"]["forward_evidence"],
        "executed"
    );
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let input = preview["data"]["validation"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    let mut apply = edit.to_vec();
    apply.extend(["--if-source-plan", plan, "--if-input-content", input]);
    fs::write(project.join("value.txt"), "2").unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success());
    assert_eq!(rejected["diagnostics"][0]["code"], "INPUT_CONTENT_MISMATCH");
    assert_eq!(
        fs::read_to_string(project.join("fixture.py")).unwrap(),
        source
    );
    fs::write(project.join("sequence.txt"), "0").unwrap();
    let (out, preview) = run(project, &dry);
    assert!(out.status.success(), "{preview:#}");
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let input = preview["data"]["validation"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    let mut apply = edit.to_vec();
    apply.extend(["--if-source-plan", plan, "--if-input-content", input]);
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success());
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_EDIT_VALIDATION_FAILED"
    );
    assert_eq!(
        rejected["data"]["verification"]["diagnostics"][0]["code"],
        "INPUT_CONTENT_MISMATCH"
    );
    assert_eq!(
        fs::read_to_string(project.join("fixture.py")).unwrap(),
        source
    );
    let snapshot = project.join("input.snapshot");
    let snapshot = snapshot.to_str().unwrap();
    let (out, probe) = run(project, &["model", "probe", "--capture-input-to", snapshot]);
    assert!(out.status.success(), "{probe:#}");
    let input = probe["data"]["batch_content"]["digest"].as_str().unwrap();
    let mut replay = edit.to_vec();
    replay.extend([
        "--dry-run",
        "--if-input-content",
        input,
        "--replay-input-from",
        snapshot,
    ]);
    let (out, preview) = run(project, &replay);
    assert!(out.status.success(), "{preview:#}");
    assert_eq!(
        preview["data"]["validation"]["batch_snapshot"]["status"],
        "replayed"
    );
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let mut apply = edit.to_vec();
    apply.extend([
        "--if-source-plan",
        plan,
        "--if-input-content",
        input,
        "--replay-input-from",
        snapshot,
    ]);
    let (out, committed) = run(project, &apply);
    assert!(out.status.success(), "{committed:#}");
    assert_eq!(
        committed["data"]["source_validation"]["batch_content"]["digest"],
        input
    );
    assert!(
        fs::read_to_string(project.join("fixture.py"))
            .unwrap()
            .contains("bias=False")
    );
}

use super::*;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

impl<B: Backend> Application<B> {
    pub(super) fn create_classification(
        &self,
        cli: &Cli,
        preset: &str,
        data: &Path,
        apply: bool,
        if_plan: Option<&str>,
    ) -> Report {
        let fail = |code, message: String| {
            Report::error(
                "create",
                code,
                message,
                "检查目录和运行环境，然后重新预览。 ",
            )
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let data_root = match root.join(data).canonicalize() {
            Ok(path) if path.is_dir() => path,
            Ok(_) => return fail("DATA_DIRECTORY_INVALID", "数据路径不是目录。".into()),
            Err(error) => return fail("DATA_DIRECTORY_INVALID", error.to_string()),
        };
        let classes = match classification_classes(&data_root) {
            Ok(classes) => classes,
            Err(error) => return fail("CLASSIFICATION_DATA_INVALID", error),
        };
        if let Err(error) = validate_classification_preset_targets(&root) {
            return fail("CREATE_TARGET_CONFLICT", error);
        }
        let model = "classification:model".to_owned();
        let batch = "classification:batch".to_owned();
        let init = |dry_run| {
            self.initialize_project(InitOptions {
                model: Some(&model),
                batch: Some(&batch),
                injections: &[],
                injection_params: &[],
                batch_binding: Some(&BatchBinding::Envelope),
                batch_params: &[],
                dry_run,
            })
        };
        let init_preview = init(true);
        if !init_preview.success() {
            return init_preview;
        }
        if preset == "resnet18" {
            let check = self.run(
                cli,
                &Command::Model {
                    command: ModelCommand::Inspect {
                        path: None,
                        inputs: Inputs {
                            model: Some("model_maven_worker.data.presets:build_resnet18".into()),
                            model_params: vec!["num_classes=2".into(), "weights=\"random\"".into()],
                            ..Inputs::default()
                        },
                    },
                },
            );
            if !check.success() {
                return check;
            }
        }
        let data_value = data_root.strip_prefix(&root).unwrap_or(&data_root);
        let Some(data_text) = data_value.to_str() else {
            return fail("DATA_DIRECTORY_INVALID", "数据路径不是 UTF-8。".into());
        };
        let source = match classification_preset_source(
            preset,
            classes.len(),
            &data_text.replace('\\', "/"),
        ) {
            Ok(source) => source,
            Err(error) => return fail("PRESET_INVALID", error),
        };
        let plan = json!({"version":1,"project_root":root,"data_root":data_root,"classes":classes,
            "preset":preset,"source":source,"config":init_preview.data["config"],
            "files":["classification.py", ".model-maven/project.json"]});
        let revision = match crate::domain::input::content_revision(&plan) {
            Ok(revision) => revision,
            Err(error) => return fail("CREATE_PLAN_FAILED", error),
        };
        if !apply {
            return Report::ok(
                "create",
                json!({"committed":false,"plan":plan,"plan_revision":revision,
                "scope":"类名和文件存在性检查；不验证图像内容，也不固定数据内容。Tiny CNN 环境需在前向时验证。"}),
            );
        }
        if if_plan != Some(revision.as_str()) {
            return fail(
                "CREATE_PLAN_CONFLICT",
                "请使用当前预览的 plan_revision 确认；数据类别或计划发生变化时需重新预览。".into(),
            );
        }
        let created =
            match create_classification_preset_files(&root, &data_root, preset, classes.len()) {
                Ok(source) => source,
                Err(error) => return fail("CREATE_SOURCE_FAILED", error),
            };
        let setup = init(false);
        if !setup.success() {
            let path = root.join("classification.py");
            if std::fs::read(&path).is_ok_and(|current| current == created.as_bytes()) {
                let _ = std::fs::remove_file(path);
            }
            return setup;
        }
        Report::ok(
            "create",
            json!({"committed":true,"plan_revision":revision,
            "files":plan["files"],"model":model,"batch":batch,"classes":classes,
            "note":"已创建可编辑的项目文件；尚未执行前向。"}),
        )
    }
}

pub(crate) fn classification_classes(root: &Path) -> Result<Vec<String>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("无法读取数据目录：{error}"))?;
    if !root.is_dir() {
        return Err("数据路径不是目录。".into());
    }
    let entries = fs::read_dir(&root).map_err(|error| format!("无法列出数据目录：{error}"))?;
    let mut classes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("读取数据目录条目失败：{error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "类别目录名必须是有效 UTF-8 文本。".to_owned())?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("无法检查数据目录条目 `{name}`：{error}"))?;
        if !name.starts_with('.') && file_type.is_dir() {
            if !classification_class_has_images(&entry.path())? {
                return Err(format!(
                    "类别 `{name}` 中没有受支持的图像文件（jpg/jpeg/png/bmp/tif/tiff/webp）。"
                ));
            }
            classes.push(name);
        }
    }
    classes.sort_by_key(|name| name.to_lowercase());
    if classes.len() < 2 {
        return Err(format!(
            "需要至少两个类别子目录；在 {} 中只找到 {} 个。每类图片应放在一个子目录里。",
            root.display(),
            classes.len()
        ));
    }
    Ok(classes)
}

fn classification_class_has_images(root: &Path) -> Result<bool, String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("无法读取类别目录 {}：{error}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("读取类别目录条目失败：{error}"))?;
            let file_type = entry
                .file_type()
                .map_err(|error| format!("无法检查图像条目：{error}"))?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
                        )
                    })
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(crate) fn classification_preset_source(
    preset: &str,
    num_classes: usize,
    data_root: &str,
) -> Result<String, String> {
    let data_root =
        serde_json::to_string(data_root).map_err(|error| format!("无法编码数据目录：{error}"))?;
    let (preset_name, builder, model_args, transforms) = match preset {
        "tiny-cnn" => (
            "Tiny CNN",
            "build_tiny_cnn",
            "",
            r#"[{"type": "Resize", "size": 32}, {"type": "ToTensor"}]"#,
        ),
        "resnet18" => (
            "ResNet18 (random initialization; no weight download)",
            "build_resnet18",
            ", weights=\"random\"",
            r#"[{"type": "Resize", "size": 224}, {"type": "ToTensor"}, {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}]"#,
        ),
        _ => return Err(format!("未知分类预设：{preset}")),
    };
    Ok(format!(
        r#"# Editable {preset_name} image-classification starter created by Model-Maven.

# Preset: {preset_name}
# Change NUM_CLASSES, DATA_ROOT, or the model/batch functions as your project grows.
# Images are read in place from DATA_ROOT/<class-name>; they are never moved.
# ResNet18 uses random weights. Preparing pretrained weights is a separate explicit operation.

from model_maven_worker.data.presets import (
    {builder},
    classification_batch as _classification_batch,
    classification_step,
)

NUM_CLASSES = {num_classes}
DATA_ROOT = {data_root}


def model():
    return {builder}(num_classes=NUM_CLASSES{model_args})


def batch():
    classification = _classification_batch(
        root=DATA_ROOT,
        batch_size=8,
        split="train",
        split_ratios={{"train": 1.0}},
        transforms={transforms},
    )
    return {{
        "args": [classification["inputs"]],
        "kwargs": {{}},
        "loss_inputs": {{
            "targets": classification["targets"],
            "dataset": classification["dataset"],
        }},
    }}


def step(model, batch, context):
    return classification_step(model, batch, context)


batch.state_dict = _classification_batch.state_dict
batch.load_state_dict = _classification_batch.load_state_dict
"#
    ))
}

pub(crate) fn validate_classification_preset_targets(root: &Path) -> Result<(), String> {
    let source_path = root.join("classification.py");
    let metadata_path = root.join(".model-maven");
    match fs::symlink_metadata(&metadata_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(".model-maven 必须是普通项目目录；为避免写入链接目标，预设已停止。".into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("无法检查 .model-maven 目录：{error}")),
    }
    let config_path = metadata_path.join("project.json");
    for (label, path) in [("项目配置", &config_path), ("分类入口", &source_path)] {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "{} 已存在，预设不会覆盖它。可先备份或选择其他项目目录。",
                    path.strip_prefix(root).unwrap_or(path).display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("无法检查{label}路径：{error}")),
        }
    }
    Ok(())
}

pub(crate) fn create_classification_preset_files(
    root: &Path,
    data_root: &Path,
    preset: &str,
    num_classes: usize,
) -> Result<String, String> {
    validate_classification_preset_targets(root)?;
    let source_path = root.join("classification.py");

    let data_value = match data_root.strip_prefix(root) {
        Ok(relative) => relative
            .to_str()
            .ok_or_else(|| "数据目录相对路径必须是有效 UTF-8 文本。".to_owned())?
            .replace('\\', "/"),
        Err(_) => data_root
            .to_str()
            .ok_or_else(|| "数据目录路径必须是有效 UTF-8 文本。".to_owned())?
            .to_owned(),
    };
    let source = classification_preset_source(preset, num_classes, &data_value)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source_path)
        .map_err(|error| format!("无法安全创建 classification.py：{error}"))?;
    if let Err(error) = file
        .write_all(source.as_bytes())
        .and_then(|()| file.sync_all())
    {
        let _ = fs::remove_file(&source_path);
        return Err(format!("写入分类入口失败：{error}"));
    }
    Ok(source)
}

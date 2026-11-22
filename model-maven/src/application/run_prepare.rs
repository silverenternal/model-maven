use super::*;
use std::path::{Path, PathBuf};

pub(super) fn variant_base_token(variant: Option<&Value>) -> Result<String, String> {
    variant.map_or_else(
        || Ok("absent".to_owned()),
        crate::domain::input::content_revision,
    )
}

pub(super) struct PreparedRunInputs {
    pub variant_name: String,
    pub variant_path: PathBuf,
    pub variant: Option<Value>,
    pub model_inputs: super::model::ModelInputs,
    pub batch: super::data::BatchInput,
    _variant_lock: Option<Box<dyn VariantLock>>,
    _source_lock: Option<Box<dyn SourceEditLock>>,
}

impl<B: Backend> Application<B> {
    pub(super) fn prepare_run_inputs(
        &self,
        cli: &Cli,
        job: &mut Job,
        root: &Path,
        source_target: bool,
    ) -> Result<PreparedRunInputs, Report> {
        if cli.if_variant_base.is_some()
            && !matches!(job.operation, "probe" | "training_probe" | "train")
            && (job.edit.is_none() || source_target || job.dry_run)
        {
            return Err(Report::error(
                "edit",
                "VARIANT_BASE_UNSUPPORTED",
                "变体基准条件只支持实际的变体编辑提交。",
                "先使用 --dry-run 获取基准，再对变体提交指定条件。",
            ));
        }
        let variant_name = cli.variant.as_deref().unwrap_or("working").to_owned();
        let external_loop = matches!(job.operation, "training_probe" | "train")
            && job.training_mode == Some("external-loop");
        let variant_path = self
            .variants
            .path(root, &variant_name)
            .map_err(|error| Report::error(job.operation, "VARIANT_NAME_INVALID", error, ""))?;
        let use_variant = cli.variant.is_some()
            || (job.layer_context && !source_target)
            || (job.inputs.model.is_none() && !external_loop)
            || job.variant_snapshot.is_some();
        let mut variant = if use_variant {
            self.variants
                .load(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_READ_FAILED", error, ""))?
        } else {
            None
        };
        if let Some(snapshot) = &job.variant_snapshot {
            variant = Some(snapshot.clone());
        }
        if variant.is_none() && cli.variant.is_some() && job.edit.is_some() {
            self.resolve_variant_edit_defaults(root, job)?;
        }
        if variant.is_none() && cli.variant.is_some() && job.edit.is_none() {
            return Err(Report::error(
                job.operation,
                "VARIANT_NOT_FOUND",
                format!("找不到变体：{variant_name}"),
                "先创建变体或省略 --variant 并提供 --model。",
            ));
        }
        if variant.is_none() && job.inputs.model.is_none() && !external_loop {
            return Err(Report::error(
                job.operation,
                "MODEL_REQUIRED",
                "缺少模型入口。",
                "例如 --model examples.models:build_shared；前向再加 --batch examples.models:shared_batch。",
            ));
        }
        let model_inputs = self.resolve_model_inputs(job, root)?;
        let batch = super::data::batch_input(&job.inputs, job.operation)?;

        // Keep both guards alive until Backend execution and edit finalization return.
        let variant_lock = if (job.edit.is_some() && !source_target && !job.dry_run)
            || (matches!(job.operation, "probe" | "training_probe" | "train")
                && cli.if_variant_base.is_some())
        {
            let lock = self
                .variants
                .lock(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_LOCKED", error, ""))?;
            variant = self
                .variants
                .load(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_READ_FAILED", error, ""))?;
            if let Some(expected) = &cli.if_variant_base {
                if expected != "absent"
                    && (expected.len() != 64
                        || !expected
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
                {
                    return Err(Report::error(
                        job.operation,
                        "VARIANT_BASE_INVALID",
                        "变体基准格式无效。",
                        "使用预览报告提供的 token。",
                    ));
                }
                let actual = variant_base_token(variant.as_ref()).map_err(|error| {
                    Report::error(
                        job.operation,
                        "VARIANT_BASE_UNAVAILABLE",
                        error,
                        "重新预览。",
                    )
                })?;
                if *expected != actual {
                    return Err(Report::error(
                        job.operation,
                        "VARIANT_BASE_CONFLICT",
                        "变体与指定的基准不同。",
                        "本次操作未执行；重新诊断并核对变体。",
                    ));
                }
            }
            if matches!(job.operation, "training_probe" | "train") {
                // The checked value is copied into the worker request; do not
                // hold the cooperative edit lock throughout a training run.
                drop(lock);
                None
            } else {
                Some(lock)
            }
        } else {
            None
        };
        let source_lock = if source_target && !job.dry_run {
            Some(self.source_edits.lock(root).map_err(|error| {
                Report::error("edit", "SOURCE_EDIT_LOCKED", error, "稍后重试源码编辑。")
            })?)
        } else {
            None
        };
        self.verify_model_inputs(job, root, &model_inputs, variant.as_ref())?;

        Ok(PreparedRunInputs {
            variant_name,
            variant_path,
            variant,
            model_inputs,
            batch,
            _variant_lock: variant_lock,
            _source_lock: source_lock,
        })
    }
}

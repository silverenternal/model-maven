use super::*;
use std::path::Path;

pub(super) fn build_worker_request(
    job: &mut Job,
    root: &Path,
    prepared: &super::run_prepare::PreparedRunInputs,
    context: ExecutionContext<'_>,
) -> Result<Value, Report> {
    if let Some(request_id) = &job.request_id {
        if !(16..=64).contains(&request_id.len())
            || !request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(Report::error(
                job.operation,
                "REQUEST_ID_INVALID",
                "训练请求标识须为 16 到 64 位字母、数字或连字符。",
                "为每次启动生成唯一标识；不要把密钥或个人信息放入标识。",
            ));
        }
    }
    if let Some(name) = job.run_name.as_mut() {
        let trimmed = name.trim();
        if trimmed.is_empty()
            || trimmed.chars().count() > 80
            || trimmed.chars().any(char::is_control)
        {
            return Err(Report::error(
                job.operation,
                "RUN_NAME_INVALID",
                "运行名称须为 1 到 80 个可见字符。",
                "移除控制字符并缩短名称。",
            ));
        }
        *name = trimmed.to_owned();
    }
    if let Some(fingerprint) = job.batch_fingerprint.as_mut() {
        let trimmed = fingerprint.trim();
        if trimmed.is_empty() || trimmed.len() > 256 || trimmed.chars().any(char::is_control) {
            return Err(Report::error(
                job.operation,
                "BATCH_FINGERPRINT_INVALID",
                "批次指纹须为 1 到 256 个可见字符。",
                "使用不含凭据的数据/切分版本标识；例如 dataset-v3/train-2026-01。",
            ));
        }
        *fingerprint = trimmed.to_owned();
    }
    let request = WorkerRequest {
        protocol_version: PROTOCOL_VERSION,
        operation: job.operation,
        project_root: root,
        model: job.inputs.model.as_deref(),
        model_params: &prepared.model_inputs.parameters,
        model_injections: &prepared.model_inputs.injections,
        batch: prepared.batch.entrypoint.as_deref(),
        batch_params: &prepared.batch.parameters,
        batch_binding: prepared.batch.binding,
        training: job.inputs.training,
        seed: job.seed,
        variant: job.variant_snapshot.as_ref().or(prepared.variant.as_ref()),
        path: job.path.as_deref(),
        layer_type: job.layer_type.as_deref(),
        search: job.search.as_deref(),
        call: &job.call,
        contract: job.contract.as_deref(),
        data_fingerprint: job.batch_fingerprint.as_deref(),
        local_probe: job.local_probe,
        loss: job.loss.as_deref(),
        training_step: job.training_step.as_deref(),
        training_loop: job.training_loop.as_deref(),
        training_mode: job.training_mode,
        training_params: &job.training_params,
        apply_supervision: job.apply_supervision,
        expected_supervision_revision: job.expected_supervision_revision.as_deref(),
        repeat_of: job.repeat_of.as_deref(),
        repeat_overrides: &job.repeat_overrides,
        parent_run_id: job.parent_run_id.as_deref(),
        run_name: job.run_name.as_deref(),
        request_id: job.request_id.as_deref(),
        expected_source_manifest: job.expected_source_manifest.as_ref(),
        expected_runtime: job.expected_runtime.as_ref(),
        runtime: &job.runtime,
        steps: job.steps,
        optimizer: job.optimizer.as_deref(),
        lr: job.lr,
        weight_decay: job.weight_decay,
        checkpoint_every: job.checkpoint_every,
        checkpoint_input: job.checkpoint_input.as_deref(),
        expected_checkpoint_sha256: job.expected_checkpoint_sha256.as_deref(),
        migration_checkpoint_input: job.migration_checkpoint_input.as_deref(),
        migration_plan: job.migration_plan.as_ref(),
        allow_state_reset: job.allow_state_reset,
        checkpoint_dir: None,
        interactive_progress: context.show_progress,
        progress_path: None,
        stop_file: None,
        edit: job.edit.as_ref(),
        require_forward: job.require_forward,
    };
    serde_json::to_value(request).map_err(|error| {
        Report::error(
            job.operation,
            "REQUEST_SERIALIZATION_FAILED",
            error.to_string(),
            "请求尚未发送到 Python worker；检查参数是否包含非有限数字。",
        )
    })
}

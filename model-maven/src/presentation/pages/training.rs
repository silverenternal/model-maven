use crate::cli::TrainingMode;

pub fn mode_label(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::ManagedStep => "managed-step",
        TrainingMode::ManualStep => "manual-step",
        TrainingMode::ExternalLoop => "external-loop",
    }
}

pub fn mode_description(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::ManagedStep => {
            "提供 training_step(model, batch, context)；CLI 管理反向传播和单组 AdamW 更新。"
        }
        TrainingMode::ManualStep => {
            "提供 training_step；用户代码管理反向传播、具名优化器和更新阶段，CLI 检查实际观测。"
        }
        TrainingMode::ExternalLoop => {
            "提供 run_loop(model, context, **params)；循环自行管理数据与更新，不要求独立 batch 或 loss。"
        }
    }
}

pub fn render_mode_guide() -> String {
    [
        "训练模式：",
        &format!(
            "1 managed-step · {}",
            mode_description(TrainingMode::ManagedStep)
        ),
        &format!(
            "2 manual-step · {}",
            mode_description(TrainingMode::ManualStep)
        ),
        &format!(
            "3 external-loop · {}",
            mode_description(TrainingMode::ExternalLoop)
        ),
        "输入序号，留空选择 managed-step；back 取消。",
    ]
    .join("\n")
}

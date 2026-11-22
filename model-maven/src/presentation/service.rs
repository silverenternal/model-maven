use crate::{
    app::Application,
    app::ExecutionContext,
    cli::{Cli, Command, LayerCommand, ModelCommand, RunsCommand, SuperviseCommand},
    domain::Report,
    ports::{Backend, PythonRuntimeControl, services::ExecutionProgress},
    presentation::contracts::TaskProgress,
};
use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::path::Path;

/// The boundary between a page and application operations. Pages receive reports and
/// do not access project files or repositories directly.
pub trait UiServices {
    fn execute(&mut self, command: &Command) -> Report;

    fn stderr_is_terminal(&self) -> bool {
        false
    }

    /// Returns unknown until a service can provide authoritative run events/state.
    fn task_progress(&mut self, _run_id: &str) -> TaskProgress {
        TaskProgress::unknown()
    }

    fn cancel_task(&mut self, _run_id: &str) -> Report {
        Report::error(
            "ui_cancel",
            "CANCELLATION_UNSUPPORTED",
            "当前 UI 服务不支持取消运行。",
            "检查 doctor 显示的运行监督能力。",
        )
    }

    fn project_defaults(&self, _root: &Path) -> Result<Option<Value>, String> {
        Err("当前 UI 服务未提供项目默认值查询。".into())
    }

    fn inspect_component(&self, _root: &Path, _id: &str) -> Result<Value, String> {
        Err("当前 UI 服务未提供组件查询。".into())
    }

    fn resolve_layer_component(
        &self,
        _root: &Path,
        _id: &str,
        _params: &Map<String, Value>,
    ) -> Result<Value, String> {
        Err("当前 UI 服务未提供 Layer 组件解析。".into())
    }

    fn resolve_layer_edit_provider(
        &self,
        _root: &Path,
        _id: &str,
        _payload: &Map<String, Value>,
    ) -> Result<Value, String> {
        Err("当前 UI 服务未提供 Layer 编辑 Provider 解析。".into())
    }
}

struct TerminalExecutionProgress;

impl ExecutionProgress for TerminalExecutionProgress {
    fn worker_starting(&self, operation: &str, timeout_seconds: u64) {
        eprintln!("正在执行 {operation}（CPU，最多 {timeout_seconds} 秒）…");
    }
}

pub struct ApplicationUiServices<'a, B: Backend> {
    application: &'a Application<B>,
    cli: &'a Cli,
}

impl<'a, B: Backend> ApplicationUiServices<'a, B> {
    pub fn new(application: &'a Application<B>, cli: &'a Cli) -> Self {
        Self { application, cli }
    }
}

impl<B: Backend> UiServices for ApplicationUiServices<'_, B> {
    fn execute(&mut self, command: &Command) -> Report {
        let interactive =
            crate::presentation::terminal::is_interactive(self.cli.json, self.cli.no_interactive);
        let mut command = command.clone();
        if interactive {
            if matches!(&command, Command::Init { model: None, .. }) {
                collect_missing_inputs(&mut command);
            } else {
                match self.application.plan_operation_inputs(self.cli, &command) {
                    Ok(Some(plan)) => {
                        collect_planned_inputs(&mut command, &plan, self.cli.variant.is_some());
                    }
                    Ok(None) => {}
                    Err(report) => return report,
                }
            }
        }
        let readiness_plan = match self.application.plan_operation_inputs(self.cli, &command) {
            Ok(plan) => plan,
            Err(report) => return report,
        };
        if let Some(plan) = readiness_plan.as_ref().filter(|plan| {
            plan.readiness.state == crate::domain::operation::ReadinessState::NeedsInput
        }) {
            let operation = plan.operation.as_str().to_owned();
            let (code, message, recovery) = readiness_diagnostic(plan, false);
            let message = readiness_message(plan, message);
            let mut report = Report::error(&operation, code, &message, recovery);
            report.data = serde_json::json!({"readiness":readiness_value(plan)});
            return report;
        }
        if let Some(plan) = readiness_plan.as_ref().filter(|plan| {
            plan.readiness.state == crate::domain::operation::ReadinessState::Unsupported
        }) {
            let capability_blocker = plan.readiness.capabilities.iter().any(|capability| {
                capability.required
                    && capability.state == crate::domain::operation::FactState::Unsupported
            });
            let (code, message, recovery) = if capability_blocker {
                (
                    "OPERATION_CAPABILITY_UNAVAILABLE",
                    "当前 Python/Worker 环境不支持该操作所需能力；目标操作没有启动。",
                    "运行 m-model doctor 查看环境和 Worker 能力，选择兼容的 Python 环境后重试。",
                )
            } else {
                readiness_diagnostic(plan, true)
            };
            let message = readiness_message(plan, message);
            let mut report = Report::error(plan.operation.as_str(), code, &message, recovery);
            report.data = serde_json::json!({"readiness":readiness_value(plan)});
            return report;
        }
        let progress = TerminalExecutionProgress;
        let mut context = ExecutionContext {
            require_assembly_confirmation: interactive,
            show_progress: !self.cli.json && self.stderr_is_terminal(),
            confirm_assembly: false,
            confirmed_assembly_revision: None,
            progress_observer: (!self.cli.json).then_some(&progress as &dyn ExecutionProgress),
        };
        let mut report = self
            .application
            .run_with_context(self.cli, &command, context);
        if report.data["confirmation_required"] == true {
            let action = report.data["action"].as_str().unwrap_or("edit");
            let slot = report.data["slot"].as_str().unwrap_or("unknown");
            eprintln!(
                "装配候选已通过构造验证：{action} 接缝 {slot}。前向检查：{}。",
                report.data["validation"]["forward"]
            );
            let confirmed = crate::presentation::terminal::prompt(
                "写入项目默认装配？输入 y/yes 确认，其他输入取消",
            )
            .is_some_and(|answer| matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes"));
            if confirmed {
                context.confirm_assembly = true;
                context.confirmed_assembly_revision = report.data["base_config_revision"].as_str();
                report = self
                    .application
                    .run_with_context(self.cli, &command, context);
            } else {
                report.data["cancelled"] = Value::Bool(true);
                report.data["confirmation_required"] = Value::Bool(false);
                report.data["note"] =
                    Value::String("候选已构造验证；本次交互已取消，project.json 未更改。".into());
            }
        }
        if let Some(plan) = readiness_plan.as_ref() {
            let readiness = readiness_value(plan);
            if !report.data.is_object() {
                report.data = serde_json::json!({"result":report.data});
            }
            report.data["readiness"] = readiness;
        }
        report
    }

    fn stderr_is_terminal(&self) -> bool {
        crate::presentation::terminal::stderr_is_terminal()
    }

    fn cancel_task(&mut self, run_id: &str) -> Report {
        self.execute(&Command::Runs {
            command: RunsCommand::Stop {
                id: Some(run_id.to_owned()),
                force: false,
            },
        })
    }

    fn project_defaults(&self, root: &Path) -> Result<Option<Value>, String> {
        self.application.project_defaults(root)
    }

    fn inspect_component(&self, root: &Path, id: &str) -> Result<Value, String> {
        self.application.components.inspect(root, id)
    }

    fn resolve_layer_component(
        &self,
        root: &Path,
        id: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String> {
        self.application.components.resolve_layer(root, id, params)
    }

    fn resolve_layer_edit_provider(
        &self,
        root: &Path,
        id: &str,
        payload: &Map<String, Value>,
    ) -> Result<Value, String> {
        self.application
            .components
            .resolve_layer_edit_provider(root, id, payload)
    }
}

fn readiness_message(plan: &crate::domain::operation::OperationPlan, message: &str) -> String {
    let detail = plan
        .readiness
        .facts
        .iter()
        .find(|fact| fact.required && fact.state != crate::domain::operation::FactState::Available)
        .and_then(|fact| fact.reason.as_deref());
    match detail {
        Some(detail) => format!("{message} 详情：{detail}"),
        None => message.to_owned(),
    }
}

fn readiness_diagnostic(
    plan: &crate::domain::operation::OperationPlan,
    unsupported: bool,
) -> (&'static str, &'static str, &'static str) {
    let blocking_fact = plan.readiness.facts.iter().find(|fact| {
        fact.required
            && fact.state
                == if unsupported {
                    crate::domain::operation::FactState::Unsupported
                } else {
                    crate::domain::operation::FactState::Missing
                }
    });
    match blocking_fact.map(|fact| fact.key.as_str()) {
        Some("component_dependencies") => (
            "COMPONENT_DEPENDENCY_RESOLUTION_FAILED",
            "组件的必需依赖闭包无法解析；目标操作没有启动。",
            "补齐兼容的组件版本后运行 m-model component check。",
        ),
        Some("component_parameters") => (
            "COMPONENT_RESOLUTION_FAILED",
            "组件参数未通过静态 Schema 校验；目标操作没有启动。",
            "检查组件参数 Schema，并使用 --param NAME=JSON 提供合法值。",
        ),
        Some("component_conformance_fixture" | "layer_provider_fixture") => (
            "COMPONENT_FIXTURE_MISSING",
            "组件缺少必需的符合性 fixture；目标操作没有启动。",
            "为组件声明有效的 conformance_fixture 后重新登记。",
        ),
        Some("layer_provider_verification") if !unsupported => (
            "COMPONENT_PROVIDER_UNVERIFIED",
            "Provider 没有与当前 payload 匹配的通过记录；目标编辑没有启动。",
            "运行 m-model component test 并提供与本次编辑相同的 payload。",
        ),
        Some("layer_provider_verification") => (
            "COMPONENT_PROVIDER_VERIFICATION_STALE",
            "Provider 验证记录与当前清单或 payload 不匹配；目标编辑没有启动。",
            "重新运行 m-model component test。",
        ),
        Some("layer_provider_capability") => {
            let conditional = blocking_fact
                .and_then(|fact| fact.reason.as_deref())
                .is_some_and(|reason| reason.contains("conditional"));
            if conditional {
                (
                    "COMPONENT_PROVIDER_CONDITION_UNVERIFIED",
                    "Provider 声明的 layer_edit 能力仍有未满足条件；目标编辑没有启动。",
                    "为当前 Provider/模型条件取得 supported 验证后重试。",
                )
            } else {
                (
                    "COMPONENT_PROVIDER_UNSUPPORTED",
                    "Provider 没有声明可用的 layer_edit 能力；目标编辑没有启动。",
                    "检查 Provider 能力声明并通过 component test。",
                )
            }
        }
        Some("layer_provider_kind") => (
            "COMPONENT_INVALID",
            "指定组件不是 Layer Edit Provider；目标编辑没有启动。",
            "选择 kind 为 layer-edit-provider 的组件。",
        ),
        Some("component_test_kind") => (
            "COMPONENT_TEST_UNSUPPORTED",
            "该组件类型没有对应的符合性测试器；Worker 未启动。",
            "当前 component test 支持 layer、layer-edit-provider、transform、dataset、loss、metric 和 trainer。",
        ),
        _ if unsupported => (
            "OPERATION_REQUIREMENT_UNAVAILABLE",
            "操作的静态前置条件未满足；目标操作没有启动。",
            "根据 readiness.facts 中的原因修复前置条件后重试。",
        ),
        _ => (
            "OPERATION_NEEDS_INPUT",
            "执行前检查发现必需输入尚未提供；目标操作没有启动。",
            "根据 readiness.actions 补齐所需入口后重试。",
        ),
    }
}

fn readiness_value(plan: &crate::domain::operation::OperationPlan) -> Value {
    let mut readiness = serde_json::to_value(plan).unwrap_or(Value::Null);
    if let Some(inputs) = readiness["inputs"].as_array_mut() {
        for input in inputs {
            if let Some(object) = input.as_object_mut() {
                object.remove("value");
            }
        }
    }
    readiness
}

fn plan_requires_input(plan: &crate::domain::operation::OperationPlan, key: &str) -> bool {
    plan.inputs.iter().any(|input| {
        input.key == key
            && input.required
            && input.state == crate::domain::operation::FactState::Missing
    })
}

fn collect_planned_inputs(
    command: &mut Command,
    plan: &crate::domain::operation::OperationPlan,
    has_variant: bool,
) {
    let needs_model = plan_requires_input(plan, "model") && !has_variant;
    let needs_batch = plan_requires_input(plan, "batch");
    let needs_step = plan_requires_input(plan, "training_step");
    let needs_loop = plan_requires_input(plan, "training_loop");
    match command {
        Command::Model { command } => {
            let inputs = match command {
                ModelCommand::Inspect { inputs, .. }
                | ModelCommand::Probe { inputs, .. }
                | ModelCommand::Mark { inputs, .. }
                | ModelCommand::Retag { inputs, .. }
                | ModelCommand::Marks { inputs }
                | ModelCommand::Verify { inputs, .. } => inputs,
                ModelCommand::Unmark { .. } => return,
                ModelCommand::Supervise {
                    command: SuperviseCommand::Guide { inputs },
                }
                | ModelCommand::Supervise {
                    command: SuperviseCommand::Preview { inputs },
                } => inputs,
                ModelCommand::Supervise {
                    command: SuperviseCommand::Rule { .. },
                } => return,
            };
            if needs_model && inputs.model.is_none() {
                inputs.model =
                    crate::presentation::terminal::prompt("原生模型入口（module:factory）");
            }
            if needs_batch && inputs.batch.is_none() {
                inputs.batch =
                    crate::presentation::terminal::prompt("示例批次入口（module:factory）");
            }
        }
        Command::Try {
            inputs,
            step,
            training_loop,
            ..
        }
        | Command::Train {
            inputs,
            step,
            training_loop,
            ..
        } => {
            if needs_model && inputs.model.is_none() {
                inputs.model =
                    crate::presentation::terminal::prompt("原生模型入口（module:factory）");
            }
            if needs_batch && inputs.batch.is_none() {
                inputs.batch =
                    crate::presentation::terminal::prompt("示例批次入口（module:factory）");
            }
            if needs_step && step.is_none() {
                *step = crate::presentation::terminal::prompt("训练步骤入口（module:callable）");
            }
            if needs_loop && training_loop.is_none() {
                *training_loop =
                    crate::presentation::terminal::prompt("用户训练循环入口（module:callable）");
            }
        }
        Command::Layer { command } => {
            let inputs = match command {
                LayerCommand::Catalog => return,
                LayerCommand::List { inputs, .. }
                | LayerCommand::Show { inputs, .. }
                | LayerCommand::Undo { inputs, .. }
                | LayerCommand::Redo { inputs, .. } => inputs,
                LayerCommand::MigrateStarter { edit, .. }
                | LayerCommand::Add { edit, .. }
                | LayerCommand::Set { edit, .. }
                | LayerCommand::Replace { edit, .. }
                | LayerCommand::Remove { edit, .. }
                | LayerCommand::Apply { edit, .. }
                | LayerCommand::Custom { edit, .. } => &mut edit.inputs,
            };
            if needs_model && inputs.model.is_none() {
                inputs.model =
                    crate::presentation::terminal::prompt("原生模型入口（module:factory）");
            }
            if needs_batch && inputs.batch.is_none() {
                inputs.batch =
                    crate::presentation::terminal::prompt("示例批次入口（module:factory）");
            }
        }
        _ => {}
    }
}

fn collect_missing_inputs(command: &mut Command) {
    let Command::Init {
        model,
        batch,
        injections,
        ..
    } = command
    else {
        return;
    };
    if model.is_none() {
        *model = crate::presentation::terminal::prompt("原生模型工厂入口（module:callable）");
        if model.is_none() {
            return;
        }
    }
    if batch.is_none() {
        let answer = crate::presentation::terminal::prompt_allow_empty(
            "可选：代表性批次工厂（module:callable；留空以后再提供）",
        );
        if answer.is_none() {
            return;
        }
        *batch = answer.filter(|value| !value.is_empty());
    }
    if injections.is_empty() {
        while let Some(value) = crate::presentation::terminal::prompt_allow_empty(
            "可选：模型接缝注入 SLOT=module:factory；留空结束",
        ) {
            if value.is_empty() {
                break;
            }
            injections.push(value);
        }
    }
}

impl<B: PythonRuntimeControl> ApplicationUiServices<'_, B> {
    pub fn select_python(application: &mut Application<B>, executable: String) {
        application.backend.select_python(executable);
        application.runtime_capabilities.borrow_mut().take();
    }
}

/// Deterministic responses for page previews and interaction tests. Exhaustion is
/// explicit, so an unconfigured mock cannot look like a successful operation.
pub struct RecordedUiServices {
    responses: VecDeque<Report>,
    pub calls: usize,
}

impl RecordedUiServices {
    pub fn new(responses: impl IntoIterator<Item = Report>) -> Self {
        Self {
            responses: responses.into_iter().collect(),
            calls: 0,
        }
    }
}

impl UiServices for RecordedUiServices {
    fn execute(&mut self, _command: &Command) -> Report {
        self.calls += 1;
        self.responses.pop_front().unwrap_or_else(|| {
            Report::error(
                "ui_mock",
                "MOCK_RESPONSE_MISSING",
                "录制的 UI 服务没有剩余响应。",
                "为此操作添加明确的模拟报告。",
            )
        })
    }
}

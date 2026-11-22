use crate::{domain::Report, presentation::widgets::recovery_actions};
use serde_json::{Value, json};

pub fn render(report: &Report, show_details: bool) -> Vec<String> {
    let Some(primary) = report.diagnostics.first() else {
        return vec!["当前操作没有可展示的结构化诊断。".into()];
    };
    let mut lines = vec![
        if report.status == "inconclusive" {
            "结果尚未判定".into()
        } else {
            "操作未成功".into()
        },
        format!("原因：{}", primary.message),
        format!("影响：{}", impact_summary(report, primary)),
    ];
    let mut actions = report
        .diagnostics
        .iter()
        .flat_map(|diagnostic| {
            diagnostic
                .details
                .iter()
                .flat_map(|details| &details.recovery_actions)
                .map(|action| {
                    json!({
                        "label":action.label,
                        "availability":action.availability,
                        "reason":action.reason,
                        "requires_confirmation":action.requires_confirmation
                    })
                })
        })
        .collect::<Vec<_>>();
    let recovery_case_id = recovery_case_for_code(&primary.code);
    let mut safety_boundary = None;
    if actions.is_empty()
        && let Some(id) = recovery_case_id
        && let Some((_, suggested_actions, guardrail)) = recovery_case_details(id)
    {
        actions = suggested_actions
            .iter()
            .map(|label| {
                json!({
                    "label":label,
                    "availability":"unknown",
                    "reason":"这是恢复建议；执行能力需要在对应页面确认。",
                    "requires_confirmation":true
                })
            })
            .collect();
        safety_boundary = Some(guardrail);
    }
    if actions.is_empty()
        && recovery_case_id.is_none()
        && let Some(hint) = primary.hint.as_ref()
    {
        actions.push(json!({"label":hint,"availability":"unknown"}));
    }
    lines.push("建议下一步：".into());
    lines.extend(recovery_actions::render(&actions));
    if let Some(boundary) = safety_boundary {
        lines.push(format!("安全边界：{boundary}"));
    }
    if show_details {
        if actions.len() > 3 {
            lines.push("其他可选动作详情：".into());
            lines.extend(recovery_actions::render_details(&actions[3..], 3));
        }
        for diagnostic in &report.diagnostics {
            lines.push(format!("诊断码：{}", diagnostic.code));
            if let Some(hint) = &diagnostic.hint {
                lines.push(format!("修复提示：{hint}"));
            }
            if let Some(details) = &diagnostic.details {
                lines.push(format!("严重级别：{:?}", details.severity));
                lines.push(format!("证据范围：{:?}", details.evidence_state));
                if let Some(scope) = &details.scope {
                    for (label, value) in [
                        ("操作", scope.operation_id.as_deref()),
                        ("运行", scope.run_id.as_deref()),
                        ("模型", scope.model_ref.as_deref()),
                        ("批次", scope.batch_ref.as_deref()),
                        ("Layer", scope.module_path.as_deref()),
                        ("源码版本", scope.source_revision.as_deref()),
                    ] {
                        if let Some(value) = value {
                            lines.push(format!("{label}：{value}"));
                        }
                    }
                }
            }
        }
        if let Some(verification) = report.data.get("provider_verification") {
            lines.push("Provider 验证记录与本次执行条件：".into());
            if let Some(fields) = verification["mismatch_fields"].as_array() {
                let fields = fields.iter().filter_map(Value::as_str).collect::<Vec<_>>();
                if !fields.is_empty() {
                    lines.push(format!("差异字段：{}", fields.join("、")));
                }
            }
            for (key, label) in [
                ("payload_sha256", "编辑 payload 指纹"),
                ("python_version", "Python 版本"),
                ("torch_version", "PyTorch 版本"),
            ] {
                let expected = verification["expected"][key].as_str().unwrap_or("未提供");
                let recorded = verification["recorded"][key].as_str().unwrap_or("未提供");
                if expected != recorded {
                    lines.push(format!("{label}：已验证 {recorded} · 当前 {expected}"));
                }
            }
            if verification["mismatch_fields"]
                .as_array()
                .is_some_and(|fields| fields.iter().any(|field| field == "source_fingerprints"))
            {
                for role in ["provider", "fixture"] {
                    let expected = &verification["expected"]["source_fingerprints"][role];
                    let recorded = &verification["recorded"]["source_fingerprints"][role];
                    if expected != recorded {
                        lines.push(format!(
                            "{role} 源码 SHA-256：已验证 {} · 当前 {}",
                            recorded["sha256"].as_str().unwrap_or("未知"),
                            expected["sha256"].as_str().unwrap_or("未知")
                        ));
                    }
                }
            }
        }
        if let Some(location) = report.data["failure_location"].as_object() {
            if let Some(call) = location
                .get("last_observed_module_call")
                .and_then(Value::as_object)
            {
                let paths = call["paths"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(" | ");
                lines.push(format!(
                    "最后观测到的模块：{} · {} · 第 {} 次调用",
                    call.get("module_type")
                        .and_then(Value::as_str)
                        .unwrap_or("未知类型"),
                    if paths.is_empty() {
                        "路径未知"
                    } else {
                        &paths
                    },
                    call.get("invocation_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                ));
            }
            for frame in location
                .get("project_stack")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                lines.push(format!(
                    "源码位置：{}:{} · {}",
                    frame["file"].as_str().unwrap_or("未知文件"),
                    frame["line"].as_u64().unwrap_or(0),
                    frame["function"].as_str().unwrap_or("未知函数")
                ));
            }
        }
        if let Some(log) = report.data["log_tail"]
            .as_str()
            .filter(|log| !log.is_empty())
        {
            lines.push("最近日志：".into());
            lines.extend(
                log.lines()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .map(str::to_owned),
            );
        }
    } else {
        lines.push("输入 details 查看诊断码、调用位置和最近日志。".into());
    }
    lines
}

pub fn recovery_case(id: &str) -> Option<Vec<String>> {
    let (problem, actions, guardrail) = recovery_case_details(id)?;
    let mut lines = vec![format!("{id} · {problem}"), "推荐动作：".into()];
    lines.extend(
        actions
            .iter()
            .enumerate()
            .map(|(index, action)| format!("{}. {action} · 需在对应页面确认执行能力", index + 1)),
    );
    lines.push(format!("安全边界：{guardrail}"));
    Some(lines)
}

fn recovery_case_details(
    id: &str,
) -> Option<(&'static str, &'static [&'static str], &'static str)> {
    Some(match id {
        "E1" => (
            "数据路径不存在或目录格式不支持",
            &[
                "在原位置选择其他目录",
                "查看支持的布局示例",
                "保存进度稍后继续",
            ],
            "修正路径后从目录探测继续；不会重建项目。",
        ),
        "E2" => (
            "当前类别样本不足以形成所选切分",
            &[
                "补充样本后重扫受影响类别",
                "指定已有验证集",
                "返回调整数据或切分策略",
            ],
            "展示每类数量和可行性；不能靠调整比例掩盖无解情况，也不自动删类别。",
        ),
        "E3" => (
            "图片损坏或无法解码",
            &["查看具体文件清单", "修复后重试", "明确排除所选文件后继续"],
            "排除项仅进入可追踪清单；不删除原文件，且继续前重验切分和数据指纹。",
        ),
        "E4" => (
            "依赖或权重下载失败",
            &[
                "在原授权范围内重试",
                "使用已验证的兼容缓存",
                "查看离线准备清单",
            ],
            "保留已完成内容；无验证缓存时不报告离线成功，也不无限重试。",
        ),
        "E5" => (
            "磁盘不足导致准备或 checkpoint 保存失败",
            &[
                "查看所需空间、目标目录与可用空间",
                "选择受支持的其他输出位置",
                "查看可安全清理的工具产物",
            ],
            "保留旧 checkpoint 和用户数据；未成功保存时不报告新恢复点。",
        ),
        _ => return None,
    })
}

fn impact_summary(report: &Report, diagnostic: &crate::domain::Diagnostic) -> String {
    if let Some(run_id) = diagnostic
        .details
        .as_ref()
        .and_then(|details| details.scope.as_ref())
        .and_then(|scope| scope.run_id.as_deref())
    {
        return format!("关联运行 {run_id}；请查看该运行状态，不能据此假定任务已停止。");
    }
    match report.operation.as_str() {
        "train" | "try" => {
            "本次训练操作没有成功结果；运行状态和已保存恢复点以运行记录为准。".into()
        }
        "preview" | "data_index" => {
            "本次数据准备或预览没有成功完成；现有数据文件不会由此诊断自动删除。".into()
        }
        "edit" | "layer_edit" => "本次修改未报告成功；请检查事务结果后再继续。".into(),
        _ => "本次操作未获得成功结果；未明确报告的状态保持未知。".into(),
    }
}

pub fn recovery_case_for_code(code: &str) -> Option<&'static str> {
    match code {
        "CLASS_DATASET_ROOT_INVALID" | "CLASS_DATASET_SCAN_FAILED" => Some("E1"),
        "DATA_SPLIT_EMPTY" | "DATA_SPLIT_MISSING" | "DATASET_SPLIT_MISSING" => Some("E2"),
        "CLASS_EXCLUSION_EMPTY" | "DATASET_SAMPLE_FAILED" | "PREVIEW_IMAGE_RENDER_FAILED" => {
            Some("E3")
        }
        "IMAGE_DECODER_UNAVAILABLE"
        | "ENVIRONMENT_RUNTIME_DEPENDENCY_MISSING"
        | "COMPONENT_DEPENDENCY_MISSING"
        | "COMPONENT_DEPENDENCY_RESOLUTION_FAILED" => Some("E4"),
        "CHECKPOINT_WRITE_FAILED" | "CHECKPOINT_STORE_FAILED" => Some("E5"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{recovery_case, recovery_case_for_code, render};
    use crate::domain::{
        Report,
        diagnostic::{
            DiagnosticDetails, DiagnosticEvidenceState, DiagnosticScope, DiagnosticSeverity,
            RecoveryAction, RecoveryAvailability,
        },
    };

    #[test]
    fn diagnostic_leads_with_reason_and_impact_then_limits_actions() {
        let mut report = Report::error("preview", "DATA_SPLIT_EMPTY", "验证切分为空", "补充样本");
        report.diagnostics[0].details = Some(DiagnosticDetails {
            severity: DiagnosticSeverity::Error,
            scope: Some(DiagnosticScope {
                operation_id: Some("preview".into()),
                model_ref: None,
                source_revision: None,
                batch_ref: Some("dataset:validation".into()),
                run_id: None,
                module_path: None,
            }),
            recovery_actions: (0..4)
                .map(|index| RecoveryAction {
                    id: format!("action-{index}"),
                    label: format!("动作 {index}"),
                    availability: RecoveryAvailability::Conditional,
                    reason: None,
                    requires_confirmation: false,
                })
                .collect(),
            evidence_state: DiagnosticEvidenceState::Partial,
        });
        report.data["log_tail"] = "line one\nline two".into();
        let concise = render(&report, false).join("\n");
        assert!(concise.contains("原因：验证切分为空"));
        assert!(concise.contains("本次数据准备或预览没有成功完成"));
        assert!(concise.contains("另有 1 个动作"));
        assert!(!concise.contains("诊断码："));
        let details = render(&report, true).join("\n");
        assert!(details.contains("诊断码：DATA_SPLIT_EMPTY"));
        assert!(details.contains("批次：dataset:validation"));
        assert!(details.contains("其他可选动作详情"));
        assert!(details.contains("line two"));
    }

    #[test]
    fn structured_action_reason_survives_diagnostic_projection() {
        let mut report = Report::error("preview", "DATA_SPLIT_EMPTY", "验证切分为空", "");
        report.diagnostics[0].details = Some(DiagnosticDetails {
            severity: DiagnosticSeverity::Error,
            scope: None,
            recovery_actions: vec![RecoveryAction {
                id: "use-existing-split".into(),
                label: "指定已有验证集".into(),
                availability: RecoveryAvailability::Conditional,
                reason: Some("目录中必须存在验证集文件夹".into()),
                requires_confirmation: true,
            }],
            evidence_state: DiagnosticEvidenceState::Partial,
        });
        let text = render(&report, false).join("\n");
        assert!(text.contains("满足条件后可用 · 需要确认"));
        assert!(text.contains("条件：目录中必须存在验证集文件夹"));
    }

    #[test]
    fn recovery_fixture_is_rendered_once_and_stays_unverified() {
        let report = Report::error(
            "data_index",
            "CLASS_DATASET_ROOT_INVALID",
            "找不到数据目录",
            "选择有效的数据目录。",
        );
        let text = render(&report, false).join("\n");
        assert_eq!(text.matches("建议下一步：").count(), 1);
        assert!(text.contains("在原位置选择其他目录 · 执行方式待确认 · 需要确认"));
        assert!(text.contains("安全边界：修正路径后从目录探测继续；不会重建项目。"));
        assert!(!text.contains("下一步：查看操作详情"));
    }

    #[test]
    fn recovery_fixtures_cover_e1_to_e5_without_claiming_unwired_actions_are_available() {
        for id in ["E1", "E2", "E3", "E4", "E5"] {
            let text = recovery_case(id).unwrap().join("\n");
            assert!(text.contains("安全边界"));
            assert!(text.contains("需在对应页面确认执行能力"));
        }
        assert!(recovery_case("E6").is_none());
        assert_eq!(
            recovery_case_for_code("CLASS_DATASET_ROOT_INVALID"),
            Some("E1")
        );
        assert_eq!(recovery_case_for_code("DATA_SPLIT_EMPTY"), Some("E2"));
        assert_eq!(recovery_case_for_code("CLASS_EXCLUSION_EMPTY"), Some("E3"));
        assert_eq!(
            recovery_case_for_code("IMAGE_DECODER_UNAVAILABLE"),
            Some("E4")
        );
        assert_eq!(
            recovery_case_for_code("CHECKPOINT_WRITE_FAILED"),
            Some("E5")
        );
    }

    #[test]
    fn missing_structured_diagnostics_are_not_fabricated() {
        let report = Report::ok("inspect", serde_json::json!({}));
        assert_eq!(
            render(&report, false),
            vec!["当前操作没有可展示的结构化诊断。"]
        );
    }

    #[test]
    fn provider_verification_drift_shows_exact_runtime_and_source_fields() {
        let mut report = Report::error(
            "edit",
            "COMPONENT_PROVIDER_UNVERIFIED",
            "Provider 没有匹配的验证记录。",
            "重新验证。",
        );
        report.data["provider_verification"] = serde_json::json!({
            "state":"stale",
            "mismatch_fields":["python_version","torch_version","source_fingerprints"],
            "expected":{
                "python_version":"3.14.10",
                "torch_version":"2.15.0",
                "source_fingerprints":{"provider":{"sha256":"current-provider"}}
            },
            "recorded":{
                "python_version":"3.13.2",
                "torch_version":"2.14.1",
                "source_fingerprints":{"provider":{"sha256":"tested-provider"}}
            }
        });
        let text = render(&report, true).join("\n");
        assert!(text.contains("Python 版本：已验证 3.13.2 · 当前 3.14.10"));
        assert!(text.contains("PyTorch 版本：已验证 2.14.1 · 当前 2.15.0"));
        assert!(
            text.contains("provider 源码 SHA-256：已验证 tested-provider · 当前 current-provider")
        );
    }
}

use crate::domain::Report;
use serde_json::Value;

/// Summarize the selected Python environment and the service's observed drift.
/// The view deliberately consumes a service report; it does not infer readiness
/// from the presence of a virtual-environment directory.
pub fn render(report: &Report) -> Vec<String> {
    let data = &report.data;
    let check = if data["environment_check"].is_object() {
        &data["environment_check"]
    } else {
        data
    };
    let mut lines = vec!["Python 环境".to_owned()];

    if let Some(python) = first_text(&[
        check["python"]["executable"].as_str(),
        check["python"].as_str(),
        data["python"].as_str(),
        data["selected_python"].as_str(),
    ]) {
        lines.push(format!("解释器：{python}"));
    }
    if let Some(version) = first_text(&[
        check["python"]["version"].as_str(),
        check["python_version"].as_str(),
    ]) {
        lines.push(format!("Python 版本：{version}"));
    }
    if let Some(abi) = first_text(&[
        check["python"]["abi"].as_str(),
        check["python_abi"].as_str(),
    ]) {
        lines.push(format!("Python ABI：{abi}"));
    }
    if let Some(venv) = data["venv"].as_str() {
        lines.push(format!("项目环境：{venv}"));
    }
    if check["system_site_packages"].is_boolean() || data["system_site_packages"].is_boolean() {
        let inherited =
            check["system_site_packages"] == true || data["system_site_packages"] == true;
        lines.push(if inherited {
            "基础 Python 包继承：已启用；锁会核对继承包，wheelhouse 不包含这些包。".into()
        } else {
            "基础 Python 包继承：未启用；项目 venv 保持隔离。".into()
        });
    }

    let observed_drift = check["python_matches"] == false
        || check["requirements_match"] == false
        || ["missing", "version_mismatch", "corrupt"]
            .iter()
            .any(|key| check[key].as_array().is_some_and(|items| !items.is_empty()));
    let matches = check["matches"].as_bool().or_else(|| {
        if observed_drift {
            Some(false)
        } else if report.operation == "environment_check" && report.success() {
            Some(true)
        } else {
            None
        }
    });
    match matches {
        Some(true) => lines.push("状态：与项目依赖锁匹配".into()),
        Some(false) => lines.push("状态：与项目依赖锁不匹配".into()),
        None if report.success() => lines.push("状态：操作已完成；尚无环境匹配结论".into()),
        None => lines.push("状态：不可用或尚未核验".into()),
    }

    if let Some(lock) = data["lock_file"].as_str() {
        lines.push(format!(
            "依赖锁：{lock} · {} 个包",
            data["package_count"].as_u64().unwrap_or(0)
        ));
    } else if data["environment_lock"].is_object() {
        lines.push(format!(
            "依赖锁：{} 个包",
            data["environment_lock"]["package_count"]
                .as_u64()
                .unwrap_or(0)
        ));
    }
    for (label, key) in [
        ("缺失依赖", "missing"),
        ("版本不符", "version_mismatch"),
        ("继承依赖缺失", "inherited_missing"),
        ("继承依赖版本不符", "inherited_version_mismatch"),
        ("额外依赖", "extra"),
        ("依赖损坏", "corrupt"),
    ] {
        if let Some(values) = check[key].as_array().filter(|items| !items.is_empty()) {
            lines.push(format!("{label}：{}", describe_items(values)));
        }
    }
    if data["python_matches"] == false {
        lines.push("原因：Python 版本、ABI 或平台与锁定目标不同。".into());
    }
    if data["requirements_match"] == false {
        lines.push("原因：requirements 输入与生成依赖锁时不同。".into());
    }
    if let Some(pip_check) = check["pip_check"]
        .as_str()
        .or_else(|| data["pip_check"].as_str())
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("依赖关系检查：{pip_check}"));
    }
    if let Some(values) = check["inherited_pip_check"]
        .as_array()
        .filter(|items| !items.is_empty())
    {
        lines.push(format!(
            "基础环境既有依赖警告（锁定基线）：{} 条",
            values.len()
        ));
        for warning in values.iter().take(3).filter_map(Value::as_str) {
            lines.push(format!("  基础环境：{warning}"));
        }
        if values.len() > 3 {
            lines.push(format!("  其余 {} 条见 JSON 详情。", values.len() - 3));
        }
    }
    if let Some(values) = check["new_pip_check"]
        .as_array()
        .filter(|items| !items.is_empty())
    {
        lines.push(format!("新增依赖冲突：{}", describe_items(values)));
    }

    if report.success() {
        if report.operation == "environment_setup" {
            lines.push(if data["created"] == true {
                "项目虚拟环境已创建并验证；后续命令可自动复用。".into()
            } else {
                "现有项目虚拟环境已复用并验证。".into()
            });
        }
        if report.operation == "environment_lock" {
            lines.push("依赖解析已完成；下一步可运行 m-model env setup。".into());
        }
    } else if report.operation == "environment_check" || report.operation == "environment_setup" {
        lines.push("建议：先查看差异；需要准备项目依赖时运行 m-model env setup。".into());
        lines.push("环境准备不会自动修改系统 Python。".into());
    }
    for diagnostic in &report.diagnostics {
        lines.push(format!("{}：{}", diagnostic.code, diagnostic.message));
        if let Some(hint) = diagnostic.hint.as_deref().filter(|hint| !hint.is_empty()) {
            lines.push(format!("下一步：{hint}"));
        }
    }
    if let Some(note) = data["note"].as_str().filter(|note| !note.is_empty()) {
        lines.push(note.into());
    }
    lines
}

fn first_text<'a>(values: &[Option<&'a str>]) -> Option<&'a str> {
    values
        .iter()
        .copied()
        .flatten()
        .find(|value| !value.is_empty())
}

fn describe_items(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| match value {
            Value::String(text) => text.clone(),
            Value::Object(_) => [
                value["name"].as_str(),
                value["expected"].as_str(),
                value["actual"].as_str(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("："),
            _ => value.to_string(),
        })
        .filter(|value| !value.is_empty())
        .take(8)
        .collect::<Vec<_>>()
        .join("、")
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::domain::Report;
    use serde_json::json;

    #[test]
    fn environment_check_explains_drift_and_safe_next_step() {
        let mut report = Report::ok(
            "environment_check",
            json!({
                "python_matches": false,
                "requirements_match": false,
                "missing": ["torch"],
                "version_mismatch": [{"name":"numpy", "expected":"2.0", "actual":"1.26"}],
                "venv": ".venv"
            }),
        );
        report.status = "error".into();
        let text = render(&report).join("\n");
        assert!(text.contains("与项目依赖锁不匹配"));
        assert!(text.contains("缺失依赖：torch"));
        assert!(text.contains("Python 版本、ABI 或平台"));
        assert!(text.contains("m-model env setup"));
        assert!(text.contains("不会自动修改系统 Python"));
    }

    #[test]
    fn successful_setup_uses_reported_environment_state() {
        let report = Report::ok(
            "environment_setup",
            json!({"created":true,"venv":".venv","python":".venv/bin/python","package_count":12,"system_site_packages":false,"environment_check":{"matches":true,"system_site_packages":false,"pip_check":"No broken requirements found."}}),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("项目环境：.venv"));
        assert!(text.contains("基础 Python 包继承：未启用"));
        assert!(text.contains("已创建并验证"));
    }

    #[test]
    fn setup_shows_inherited_runtime_and_base_warnings_separately_from_new_conflicts() {
        let report = Report::ok(
            "environment_setup",
            json!({
                "created":true,
                "venv":".venv",
                "system_site_packages":true,
                "environment_check":{
                    "matches":true,
                    "system_site_packages":true,
                    "pip_check":"No new dependency conflicts beyond the inherited base environment. Inherited base warnings: 1.",
                    "inherited_pip_check":["old-base-package 1.0 requires missing-lib, which is not installed."],
                    "new_pip_check":[]
                }
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("基础 Python 包继承：已启用"));
        assert!(text.contains("基础环境既有依赖警告（锁定基线）：1 条"));
        assert!(text.contains("old-base-package 1.0 requires missing-lib"));
        assert!(!text.contains("新增依赖冲突："));
    }
}

use serde_json::Value;

pub fn render(delivery: &Value) -> Vec<String> {
    match delivery["mode"].as_str().unwrap_or("unknown") {
        "local" => {
            let Some(path) = delivery["path"].as_str() else {
                return vec!["本地交付位置未知；请检查预览报告中的产物路径。".into()];
            };
            let mut lines = vec![format!("本机 HTML 预览：{path}")];
            match delivery["opened"].as_bool() {
                Some(true) => lines.push("预览已打开".into()),
                Some(false) => {
                    lines.push("尚未自动打开；可在当前执行环境中自行打开该文件。".into());
                }
                None => lines.push("打开状态未知；请按路径检查文件，不推断是否已打开。".into()),
            }
            lines
        }
        "ssh" => {
            let mut lines = Vec::new();
            if let Some(path) = delivery["remote_path"].as_str() {
                lines.push(format!("远端预览文件：{path}"));
            }
            if let Some(path) = delivery["local_destination"].as_str() {
                lines.push(format!("本机保存位置：{path}"));
            }
            if let Some(command) = delivery["scp_command"].as_str() {
                lines.push("检查后在目标本机执行：".into());
                lines.push(format!("  {command}"));
            } else {
                lines.push("缺少完整 SSH 取回命令；没有猜测主机、端口或路径。".into());
            }
            if delivery["transferred"] == true {
                lines.push("文件已传输".into());
            } else {
                lines.push("Model-Maven 不会自动连接或传输文件。".into());
            }
            lines
        }
        _ => vec!["交付方式未知；请按报告中的明确路径自行检查，不会自动打开或传输。".into()],
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn local_delivery_shows_the_actual_path_without_claiming_it_was_opened() {
        let lines = render(&json!({"mode":"local", "path":"/tmp/my preview.html", "opened":false}));
        let text = lines.join("\n");
        assert!(text.contains("/tmp/my preview.html"));
        assert!(text.contains("尚未自动打开"));
    }

    #[test]
    fn ssh_delivery_preserves_the_confirmed_command_and_destination() {
        let lines = render(&json!({
            "mode":"ssh",
            "remote_path":"/tmp/project/previews/preview.html",
            "local_destination":"./my preview.html",
            "scp_command":"scp -P 2244 user@host:'/tmp/project/previews/preview.html' './my preview.html'",
            "transferred":false
        }));
        let text = lines.join("\n");
        assert!(text.contains("2244"));
        assert!(text.contains("./my preview.html"));
        assert!(text.contains("不会自动连接或传输"));
    }

    #[test]
    fn ssh_delivery_with_missing_connection_fields_does_not_guess_them() {
        let text = render(&json!({
            "mode":"ssh",
            "remote_path":"/tmp/model preview.html",
            "local_destination":"./local preview.html",
            "ssh_host":null,
            "ssh_port":null
        }))
        .join("\n");
        assert!(text.contains("/tmp/model preview.html"));
        assert!(text.contains("./local preview.html"));
        assert!(text.contains("缺少完整 SSH 取回命令"));
        assert!(!text.contains("scp -P"));
        assert!(!text.contains("@"));
    }

    #[test]
    fn local_delivery_with_unknown_open_state_does_not_claim_not_opened() {
        let text = render(&json!({"mode":"local", "path":"./preview.html"})).join("\n");
        assert!(text.contains("打开状态未知"));
        assert!(!text.contains("尚未自动打开"));
    }
}

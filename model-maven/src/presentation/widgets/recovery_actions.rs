use serde_json::Value;

const MAX_VISIBLE_ACTIONS: usize = 3;

pub fn render(actions: &[Value]) -> Vec<String> {
    if actions.is_empty() {
        return vec!["下一步：查看操作详情并按上方提示修复后重试。".into()];
    }
    let mut lines = Vec::new();
    for (index, action) in actions.iter().take(MAX_VISIBLE_ACTIONS).enumerate() {
        let availability = match action["availability"].as_str().unwrap_or("unknown") {
            "available" => "可用",
            "conditional" => "满足条件后可用",
            "unsupported" => "当前不支持",
            _ => "执行方式待确认",
        };
        let confirmation = if action["requires_confirmation"] == true {
            " · 需要确认"
        } else {
            ""
        };
        lines.push(format!(
            "{}. {} · {}{}",
            index + 1,
            action["label"].as_str().unwrap_or("查看详情"),
            availability,
            confirmation
        ));
        if let Some(reason) = action["reason"]
            .as_str()
            .filter(|reason| !reason.is_empty())
        {
            lines.push(format!("   条件：{reason}"));
        }
    }
    if actions.len() > MAX_VISIBLE_ACTIONS {
        lines.push(format!(
            "另有 {} 个动作，可在诊断详情中查看。",
            actions.len() - MAX_VISIBLE_ACTIONS
        ));
    }
    lines
}

pub fn render_details(actions: &[Value], index_offset: usize) -> Vec<String> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let availability = match action["availability"].as_str().unwrap_or("unknown") {
                "available" => "可用",
                "conditional" => "满足条件后可用",
                "unsupported" => "当前不支持",
                _ => "执行方式待确认",
            };
            let confirmation = if action["requires_confirmation"] == true {
                " · 需要确认"
            } else {
                ""
            };
            let reason = action["reason"]
                .as_str()
                .map(|reason| format!(" · {reason}"))
                .unwrap_or_default();
            format!(
                "{}. {} · {}{}{}",
                index + index_offset + 1,
                action["label"].as_str().unwrap_or("查看详情"),
                availability,
                confirmation,
                reason
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{render, render_details};
    use serde_json::json;

    #[test]
    fn recovery_actions_show_three_recommendations_and_keep_availability_visible() {
        let actions = vec![
            json!({"label":"查看文件","availability":"available"}),
            json!({"label":"修复后重试","availability":"conditional","requires_confirmation":true}),
            json!({"label":"查看日志","availability":"unknown"}),
            json!({"label":"其他动作","availability":"unsupported"}),
        ];
        let text = render(&actions).join("\n");
        assert!(text.contains("1. 查看文件 · 可用"));
        assert!(text.contains("2. 修复后重试 · 满足条件后可用 · 需要确认"));
        assert!(text.contains("另有 1 个动作"));
        assert!(!text.contains("4. 其他动作"));
    }

    #[test]
    fn detailed_overflow_actions_continue_the_original_numbering() {
        let actions = vec![json!({"label":"第 4 个动作","availability":"unsupported"})];
        let text = render_details(&actions, 3).join("\n");
        assert!(text.starts_with("4. 第 4 个动作 · 当前不支持"));
    }
}

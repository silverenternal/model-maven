use crate::presentation::contracts::{ActionDescriptor, Availability, PageDescriptor, PageId};

pub fn render(
    project_path: &str,
    model: &str,
    batch: Option<&str>,
    needs_revalidation: bool,
) -> PageDescriptor {
    let source_status = if needs_revalidation {
        "\n项目配置或模型源码已变化，或无法确认源码版本；先显式重新检查入口，再恢复到工作台。"
    } else {
        ""
    };
    let mut actions = vec![ActionDescriptor {
        id: "restore".into(),
        label: "恢复选择".into(),
        availability: if needs_revalidation {
            Availability::Conditional("配置或源码需要重新检查；完成检查后才能恢复。".into())
        } else {
            Availability::Available
        },
        confirmation: None,
    }];
    if needs_revalidation {
        actions.push(ActionDescriptor::available(
            "revalidate",
            "重新检查模型入口",
        ));
    }
    actions.extend([
        ActionDescriptor::available("discard", "丢弃会话草稿"),
        ActionDescriptor::available("return_onboarding", "返回向导"),
    ]);
    PageDescriptor {
        id: PageId::new("session_restore"),
        title: "恢复上次接入".into(),
        summary: format!(
            "项目：{project_path}\n根模型：{model}\n代表批次：{}\n\n这只是上次确认的临时入口；恢复不会重新运行模型，不会保存工程配置，也不会重放训练。恢复后可检查、修改或显式保存入口。{source_status}",
            batch.unwrap_or("未设置")
        ),
        actions,
    }
}

#[cfg(test)]
mod tests {
    use super::render;

    #[test]
    fn recovery_page_never_claims_to_replay_a_model_or_training_run() {
        let page = render("/project", "models:build", None, false);
        assert!(page.summary.contains("不会重新运行模型"));
        assert!(page.summary.contains("不会保存工程配置"));
        assert!(page.actions.iter().any(|action| action.id == "restore"));
        assert!(page.actions.iter().any(|action| action.id == "discard"));
    }

    #[test]
    fn stale_sources_require_an_explicit_revalidation_action() {
        let page = render("/project", "models:build", None, true);
        assert!(page.summary.contains("项目配置或模型源码已变化"));
        assert!(page.actions.iter().any(|action| action.id == "revalidate"));
        assert!(matches!(
            page.actions
                .iter()
                .find(|action| action.id == "restore")
                .unwrap()
                .availability,
            crate::presentation::contracts::Availability::Conditional(_)
        ));
    }
}

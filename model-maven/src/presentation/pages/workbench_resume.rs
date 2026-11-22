use crate::presentation::contracts::{ActionDescriptor, PageDescriptor, PageId};

pub fn render(page_name: &str, config_changed: bool) -> PageDescriptor {
    let config_note = if config_changed {
        "\n项目配置已变化；进入页面后将采用当前配置。"
    } else {
        ""
    };
    PageDescriptor {
        id: PageId::new("workbench_resume"),
        title: "继续上次的工作台位置".into(),
        summary: format!(
            "上次离开时停在：{page_name}。继续只恢复页面位置，不会重放模型检查、前向试跑、编辑、训练或其它操作；页面将使用当前项目状态。{config_note}"
        ),
        actions: vec![
            ActionDescriptor::available("resume_page", "打开上次页面"),
            ActionDescriptor::available("discard", "丢弃页面恢复点并回到首页"),
        ],
    }
}

pub fn page_name(page_id: &str) -> &str {
    match page_id {
        "model" => "模型",
        "edit" => "编辑模型",
        "runs" => "试跑与训练",
        "components" => "组件",
        "environment" => "环境与能力",
        "outputs" => "数据预览、产物与清理",
        "project" => "项目配置与历史",
        _ => "最近访问的工作台页面",
    }
}

#[cfg(test)]
mod tests {
    use super::{page_name, render};

    #[test]
    fn resume_page_explains_it_will_not_replay_operations() {
        let page = render("组件", true);
        assert!(page.summary.contains("不会重放模型检查"));
        assert!(page.summary.contains("项目配置已变化"));
        assert!(page.actions.iter().any(|action| action.id == "resume_page"));
        assert_eq!(page_name("components"), "组件");
    }
}

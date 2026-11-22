use crate::presentation::contracts::{ActionDescriptor, PageDescriptor, PageId};

pub fn render(project_path: &str, has_project: bool, python: &str) -> PageDescriptor {
    let project_state = if has_project {
        "已识别 Model-Maven 项目；可打开现有配置继续工作。"
    } else {
        "当前目录还没有 Model-Maven 项目配置。"
    };
    let mut actions = vec![
        ActionDescriptor::available("connect", "接入已有 Python 模型"),
        ActionDescriptor::available("page:project_open", "打开一个项目目录"),
    ];
    actions.push(ActionDescriptor::available("preset", "从分类预设开始"));
    actions.push(ActionDescriptor::available(
        "environment",
        "检查 Python 环境",
    ));
    actions.push(ActionDescriptor::available(
        "continue_workbench",
        "进入模型工作台",
    ));
    actions.push(ActionDescriptor::available("quit", "退出"));
    PageDescriptor {
        id: PageId::new("onboarding"),
        title: "Model-Maven · 开始一个模型项目".into(),
        summary: format!(
            "{project_state}\n目录：{project_path}\n运行环境：{python}\n\n最快接入方式：提供根模型工厂；代表批次可稍后补充。你可以先临时查看，不会自动写入项目配置；只有明确选择保存后才会创建默认入口。\n也可以从 Tiny CNN 或 ResNet18 分类预设开始：选择本地类别目录后，工具会显示识别到的类别数，并在你确认后创建可编辑的 classification.py 和项目入口。Tiny CNN 不需要 torchvision；ResNet18 使用随机初始化并需要兼容的 torchvision，工具不会自动下载权重。已有文件不会被覆盖。"
        ),
        actions,
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::presentation::contracts::{Availability, PageId};

    #[test]
    fn onboarding_keeps_code_entry_and_verified_classification_preset_available() {
        let page = render("/tmp/project", false, "python3");
        assert_eq!(page.id, PageId::new("onboarding"));
        assert!(page.actions.iter().any(|action| action.id == "connect"));
        assert!(page.actions.iter().any(|action| action.id == "environment"));
        assert_eq!(
            page.actions
                .iter()
                .find(|action| action.id == "preset")
                .unwrap()
                .availability,
            Availability::Available
        );
        assert!(page.summary.contains("不会自动写入项目配置"));
        assert!(page.summary.contains("Tiny CNN 或 ResNet18 分类预设"));
        assert!(page.summary.contains("不会自动下载权重"));
    }
}

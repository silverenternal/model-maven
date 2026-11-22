use crate::presentation::contracts::{ActionDescriptor, Availability, PageDescriptor, PageId};

pub fn register_pages(
    project: &str,
    model: Option<&str>,
    model_source: Option<&str>,
    batch: Option<&str>,
    variant: Option<&str>,
    current_work: &str,
) -> Result<crate::presentation::contracts::PageRegistry, String> {
    let mut pages = crate::presentation::contracts::PageRegistry::default();
    pages.register(home_page(
        project,
        model,
        model_source,
        batch,
        variant,
        current_work,
    ))?;
    pages.register(category_page(
        "model",
        "模型",
        "浏览结构、做前向试跑，并选择当前会话使用的模型入口。",
        &[
            ("1", "检查模型网络"),
            ("2", "运行前向试跑"),
            ("3", "搜索并查看 Layer"),
            ("11", "更换本次会话的模型/批次"),
            ("24", "查看或切换模型变体"),
        ],
    ))?;
    pages.register(category_page(
        "edit",
        "编辑模型",
        "对 Layer、根工厂接缝或受支持的 Python 源码进行可验证修改。",
        &[
            ("4", "添加 Layer"),
            ("5", "修改变体 Layer 参数"),
            ("6", "替换 Layer"),
            ("7", "删除 Layer"),
            ("8", "撤销变体编辑"),
            ("9", "重做变体编辑"),
            ("23", "运行自定义 Layer 编辑 Provider"),
            ("25", "编辑 Python 源码中的 Layer"),
            ("26", "撤销源码编辑"),
            ("27", "重做源码编辑"),
            ("28", "成组修改源码 Layer 参数"),
            ("29", "成组增删替换源码结构"),
            ("33", "管理根工厂装配接缝"),
            ("36", "成组修复变体 Layer 参数"),
            ("38", "调整 Layer 共享关系"),
        ],
    ))?;
    pages.register(category_page(
        "runs",
        "试跑与训练",
        "验证损失和梯度，启动、恢复训练，浏览运行记录、进度与恢复线索，并对照试跑。",
        &[
            ("12", "验证损失与梯度"),
            ("13", "试跑训练步骤"),
            ("14", "开始正式训练"),
            ("15", "恢复训练"),
            ("16", "浏览运行列表与详情"),
            ("17", "查看最近运行详情"),
            ("19", "附加到后台训练"),
            ("20", "停止后台训练"),
            ("21", "查看运行占用"),
            ("35", "对照两次试跑"),
        ],
    ))?;
    pages.register(category_page(
        "components",
        "组件",
        "检查已登记组件，发现当前 Python 环境中已安装的扩展。",
        &[("22", "浏览组件目录"), ("30", "发现已安装组件")],
    ))?;
    pages.register(category_page(
        "environment",
        "环境与能力",
        "检查当前可用能力，准备或核对项目 Python 环境。",
        &[("10", "查看能力与环境诊断"), ("31", "管理项目 Python 环境")],
    ))?;
    let mut outputs = category_page(
        "outputs",
        "数据预览、产物与清理",
        "启动有来源的后台数据预览，导出训练工程或按计划清理运行文件。",
        &[("18", "导出最近训练"), ("32", "清理运行文件")],
    );
    let back = outputs
        .actions
        .pop()
        .expect("category page has a back action");
    outputs.actions.push(back);
    outputs
        .actions
        .push(ActionDescriptor::available("39", "配置并启动后台数据预览"));
    pages.register(outputs)?;
    pages.register(category_page(
        "project",
        "项目配置",
        "查看本项目模型入口的来源与历史，恢复字段，或打开/切换项目目录。",
        &[
            ("37", "查看当前配置与本次输入来源"),
            ("34", "查看历史并恢复字段"),
            ("40", "打开/切换项目目录"),
        ],
    ))?;
    Ok(pages)
}

pub fn home_page(
    project: &str,
    model: Option<&str>,
    model_source: Option<&str>,
    batch: Option<&str>,
    variant: Option<&str>,
    current_work: &str,
) -> PageDescriptor {
    let model_summary = model.unwrap_or("尚未选择");
    let source_summary = model_source.unwrap_or("无");
    let batch_summary = batch.unwrap_or("尚未设置（前向验证时再提供）");
    let variant_summary = variant.unwrap_or("默认模型");
    let next_step = if model.is_none() {
        "下一步：进入“模型”，选择模型入口并先检查结构。"
    } else if batch.is_none() {
        "下一步：先检查模型结构；需要前向验证时再补充代表批次。"
    } else {
        "下一步：前向试跑验证当前模型与代表批次。"
    };
    PageDescriptor {
        id: PageId::new("home"),
        title: "Model-Maven · 模型工作台".into(),
        summary: format!(
            "项目：{project}\n模型：{model_summary}（来源：{source_summary}）\n批次：{batch_summary}\n当前结构：{variant_summary}\n当前工作：{current_work}\n{next_step}\n输入序号、动作名称或关键词；输入 back 返回。"
        ),
        actions: vec![
            page_action("model", "模型与 Layer"),
            page_action("edit", "编辑模型"),
            page_action("runs", "试跑与训练"),
            page_action("components", "组件"),
            page_action("environment", "环境与能力"),
            page_action("outputs", "数据预览、产物与清理"),
            page_action("project", "项目配置与历史"),
            ActionDescriptor::available("quit", "退出"),
        ],
    }
}

fn category_page(id: &str, title: &str, summary: &str, actions: &[(&str, &str)]) -> PageDescriptor {
    let mut actions = actions
        .iter()
        .map(|(id, label)| ActionDescriptor::available(*id, *label))
        .collect::<Vec<_>>();
    actions.push(ActionDescriptor::available("back", "返回首页"));
    PageDescriptor {
        id: PageId::new(id),
        title: title.into(),
        summary: format!("{summary}\n输入序号、动作名称或关键词；输入 back 返回首页。"),
        actions,
    }
}

fn page_action(id: &str, label: &str) -> ActionDescriptor {
    ActionDescriptor {
        id: format!("page:{id}"),
        label: label.into(),
        availability: Availability::Available,
        confirmation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::register_pages;
    use crate::presentation::contracts::PageId;

    #[test]
    fn runs_category_offers_human_trial_comparison() {
        let pages = register_pages("project", None, None, None, None, "没有活动运行").unwrap();
        let runs = pages.get(&PageId::new("runs")).unwrap();
        assert!(
            runs.actions
                .iter()
                .any(|action| { action.id == "35" && action.label == "对照两次试跑" })
        );
        assert!(matches!(
            crate::presentation::pages::navigation::resolve_selection(runs, "对照两次试跑"),
            crate::presentation::pages::navigation::Selection::Navigate(
                crate::presentation::contracts::NavigationResult::RunAction(action)
            ) if action == "35"
        ));
    }

    #[test]
    fn data_preview_is_accessible_without_shifting_home_exit_shortcut() {
        let pages = register_pages("project", None, None, None, None, "没有活动运行").unwrap();
        let home = pages.get(&PageId::new("home")).unwrap();
        assert_eq!(home.actions.len(), 8);
        assert_eq!(home.actions[7].id, "quit");
        let outputs = pages.get(&PageId::new("outputs")).unwrap();
        assert!(outputs.actions.iter().any(|action| action.id == "39"));
        assert_eq!(outputs.actions[2].id, "back");
        assert_eq!(outputs.actions[3].id, "39");
    }
}

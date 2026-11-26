use clap::{Args, Parser, Subcommand, ValueEnum};

pub(crate) const CLASSIFICATION_CREATE_PRESETS: [&str; 2] = ["tiny-cnn", "resnet18"];
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "m-model",
    version,
    about = "检查、编辑、训练并打包原生 Python 模型工程"
)]
pub struct Cli {
    #[arg(long, global = true, help = "输出无需 Python 的 CLI 接口能力目录")]
    pub describe_interface: bool,
    #[arg(long, global = true, help = "机器输出；不等待交互输入")]
    pub json: bool,
    #[arg(
        long = "no-interactive",
        global = true,
        help = "即使连接终端也不提示；缺少必需输入时返回诊断"
    )]
    pub no_interactive: bool,
    #[arg(
        long,
        global = true,
        help = "Python 解释器；未指定时按操作使用项目选择、项目环境或 PATH 中发现的解释器"
    )]
    pub python: Option<String>,
    #[arg(skip)]
    pub python_explicit: bool,
    #[arg(
        long,
        global = true,
        value_parser = clap::value_parser!(u64).range(1..),
        help = "worker 超时秒数；正式训练到时最多再等 10 秒步骤边界 checkpoint，之后强制终止"
    )]
    pub timeout: Option<u64>,
    #[arg(long, global = true, help = "当前目录中的持久模型变体名称")]
    pub variant: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "ABSENT_OR_SHA256",
        help = "变体编辑提交或显式变体诊断：要求锁内读取的变体内容与指定 token 一致"
    )]
    pub if_variant_base: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "SHA256",
        help = "要求实际前向输入内容与指定 sha256-forward-input-v1 指纹匹配；不复用历史前向"
    )]
    pub if_input_content: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "SHA256",
        help = "要求候选记录的项目源码清单与预览摘要一致"
    )]
    pub if_source_manifest: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "SHA256",
        help = "要求源码编辑候选与先前预览计划一致"
    )]
    pub if_source_plan: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "SHA256",
        help = "要求源码编辑使用与预览一致的项目配置"
    )]
    pub if_source_config: Option<String>,
    #[arg(
        long,
        global = true,
        value_name = "FILE",
        conflicts_with = "replay_input_from",
        help = "诊断时独占创建私有输入快照文件；文件包含实际输入数据"
    )]
    pub capture_input_to: Option<std::path::PathBuf>,
    #[arg(
        long,
        global = true,
        value_name = "FILE",
        conflicts_with = "capture_input_to",
        help = "从输入快照重放；需要 --if-input-content 校验原输入内容"
    )]
    pub replay_input_from: Option<std::path::PathBuf>,
    #[arg(
        long,
        global = true,
        help = "随机种子；新变体条件提交需复用预览的 initialization_seed_decimal；已有变体的 Layer 编辑使用变体固定种子"
    )]
    pub seed: Option<u64>,
    #[arg(
        long,
        global = true,
        help = "显示 Python worker 的诊断日志，便于排查模型构造和运行错误"
    )]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    #[command(about = "试跑用户自有的纯 Python 循环，不自动导入或绑定机器学习框架")]
    LoopTrial {
        #[arg(long = "loop", help = "用户循环入口 module:callable，签名为 loop(None, context, **params)")]
        training_loop: String,
        #[arg(long = "loop-interface", default_value = "model-context", value_parser = ["context", "model-context"], help = "循环签名：context 为 loop(context, **params)，model-context 为 loop(None, context, **params)")]
        loop_interface: String,
        #[arg(long = "loop-param", value_name = "NAME=JSON", help = "循环参数，可重复；请勿放入密钥")]
        params: Vec<String>,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=1000), help = "最多接收的步骤报告数")]
        steps: u32,
        #[arg(long, help = "试跑记录名称")]
        name: Option<String>,
    },
    #[command(about = "运行用户自有的纯 Python 循环并保存用户报告的 JSON 状态")]
    LoopRun {
        #[arg(long = "loop", help = "新运行的 module:callable 入口；恢复时沿用原入口")]
        training_loop: Option<String>,
        #[arg(long = "loop-interface", value_parser = ["context", "model-context"], help = "循环签名；新运行默认 model-context，恢复时沿用原签名")]
        loop_interface: Option<String>,
        #[arg(long = "loop-param", value_name = "NAME=JSON", help = "新运行的循环参数，可重复；不要填写密钥")]
        params: Vec<String>,
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1_000_000), help = "总步骤目标，恢复时包含已有步数")]
        steps: u32,
        #[arg(long, help = "从已停止的纯 Python 循环运行恢复")]
        resume: Option<String>,
        #[arg(long = "if-checkpoint-sha256", help = "恢复时确认的原 checkpoint SHA-256")]
        expected_checkpoint: Option<String>,
        #[arg(long, help = "运行名称")]
        name: Option<String>,
        #[arg(long, help = "脱离终端后台运行（当前仅 Linux）；需要唯一请求标识")]
        detach: bool,
        #[arg(long, help = "16 到 64 位字母、数字或连字符，用于核对启动结果")]
        request_id: Option<String>,
    },
    #[command(about = "只读扫描项目 Python 源码中的常见入口候选；不导入或运行项目")]
    Discover {
        #[arg(long = "import-root", value_name = "PROJECT_RELATIVE_DIR", help = "优先扫描的项目模块搜索目录，可重复；不导入代码")]
        import_roots: Vec<String>,
    },
    #[command(about = "执行用户训练步骤或完整自有循环并进行有限试跑")]
    Try {
        #[command(flatten)]
        inputs: TrainingInputs,
        #[arg(long, num_args = 0..=1, default_missing_value = "latest", help = "复用最近或指定运行的输入快照；也可覆盖显式参数")]
        repeat: Option<String>,
        #[arg(long, help = "为本次试跑添加便于查找的名称")]
        name: Option<String>,
        #[arg(
            long = "training-mode",
            value_enum,
            help = "训练模式：managed-step、manual-step 或 external-loop"
        )]
        training_mode: Option<TrainingMode>,
        #[arg(
            long,
            conflicts_with = "training_loop",
            help = "training_step(model, batch, context, **params)"
        )]
        step: Option<String>,
        #[arg(
            long = "loop",
            conflicts_with = "step",
            help = "run_loop(model, context, **params)；用户循环控制数据与更新"
        )]
        training_loop: Option<String>,
        #[arg(
            long,
            value_parser = clap::value_parser!(u32).range(1..=1000),
            help = "本次有限试跑执行的最大步骤数"
        )]
        steps: Option<u32>,
        #[arg(long, help = "AdamW 学习率")]
        lr: Option<f64>,
        #[arg(long, help = "AdamW weight decay")]
        weight_decay: Option<f64>,
        #[arg(
            long,
            help = "明确启用绑定到人工语义标记的 require/penalty 规则；penalty 会改变训练目标"
        )]
        apply_supervision: bool,
        #[arg(long, conflicts_with = "apply_supervision", help = "本次试跑不启用监督，即使重复的历史试跑启用了监督")]
        no_supervision: bool,
        #[arg(
            long = "step-param",
            value_name = "NAME=JSON",
            help = "训练步骤关键字参数，可重复"
        )]
        params: Vec<String>,
        #[arg(
            long = "loop-param",
            value_name = "NAME=JSON",
            help = "external-loop 关键字参数，可重复"
        )]
        loop_params: Vec<String>,
    },
    #[command(about = "保存现有模型与示例批次入口，供后续命令和工作台复用")]
    Init {
        #[arg(long, help = "原生模型工厂入口，如 my_project.model:build_model")]
        model: Option<String>,
        #[arg(long, help = "可选的代表性批次工厂入口")]
        batch: Option<String>,
        #[arg(
            long = "inject",
            value_name = "SLOT=MODULE:FACTORY|component:ID@VERSION",
            help = "模型工厂组件接缝；可传 Python 工厂或已登记 Layer 组件"
        )]
        injections: Vec<String>,
        #[arg(
            long = "inject-param",
            value_name = "SLOT.NAME=JSON",
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::injection_param_completer),
            help = "已登记注入组件的参数，可重复，例如 encoder.depth=50"
        )]
        injection_params: Vec<String>,
        #[arg(
            long,
            value_enum,
            help = "批次绑定方式；默认按 Tensor 或 envelope 推断"
        )]
        batch_binding: Option<BatchBinding>,
        #[arg(long, help = "只预览将保存的入口，不写文件")]
        dry_run: bool,
    },
    #[command(about = "运行用户训练逻辑并周期性保存可恢复 checkpoint")]
    Train {
        #[command(flatten)]
        inputs: TrainingInputs,
        #[arg(long, num_args = 0..=1, default_missing_value = "latest", help = "从最近或指定的正式训练运行恢复")]
        resume: Option<String>,
        #[arg(
            long,
            requires = "resume",
            value_name = "SHA256",
            help = "仅恢复与此 SHA-256 一致的 checkpoint"
        )]
        if_checkpoint_sha256: Option<String>,
        #[arg(
            long,
            conflicts_with = "resume",
            help = "从指定正式训练运行的 latest checkpoint 显式迁移权重，并创建新的子运行"
        )]
        migrate_from: Option<String>,
        #[arg(
            long,
            requires = "migrate_from",
            conflicts_with = "resume",
            help = "权重迁移计划 JSON；每个状态需显式映射、重置或丢弃"
        )]
        migration_plan: Option<PathBuf>,
        #[arg(long, help = "为本次正式训练添加便于查找的名称")]
        name: Option<String>,
        #[arg(long, help = "本次训练启动的唯一请求标识；可用来查找回执丢失的运行")]
        request_id: Option<String>,
        #[arg(
            long = "training-mode",
            value_enum,
            help = "训练模式：managed-step、manual-step 或 external-loop"
        )]
        training_mode: Option<TrainingMode>,
        #[arg(
            long,
            conflicts_with = "training_loop",
            help = "training_step(model, batch, context, **params)"
        )]
        step: Option<String>,
        #[arg(
            long = "loop",
            conflicts_with = "step",
            help = "run_loop(model, context, **params)；用户循环控制数据与更新"
        )]
        training_loop: Option<String>,
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1_000_000), help = "新训练的总步骤数")]
        steps: Option<u32>,
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1_000_000), help = "恢复后额外执行的步骤数")]
        additional_steps: Option<u32>,
        #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u32).range(1..=1_000_000), help = "checkpoint 保存间隔")]
        checkpoint_every: u32,
        #[arg(long, help = "AdamW 学习率；恢复时继承原设置")]
        lr: Option<f64>,
        #[arg(long, help = "AdamW weight decay；恢复时继承原设置")]
        weight_decay: Option<f64>,
        #[arg(
            long,
            help = "明确启用绑定到人工语义标记的 require/penalty 规则；penalty 会改变训练目标"
        )]
        apply_supervision: bool,
        #[arg(
            long = "step-param",
            value_name = "NAME=JSON",
            help = "训练步骤关键字参数，可重复"
        )]
        params: Vec<String>,
        #[arg(
            long = "loop-param",
            value_name = "NAME=JSON",
            help = "external-loop 关键字参数，可重复"
        )]
        loop_params: Vec<String>,
        #[arg(
            long,
            help = "明确允许未声明的 batch/step/loop 自定义状态从初始状态重置"
        )]
        allow_state_reset: bool,
        #[arg(long, help = "脱离当前终端启动受监督训练；可用 runs attach/stop 管理")]
        detach: bool,
    },
    #[command(about = "使用项目评估入口评估已保存训练权重并记录具名指标")]
    Evaluate {
        #[arg(long, help = "要评估的正式训练运行 ID")]
        run: String,
        #[arg(
            long,
            default_value = "latest",
            help = "训练运行中可验证的 checkpoint；当前支持 latest"
        )]
        checkpoint: String,
        #[arg(
            long,
            value_name = "SHA256",
            help = "仅评估与此 SHA-256 一致的 checkpoint"
        )]
        if_checkpoint_sha256: Option<String>,
        #[arg(long, default_value = "test", help = "传给评估入口的数据切分名称")]
        split: String,
        #[arg(long, help = "项目评估入口 module:evaluate")]
        evaluator: String,
        #[arg(long, help = "评估模型设备；输入由评估函数自行迁移")]
        device: Option<String>,
        #[arg(long, help = "验证/测试数据工厂 module:build_data")]
        data: String,
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..), help = "传给评估入口的样本目标；入口负责限量并报告实际数量")]
        samples: Option<u64>,
        #[arg(
            long = "preview-samples",
            value_parser = clap::value_parser!(u64).range(1..=32),
            help = "请求评估器返回最多 N 条结构化预测预览；入口须调用 context.report_prediction"
        )]
        preview_samples: Option<u64>,
        #[arg(long, help = "可复用的数据版本/快照标识；不含凭据")]
        data_fingerprint: Option<String>,
        #[arg(
            long = "param",
            value_name = "NAME=JSON",
            help = "传给评估入口的参数，可重复"
        )]
        params: Vec<String>,
        #[arg(
            long = "data-param",
            value_name = "NAME=JSON",
            help = "传给数据工厂的参数，可重复"
        )]
        data_params: Vec<String>,
    },
    #[command(about = "查看模型结构或实际前向")]
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    #[command(about = "预览真实数据样本，检查数据、变换顺序并生成 PNG/HTML")]
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },
    #[command(about = "管理项目中按 train/eval 切分的有序数据增强步骤")]
    Augment {
        #[command(subcommand)]
        command: AugmentCommand,
    },
    #[command(
        about = "Layer 查询、增删改、Provider 扩展、原子计划和撤销；可选变体或受支持源码目标"
    )]
    Layer {
        #[command(subcommand)]
        command: LayerCommand,
    },
    #[command(about = "查看并编辑 Python 根工厂显式暴露的模型装配接缝")]
    Assembly {
        #[command(subcommand)]
        command: AssemblyCommand,
    },
    #[command(about = "预览或创建可编辑的分类模型项目")]
    Create {
        #[arg(long, required_unless_present = "recipe", conflicts_with = "recipe", value_parser = clap::builder::PossibleValuesParser::new(CLASSIFICATION_CREATE_PRESETS), help = "分类模型预设")]
        preset: Option<String>,
        #[arg(long, required_unless_present = "recipe", conflicts_with = "recipe", value_name = "DIRECTORY", help = "按类别分目录的图像数据根目录")]
        data: Option<PathBuf>,
        #[arg(long, value_name = "FILE", help = "从分类项目声明重建；数据路径相对于目标项目")]
        recipe: Option<PathBuf>,
        #[arg(long, help = "确认创建；需要 --if-plan 使用预览返回的修订")]
        apply: bool,
        #[arg(
            long,
            requires = "apply",
            value_name = "REVISION",
            help = "只提交与预览一致的计划修订"
        )]
        if_plan: Option<String>,
    },
    #[command(about = "审核并更新图像分类声明中的训练、监督和源码引用")]
    RecipeUpdate {
        #[arg(long, value_name = "JSON", help = "训练声明对象；使用 null 清除")]
        training: Option<String>,
        #[arg(long, value_name = "JSON", help = "监督文件引用对象；使用 null 清除")]
        supervision: Option<String>,
        #[arg(long, value_name = "JSON", help = "源码快照引用对象；使用 null 清除")]
        source: Option<String>,
        #[arg(long, value_name = "JSON", help = "项目内相对运行目录与模块搜索目录；使用 null 清除")]
        runtime: Option<String>,
        #[arg(long, value_name = "SHA256", help = "当前 model-maven.json 的摘要")]
        if_recipe: String,
        #[arg(long, help = "提交已审核的计划")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "REVISION")]
        if_plan: Option<String>,
    },
    #[command(about = "独立审核并同步通用模型声明中的项目输入或训练设置")]
    RecipeSync {
        #[arg(long, required_unless_present = "training_profile", help = "同步当前项目已保存的输入")]
        data: bool,
        #[arg(long, value_name = "JSON", help = "训练声明对象；使用 null 清除")]
        training_profile: Option<String>,
        #[arg(long, value_name = "SHA256", help = "当前模型声明的文件摘要")]
        if_recipe: String,
        #[arg(long, value_name = "SHA256", help = "当前项目配置修订")]
        if_config_revision: String,
        #[arg(long, help = "提交已审核的声明变更")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "REVISION")]
        if_plan: Option<String>,
    },
    #[command(about = "按稳定层 ID 预览草稿内同组移动；不会修改声明或运行模型")]
    RecipeLayerPlan {
        #[arg(long, value_name = "FILE", help = "要规划的草稿声明 JSON")]
        recipe: PathBuf,
        #[arg(long, value_name = "SHA256", help = "输入文件的 SHA-256")]
        if_recipe: String,
        #[arg(long, value_name = "JSON", help = "移动数组，如 [{\"id\":\"n_...\",\"before\":\"n_...\"}]；before 为 null 表示末尾")]
        moves: String,
    },
    #[command(about = "按稳定 ID 批量预览声明层的增删改查；不写项目文件")]
    RecipeLayerEditPlan {
        #[arg(long, value_name = "FILE", help = "草稿声明 JSON 文件")]
        recipe: PathBuf,
        #[arg(long, value_name = "SHA256", help = "草稿文件摘要")]
        if_recipe: String,
        #[arg(long, value_name = "FILE", help = "操作数组 JSON 文件；避免将参数暴露在命令行中")]
        operations: PathBuf,
        #[arg(long, value_name = "SHA256", help = "操作文件摘要")]
        if_operations: String,
    },
    #[command(about = "只读规划显式节点、端口和共享组件图；不运行模型")]
    GraphPlan {
        #[arg(long, value_name = "FILE", help = "图声明 JSON 文件")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256", help = "图文件内容摘要")]
        if_graph: String,
    },
    #[command(about = "按稳定 ID 只读预览一组图编辑；不写入项目或运行模型")]
    GraphEditPlan {
        #[arg(long, value_name = "FILE")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_graph: String,
        #[arg(long, value_name = "FILE")]
        operations: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_operations: String,
    },
    #[command(about = "只读预览将顺序声明包装为显式图；保留原声明和源码")]
    GraphImportScaffold {
        #[arg(long, value_name = "FILE", help = "现有 model-maven.json")]
        recipe: PathBuf,
        #[arg(long, value_name = "SHA256", help = "声明文件内容摘要")]
        if_recipe: String,
        #[arg(long, value_name = "MODULE:CALLABLE", help = "现有模型工厂；不导入或运行")]
        model: String,
        #[arg(long, value_name = "SHA256", help = "可选：核对项目配置版本并在声明未提供数据时保留当前批次入口")]
        if_project: Option<String>,
        #[arg(long, help = "从受支持的声明层栈构造可编辑图草稿；不推断用户修改的源码或权重")]
        editable: bool,
    },
    #[command(about = "只读预览无分支组件链的顺序调整；不写入或运行模型")]
    GraphReorderPlan {
        #[arg(long, value_name = "FILE")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_graph: String,
        #[arg(long, value_name = "NODE_ID")]
        node: String,
        #[arg(long, value_name = "NODE_ID", conflicts_with = "after", required_unless_present = "after")]
        before: Option<String>,
        #[arg(long, value_name = "NODE_ID", conflicts_with = "before", required_unless_present = "before")]
        after: Option<String>,
    },
    #[command(about = "预览图 Python 源码，或按计划创建/条件更新文件；不运行模型")]
    GraphGenerate {
        #[arg(long, value_name = "FILE")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_graph: String,
        #[arg(long, value_name = "PYTHON_FILE", help = "项目根目录下的新 Python 模块文件")]
        output: Option<String>,
        #[arg(long, requires = "output", value_name = "SHA256", help = "显式审核并条件更新已有源码的摘要")]
        if_source: Option<String>,
        #[arg(long, requires_all = ["output", "if_plan"], help = "按已审核计划创建新文件")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "SHA256")]
        if_plan: Option<String>,
    },
    #[command(about = "在独立 worker 中用已捕获输入预览未保存的图草稿")]
    GraphDraftProbe {
        #[arg(long, value_name = "FILE")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_graph: String,
        #[arg(long, value_name = "FILE")]
        snapshot: PathBuf,
        #[arg(long, value_name = "SHA256")]
        input_sha256: String,
        #[arg(long, value_name = "MODULE:CALLABLE")]
        batch: Option<String>,
        #[arg(long, value_name = "DEVICE", help = "预览模型设备；不会自动迁移捕获的输入")]
        model_device: Option<String>,
        #[arg(long, value_name = "SHA256")]
        if_config: String,
    },
    #[command(about = "审核并有条件地将已保存图源码和声明数据一起绑定到现有项目")]
    GraphBind {
        #[arg(long, value_name = "FILE")]
        graph: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_graph: String,
        #[arg(long, value_name = "PYTHON_FILE")]
        source: String,
        #[arg(long, value_name = "ABSENT_OR_SHA256")]
        if_config: String,
        #[arg(long, help = "同时应用图中声明的数据绑定")]
        include_data: bool,
        #[arg(long, requires = "if_plan")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "SHA256")]
        if_plan: Option<String>,
    },
    #[command(about = "有条件保存显式图草稿；保留不完整图，绝不覆盖未知内容")]
    GraphSave {
        #[arg(long, value_name = "FILE")]
        draft: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_draft: String,
        #[arg(long, value_name = "JSON_FILE")]
        output: String,
        #[arg(long, value_name = "ABSENT_OR_SHA256")]
        if_base: String,
        #[arg(long, requires = "if_plan")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "SHA256")]
        if_plan: Option<String>,
    },
    #[command(about = "统一审核图草稿、拟生成源码和当前项目绑定；只读，不提交")]
    GraphDeliveryPlan {
        #[arg(long, value_name = "FILE")]
        draft: PathBuf,
        #[arg(long, value_name = "SHA256")]
        if_draft: String,
        #[arg(long, value_name = "JSON_FILE")]
        output: String,
        #[arg(long, value_name = "ABSENT_OR_SHA256")]
        if_base: String,
        #[arg(long, value_name = "PYTHON_FILE")]
        source: String,
    },
    #[command(about = "根据联合审核执行可恢复的图、源码与项目分阶段提交")]
    GraphDeliveryStart {
        #[arg(long)] draft: PathBuf,
        #[arg(long)] if_draft: String,
        #[arg(long)] output: String,
        #[arg(long)] if_base: String,
        #[arg(long)] source: String,
        #[arg(long)] if_plan: String,
    },
    #[command(about = "查看未完成图交付的各阶段状态，不执行提交")]
    GraphDeliveryStatus,
    #[command(about = "归档待恢复交付记录，不回滚已写入的文件")]
    GraphDeliveryArchive {
        #[arg(long, value_name = "SHA256")]
        if_plan: String,
    },
    #[command(about = "继续当前项目中断的图交付，遇到外部修改时停止")]
    GraphDeliveryResume {
        #[arg(long, value_name = "SHA256", help = "要求待恢复记录与已审核计划一致")]
        if_plan: Option<String>,
    },
    #[command(about = "预览或导出项目的可迁移文件包")]
    RecipeBundle {
        #[arg(long, conflicts_with = "scaffold", help = "打包 model-graph.json 显式图；源码、数据和环境文件需显式选择")]
        graph: bool,
        #[arg(long, help = "打包 model-maven.json 通用骨架声明及其监督引用")]
        scaffold: bool,
        #[arg(long, value_name = "DIRECTORY", help = "项目根目录下的新目录名")]
        output: String,
        #[arg(long, value_name = "PATH", help = "额外包含的项目内文件或目录，可重复；数据和环境文件需要显式选择")]
        include: Vec<String>,
        #[arg(long, help = "创建已审核的项目包")]
        apply: bool,
        #[arg(long, requires = "apply", value_name = "REVISION")]
        if_plan: Option<String>,
    },
    #[command(about = "从具名输入创建可编辑的自定义多输入模型骨架")]
    Scaffold {
        #[arg(long, value_name = "FILE", conflicts_with_all = ["module", "batch", "batch_binding", "batch_params", "injections", "injection_params", "component_params_slots", "inputs", "routes", "native_encoders", "encoder_layers", "encoder_widths", "training_profile", "supervision_reference", "output_dim", "outputs", "loss"], help = "读取声明式骨架配置，预览后可按修订创建")]
        recipe: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["module", "output_dim", "loss"], help = "创建时同时写入可复用的 model-maven.json")]
        write_recipe: bool,
        #[arg(long, value_name = "SHA256", requires = "write_recipe", help = "按当前 model-maven.json 的文件摘要生成并切换到新构建")]
        update_recipe: Option<String>,
        #[arg(long, requires = "update_recipe", help = "将组件绑定、输入接线和编码器输出方式同步至画布声明，需审核生成计划")]
        reconcile_assembly: bool,
        #[arg(long, requires = "write_recipe", value_name = "JSON", help = "声明训练模式、入口和参数；不会自动执行")]
        training_profile: Option<String>,
        #[arg(long = "supervision-ref", requires = "write_recipe", value_name = "JSON", help = "引用项目内监督声明及摘要；重建后需显式导入并绑定")]
        supervision_reference: Option<String>,
        #[arg(long = "component-params-slot", requires = "write_recipe", help = "明确声明组件参数对象，包括空对象；可重复")]
        component_params_slots: Vec<String>,
        #[arg(
            long,
            value_name = "MODULE",
            help = "生成 Python 模块路径，例如 models.custom；默认 model_starter"
        )]
        module: Option<String>,
        #[arg(
            long,
            value_name = "MODULE:CALLABLE",
            help = "可选已有批次工厂；省略则使用生成的合成批次"
        )]
        batch: Option<String>,
        #[arg(
            long,
            requires = "batch",
            value_enum,
            help = "已有批次的模型输入绑定方式；省略时自动推断"
        )]
        batch_binding: Option<BatchBinding>,
        #[arg(
            long = "batch-param",
            requires = "batch",
            value_name = "NAME=JSON",
            help = "已有批次工厂参数，可重复；保存在本地项目配置，请勿填写凭据"
        )]
        batch_params: Vec<String>,
        #[arg(
            long = "inject",
            value_name = "SLOT=MODULE:FACTORY|component:ID@VERSION",
            help = "预先绑定生成骨架的编码器、融合或预测头接缝，可重复"
        )]
        injections: Vec<String>,
        #[arg(
            long = "inject-param",
            requires = "injections",
            value_name = "SLOT.NAME=JSON",
            help = "已登记组件参数，可重复；存入本地配置，请勿填写凭据"
        )]
        injection_params: Vec<String>,
        #[arg(
            long = "input",
            required_unless_present = "recipe",
            value_name = "NAME[:DTYPE]=DIM,DIM",
            help = "输入名、可选 dtype 和不含批次轴的形状；?128 表示可变轴及代表长度，可重复；省略 dtype 使用批次执行时 PyTorch 默认浮点类型"
        )]
        inputs: Vec<String>,
        #[arg(
            long = "route",
            value_name = "ENCODER=INPUT",
            help = "将编码器接缝接到具名输入，可重复；未指定时接同名输入"
        )]
        routes: Vec<String>,
        #[arg(
            long = "native-encoder",
            value_name = "SLOT",
            help = "保留指定编码器的原始输出，不自动平均或展平；可重复"
        )]
        native_encoders: Vec<String>,
        #[arg(long = "encoder-layers", value_name = "SLOT=JSON", requires = "write_recipe", help = "声明编码器的有序层配置，可重复")]
        encoder_layers: Vec<String>,
        #[arg(long = "encoder-width", value_name = "SLOT=WIDTH", requires = "write_recipe", help = "声明组件编码器的输出特征数，可重复")]
        encoder_widths: Vec<String>,
        #[arg(long, value_name = "FEATURES", help = "输出特征数；默认使用融合特征数")]
        output_dim: Option<usize>,
        #[arg(
            long = "output",
            value_name = "NAME=FEATURES[:mse|ce|custom:MODULE:CALLABLE]",
            conflicts_with_all = ["output_dim", "loss"],
            help = "具名输出头，可重复；custom:module:callable 在训练时调用函数(prediction, target)，须返回标量张量"
        )]
        outputs: Vec<String>,
        #[arg(long, value_parser = ["mse", "cross_entropy"], help = "单输出的示例损失；cross_entropy 使用整数类别目标")]
        loss: Option<String>,
        #[arg(long, help = "确认创建；需同时提供预览修订")]
        apply: bool,
        #[arg(long, requires = "apply", help = "只提交与预览一致的骨架修订")]
        if_plan: Option<String>,
    },
    #[command(about = "浏览项目配置历史，并按字段恢复之前的模型设置")]
    Config {
        #[command(subcommand)]
        command: ProjectConfigCommand,
    },
    #[command(about = "查看本项目的训练与试跑记录")]
    Runs {
        #[command(subcommand)]
        command: RunsCommand,
    },
    #[command(about = "登记、发现、检查、卸载和测试项目组件")]
    Component {
        #[command(subcommand)]
        command: ComponentCommand,
    },
    #[command(about = "创建并核对项目 Python 环境与可哈希依赖锁")]
    Env {
        #[command(subcommand)]
        command: EnvironmentCommand,
    },
    #[command(about = "列出、查看并复制持久模型变体")]
    Variant {
        #[command(subcommand)]
        command: VariantCommand,
    },
    #[command(about = "工作区内部只读操作")]
    Workspace {
        #[command(subcommand)]
        command: WorkspaceCommand,
    },
    #[command(about = "将正式训练的代码快照与 checkpoint 导出为可验证工程包")]
    Export {
        /// Export destination directory.
        directory: PathBuf,
        #[arg(long, help = "正式训练运行 ID；默认选择最近含 checkpoint 的训练")]
        run: Option<String>,
        #[arg(
            long,
            requires = "run",
            value_name = "SHA256",
            help = "仅导出与此 SHA-256 一致的 checkpoint"
        )]
        if_checkpoint_sha256: Option<String>,
        #[arg(long, help = "目标已存在时先备份，再原子替换")]
        overwrite: bool,
    },
    #[command(about = "检查 Python/PyTorch 和当前实际能力")]
    Doctor,
    #[command(about = "输出当前 shell 的动态补全加载命令")]
    Completions {
        #[arg(value_parser = ["bash", "elvish", "fish", "powershell", "zsh"], help = "目标 shell")]
        shell: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum WorkspaceCommand {
    #[command(hide = true, about = "为目标操作解析可复核授权身份；不会启动 Python")]
    AuthorizationIdentity {
        #[arg(long = "request-json", value_name = "JSON", help = "规范化目标 CLI 参数的 JSON 字符串数组")]
        request_json: String,
    },
    #[command(hide = true, about = "使用后端身份条件执行目标工作区操作")]
    Run {
        #[arg(long = "request-json", value_name = "JSON", help = "规范化目标 CLI 参数的 JSON 字符串数组")]
        request_json: String,
        #[arg(long = "if-authorization-identity", value_name = "SHA256", help = "要求解析后的操作身份与工作区预检完全相同")]
        if_authorization_identity: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum EnvironmentCommand {
    #[command(about = "为后续项目环境明确设置是否继承基础 Python 的已安装包")]
    Configure {
        #[arg(
            long,
            conflicts_with = "isolate_system_site_packages",
            required_unless_present = "isolate_system_site_packages",
            help = "让新 venv 复用当前基础解释器包，例如已安装的 CUDA/ROCm PyTorch"
        )]
        system_site_packages: bool,
        #[arg(
            long,
            conflicts_with = "system_site_packages",
            required_unless_present = "system_site_packages",
            help = "新 venv 仅安装项目锁定的包（默认隔离策略）"
        )]
        isolate_system_site_packages: bool,
        #[arg(
            long,
            help = "设置策略作用的项目虚拟环境路径；旧环境不会被改写，目标必须尚不存在"
        )]
        venv: Option<PathBuf>,
    },
    #[command(about = "解析 requirements.txt 并生成带 SHA-256 的 requirements.lock.txt")]
    Lock {
        #[arg(long, default_value = "requirements.txt", help = "项目依赖输入文件")]
        requirements: PathBuf,
        #[arg(long, help = "重新解析依赖并更新锁，即使现有锁仍有效")]
        refresh: bool,
    },
    #[command(about = "在项目虚拟环境安装锁定依赖并运行 pip check")]
    Setup {
        #[arg(long, default_value = "requirements.txt", help = "项目依赖输入文件")]
        requirements: PathBuf,
        #[arg(long, help = "项目虚拟环境目录；默认复用上次 setup 选择的环境或 .venv")]
        venv: Option<PathBuf>,
        #[arg(long, help = "重新解析依赖并更新锁；已有不兼容环境仍不会被覆盖")]
        update_lock: bool,
    },
    #[command(about = "核对当前解释器或项目虚拟环境是否符合依赖锁")]
    Check {
        #[arg(long, help = "要核对的项目虚拟环境目录；默认使用上次 setup 选择的环境")]
        venv: Option<PathBuf>,
    },
    #[command(about = "只读查看所选 Python 环境中的发行包元数据")]
    Libraries {
        #[arg(long, help = "要检查的项目虚拟环境目录；默认使用上次 setup 选择的环境")]
        venv: Option<PathBuf>,
        #[arg(
            long = "library",
            value_name = "DISTRIBUTION",
            help = "只检查指定发行包；可重复指定，默认列出有界环境清单"
        )]
        libraries: Vec<String>,
    },
    #[command(about = "显式导入 PyTorch 并查询该解释器可见的训练设备；不安装依赖")]
    Devices {
        #[arg(long, help = "要查询的项目虚拟环境目录；默认使用当前所选环境")]
        venv: Option<PathBuf>,
    },
    #[command(
        about = "显式导入用户点名的 Python 模块；可选 --check-runtime 运行支持的有限 CPU smoke；不会安装依赖或绑定训练后端"
    )]
    Probe {
        #[arg(long, help = "要使用的项目虚拟环境目录；默认使用上次 setup 选择的环境")]
        venv: Option<PathBuf>,
        #[arg(
            long = "library",
            value_name = "ID",
            required = true,
            num_args = 1..,
            help = "要显式导入的模块名或已知库 ID；可重复指定，发行包名不一定是模块名"
        )]
        libraries: Vec<String>,
        #[arg(
            long,
            help = "额外运行支持的库计算；JAX 使用环境默认后端并可能初始化加速器，不支持的库标记 unsupported，不自动绑定训练后端"
        )]
        check_runtime: bool,
        #[arg(long, help = "从显式导入的 torch.nn 发现公开 Module 类及其可观察签名；不会构造 Layer")]
        layer_catalog: bool,
        #[arg(long, help = "列出显式指定模块中的公开类和函数及可观察签名；不构造对象，不代表模型兼容")]
        operator_catalog: bool,
    },
    #[command(about = "归档并离线重建与当前 Python ABI 匹配的锁定依赖")]
    Wheelhouse {
        #[command(subcommand)]
        command: WheelhouseCommand,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum WheelhouseCommand {
    #[command(about = "从当前配置的 pip 源归档锁定 wheel")]
    Archive {
        #[arg(
            long,
            value_name = "DIR",
            help = "项目内归档目录；默认按 lock 指纹创建"
        )]
        path: Option<PathBuf>,
    },
    #[command(about = "只读检查 wheel 索引、ABI、缺失文件和哈希")]
    Check {
        #[arg(
            long,
            value_name = "DIR",
            help = "项目内归档目录；默认使用 lock 指纹目录"
        )]
        path: Option<PathBuf>,
    },
    #[command(about = "从完整归档离线创建新虚拟环境，不修改已有环境")]
    Replay {
        #[arg(
            long,
            value_name = "DIR",
            help = "尚不存在的项目内目标虚拟环境"
        )]
        venv: PathBuf,
        #[arg(long, required = true, help = "仅创建全新目标环境并离线安装；目标必须不存在")]
        create: bool,
        #[arg(
            long,
            value_name = "DIR",
            help = "项目内归档目录；默认使用 lock 指纹目录"
        )]
        path: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ComponentCommand {
    #[command(about = "生成可独立构建的 Layer 或 Layer Edit Provider 扩展包骨架")]
    Scaffold {
        #[arg(help = "组件 ID，格式 namespace/name")]
        id: String,
        #[arg(long, value_parser = ["layer", "layer-edit-provider"], help = "扩展组件类型")]
        kind: String,
        #[arg(long, default_value = "0.1.0", help = "初始组件版本 MAJOR.MINOR.PATCH")]
        version: String,
        #[arg(long, value_name = "DIR", help = "项目内尚不存在的输出目录")]
        output: PathBuf,
    },
    #[command(about = "构建本地扩展 wheel，在项目隔离存储中安装并自动发现登记")]
    Install {
        #[arg(value_name = "DIR", help = "项目内包含 pyproject.toml 的扩展包目录")]
        path: PathBuf,
        #[arg(
            long,
            help = "使用当前 Python 环境中的构建依赖；缺少依赖时立即报错，不创建隔离构建环境"
        )]
        no_build_isolation: bool,
    },
    #[command(about = "预览或移除已注册组件版本及其不再使用的项目扩展包")]
    Uninstall {
        #[arg(
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::exact_component_completer),
            help = "精确组件版本，格式 namespace/name@MAJOR.MINOR.PATCH"
        )]
        id: String,
        #[arg(long, help = "确认应用预览中的移除计划")]
        apply: bool,
        #[arg(
            long,
            requires = "apply",
            value_name = "SHA256",
            help = "要求移除影响与先前预览的计划一致"
        )]
        if_plan: Option<String>,
        #[arg(
            long,
            help = "允许移除仍被项目配置/变体/源码引用或源码扫描不完整的组件"
        )]
        force: bool,
    },
    #[command(about = "从当前 Python 环境发现已安装扩展包中的静态组件描述")]
    Discover {
        #[arg(long, help = "只扫描指定 distribution 名称")]
        distribution: Option<String>,
        #[arg(
            long = "path",
            value_name = "SITE_PACKAGES",
            help = "额外扫描的 Python 包目录；可重复"
        )]
        paths: Vec<PathBuf>,
    },
    #[command(about = "列出或搜索本项目注册的静态组件描述")]
    List {
        #[arg(long, help = "按组件 ID 或说明筛选")]
        query: Option<String>,
    },
    #[command(about = "显示组件入口、版本、参数 Schema 和能力")]
    Show {
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        id: String,
    },
    #[command(about = "检查组件描述与本地必需依赖版本，不导入或执行用户 Python 代码")]
    Check {
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        id: String,
    },
    #[command(about = "按组件声明验证 Layer 工厂、Layer Edit Provider 或 transform")]
    Test {
        #[arg(
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer),
            help = "组件 ID，可指定精确版本 namespace/name@x.y.z"
        )]
        id: String,
        #[arg(
            long = "param",
            value_name = "NAME=JSON",
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_param_completer),
            help = "工厂参数或 Provider payload；Schema 默认值自动填充"
        )]
        params: Vec<String>,
        #[arg(long, help = "Layer 的可选前向批次工厂；声明 forward_probe 时使用")]
        batch: Option<String>,
        #[arg(
            long = "batch-param",
            value_name = "NAME=JSON",
            help = "批次工厂参数，可重复"
        )]
        batch_params: Vec<String>,
        #[arg(
            long,
            value_enum,
            help = "批次绑定方式；默认按 Tensor 或 envelope 推断"
        )]
        batch_binding: Option<BatchBinding>,
    },
    #[command(about = "注册一个现有 Python 工厂，不安装或复制 Python 包")]
    Add {
        id: String,
        #[arg(long, value_parser = ["model", "layer", "dataset", "transform", "loss", "metric", "trainer", "layer-edit-provider"], help = "组件类型；必须与工厂的接口和声明能力一致")]
        kind: String,
        #[arg(long, help = "工厂入口，如 my_project.layers:make_activation")]
        entrypoint: String,
        #[arg(long, default_value = "0.1.0", help = "组件版本 MAJOR.MINOR.PATCH")]
        version: String,
        #[arg(long, default_value = "", help = "组件用途说明")]
        description: String,
        #[arg(long, help = "参数 JSON Schema 子集；默认不接受参数")]
        params_schema: Option<String>,
        #[arg(long, help = "能力 JSON 对象；只记录，不代表已通过运行时符合性测试")]
        capabilities: Option<String>,
        #[arg(
            long,
            help = "符合性 fixture module:callable；Layer、Provider、Transform、值组件和 Trainer 测试使用"
        )]
        conformance_fixture: Option<String>,
        #[arg(
            long,
            help = "组件依赖 JSON 数组，版本使用 SemVer range；例如 [{\"component_id\":\"demo/kernel\",\"version\":\"^1.2\"}]"
        )]
        dependencies: Option<String>,
        #[arg(
            long = "dependency",
            value_name = "ID@RANGE",
            help = "必需组件依赖，可重复；例如 demo/kernel@^1.2.0"
        )]
        required_dependencies: Vec<String>,
        #[arg(
            long = "optional-dependency",
            value_name = "ID@RANGE",
            help = "可选组件依赖，可重复；例如 demo/metrics@~2.0.0"
        )]
        optional_dependencies: Vec<String>,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum VariantCommand {
    #[command(about = "列出本项目的模型变体与编辑进度")]
    List {
        #[arg(long, help = "按名称或模型入口筛选")]
        query: Option<String>,
    },
    #[command(about = "查看变体来源、操作历史和完整结构配方")]
    Show { name: String },
    #[command(about = "复制变体作为独立编辑分支；目标已存在时拒绝覆盖")]
    Copy { source: String, destination: String },
}

#[derive(Debug, Clone, Subcommand)]
pub enum RunsCommand {
    #[command(about = "按训练请求标识查找运行，不读取 checkpoint 或进度")]
    FindRequest {
        #[arg(help = "启动时指定的请求标识")]
        request_id: String,
    },
    #[command(about = "按最近时间列出训练与试跑记录")]
    List {
        #[arg(long, help = "按名称或运行 ID 子串筛选")]
        query: Option<String>,
        #[arg(long, value_parser = ["active", "failed", "interrupted", "completed", "unknown"], help = "按状态筛选；active 包含 running/stopping")]
        status: Option<String>,
        #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true, help = "只看保留项；--kept=false 排除保留项")]
        kept: Option<bool>,
    },
    #[command(about = "查看记录输入快照与完整结果")]
    Show { id: Option<String> },
    #[command(about = "审核并显式解除未知运行对新任务的阻挡；不会判定进程已停止")]
    AcknowledgeUnknown {
        id: String,
        #[arg(long, help = "使用预览返回的运行记录修订")]
        if_revision: Option<String>,
        #[arg(long, requires = "if_revision", help = "确认已检查进程并接受运行状态仍未知")]
        acknowledge: bool,
        #[arg(long, requires = "acknowledge", help = "旧记录缺少主机/PID 命名空间身份时，确认已在原运行环境检查")]
        accept_unverified_host: bool,
        #[arg(long, requires = "acknowledge", help = "启动环境与当前环境不同时，明确接受无法检查原环境进程")]
        accept_different_scope: bool,
    },
    #[command(about = "只读比较两个运行的输入、代码、数据、预算、设备、权重与指标")]
    Compare {
        #[arg(help = "左侧运行 ID")]
        left: String,
        #[arg(help = "右侧运行 ID")]
        right: String,
    },
    #[command(about = "查看后台训练进度；TTY 下默认持续跟随")]
    Attach {
        id: Option<String>,
        #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true, help = "持续跟随进度；--follow=false 仅查看一次")]
        follow: Option<bool>,
    },
    #[command(about = "请求后台训练在安全检查点停止")]
    Stop {
        id: Option<String>,
        #[arg(long, help = "立即中断 worker，回退到最近已写入的 checkpoint")]
        force: bool,
    },
    #[command(about = "查看运行记录、日志、进度和 checkpoint 的磁盘占用")]
    Storage { id: Option<String> },
    #[command(about = "修改运行名称，不改配置、指标或权重")]
    Rename { id: String, name: String },
    #[command(about = "将运行标记为保留")]
    Keep { id: String },
    #[command(about = "移除运行的保留标记")]
    Unkeep { id: String },
    #[command(about = "预览并按固定计划清理结束运行的日志、进度和 checkpoint 文件")]
    Clean {
        #[arg(long, help = "只规划指定运行")]
        run: Option<String>,
        #[arg(
            long,
            conflicts_with = "apply",
            help = "将已结束运行的 worker.log 纳入清理预览"
        )]
        include_logs: bool,
        #[arg(
            long,
            conflicts_with = "apply",
            help = "将已结束运行的 progress.jsonl 纳入清理预览"
        )]
        include_progress: bool,
        #[arg(
            long,
            requires_all = ["run", "dry_run"],
            conflicts_with = "apply",
            help = "将指定运行的最新恢复权重列为显式释放候选"
        )]
        include_checkpoints: bool,
        #[arg(
            long,
            value_name = "ARTIFACT",
            requires_all = ["run", "include_checkpoints", "dry_run"],
            conflicts_with = "apply",
            help = "只将指定 checkpoint artifact ID 纳入释放预览"
        )]
        artifact: Option<String>,
        #[arg(long, conflicts_with = "apply", help = "只生成清理计划，不删除文件")]
        dry_run: bool,
        #[arg(
            long,
            value_name = "PLAN",
            conflicts_with_all = ["dry_run", "run", "include_logs", "include_progress", "include_checkpoints"],
            help = "执行之前预览得到的固定计划"
        )]
        apply: Option<String>,
        #[arg(long, requires = "apply", help = "明确接受移除已结束运行的恢复权重")]
        release_checkpoints: bool,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum TrainingMode {
    #[default]
    ManagedStep,
    ManualStep,
    ExternalLoop,
}

#[derive(Debug, Clone, Default, Args)]
pub struct Inputs {
    #[arg(skip)]
    pub no_project_model: bool,
    #[arg(long, help = "原生模型类或工厂，如 examples.models:build_shared")]
    pub model: Option<String>,
    #[arg(long, help = "示例批次函数，不需要 DataLoader 或训练步骤")]
    pub batch: Option<String>,
    #[arg(
        long = "model-param",
        value_name = "NAME=JSON",
        help = "模型工厂参数，可重复"
    )]
    pub model_params: Vec<String>,
    #[arg(
        long = "inject",
        value_name = "SLOT=MODULE:FACTORY|component:ID@VERSION",
        help = "给根工厂接缝注入 nn.Module 工厂或已登记 Layer 组件；可重复"
    )]
    pub injections: Vec<String>,
    #[arg(
        long = "inject-param",
        value_name = "SLOT.NAME=JSON",
        add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::injection_param_completer),
        help = "注入工厂或已登记组件的参数，可重复，例如 encoder.depth=50"
    )]
    pub injection_params: Vec<String>,
    #[arg(skip)]
    pub saved_injections: Option<Value>,
    #[arg(
        long = "batch-param",
        value_name = "NAME=JSON",
        help = "批次工厂参数，可重复"
    )]
    pub batch_params: Vec<String>,
    #[arg(
        long,
        value_enum,
        help = "批次与模型前向参数的绑定方式；single-raw 整体传入并跳过自动数据集采样"
    )]
    pub batch_binding: Option<BatchBinding>,
    #[arg(long, help = "按训练模式前向（仍不反向/更新）；默认 eval")]
    pub training: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub struct TrainingInputs {
    #[arg(long, value_name = "FILE", help = "使用项目、分类配置或模型图 JSON 的训练声明；不能与训练入口或恢复选项同时指定")]
    pub training_recipe: Option<PathBuf>,
    #[arg(long = "no-model", conflicts_with_all = ["model", "model_params", "injections", "injection_params"],
        help = "external-loop 新运行不使用项目默认模型；循环收到 None")]
    pub no_model: bool,
    #[arg(long = "model-initialization", value_parser = ["runtime", "seeded"],
        help = "模型工厂的随机状态策略；seeded 可使用独立的 --model-initialization-seed")]
    pub model_initialization: Option<String>,
    #[arg(long = "model-initialization-seed", requires = "model_initialization",
        help = "模型构造种子，与训练执行种子分别控制")]
    pub model_initialization_seed: Option<u64>,
    #[arg(long = "batch-device-policy", value_parser = ["preserve", "recursive-strict"],
        help = "步骤前保留批次对象或严格递归迁移；新运行默认严格迁移，续训沿用原策略")]
    pub batch_device_policy: Option<String>,
    #[arg(long = "if-repeat-inputs-sha256", value_name = "SHA256", help = "仅当重复试跑的历史输入与 runs show 回执一致时执行")]
    pub if_repeat_inputs_sha256: Option<String>,
    #[arg(long = "optimizer-factory", help = "managed-step 优化器工厂 module:callable；接收模型和显式参数")]
    pub optimizer_factory: Option<String>,
    #[arg(long = "optimizer-param", value_name = "NAME=JSON", requires = "optimizer_factory", help = "优化器工厂参数；会记录到运行与检查点中，勿填写密钥")]
    pub optimizer_params: Vec<String>,
    #[arg(
        long = "if-config-revision",
        value_name = "SHA256",
        help = "仅当启动时项目配置与预览修订一致时开始新训练或试跑"
    )]
    pub if_config_revision: Option<String>,
    #[arg(
        long = "if-supervision-revision",
        value_name = "SHA256",
        help = "仅使用与确认计划一致的监督标记和规则"
    )]
    pub if_supervision_revision: Option<String>,
    #[arg(
        long = "data-fingerprint",
        value_name = "VERSION",
        help = "用户声明的数据集/切分版本；不是自动验证的数据内容哈希"
    )]
    pub data_fingerprint: Option<String>,
    #[arg(long, help = "训练执行设备：cpu、auto 或所选 PyTorch 支持的设备标识；默认 cpu")]
    pub device: Option<String>,
    #[arg(long, help = "原生模型类或工厂，如 examples.models:build_classifier")]
    pub model: Option<String>,
    #[arg(long, help = "每步重新调用的批次工厂；managed/manual-step 需要")]
    pub batch: Option<String>,
    #[arg(
        long = "model-param",
        value_name = "NAME=JSON",
        help = "模型工厂参数，可重复"
    )]
    pub model_params: Vec<String>,
    #[arg(
        long = "inject",
        value_name = "SLOT=MODULE:FACTORY|component:ID@VERSION",
        help = "给根工厂接缝注入 nn.Module 工厂或已登记 Layer 组件；可重复"
    )]
    pub injections: Vec<String>,
    #[arg(
        long = "inject-param",
        value_name = "SLOT.NAME=JSON",
        add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::injection_param_completer),
        help = "注入工厂或已登记组件的参数，可重复，例如 encoder.depth=50"
    )]
    pub injection_params: Vec<String>,
    #[arg(skip)]
    pub saved_injections: Option<Value>,
    #[arg(
        long = "batch-param",
        value_name = "NAME=JSON",
        help = "批次工厂参数，可重复"
    )]
    pub batch_params: Vec<String>,
    #[arg(long, value_enum, help = "批次与模型前向参数的绑定方式；single-raw 整体传入并跳过自动数据集采样")]
    pub batch_binding: Option<BatchBinding>,
}

impl From<TrainingInputs> for Inputs {
    fn from(inputs: TrainingInputs) -> Self {
        Self {
            no_project_model: inputs.no_model,
            model: inputs.model,
            batch: inputs.batch,
            model_params: inputs.model_params,
            injections: inputs.injections,
            injection_params: inputs.injection_params,
            saved_injections: inputs.saved_injections,
            batch_params: inputs.batch_params,
            batch_binding: inputs.batch_binding,
            ..Self::default()
        }
    }
}

impl From<Inputs> for TrainingInputs {
    fn from(inputs: Inputs) -> Self {
        Self {
            training_recipe: None,
            no_model: inputs.no_project_model,
            model_initialization: None,
            model_initialization_seed: None,
            if_repeat_inputs_sha256: None,
            optimizer_factory: None,
            optimizer_params: Vec::new(),
            device: None,
            if_config_revision: None,
            if_supervision_revision: None,
            data_fingerprint: None,
            batch_device_policy: None,
            model: inputs.model,
            batch: inputs.batch,
            model_params: inputs.model_params,
            injections: inputs.injections,
            injection_params: inputs.injection_params,
            saved_injections: inputs.saved_injections,
            batch_params: inputs.batch_params,
            batch_binding: inputs.batch_binding,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum BatchBinding {
    Envelope,
    Single,
    SingleRaw,
    Args,
    Kwargs,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelCommand {
    #[command(about = "检查模型层级、模块类型和参数规模；不要求示例批次")]
    Inspect {
        path: Option<String>,
        #[command(flatten)]
        inputs: Inputs,
    },
    #[command(about = "对示例批次执行真实前向并报告逐调用输入/输出与契约检查")]
    Probe {
        path: Option<String>,
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, default_value = "all", help = "本次选中模块的调用序号，或 all")]
        call: String,
        #[arg(
            long,
            value_name = "FILE",
            help = "可选 JSON 端口契约；声明输入/输出 shape、dtype、layout 与符号维度"
        )]
        contract: Option<PathBuf>,
        #[arg(long, requires = "contract", value_name = "SHA256", help = "仅使用与此 SHA-256 一致的契约文件")]
        if_contract_sha256: Option<String>,
        #[arg(
            long,
            value_name = "ID",
            help = "可选数据/切分指纹；配合 Dataset 的 sample_id(index) 启用安全证据复用"
        )]
        batch_fingerprint: Option<String>,
        #[arg(long, help = "显式迁移模型到所选 PyTorch 设备；输入不会自动迁移")]
        model_device: Option<String>,
        #[arg(
            long,
            requires = "path",
            help = "复制已观察到的调用输入，隔离重放所选 Layer；需同时指定 --path"
        )]
        local: bool,
    },
    #[command(about = "把人工语义标签绑定到本次真实 Probe 观察到的模型端口")]
    Mark {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(
            long,
            required = true,
            help = "模型根路径 model，或完整模块树路径，例如 model.encoder.projection"
        )]
        path: String,
        #[arg(
            long,
            required = true,
            help = "Probe 显示的端口路径，例如 outputs.features"
        )]
        port: String,
        #[arg(
            long,
            required = true,
            help = "自由命名的语义角色，例如 embedding 或 action"
        )]
        role: String,
        #[arg(long, required = true, help = "人工编写的简短语义说明")]
        meaning: String,
        #[arg(long, help = "可选的稳定标记 ID；省略时自动生成")]
        id: Option<String>,
        #[arg(long, value_name = "JSON", help = "仅当重跑观察到相同实现来源时保存标记")]
        if_anchor: Option<String>,
        #[arg(long, help = "显式选择标记观察时的模型设备；输入不会自动迁移")]
        model_device: Option<String>,
    },
    #[command(about = "重新探测并更新一个已有语义标记的实现块或含义")]
    Retag {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, required = true, help = "要替换绑定的精确标记 ID")]
        id: String,
        #[arg(long, required = true, help = "当前模块树路径")]
        path: String,
        #[arg(long, required = true, help = "本次 Probe 中观察到的端口路径")]
        port: String,
        #[arg(long, required = true, help = "自由命名的语义角色")]
        role: String,
        #[arg(long, required = true, help = "人工编写的简短语义说明")]
        meaning: String,
        #[arg(long, value_name = "SHA256", help = "仅当语义状态与已查看修订一致时重新绑定")]
        if_state_revision: Option<String>,
        #[arg(long, value_name = "JSON", help = "仅当重跑观察到相同实现来源时重新绑定")]
        if_anchor: Option<String>,
        #[arg(long, help = "显式选择重新绑定观察时的模型设备；输入不会自动迁移")]
        model_device: Option<String>,
    },
    #[command(about = "列出项目中的人工语义标记，并检查源码锚点是否过期")]
    Marks {
        #[command(flatten)]
        inputs: Inputs,
    },
    #[command(about = "按精确 ID 删除一个人工语义标记")]
    Unmark {
        #[arg(long, required = true, help = "model mark 返回的精确标记 ID")]
        id: String,
    },
    #[command(about = "给人工标记添加监督/约束规则，或预览规则在当前批次上的结果")]
    Supervise {
        #[command(subcommand)]
        command: SuperviseCommand,
    },
    #[command(about = "执行真实前向与损失反向传播，检查损失和梯度")]
    Verify {
        #[arg(
            long,
            help = "接收 model_output 和原始 batch、返回标量 Tensor 的损失函数"
        )]
        loss: String,
        #[command(flatten)]
        inputs: Inputs,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum SuperviseCommand {
    #[command(about = "逐步引导人工标记真实实现块、添加规则并预览；只在终端交互中运行")]
    Guide {
        #[command(flatten)]
        inputs: Inputs,
    },
    #[command(about = "管理绑定到语义标记的规则")]
    Rule {
        #[command(subcommand)]
        command: SupervisionRuleCommand,
    },
    #[command(about = "导出监督标记与规则声明；不写文件")]
    Export {
        #[arg(long, required = true, help = "已审核语义状态的修订")]
        if_state_revision: String,
    },
    #[command(about = "预览或导入监督声明；导入的标记必须重新绑定")]
    Import {
        #[arg(long, value_name = "FILE", required = true)]
        file: PathBuf,
        #[arg(long, required = true, help = "已审核声明文件的 SHA-256")]
        if_sha256: String,
        #[arg(long, required = true, help = "目标语义状态修订")]
        if_state_revision: String,
        #[arg(long, help = "应用已预览的导入")]
        apply: bool,
    },
    #[command(about = "对真实模型前向执行只读规则预览；不反向、不更新参数")]
    Preview {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, help = "显式选择监督预览时的模型设备；输入不会自动迁移")]
        model_device: Option<String>,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum SupervisionRuleCommand {
    #[command(about = "添加绑定到人工语义标记的监督规则")]
    Add {
        #[arg(long, required = true, help = "model mark 返回的精确语义标记 ID")]
        mark: String,
        #[arg(
            long,
            value_enum,
            required = true,
            help = "规则类型：finite、range、batch-alignment、custom-validator 或 custom-penalty"
        )]
        kind: SupervisionRuleKind,
        #[arg(
            long,
            value_enum,
            required = true,
            help = "observe 记录结果、require 在更新前拒绝违规、penalty 显式加入加权 differentiable loss"
        )]
        mode: SupervisionRuleMode,
        #[arg(long, allow_hyphen_values = true, help = "range 下界；可选")]
        min: Option<f64>,
        #[arg(long, allow_hyphen_values = true, help = "range 上界；可选")]
        max: Option<f64>,
        #[arg(long, help = "penalty 模式必须显式提供大于 0 的权重")]
        weight: Option<f64>,
        #[arg(
            long,
            help = "batch-alignment 的人工目标路径，例如 batch.args[1] 或 batch.kwargs.labels"
        )]
        target_path: Option<String>,
        #[arg(long, value_parser = clap::value_parser!(u32), help = "batch-alignment 的目标维度；finite/range/custom-validator 可用它指定输出样本轴")]
        axis: Option<u32>,
        #[arg(long, help = "可选稳定规则 ID；省略时自动生成")]
        id: Option<String>,
        #[arg(long = "if-state-revision", help = "仅当语义状态仍是所选完整修订时添加规则")]
        if_state_revision: Option<String>,
        #[arg(
            long,
            help = "自定义规则 module:callable；函数签名为 (value, batch, params)"
        )]
        plugin: Option<String>,
        #[arg(long, value_parser = ["project", "environment"], help = "规则来源：project（默认）或所选 Python environment")]
        plugin_source: Option<String>,
        #[arg(
            long = "plugin-param",
            value_name = "NAME=JSON",
            help = "自定义规则 JSON 参数，可重复"
        )]
        plugin_params: Vec<String>,
    },
    #[command(about = "将监督规则完整复制到其他人工标记，保留插件源码身份")]
    Copy {
        #[arg(long, required = true, help = "源规则 ID")]
        id: String,
        #[arg(long, required = true, help = "目标语义标记 ID")]
        mark: String,
        #[arg(long, required = true, help = "仅当标记与规则状态修订一致时复制")]
        if_state_revision: String,
    },
    #[command(about = "列出监督规则及其当前可用性")]
    List,
    #[command(about = "按精确 ID 移除一条监督规则")]
    Remove {
        #[arg(long, required = true, help = "精确规则 ID")]
        id: String,
        #[arg(long, help = "仅当规则与列出的 JSON 完全一致时删除")]
        expected_rule: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum SupervisionRuleKind {
    Finite,
    Range,
    #[value(name = "batch-alignment")]
    BatchAlignment,
    #[value(name = "custom-validator")]
    CustomValidator,
    #[value(name = "custom-penalty")]
    CustomPenalty,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum SupervisionRuleMode {
    Observe,
    Require,
    Penalty,
}

#[derive(Debug, Clone, Subcommand)]
// Keep the flat fields so Clap can expose the established, documented flags directly.
#[allow(clippy::large_enum_variant)]
pub enum DataCommand {
    #[command(about = "浏览或搜索内置模型预设；只读，不下载权重或导入 PyTorch")]
    Presets {
        #[arg(
            value_name = "QUERY",
            help = "可选名称或任务关键词，例如 resnet 或 classification"
        )]
        query: Option<String>,
    },
    #[command(about = "扫描分类目录、验证图像，并按确认结果记录显式排除协议")]
    Index {
        #[arg(long, required = true, help = "分类目录；每个类别对应一个子目录")]
        root: PathBuf,
        #[arg(long, help = "解码并验证全部已索引图像；可能需要 Pillow")]
        verify: bool,
        #[arg(
            long = "exclude",
            value_name = "RELATIVE_PATH",
            requires = "apply_exclusions",
            help = "显式选择要排除的损坏相对路径；可重复指定"
        )]
        exclude_files: Vec<String>,
        #[arg(
            long,
            requires = "verify",
            help = "复验当前内容后把所选排除协议写入项目元数据；不会删除图像"
        )]
        apply_exclusions: bool,
        #[arg(
            long,
            requires = "verify",
            conflicts_with = "apply_exclusions",
            help = "所有图像修复并复验通过后，清除项目中的排除协议"
        )]
        clear_exclusions: bool,
    },
    #[command(about = "预览或创建可编辑的数值 CSV 批次模板")]
    CsvTemplate {
        #[arg(long, help = "确认创建新文件；必须提供预览修订")]
        apply: bool,
        #[arg(long, requires = "apply", help = "只提交与预览一致的文件计划")]
        if_plan: Option<String>,
    },
    #[command(about = "从项目 Dataset 或批次工厂采样并生成可追溯预览")]
    Preview {
        #[arg(
            long,
            conflicts_with = "batch",
            help = "Dataset 工厂入口，如 project.data:build_dataset"
        )]
        dataset: Option<String>,
        #[arg(
            long,
            conflicts_with = "dataset",
            help = "已准备批次工厂入口，如 project.data:example_batch"
        )]
        batch: Option<String>,
        #[arg(long, requires = "batch", help = "将批次工厂整个返回值作为一项预览，不自动采样 Dataset")]
        whole_batch: bool,
        #[arg(
            long = "param",
            value_name = "NAME=JSON",
            help = "所选 Dataset/批次工厂参数，可重复"
        )]
        parameters: Vec<String>,
        #[arg(long, default_value = "train", help = "数据切分名称；默认 train")]
        split: String,
        #[arg(long, help = "不请求数据切分；优先于 --split，且不自动套用项目 train/eval 增强")]
        no_split: bool,
        #[arg(long, value_parser = ["train", "eval", "none"], conflicts_with = "transforms", help = "显式选择项目增强流水线；none 不套用增强，缺省时按切分选择")]
        augment_split: Option<String>,
        #[arg(long, default_value_t = 12, value_parser = parse_preview_samples, help = "最多预览的样本数（1 到 64）")]
        samples: usize,
        #[arg(long, requires = "dataset", value_parser = clap::value_parser!(u32).range(1..=100_000), help = "流式 Dataset 最多读取的样本数；从该窗口抽样，不代表整个数据流")]
        scan_limit: Option<u32>,
        #[arg(long, help = "用户提供的数据/切分稳定指纹；缺省时本次预览不复用缓存")]
        fingerprint: Option<String>,
        #[arg(long, help = "包含有序变换声明的 JSON 数组文件")]
        transforms: Option<PathBuf>,
        #[arg(long, value_parser = ["auto", "summary", "custom"], default_value = "auto", help = "auto 自动生成图像网格/非图像摘要；summary 仅生成结构摘要；custom 调用项目 renderer")]
        renderer: String,
        #[arg(
            long,
            required_if_eq("renderer", "custom"),
            help = "项目 renderer 工厂入口 module:callable；返回的 artifacts 文件须位于请求目录"
        )]
        renderer_entrypoint: Option<String>,
        #[arg(
            long,
            required_if_eq("renderer", "custom"),
            help = "项目 renderer 版本标记，内容变化时必须更新以使缓存失效"
        )]
        renderer_version: Option<String>,
        #[arg(long, value_enum, default_value_t = PreviewDelivery::Local, help = "本机路径或生成需用户本机执行的 SSH 取回命令")]
        delivery: PreviewDelivery,
        #[arg(
            long,
            requires = "ssh_user",
            requires = "ssh_port",
            required_if_eq("delivery", "ssh"),
            help = "用户确认的 SSH 主机名/IP"
        )]
        ssh_host: Option<String>,
        #[arg(
            long,
            requires = "ssh_host",
            requires = "ssh_port",
            required_if_eq("delivery", "ssh"),
            help = "用户确认的 SSH 用户名"
        )]
        ssh_user: Option<String>,
        #[arg(
            long,
            requires = "ssh_host",
            requires = "ssh_user",
            required_if_eq("delivery", "ssh"),
            help = "用户确认的 SSH 端口"
        )]
        ssh_port: Option<u16>,
        #[arg(
            long,
            default_value = "./preview.html",
            help = "SSH 取回命令的本机目标路径"
        )]
        destination: PathBuf,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum AugmentCommand {
    #[command(about = "列出当前增强步骤")]
    List {
        #[arg(long, value_parser = ["train", "eval"], default_value = "train", help = "要查看的增强流水线：train 或 eval")]
        split: String,
    },
    #[command(about = "在流水线末尾添加增强步骤")]
    Add {
        #[command(flatten)]
        edit: AugmentEditOptions,
        #[arg(
            help = "Resize、RandomResizedCrop、ToTensor、Normalize、RandomHorizontalFlip 或 custom"
        )]
        kind: String,
        #[arg(long, value_parser = ["train", "eval"], default_value = "train", help = "步骤所属的数据切分：train 或 eval")]
        split: String,
        #[arg(long, required = true, help = "稳定的步骤 ID")]
        id: String,
        #[arg(long, help = "自定义变换工厂 module:callable；仅用于 custom")]
        entrypoint: Option<String>,
        #[arg(
            long,
            help = "训练 batch 中要变换的数据路径，如 args.0 或 kwargs.images"
        )]
        target: Option<String>,
        #[arg(long, value_name = "MIN,MAX", help = "随机裁剪面积比例，如 0.08,1.0")]
        scale: Option<String>,
        #[arg(long = "param", value_name = "NAME=JSON", help = "变换参数，可重复")]
        parameters: Vec<String>,
    },
    #[command(about = "修改已有步骤参数")]
    Set {
        #[command(flatten)]
        edit: AugmentEditOptions,
        id: String,
        #[arg(long, value_parser = ["train", "eval"], default_value = "train", help = "要修改的增强流水线：train 或 eval")]
        split: String,
        #[arg(long, value_name = "MIN,MAX", help = "更新随机裁剪面积比例")]
        scale: Option<String>,
        #[arg(long, help = "更新训练 batch 中的数据路径，如 args.0")]
        target: Option<String>,
        #[arg(long = "param", value_name = "NAME=JSON", help = "替换的参数，可重复")]
        parameters: Vec<String>,
    },
    #[command(about = "移动步骤；省略 --before 时移至末尾")]
    Move {
        #[command(flatten)]
        edit: AugmentEditOptions,
        id: String,
        #[arg(long, value_parser = ["train", "eval"], default_value = "train", help = "要移动步骤的增强流水线：train 或 eval")]
        split: String,
        #[arg(long, help = "放到此步骤之前")]
        before: Option<String>,
    },
    #[command(about = "删除已有步骤")]
    Remove {
        #[command(flatten)]
        edit: AugmentEditOptions,
        id: String,
        #[arg(long, value_parser = ["train", "eval"], default_value = "train", help = "要移除步骤的增强流水线：train 或 eval")]
        split: String,
    },
}

#[derive(Debug, Clone, Args)]
pub struct AugmentEditOptions {
    #[arg(long, help = "预览已校验的增强配置，不写入项目")]
    pub dry_run: bool,
    #[arg(long, help = "仅当项目配置仍为该内容版本时执行编辑")]
    pub if_config_revision: Option<String>,
}

fn parse_preview_samples(value: &str) -> Result<usize, String> {
    let samples = value
        .parse::<usize>()
        .map_err(|_| "样本数必须是整数。".to_owned())?;
    if !(1..=64).contains(&samples) {
        return Err("样本数必须在 1 到 64 之间。".into());
    }
    Ok(samples)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum PreviewDelivery {
    Local,
    Ssh,
}

#[derive(Debug, Clone, Args)]
pub struct EditOptions {
    #[command(flatten)]
    pub inputs: Inputs,
    #[arg(
        long,
        default_value = "variant",
        value_parser = ["variant", "source"],
        help = "写入持久 variant；source 支持 Sequential/ModuleList/ModuleDict CRUD 与跨文件 set/结构事务"
    )]
    pub target: String,
    #[arg(long = "variant-device-policy", value_parser = ["legacy-cpu", "factory"],
        help = "新变体默认保留基础模型和新增层的工厂设备；替换层参考旧层首个状态；已有变体保持原策略")]
    pub variant_device_policy: Option<String>,
    #[arg(long = "base-rng-scope", value_parser = ["cpu", "selected"], help = "新变体基础模型随机范围：cpu 或 selected（CPU 和明确指定的设备）")]
    pub base_rng_scope: Option<String>,
    #[arg(long = "base-rng-device", value_name = "BACKEND:INDEX", requires = "base_rng_scope", help = "selected 范围内的基础模型随机设备；可重复，会初始化所选设备")]
    pub base_rng_devices: Vec<String>,
    #[arg(long = "layer-rng-device", value_name = "BACKEND:INDEX", help = "变体 add/replace 时额外固定的随机设备；可重复，同一后端，例如 cuda:0；会初始化指定设备")]
    pub layer_rng_devices: Vec<String>,
    #[arg(
        long = "param",
        value_name = "NAME=JSON",
        add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_param_completer),
        help = "Layer 构造参数；按类型/组件字段校验，可重复指定"
    )]
    pub params: Vec<String>,
    #[arg(long, help = "只预览候选改动和已执行检查，不写入变体或源码")]
    pub dry_run: bool,
    #[arg(long, help = "要求提供 --batch，并对候选模型执行真实前向")]
    pub require_forward: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub enum LayerCommand {
    #[command(about = "预览并迁移可识别的旧版自定义骨架；需要 --target source")]
    MigrateStarter {
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "列出部分 PyTorch Layer 的离线参数示例；非可用类型清单，不导入 PyTorch")]
    Catalog,
    #[command(about = "列出模型中的 Layer；可按父路径、关键词或模块类型筛选")]
    List {
        #[arg(help = "可选父路径；省略时列出整个模型树")]
        parent: Option<String>,
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long = "type", help = "只显示此模块类型，如 Linear 或 LayerNorm")]
        layer_type: Option<String>,
        #[arg(
            long,
            value_name = "KEYWORD",
            help = "不区分大小写地匹配模块路径、名称、类型或共享别名"
        )]
        search: Option<String>,
    },
    #[command(about = "查看目标 Layer 的参数、shape、来源和编辑能力")]
    Show {
        #[arg(help = "模型树路径，如 model.encoder.proj")]
        path: String,
        #[command(flatten)]
        inputs: Inputs,
    },
    #[command(about = "向 nn.Sequential、nn.ModuleList、nn.ModuleDict 或其子类添加 Layer")]
    Add {
        #[arg(
            long,
            help = "父容器路径；variant 支持原生容器及子类，source 支持直接赋值的原生容器"
        )]
        parent: String,
        #[arg(
            long,
            conflicts_with = "after",
            help = "插入到子项之前：Sequential/ModuleDict 用名称，ModuleList 用数字索引"
        )]
        before: Option<String>,
        #[arg(
            long,
            conflicts_with = "before",
            help = "插入到子项之后：Sequential/ModuleDict 用名称，ModuleList 用数字索引"
        )]
        after: Option<String>,
        #[arg(
            long,
            default_value = "",
            help = "Sequential 子层名 / ModuleDict 分支名（须为无点号 Python 标识符）；ModuleList 自动使用数字索引"
        )]
        name: String,
        #[arg(
            long = "type",
            required_unless_present = "component",
            conflicts_with = "component",
            help = "内置 Layer 类型，如 Linear、ReLU、Dropout"
        )]
        layer_type: Option<String>,
        #[arg(
            long,
            required_unless_present = "layer_type",
            conflicts_with = "layer_type",
            help = "Python 工厂 module:callable，或已登记的 Layer 组件 ID（可带版本约束）"
        )]
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        component: Option<String>,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "修改目标 Layer 已支持的构造参数")]
    Set {
        #[arg(help = "模型树路径，如 model.encoder.proj")]
        path: String,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "用内置 Layer、Python 工厂或已登记组件替换目标 Layer")]
    Replace {
        #[arg(help = "模型树路径，如 model.encoder.proj")]
        path: String,
        #[arg(
            long = "type",
            required_unless_present = "component",
            conflicts_with = "component",
            help = "内置 Layer 类型，如 Linear、ReLU、Dropout"
        )]
        layer_type: Option<String>,
        #[arg(
            long,
            required_unless_present = "layer_type",
            conflicts_with = "layer_type",
            help = "Python 工厂 module:callable，或已登记的 Layer 组件 ID（可带版本约束）"
        )]
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        component: Option<String>,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "从有序容器删除 Layer，或将单输入 Layer 替换为 Identity")]
    Remove {
        #[arg(help = "模型树路径，如 model.encoder.proj")]
        path: String,
        #[arg(
            long,
            default_value = "remove",
            value_parser = ["remove", "bypass"],
            help = "remove 真正删除；bypass 将单输入 Layer 替换为 Identity"
        )]
        strategy: String,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "把多个 Layer 操作作为一个候选计划统一检查和提交")]
    Apply {
        #[arg(
            long = "operation",
            required = true,
            value_name = "JSON",
            help = "可重复；JSON 操作对象，如 '{\"action\":\"set\",\"path\":\"model.encoder.proj\",\"params\":{\"out_features\":16}}'"
        )]
        operations: Vec<String>,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "通过项目 Layer Edit Provider 执行自定义图编辑")]
    Custom {
        #[arg(help = "已注册的 namespace/name[@version]，或 module:callable")]
        provider: String,
        #[arg(
            long,
            default_value = "{}",
            value_name = "JSON",
            help = "提供者 Schema 定义的编辑输入"
        )]
        payload: String,
        #[command(flatten)]
        edit: EditOptions,
    },
    #[command(about = "撤销最近一次 Layer 编辑；源码模式会检查文件冲突")]
    Undo {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, default_value = "variant", value_parser = ["variant", "source"], help = "撤销目标；source 会检查文件指纹冲突")]
        target: String,
        #[arg(long, help = "只预览变体撤销，不保存")]
        dry_run: bool,
        #[arg(long, help = "对撤销后的变体执行实际前向")]
        require_forward: bool,
    },
    #[command(about = "重做最近一次已撤销的 Layer 编辑")]
    Redo {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, default_value = "variant", value_parser = ["variant", "source"], help = "重做目标；source 会检查文件指纹冲突")]
        target: String,
        #[arg(long, help = "只预览变体重做，不保存")]
        dry_run: bool,
        #[arg(long, help = "对重做后的变体执行实际前向")]
        require_forward: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum AssemblyCommand {
    #[command(about = "显示项目装配接缝、组件参数与模型中的实际注册路径")]
    Show {
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(
            long,
            help = "只读取项目配置，不导入模型或运行前向；用于恢复损坏的装配"
        )]
        config_only: bool,
    },
    #[command(about = "一次预览并条件提交多个装配接缝，支持修复多处故障")]
    Patch {
        #[arg(
            long = "set",
            value_name = "SLOT=FACTORY",
            help = "新增或替换接缝，支持 module:factory 或 component:ID@VERSION；可重复"
        )]
        sets: Vec<String>,
        #[arg(
            long = "remove",
            value_name = "SLOT",
            help = "移除已配置的接缝；可重复"
        )]
        removes: Vec<String>,
        #[arg(
            long = "param",
            value_name = "SLOT.NAME=JSON",
            help = "已登记组件的参数；可重复"
        )]
        params: Vec<String>,
        #[arg(
            long = "encoder-output",
            value_name = "SLOT=native|flatten",
            help = "同时更改骨架编码器输出方式；可重复"
        )]
        encoder_outputs: Vec<String>,
        #[arg(
            long = "route",
            value_name = "SLOT=INPUT",
            help = "同时更改骨架编码器输入接线；可重复"
        )]
        route_changes: Vec<String>,
        #[arg(
            long,
            requires = "apply",
            value_name = "SHA256",
            help = "提交时要求观察到的项目源码与预览摘要一致"
        )]
        if_source_revision: Option<String>,
        #[arg(
            long,
            requires = "apply",
            value_name = "SHA256",
            help = "提交时要求前向输入内容与预览一致"
        )]
        if_input_sha256: Option<String>,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "提交已验证候选；需要 --if-config-revision")]
        apply: bool,
    },
    #[command(about = "向根工厂新增一个已显式声明并实际使用的模块接缝")]
    Add {
        #[arg(help = "根工厂关键字参数名，例如 encoder")]
        slot: String,
        #[arg(
            long,
            conflicts_with = "factory",
            help = "已登记 Layer 组件 ID@精确版本"
        )]
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        component: Option<String>,
        #[arg(
            long,
            conflicts_with = "component",
            help = "项目 Python 零参模块工厂 module:callable"
        )]
        factory: Option<String>,
        #[arg(
            long = "param",
            value_name = "NAME=JSON",
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_param_completer),
            help = "注册组件参数；未指定项采用 Schema 默认值"
        )]
        params: Vec<String>,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "验证候选但不更新 project.json")]
        dry_run: bool,
    },
    #[command(about = "预览或条件更新自定义骨架的输入到编码器连接")]
    Route {
        #[arg(help = "目标编码器接缝，例如 image_encoder")]
        slot: String,
        #[arg(help = "要连接的具名输入")]
        input: String,
        #[arg(long, value_name = "SHA256", help = "要求已观察项目源码与预览一致")]
        if_source_revision: Option<String>,
        #[arg(long, value_name = "SHA256", help = "要求代表前向输入内容与预览一致")]
        if_input_sha256: Option<String>,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "真实前向验证候选但不修改配置")]
        dry_run: bool,
    },
    #[command(about = "预览或条件更新骨架编码器输出的处理方式")]
    EncoderOutput {
        #[arg(help = "编码器接缝，例如 text_encoder")]
        slot: String,
        #[arg(help = "native 保留原始输出；flatten 使用默认归约和展平")]
        mode: String,
        #[arg(long, value_name = "SHA256", help = "要求已观察项目源码与预览一致")]
        if_source_revision: Option<String>,
        #[arg(long, value_name = "SHA256", help = "要求代表前向输入内容与预览一致")]
        if_input_sha256: Option<String>,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "真实前向验证候选但不修改配置")]
        dry_run: bool,
    },
    #[command(about = "替换一个已配置的模块接缝及其组件参数")]
    Set {
        #[arg(help = "根工厂关键字参数名，例如 encoder")]
        slot: String,
        #[arg(
            long,
            conflicts_with = "factory",
            help = "已登记 Layer 组件 ID@精确版本"
        )]
        #[arg(add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_id_completer))]
        component: Option<String>,
        #[arg(
            long,
            conflicts_with = "component",
            help = "项目 Python 零参模块工厂 module:callable"
        )]
        factory: Option<String>,
        #[arg(
            long = "param",
            value_name = "NAME=JSON",
            add = clap_complete::engine::ArgValueCompleter::new(crate::presentation::completion::component_param_completer),
            help = "注册组件参数；未指定项采用 Schema 默认值"
        )]
        params: Vec<String>,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "验证候选但不更新 project.json")]
        dry_run: bool,
    },
    #[command(about = "移除一个接缝绑定；根工厂必须能在没有该参数时继续构造")]
    Remove {
        #[arg(help = "根工厂关键字参数名")]
        slot: String,
        #[command(flatten)]
        inputs: AssemblyInputs,
        #[arg(long, help = "验证候选但不更新 project.json")]
        dry_run: bool,
    },
    #[command(about = "从项目装配或所选变体生成可读 Python 适配器，不覆盖已有文件")]
    Generate {
        #[arg(
            long,
            default_value = "generated/model_assembly.py",
            help = "项目目录内的新生成文件路径"
        )]
        output: PathBuf,
        #[command(flatten)]
        inputs: AssemblyInputs,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ProjectConfigCommand {
    #[command(about = "列出项目配置的历史版本")]
    History,
    #[command(about = "预览或条件保存项目默认 Python 解释器")]
    SetPython {
        #[arg(long, required_unless_present = "clear", conflicts_with = "clear")]
        executable: Option<String>,
        #[arg(long, conflicts_with = "executable")]
        clear: bool,
        #[arg(long, requires = "if_current")]
        apply: bool,
        #[arg(long = "if-current", requires = "apply")]
        if_current: Option<String>,
    },
    #[command(about = "预览或条件保存项目运行目录与模块搜索目录")]
    SetRuntimePaths {
        #[arg(long, help = "已有的绝对运行目录；省略时使用项目目录")]
        working_directory: Option<String>,
        #[arg(long = "import-root", help = "项目内相对模块搜索目录，按优先级重复指定")]
        import_roots: Vec<String>,
        #[arg(long, requires = "if_current", help = "按预览修订保存")]
        apply: bool,
        #[arg(long = "if-current", requires = "apply")]
        if_current: Option<String>,
    },
    #[command(about = "预览或条件更新默认模型入口与参数")]
    SetModel {
        #[arg(long, required = true, help = "新的模型工厂 module:callable")]
        model: String,
        #[arg(long = "model-param", value_name = "NAME=JSON", help = "模型工厂参数；会保存在项目配置中，请勿保存凭据")]
        model_params: Vec<String>,
        #[arg(long, help = "同时清除已有模型装配注入")]
        clear_injections: bool,
        #[arg(long, requires = "if_current", help = "按预览的当前修订提交")]
        apply: bool,
        #[arg(long = "if-current", requires = "apply", help = "预览时返回的完整配置修订")]
        if_current: Option<String>,
    },
    #[command(about = "预览或条件更新默认批次入口，不改变模型和装配")]
    SetBatch {
        #[arg(long, required = true, help = "新的批次工厂 module:callable")]
        batch: String,
        #[arg(long, value_enum, help = "批次绑定；省略为自动推断")]
        batch_binding: Option<BatchBinding>,
        #[arg(
            long = "batch-param",
            value_name = "NAME=JSON",
            help = "批次工厂参数，可重复；会写入本地项目配置，请勿保存凭据"
        )]
        batch_params: Vec<String>,
        #[arg(long, requires = "if_current", help = "按预览的当前修订提交")]
        apply: bool,
        #[arg(
            long = "if-current",
            requires = "apply",
            help = "预览时返回的完整配置修订"
        )]
        if_current: Option<String>,
    },
    #[command(about = "从历史版本恢复指定字段；默认仅预览，添加 --apply 后提交")]
    Restore {
        #[arg(help = "完整 revision 或唯一前缀")]
        revision: String,
        #[arg(
            long = "field",
            value_name = "FIELD",
            value_parser = ["model", "model_params", "batch", "batch_binding", "batch_params", "model_injections", "runtime_paths", "runtime_python"],
            required = true,
            help = "恢复的配置字段，可重复指定"
        )]
        fields: Vec<String>,
        #[arg(
            long,
            requires = "if_current",
            help = "应用字段恢复；必须同时指定预览时的当前 revision"
        )]
        apply: bool,
        #[arg(
            long = "if-current",
            requires = "apply",
            value_name = "REVISION",
            help = "仅当当前配置仍等于预览时的 revision 才提交"
        )]
        if_current: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Args)]
pub struct AssemblyInputs {
    #[arg(
        long = "if-config-revision",
        value_name = "SHA256",
        help = "仅在启动时项目配置匹配指定修订时执行装配编辑或代码生成"
    )]
    pub if_config_revision: Option<String>,
    #[arg(
        long = "model-param",
        value_name = "NAME=JSON",
        help = "根模型工厂构造参数，可重复"
    )]
    pub model_params: Vec<String>,
    #[arg(
        long = "batch-param",
        value_name = "NAME=JSON",
        help = "批次工厂构造参数，可重复"
    )]
    pub batch_params: Vec<String>,
    #[arg(long, help = "临时覆盖示例批次入口；省略时使用项目默认值")]
    pub batch: Option<String>,
    #[arg(long, value_enum, help = "示例批次绑定方式")]
    pub batch_binding: Option<BatchBinding>,
}

#[cfg(test)]
mod public_help_contract_tests {
    use super::Cli;
    use clap::{Command as ClapCommand, CommandFactory};
    use serde_json::Value;

    const OPERATIONS: &str = include_str!("../docs/contracts/operations.json");
    const COMPATIBILITY: &str = include_str!("../docs/contracts/compatibility.json");

    fn collect_help_gaps(command: &ClapCommand, parent: &str, gaps: &mut Vec<String>) {
        for child in command.get_subcommands() {
            let path = if parent.is_empty() {
                child.get_name().to_owned()
            } else {
                format!("{parent} {}", child.get_name())
            };
            let about = child
                .get_about()
                .or_else(|| child.get_long_about())
                .map(ToString::to_string)
                .unwrap_or_default();
            if about.trim().is_empty() {
                gaps.push(format!("公开命令 `{path}` 缺少 help 描述"));
            }
            for argument in child.get_arguments() {
                if argument.get_long().is_none()
                    || argument.is_hide_set()
                    || matches!(argument.get_id().as_str(), "help" | "version")
                {
                    continue;
                }
                let help = argument
                    .get_help()
                    .or_else(|| argument.get_long_help())
                    .map(ToString::to_string)
                    .unwrap_or_default();
                if help.trim().is_empty() {
                    gaps.push(format!(
                        "公开选项 `--{}`（命令 `{path}`）缺少 help 描述",
                        argument.get_long().unwrap_or_default()
                    ));
                }
            }
            collect_help_gaps(child, &path, gaps);
        }
    }

    fn resolve_documented_cli_path(path: &str, root: &ClapCommand) -> Vec<ClapCommand> {
        let mut commands = vec![root.clone()];
        for token in path.split_whitespace() {
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }
            if let Some(option) = token.strip_prefix("--") {
                let option = option.split('=').next().unwrap_or(option);
                commands.retain(|command| {
                    command
                        .get_arguments()
                        .any(|argument| argument.get_long() == Some(option))
                });
                assert!(
                    !commands.is_empty(),
                    "文档 CLI 路径 `{path}` 引用了不存在的选项 `--{option}`"
                );
                continue;
            }

            let alternatives = token.split('/').collect::<Vec<_>>();
            let mut next = Vec::new();
            for command in &commands {
                for alternative in &alternatives {
                    if let Some(child) = command.find_subcommand(alternative) {
                        next.push(child.clone());
                    }
                }
            }
            assert!(
                !next.is_empty(),
                "文档 CLI 路径 `{path}` 在真实 Clap 命令树中找不到 `{token}`"
            );
            commands = next;
        }
        commands
    }

    #[test]
    fn every_public_command_and_option_has_a_human_help_description() {
        let command = Cli::command();
        let mut gaps = Vec::new();
        collect_help_gaps(&command, "", &mut gaps);
        assert!(gaps.is_empty(), "{}", gaps.join("\n"));
    }

    #[test]
    fn documented_operations_resolve_to_live_help_and_registered_execution() {
        let operations: Value = serde_json::from_str(OPERATIONS).expect("operations contract");
        let compatibility: Value =
            serde_json::from_str(COMPATIBILITY).expect("compatibility contract");
        let worker_operations = compatibility["operations"]
            .as_object()
            .expect("compatibility worker operation registry");
        let command = Cli::command();
        let catalog = operations["operation_catalog"]["operations"]
            .as_object()
            .expect("operation catalog");

        for (operation_id, declaration) in catalog {
            let cli_path = declaration["current_cli"]
                .as_str()
                .unwrap_or_else(|| panic!("{operation_id} must declare current_cli"));
            let resolved = resolve_documented_cli_path(cli_path, &command);
            assert!(
                resolved.iter().all(|item| {
                    item.get_about()
                        .or_else(|| item.get_long_about())
                        .is_some_and(|about| !about.to_string().trim().is_empty())
                }),
                "{operation_id} resolves to a CLI command without help: {cli_path}"
            );

            match declaration["execution_owner"].as_str() {
                Some("python_worker") => {
                    let worker_operation = declaration["worker_operation"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{operation_id} must declare worker_operation"));
                    assert!(
                        worker_operations.contains_key(worker_operation),
                        "{operation_id} references unregistered worker operation `{worker_operation}`"
                    );
                }
                Some("rust_cli") => assert!(
                    declaration.get("worker_operation").is_none(),
                    "Rust-owned operation {operation_id} must not claim a worker operation"
                ),
                owner => panic!("{operation_id} has invalid execution_owner: {owner:?}"),
            }
        }
    }
}

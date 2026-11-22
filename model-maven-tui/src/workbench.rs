use crate::{
    cli::{CapabilityCheck, Cli, Report},
    scaffold_draft::{CustomScaffoldDraft, DraftStore},
    terminal::{
        layout,
        locale::{LanguageChoice, Locale},
        text,
    },
    welcome::{self, ProjectState},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, OpenOptions},
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    OpenProject,
    ProjectStatus,
    Init,
    CreateClassification,
    CreateCustom,
    Assembly,
    LayerEdit,
    SourceEdit,
    DataPreview,
    Environment,
    Components,
    Supervision,
    SupervisionRules,
    Training,
    Runs,
    Inspect,
    Probe,
    Doctor,
    LastReport,
    LastDiagnosis,
    LastPreview,
    Calls,
    Details,
    Structure,
    Preview,
    Locate,
    Settings,
    Quit,
}

#[derive(Clone, Copy)]
struct Binding {
    key: &'static str,
    action: Action,
}

const DEFAULT_BINDINGS: [Binding; 28] = [
    Binding {
        key: "1",
        action: Action::Inspect,
    },
    Binding {
        key: "2",
        action: Action::Probe,
    },
    Binding {
        key: "3",
        action: Action::Doctor,
    },
    Binding {
        key: "4",
        action: Action::LastReport,
    },
    Binding {
        key: "5",
        action: Action::Details,
    },
    Binding {
        key: "6",
        action: Action::Structure,
    },
    Binding {
        key: "7",
        action: Action::Preview,
    },
    Binding {
        key: "8",
        action: Action::Locate,
    },
    Binding {
        key: "9",
        action: Action::LastDiagnosis,
    },
    Binding {
        key: "0",
        action: Action::LastPreview,
    },
    Binding {
        key: "o",
        action: Action::OpenProject,
    },
    Binding {
        key: "n",
        action: Action::Init,
    },
    Binding {
        key: "b",
        action: Action::CreateClassification,
    },
    Binding {
        key: "m",
        action: Action::CreateCustom,
    },
    Binding {
        key: "a",
        action: Action::Assembly,
    },
    Binding {
        key: "w",
        action: Action::LayerEdit,
    },
    Binding {
        key: "h",
        action: Action::SourceEdit,
    },
    Binding {
        key: "x",
        action: Action::DataPreview,
    },
    Binding {
        key: "y",
        action: Action::Environment,
    },
    Binding {
        key: "z",
        action: Action::Components,
    },
    Binding {
        key: "t",
        action: Action::Supervision,
    },
    Binding {
        key: "d",
        action: Action::SupervisionRules,
    },
    Binding {
        key: "r",
        action: Action::Training,
    },
    Binding {
        key: "j",
        action: Action::Runs,
    },
    Binding {
        key: "c",
        action: Action::ProjectStatus,
    },
    Binding {
        key: "k",
        action: Action::Calls,
    },
    Binding {
        key: "s",
        action: Action::Settings,
    },
    Binding {
        key: "q",
        action: Action::Quit,
    },
];

const SHORTCUT_BINDINGS: [Binding; 28] = [
    Binding {
        key: "i",
        action: Action::Inspect,
    },
    Binding {
        key: "p",
        action: Action::Probe,
    },
    Binding {
        key: "d",
        action: Action::Doctor,
    },
    Binding {
        key: "l",
        action: Action::LastReport,
    },
    Binding {
        key: "v",
        action: Action::Details,
    },
    Binding {
        key: "g",
        action: Action::Structure,
    },
    Binding {
        key: "e",
        action: Action::Preview,
    },
    Binding {
        key: "f",
        action: Action::Locate,
    },
    Binding {
        key: "h",
        action: Action::LastDiagnosis,
    },
    Binding {
        key: "u",
        action: Action::LastPreview,
    },
    Binding {
        key: "o",
        action: Action::OpenProject,
    },
    Binding {
        key: "n",
        action: Action::Init,
    },
    Binding {
        key: "b",
        action: Action::CreateClassification,
    },
    Binding {
        key: "m",
        action: Action::CreateCustom,
    },
    Binding {
        key: "a",
        action: Action::Assembly,
    },
    Binding {
        key: "w",
        action: Action::LayerEdit,
    },
    Binding {
        key: "1",
        action: Action::SourceEdit,
    },
    Binding {
        key: "x",
        action: Action::DataPreview,
    },
    Binding {
        key: "y",
        action: Action::Environment,
    },
    Binding {
        key: "z",
        action: Action::Components,
    },
    Binding {
        key: "t",
        action: Action::Supervision,
    },
    Binding {
        key: "2",
        action: Action::SupervisionRules,
    },
    Binding {
        key: "r",
        action: Action::Training,
    },
    Binding {
        key: "j",
        action: Action::Runs,
    },
    Binding {
        key: "c",
        action: Action::ProjectStatus,
    },
    Binding {
        key: "k",
        action: Action::Calls,
    },
    Binding {
        key: "s",
        action: Action::Settings,
    },
    Binding {
        key: "q",
        action: Action::Quit,
    },
];

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Style {
    #[default]
    Default,
    Compact,
    Story,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Preset {
    #[default]
    Default,
    Shortcuts,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Preferences {
    pub style: Style,
    pub preset: Preset,
}

impl Preferences {
    pub(crate) fn path() -> Option<PathBuf> {
        if let Some(path) =
            std::env::var_os("MODEL_MAVEN_TUI_PREFERENCES").filter(|path| !path.is_empty())
        {
            return Some(PathBuf::from(path));
        }
        #[cfg(windows)]
        let root = std::env::var_os("APPDATA").map(PathBuf::from);
        #[cfg(target_os = "macos")]
        let root = std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library/Application Support"));
        #[cfg(all(unix, not(target_os = "macos")))]
        let root = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|path| Path::new(path).is_absolute())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")));
        #[cfg(not(any(unix, windows)))]
        let root: Option<PathBuf> = None;
        root.map(|root| root.join("model-maven-tui").join("preferences.json"))
    }

    pub(crate) fn load(path: &Path) -> io::Result<Self> {
        Self::read(path)
    }

    fn load_language(path: &Path) -> io::Result<LanguageChoice> {
        let document: serde_json::Map<String, Value> = Self::read(path)?;
        document
            .get("language")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map(|value| value.unwrap_or_default())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    fn save_language(path: &Path, language: LanguageChoice) -> io::Result<()> {
        let mut document: serde_json::Map<String, Value> = Self::read(path)?;
        document.insert(
            "language".into(),
            serde_json::to_value(language).map_err(io::Error::other)?,
        );
        write_preferences_document(path, &document)
    }

    pub(crate) fn load_for_diagnosis(path: &Path) -> io::Result<(Self, bool)> {
        fn setting<T: DeserializeOwned + Default>(entry: Option<Value>) -> (T, bool) {
            match entry.map(serde_json::from_value).transpose() {
                Ok(entry) => (entry.unwrap_or_default(), false),
                Err(_) => (T::default(), true),
            }
        }
        let mut document: serde_json::Map<String, Value> = Self::read(path)?;
        let (style, style_invalid) = setting(document.remove("style"));
        let (preset, preset_invalid) = setting(document.remove("preset"));
        Ok((Self { style, preset }, style_invalid || preset_invalid))
    }

    fn read<T: DeserializeOwned + Default>(path: &Path) -> io::Result<T> {
        match fs::read(path) {
            Ok(data) => serde_json::from_slice(&data)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(T::default()),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn save(&self, path: &Path) -> io::Result<()> {
        let mut document: serde_json::Map<String, Value> = match Self::read(path) {
            Ok(document) => document,
            Err(error) if error.kind() == io::ErrorKind::InvalidData => serde_json::Map::new(),
            Err(error) => return Err(error),
        };
        document.insert(
            "style".into(),
            serde_json::to_value(self.style).map_err(io::Error::other)?,
        );
        document.insert(
            "preset".into(),
            serde_json::to_value(self.preset).map_err(io::Error::other)?,
        );
        write_preferences_document(path, &document)
    }

    pub(crate) fn cycle_style(&mut self) {
        self.style = match self.style {
            Style::Default => Style::Compact,
            Style::Compact => Style::Story,
            Style::Story => Style::Default,
        };
    }

    pub(crate) fn cycle_preset(&mut self) {
        self.preset = match self.preset {
            Preset::Default => Preset::Shortcuts,
            Preset::Shortcuts => Preset::Default,
        };
    }
}

pub(crate) fn preferred_locale() -> Locale {
    Preferences::path()
        .as_deref()
        .and_then(|path| Preferences::load_language(path).ok())
        .unwrap_or_default()
        .resolve()
}

fn write_preferences_document(
    path: &Path,
    document: &serde_json::Map<String, Value>,
) -> io::Result<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let directory = path.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "missing preferences directory")
    })?;
    let data = serde_json::to_vec_pretty(document).map_err(io::Error::other)?;
    fs::create_dir_all(directory)?;
    let temp = directory.join(format!(
        ".preferences-{}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct WorkbenchKeymaps {
    default: BTreeMap<String, String>,
    shortcuts: BTreeMap<String, String>,
}
const MENU_KEYS: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
];
impl WorkbenchKeymaps {
    fn map(&self, preset: Preset) -> &BTreeMap<String, String> {
        match preset {
            Preset::Default => &self.default,
            Preset::Shortcuts => &self.shortcuts,
        }
    }
    fn map_mut(&mut self, preset: Preset) -> &mut BTreeMap<String, String> {
        match preset {
            Preset::Default => &mut self.default,
            Preset::Shortcuts => &mut self.shortcuts,
        }
    }
    fn bindings(&self, preset: Preset) -> Result<Vec<Binding>, String> {
        let overrides = self.map(preset);
        let defaults = preset.bindings();
        for (id, value) in overrides {
            let Some(binding) = defaults.iter().find(|binding| binding.action.id() == id) else {
                return Err(format!("Unknown action: {id}"));
            };
            if matches!(binding.action, Action::Quit | Action::Settings) {
                return Err(format!("Protected action: {id}"));
            }
            if value == "b" || value == "q" || !MENU_KEYS.contains(&value.as_str()) {
                return Err(format!("Reserved or invalid key: {value}"));
            }
        }
        let mut used = HashMap::new();
        let mut result = Vec::with_capacity(defaults.len());
        for binding in defaults {
            let key = overrides
                .get(binding.action.id())
                .map_or(binding.key, String::as_str);
            let key = MENU_KEYS
                .iter()
                .copied()
                .find(|candidate| *candidate == key)
                .ok_or_else(|| format!("Invalid key: {key}"))?;
            if let Some(previous) = used.insert(key, binding.action.id()) {
                return Err(format!("Key {key} is already used by {previous}"));
            }
            result.push(Binding {
                key,
                action: binding.action,
            });
        }
        Ok(result)
    }
    fn load(path: &Path) -> io::Result<Self> {
        let document: serde_json::Map<String, Value> = Preferences::read(path)?;
        let keymaps: Self = document
            .get("workbench_keys")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .unwrap_or_default();
        for preset in [Preset::Default, Preset::Shortcuts] {
            keymaps
                .bindings(preset)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        }
        Ok(keymaps)
    }
    fn save(&self, path: &Path, preferences: Preferences) -> io::Result<()> {
        let mut document: serde_json::Map<String, Value> = match Preferences::read(path) {
            Ok(document) => document,
            Err(error) if error.kind() == io::ErrorKind::InvalidData => serde_json::Map::new(),
            Err(error) => return Err(error),
        };
        document.insert(
            "style".into(),
            serde_json::to_value(preferences.style).map_err(io::Error::other)?,
        );
        document.insert(
            "preset".into(),
            serde_json::to_value(preferences.preset).map_err(io::Error::other)?,
        );
        document.insert(
            "workbench_keys".into(),
            serde_json::to_value(self).map_err(io::Error::other)?,
        );
        write_preferences_document(path, &document)
    }
}

impl Preset {
    fn bindings(self) -> &'static [Binding] {
        match self {
            Self::Default => &DEFAULT_BINDINGS,
            Self::Shortcuts => &SHORTCUT_BINDINGS,
        }
    }
}

impl Action {
    fn id(self) -> &'static str {
        match self {
            Self::OpenProject => "open_project",
            Self::ProjectStatus => "project_status",
            Self::Init => "init",
            Self::CreateClassification => "create_classification",
            Self::CreateCustom => "create_custom",
            Self::Assembly => "assembly",
            Self::LayerEdit => "layer_edit",
            Self::SourceEdit => "source_edit",
            Self::DataPreview => "data_preview",
            Self::Environment => "environment",
            Self::Components => "components",
            Self::Supervision => "supervision",
            Self::SupervisionRules => "supervision_rules",
            Self::Training => "training",
            Self::Runs => "runs",
            Self::Inspect => "inspect",
            Self::Probe => "probe",
            Self::Doctor => "doctor",
            Self::LastReport => "last_report",
            Self::LastDiagnosis => "last_diagnosis",
            Self::LastPreview => "last_preview",
            Self::Calls => "calls",
            Self::Details => "details",
            Self::Structure => "structure",
            Self::Preview => "preview",
            Self::Locate => "locate",
            Self::Settings => "settings",
            Self::Quit => "quit",
        }
    }

    fn label(self, locale: Locale) -> &'static str {
        match self {
            Self::OpenProject => locale.text("Open project", "打开项目", "開啟專案"),
            Self::ProjectStatus => locale.text(
                "Project and CLI status",
                "项目与 CLI 状态",
                "專案與 CLI 狀態",
            ),
            Self::Init => locale.text("Set up project", "设置项目入口", "設定專案入口"),
            Self::CreateClassification => locale.text(
                "Create classification model",
                "创建分类模型",
                "建立分類模型",
            ),
            Self::CreateCustom => {
                locale.text("Create custom model", "创建自定义模型", "建立自訂模型")
            }
            Self::Assembly => locale.text("Assemble components", "装配组件", "裝配元件"),
            Self::LayerEdit => locale.text("Edit model layers", "编辑模型层", "編輯模型層"),
            Self::SourceEdit => locale.text(
                "Edit Python source layers",
                "编辑 Python 源码层",
                "編輯 Python 原始碼層",
            ),
            Self::DataPreview => locale.text("Preview data", "预览数据", "預覽資料"),
            Self::Environment => locale.text("Project environment", "项目环境", "專案環境"),
            Self::Components => locale.text("Project components", "项目组件", "專案元件"),
            Self::Supervision => locale.text(
                "Mark a port and add a rule",
                "标记端口并添加约束",
                "標記連接埠並新增約束",
            ),
            Self::SupervisionRules => {
                locale.text("Manage supervision rules", "管理监督规则", "管理監督規則")
            }
            Self::Training => locale.text("Start training", "启动训练", "啟動訓練"),
            Self::Runs => locale.text("Training runs", "训练记录", "訓練記錄"),
            Self::Inspect => locale.text("Inspect model structure", "检查模型结构", "檢查模型結構"),
            Self::Probe => locale.text("Probe a batch", "前向诊断", "前向診斷"),
            Self::Doctor => locale.text(
                "Check Python and PyTorch",
                "检查 Python 和 PyTorch",
                "檢查 Python 和 PyTorch",
            ),
            Self::Settings => locale.text("Settings", "设置", "設定"),
            Self::LastReport => locale.text("Last result", "最近结果", "最近結果"),
            Self::LastDiagnosis => locale.text("Last diagnosis", "最近诊断", "最近診斷"),
            Self::LastPreview => locale.text(
                "Last preview / receipt",
                "最近预览 / 回执",
                "最近預覽 / 回執",
            ),
            Self::Calls => locale.text("Locate recorded calls", "定位已记录调用", "定位已記錄呼叫"),
            Self::Details => locale.text("View item details", "查看条目详情", "檢視項目詳情"),
            Self::Structure => locale.text("3D structure", "3D 模型结构", "3D 模型結構"),
            Self::Preview => locale.text(
                "Preview Linear input size",
                "预览 Linear 输入维度",
                "預覽 Linear 輸入維度",
            ),
            Self::Locate => locale.text(
                "Locate last observed call",
                "定位最后观测调用",
                "定位最後觀測呼叫",
            ),
            Self::Quit => locale.text("Quit", "退出", "離開"),
        }
    }

    fn description(self, locale: Locale) -> &'static str {
        match self {
            Self::OpenProject => locale.text(
                "Choose another project directory.",
                "选择另一个项目目录。",
                "選擇另一個專案目錄。",
            ),
            Self::ProjectStatus => locale.text(
                "Review or retry the project check.",
                "查看或重试项目检查。",
                "檢視或重試專案檢查。",
            ),
            Self::Init => locale.text(
                "Preview factories, or enter :preset to create a classification model.",
                "预览已有工厂，或输入 :preset 创建分类模型。",
                "預覽現有工廠，或輸入 :preset 建立分類模型。",
            ),
            Self::CreateClassification => locale.text(
                "Create a model from labeled image folders.",
                "从分类图片目录创建模型。",
                "從分類圖片目錄建立模型。",
            ),
            Self::CreateCustom => locale.text(
                "Start with named inputs and editable Python seams.",
                "从具名输入和可编辑 Python 接缝开始。",
                "從具名輸入和可編輯 Python 接縫開始。",
            ),
            Self::Assembly => locale.text(
                "Preview and bind model factory seams.",
                "预览并绑定模型工厂接缝。",
                "預覽並綁定模型工廠接縫。",
            ),
            Self::LayerEdit => locale.text(
                "Preview and edit layers in a project variant.",
                "预览并编辑项目变体中的模型层。",
                "預覽並編輯專案變體中的模型層。",
            ),
            Self::SourceEdit => locale.text(
                "Preview and conditionally apply supported Python source edits.",
                "预览并按条件应用受支持的 Python 源码编辑。",
                "預覽並按條件套用受支援的 Python 原始碼編輯。",
            ),
            Self::DataPreview => locale.text(
                "Preview a batch or dataset in the background.",
                "在后台预览批次或数据集。",
                "在背景預覽批次或資料集。",
            ),
            Self::Environment => locale.text(
                "Check, probe or explicitly prepare the project environment.",
                "检查、探测或明确准备项目环境。",
                "檢查、探測或明確準備專案環境。",
            ),
            Self::Components => locale.text(
                "Find, inspect and validate project components.",
                "查找、查看和验证项目组件。",
                "尋找、檢視和驗證專案元件。",
            ),
            Self::Supervision => locale.text(
                "Attach meaning and preview a rule on an observed port.",
                "为已观察端口标记语义并预览规则。",
                "為已觀察連接埠標記語義並預覽規則。",
            ),
            Self::SupervisionRules => locale.text(
                "Inspect or remove saved rules.",
                "查看或移除已保存的规则。",
                "檢視或移除已儲存的規則。",
            ),
            Self::Training => locale.text(
                "Review inputs and start a trial or background run.",
                "确认输入并启动试跑或后台训练。",
                "確認輸入並啟動試跑或背景訓練。",
            ),
            Self::Runs => locale.text(
                "Inspect, stop or resume saved runs.",
                "查看、停止或续跑已有运行。",
                "檢視、停止或續跑現有執行。",
            ),
            Self::Inspect => locale.text(
                "Load and explore the model's modules.",
                "查看模型中的模块与层级。",
                "檢視模型中的模組與層級。",
            ),
            Self::Probe => locale.text(
                "Run a representative batch through the model.",
                "用代表批次观察前向调用。",
                "用代表批次觀察前向呼叫。",
            ),
            Self::Doctor => locale.text(
                "Review the Python and PyTorch environment.",
                "查看 Python 和 PyTorch 环境。",
                "檢視 Python 和 PyTorch 環境。",
            ),
            Self::LastReport => locale.text(
                "Return to the latest result in this session.",
                "回到本次会话的最近结果。",
                "回到本次工作階段的最近結果。",
            ),
            Self::LastDiagnosis => locale.text(
                "Return to the retained probe report.",
                "回到保留的前向诊断报告。",
                "回到保留的前向診斷報告。",
            ),
            Self::LastPreview => locale.text(
                "Review the retained preview or commit receipt from this session.",
                "查看本次会话保留的预览或提交回执。",
                "檢視本次工作階段保留的預覽或提交回執。",
            ),
            Self::Calls => locale.text(
                "Choose a recorded invocation in the inspected structure.",
                "在已检查结构中选择一次已记录调用。",
                "在已檢查結構中選擇一次已記錄呼叫。",
            ),
            Self::Details => locale.text(
                "Explore the reported modules or calls.",
                "查看报告中的模块或调用详情。",
                "檢視報告中的模組或呼叫詳情。",
            ),
            Self::Structure => locale.text(
                "Drag to orbit and hover to inspect model layers.",
                "在立体视图中观察并选择模型层。",
                "在立體視圖中觀察並選擇模型層。",
            ),
            Self::Preview => locale.text(
                "Preview a change to the last observed Linear call.",
                "预览最后观测到的 Linear 调用的参数修改。",
                "預覽最後觀測到的 Linear 呼叫的參數修改。",
            ),
            Self::Locate => locale.text(
                "Open the inspected structure at the observed module.",
                "在已检查结构中定位观测到的模块。",
                "在已檢查結構中定位觀測到的模組。",
            ),
            Self::Settings => locale.text(
                "Choose the interface style and controls.",
                "选择界面风格和操作方式。",
                "選擇介面風格和操作方式。",
            ),
            Self::Quit => locale.text("Close the workbench.", "离开工作台。", "離開工作臺。"),
        }
    }
}

struct MenuPage<'a> {
    project: &'a std::path::Path,
    project_state: ProjectState,
    variant: &'a str,
    bindings: &'a [Binding],
    context: Option<&'a (String, Option<String>)>,
    has_report: bool,
    has_diagnosis: bool,
    diagnosis_retry_issue: Option<RetryIssue>,
    has_saved_preview: bool,
    has_details: bool,
    has_structure: bool,
    has_preview: bool,
    has_location: bool,
    has_calls: bool,
}

impl MenuPage<'_> {
    fn next_action(&self) -> Action {
        match self.project_state {
            _ if self.diagnosis_retry_issue.is_some() => Action::Probe,
            ProjectState::Configured if self.has_diagnosis => Action::LastDiagnosis,
            ProjectState::Configured => Action::Probe,
            ProjectState::Unconfigured => Action::Init,
            ProjectState::NeedsReview | ProjectState::Unavailable => Action::ProjectStatus,
        }
    }

    fn available(&self, action: Action) -> bool {
        match action {
            Action::Init | Action::CreateClassification | Action::CreateCustom => {
                matches!(self.project_state, ProjectState::Unconfigured)
            }
            Action::SupervisionRules
            | Action::SourceEdit
            | Action::Assembly
            | Action::LayerEdit
            | Action::DataPreview
            | Action::Environment
            | Action::Components => {
                matches!(self.project_state, ProjectState::Configured)
            }
            Action::Training => matches!(self.project_state, ProjectState::Configured),
            Action::Supervision => self.has_diagnosis && self.diagnosis_retry_issue.is_none(),
            Action::LastReport => self.has_report,
            Action::LastDiagnosis => self.has_diagnosis,
            Action::LastPreview => self.has_saved_preview,
            Action::Details => self.has_details,
            Action::Preview => self.has_preview,
            Action::Locate => self.has_location,
            Action::Calls => self.has_calls,
            _ => true,
        }
    }

    fn action(&self, input: &str) -> Option<Action> {
        if input.is_empty() {
            let next = self.next_action();
            return self
                .bindings
                .iter()
                .find(|binding| binding.action == next && self.available(binding.action))
                .or_else(|| {
                    self.bindings
                        .iter()
                        .find(|binding| self.available(binding.action))
                })
                .map(|binding| binding.action);
        }
        self.bindings
            .iter()
            .find(|binding| {
                binding.key.eq_ignore_ascii_case(input) && self.available(binding.action)
            })
            .map(|binding| binding.action)
    }
}

enum Page {
    Menu,
    Settings,
    Details,
    Structure,
    Preview,
    SavedPreview,
    Locate,
    Project { draft: String },
    ProjectStatus,
    Calls,
    Initialize,
    CreateClassification,
    CreateCustom,
    Assembly,
    LayerEdit,
    SourceEdit,
    DataPreview,
    Environment,
    Components,
    Supervision,
    SupervisionRules,
    Training,
    Runs,
    Model { action: Action, draft: String },
    Batch { model: String, draft: String },
    Report,
}

struct Workbench {
    page: Page,
    style: Style,
    preset: Preset,
    variant: String,
    variant_revision: Option<String>,
    last_result: Option<io::Result<Report>>,
    last_command: Option<Command>,
    diagnosis: Option<RecordedDiagnosis>,
    diagnosis_retry_issue: Option<RetryIssue>,
    saved_preview: Option<PreviewRecord>,
    saved_receipt: Option<PreviewRecord>,
    report_focus: ReportFocus,
    form_context: Option<(String, Option<String>)>,
    probe_options_draft: Option<ProbeOptions>,
    model_context: Option<(String, BTreeMap<String, Value>)>,
    model_draft: Option<BTreeMap<String, Value>>,
    model_draft_owner: Option<String>,
    inspected_model_params: BTreeMap<String, Value>,
    inspected_structure: Option<(String, Vec<Value>)>,
    call_focus: usize,
    call_query: String,
    call_views: HashMap<usize, crate::graph::FocusedViewState>,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct ProbeOptions {
    binding: String,
    fingerprint: String,
    batch_params: BTreeMap<String, Value>,
    model_params: BTreeMap<String, Value>,
    training: bool,
    capture_snapshot: bool,
}

#[derive(Clone)]
struct Command {
    action: Action,
    model: String,
    batch: String,
    variant: String,
    probe_options: ProbeOptions,
}

struct RecordedDiagnosis {
    command: Command,
    report: Report,
    batch_comparison: Option<BatchComparison>,
    snapshot: Option<tempfile::TempDir>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchComparison {
    Same,
    Different,
    Unknown,
}

fn compare_batch_ids(previous: &Report, current: &Report) -> BatchComparison {
    let stable_id = |report: &Report| {
        (report.data["batch_identity_stable"] == true)
            .then(|| {
                report.data["batch_id"]
                    .as_str()
                    .filter(|id| !id.trim().is_empty())
                    .map(str::to_owned)
            })
            .flatten()
    };
    match (stable_id(previous), stable_id(current)) {
        (Some(a), Some(b)) if a == b => BatchComparison::Same,
        (Some(_), Some(_)) => BatchComparison::Different,
        _ => BatchComparison::Unknown,
    }
}

pub(crate) fn reported_training_mode(mode: &Value) -> Option<bool> {
    match mode.as_str() {
        Some("train") => Some(true),
        Some("eval") => Some(false),
        _ => None,
    }
}

pub(crate) fn mode_mismatch(mode: &Value, requested_training: bool) -> bool {
    reported_training_mode(mode).is_some_and(|actual| actual != requested_training)
}

#[derive(Debug, PartialEq, Eq)]
enum InputContentComparison {
    Same,
    Different,
    Unknown,
}

fn compare_input_content(diagnosis: &Value, preview: &Value) -> InputContentComparison {
    fn digest(value: &Value) -> Option<&str> {
        if value["status"] != "available" || value["algorithm"] != "sha256-forward-input-v1" {
            return None;
        }
        value["digest"].as_str().filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    }
    match (digest(diagnosis), digest(preview)) {
        (Some(left), Some(right)) if left == right => InputContentComparison::Same,
        (Some(_), Some(_)) => InputContentComparison::Different,
        _ => InputContentComparison::Unknown,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallSearchResult {
    Call(usize),
    Diagnostic(usize),
}

struct ReportDiagnostic {
    operation: String,
    item: crate::cli::Diagnostic,
    raw: std::sync::Arc<str>,
}

impl CallSearchResult {
    fn key(self) -> String {
        match self {
            Self::Call(index) => (index + 1).to_string(),
            Self::Diagnostic(index) => format!("d{}", index + 1),
        }
    }
}

fn select_line_call(
    input: &str,
    matches: &mut Vec<CallSearchResult>,
    count: usize,
) -> Option<usize> {
    let input = input.trim();
    if let Some(index) = input
        .strip_prefix('@')
        .and_then(|n| n.parse::<usize>().ok())
        .and_then(|n| n.checked_sub(1))
        .filter(|&index| index < count)
    {
        let target = CallSearchResult::Call(index);
        return Some(
            matches
                .iter()
                .position(|&item| item == target)
                .unwrap_or_else(|| {
                    matches.push(target);
                    matches.len() - 1
                }),
        );
    }
    matches
        .iter()
        .position(|result| result.key().eq_ignore_ascii_case(input))
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct CallNesting {
    depth: usize,
    incomplete: bool,
}

/// Only link a recorded parent when its invocation ID is unique, earlier and
/// consistent with any shared execution identities. Nesting is not tensor flow.
fn call_nesting(calls: &[Value]) -> Vec<CallNesting> {
    let mut ids: HashMap<&str, Option<usize>> = HashMap::new();
    for (index, call) in calls.iter().enumerate() {
        if let Some(id) = call["invocation_id"].as_str().filter(|id| !id.is_empty()) {
            ids.entry(id)
                .and_modify(|entry| *entry = None)
                .or_insert(Some(index));
        }
    }
    let mut result: Vec<CallNesting> = Vec::with_capacity(calls.len());
    for (index, call) in calls.iter().enumerate() {
        let parent = call.get("parent_invocation_id");
        let nesting = match parent {
            None | Some(Value::Null) => CallNesting::default(),
            Some(Value::String(id)) if !id.is_empty() => {
                let candidate = ids
                    .get(id.as_str())
                    .copied()
                    .flatten()
                    .filter(|&parent_index| parent_index < index)
                    .filter(|&parent_index| {
                        ["forward_id", "probe_id", "batch_id"].iter().all(|field| {
                            match (calls[parent_index][*field].as_str(), call[*field].as_str()) {
                                (Some(a), Some(b)) => a == b,
                                _ => true,
                            }
                        })
                    });
                match candidate {
                    Some(parent_index) => CallNesting {
                        depth: result[parent_index].depth.saturating_add(1),
                        incomplete: result[parent_index].incomplete,
                    },
                    None => CallNesting {
                        depth: 0,
                        incomplete: true,
                    },
                }
            }
            Some(_) => CallNesting {
                depth: 0,
                incomplete: true,
            },
        };
        result.push(nesting);
    }
    result
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetryIssue {
    TimedOut,
    Cancelled,
}

struct PreviewRecord {
    content: String,
    #[cfg(unix)]
    diff: Option<crate::graph::diff::Diff>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReportFocus {
    Latest,
    Diagnosis,
}

impl Command {
    fn execute(&self, cli: &Cli) -> io::Result<Report> {
        self.execute_with_snapshot(cli, None)
    }

    fn execute_with_snapshot(&self, cli: &Cli, snapshot: Option<&str>) -> io::Result<Report> {
        let batch_params: Vec<String> = self
            .probe_options
            .batch_params
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        let model_params: Vec<String> = self
            .probe_options
            .model_params
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        let mut args = match self.action {
            Action::Inspect => vec!["model", "inspect"],
            Action::Probe => vec!["model", "probe"],
            Action::Doctor => vec!["doctor"],
            Action::Init => vec!["init", "--dry-run"],
            Action::LastReport
            | Action::CreateClassification
            | Action::CreateCustom
            | Action::Assembly
            | Action::LayerEdit
            | Action::SourceEdit
            | Action::DataPreview
            | Action::Environment
            | Action::Components
            | Action::Supervision
            | Action::SupervisionRules
            | Action::Training
            | Action::Runs
            | Action::LastDiagnosis
            | Action::LastPreview
            | Action::OpenProject
            | Action::ProjectStatus
            | Action::Calls
            | Action::Details
            | Action::Structure
            | Action::Preview
            | Action::Locate
            | Action::Settings
            | Action::Quit => {
                unreachable!()
            }
        };
        if !self.model.is_empty() {
            args.extend(["--model", &self.model]);
        }
        if !self.batch.is_empty() {
            args.extend(["--batch", &self.batch]);
        }
        if matches!(self.action, Action::Inspect | Action::Probe) && !self.variant.is_empty() {
            args.extend(["--variant", &self.variant]);
        }
        if matches!(self.action, Action::Inspect | Action::Probe) {
            for parameter in &model_params {
                args.extend(["--model-param", parameter]);
            }
        }
        if self.action == Action::Probe {
            if let Some(snapshot) = snapshot {
                args.extend(["--capture-input-to", snapshot]);
            }
            if self.probe_options.training {
                args.push("--training");
            }
            for parameter in &batch_params {
                args.extend(["--batch-param", parameter]);
            }
            if !self.probe_options.binding.is_empty() {
                args.extend(["--batch-binding", &self.probe_options.binding]);
            }
            if !self.probe_options.fingerprint.is_empty() {
                args.extend(["--batch-fingerprint", &self.probe_options.fingerprint]);
            }
        }
        let report = cli.execute(&args)?;
        let expected = match self.action {
            Action::Inspect => "inspect",
            Action::Probe => "probe",
            Action::Doctor => "doctor",
            Action::Init => "init",
            _ => unreachable!(),
        };
        if report.status == "success" && report.operation != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "CLI protocol error: expected {expected} report, received {}",
                    text::plain_text(&report.operation)
                ),
            ));
        }
        Ok(report)
    }

    fn announce(
        &self,
        renderer: &Renderer,
        target: Option<&str>,
        cancellable: bool,
    ) -> io::Result<()> {
        let mut output = String::new();
        renderer.heading(
            &mut output,
            renderer
                .locale
                .text("Running CLI", "CLI 正在运行", "CLI 正在執行"),
        );
        renderer.line(&mut output, self.action.label(renderer.locale));
        if matches!(
            self.action,
            Action::Inspect | Action::Probe | Action::Preview | Action::Init
        ) {
            let model = if self.model.is_empty() {
                renderer
                    .locale
                    .text("Project default", "项目默认值", "專案預設值")
            } else {
                &self.model
            };
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text("Model", "模型", "模型"),
                    text::plain_text(model)
                ),
            );
        }
        if matches!(self.action, Action::Probe | Action::Preview | Action::Init) {
            let batch = if self.batch.is_empty() {
                renderer
                    .locale
                    .text("Project default", "项目默认值", "專案預設值")
            } else {
                &self.batch
            };
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text("Batch", "批次", "批次"),
                    text::plain_text(batch)
                ),
            );
        }
        if matches!(
            self.action,
            Action::Inspect | Action::Probe | Action::Preview
        ) {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text("Variant", "变体", "變體"),
                    if self.variant.is_empty() {
                        renderer.locale.text("CLI default", "CLI 默认", "CLI 預設")
                    } else {
                        &self.variant
                    }
                ),
            );
        }
        if self.action == Action::Probe
            && (!self.probe_options.binding.is_empty()
                || !self.probe_options.fingerprint.is_empty())
        {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer
                        .locale
                        .text("Batch binding", "批次绑定", "批次繫結"),
                    if self.probe_options.binding.is_empty() {
                        "auto"
                    } else {
                        &self.probe_options.binding
                    }
                ),
            );
            if !self.probe_options.fingerprint.is_empty() {
                renderer.line(
                    &mut output,
                    &format!(
                        "{}  {}",
                        renderer
                            .locale
                            .text("Dataset fingerprint", "数据集指纹", "資料集指紋"),
                        text::plain_text(&self.probe_options.fingerprint)
                    ),
                );
            }
        }
        if matches!(
            self.action,
            Action::Inspect | Action::Probe | Action::Preview
        ) && !self.probe_options.model_params.is_empty()
        {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text(
                        "Model factory parameters",
                        "模型工厂参数",
                        "模型工廠參數"
                    ),
                    self.probe_options.model_params.len()
                ),
            );
        }
        if matches!(self.action, Action::Probe | Action::Preview)
            && !self.probe_options.batch_params.is_empty()
        {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text(
                        "Batch factory parameters",
                        "批次工厂参数",
                        "批次工廠參數"
                    ),
                    self.probe_options.batch_params.len()
                ),
            );
        }
        if matches!(self.action, Action::Probe | Action::Preview) {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer
                        .locale
                        .text("Requested mode", "请求模式", "請求模式"),
                    if self.probe_options.training {
                        "train"
                    } else {
                        "eval"
                    }
                ),
            );
        }
        if self.action == Action::Probe && self.probe_options.capture_snapshot {
            renderer.line(&mut output, renderer.locale.text(
                "A private temporary file will contain input values; abnormal termination may leave it behind.",
                "临时私有文件将包含输入值；异常退出可能留下文件。",
                "暫存私有檔案將包含輸入值；異常結束可能留下檔案。",
            ));
        }
        if let Some(target) = target {
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    renderer.locale.text("Target", "目标", "目標"),
                    text::plain_text(target)
                ),
            );
        }
        #[cfg(any(unix, windows))]
        if cancellable && io::stdin().is_terminal() {
            renderer.line(
                &mut output,
                renderer.locale.text(
                    "Ctrl+C cancels this operation",
                    "Ctrl+C 取消本次操作",
                    "Ctrl+C 取消本次操作",
                ),
            );
        }
        #[cfg(any(unix, windows))]
        if !cancellable && io::stdin().is_terminal() {
            renderer.line(
                &mut output,
                renderer.locale.text(
                    "Saving; Ctrl+C waits for completion",
                    "正在保存；Ctrl+C 不会中断保存",
                    "正在儲存；Ctrl+C 不會中斷儲存",
                ),
            );
        }
        #[cfg(not(any(unix, windows)))]
        let _ = cancellable;
        let mut stderr = io::stderr().lock();
        stderr.write_all(output.as_bytes())?;
        stderr.flush()
    }
}

fn execute_and_record(
    workbench: &mut Workbench,
    command: Command,
    cli: &Cli,
    renderer: &Renderer,
) -> io::Result<()> {
    command.announce(renderer, None, true)?;
    let mut temporary = None;
    let result = if command.action == Action::Probe && command.probe_options.capture_snapshot {
        match cli.supports_input_snapshot() {
            Ok(true) => match tempfile::Builder::new()
                .prefix("model-maven-input-")
                .tempdir()
            {
                Ok(directory) => {
                    let path = directory.path().join("input.snapshot");
                    let result = match path.to_str() {
                        Some(path) => command.execute_with_snapshot(cli, Some(path)),
                        None => Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "temporary snapshot path is not UTF-8",
                        )),
                    };
                    temporary = Some(directory);
                    result
                }
                Err(error) => Err(error),
            },
            Ok(false) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "CLI does not advertise input snapshot capture and replay",
            )),
            Err(error) => Err(error),
        }
    } else {
        command.execute(cli)
    };
    let captured = result.as_ref().is_ok_and(|report| {
        report.operation == "probe"
            && report.data["batch_snapshot"]["status"] == "captured"
            && report.data["batch_snapshot"]["format"] == "model-maven-forward-input-snapshot-v1"
            && report.data["batch_content"]["status"] == "available"
            && report.data["batch_content"]["algorithm"] == "sha256-forward-input-v1"
            && report.data["batch_content"]["digest"]
                .as_str()
                .is_some_and(|digest| {
                    digest.len() == 64
                        && digest
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
    });
    workbench.record_result(command, result);
    if captured
        && let Some(directory) = temporary
        && directory.path().join("input.snapshot").is_file()
        && let Some(diagnosis) = &mut workbench.diagnosis
    {
        diagnosis.snapshot = Some(directory);
    }
    Ok(())
}

fn preview_alias<'a>(call: &'a Value, modules: Option<&'a [Value]>) -> Option<&'a str> {
    fn valid_alias(value: &Value) -> Option<&str> {
        value.as_str().filter(|path| !path.trim().is_empty())
    }
    let aliases = call["alias_paths"].as_array()?;
    if let Some(modules) = modules {
        let mut matches = modules.iter().filter_map(|module| {
            let path = module["path"]
                .as_str()
                .filter(|path| !path.trim().is_empty())?;
            aliases
                .iter()
                .filter_map(valid_alias)
                .any(|alias| alias == path)
                .then_some(module)
        });
        let module = matches.next()?;
        if matches.next().is_some() || module["type"].as_str() != Some("Linear") {
            return None;
        }
        return module["path"].as_str();
    }
    aliases.iter().find_map(valid_alias)
}

impl Workbench {
    fn clear_project_context(&mut self) {
        self.page = Page::Menu;
        self.last_result = None;
        self.last_command = None;
        self.diagnosis = None;
        self.diagnosis_retry_issue = None;
        self.saved_preview = None;
        self.saved_receipt = None;
        self.report_focus = ReportFocus::Latest;
        self.form_context = None;
        self.probe_options_draft = None;
        self.model_context = None;
        self.model_draft = None;
        self.model_draft_owner = None;
        self.inspected_model_params.clear();
        self.inspected_structure = None;
        self.call_focus = 0;
        self.call_query.clear();
        self.call_views.clear();
    }

    fn displayed_report(&self) -> Option<&Report> {
        match self.report_focus {
            ReportFocus::Latest => self.last_result.as_ref()?.as_ref().ok(),
            ReportFocus::Diagnosis => self.diagnosis.as_ref().map(|diagnosis| &diagnosis.report),
        }
    }

    fn displayed_command(&self) -> Option<&Command> {
        match self.report_focus {
            ReportFocus::Latest => self.last_command.as_ref(),
            ReportFocus::Diagnosis => self.diagnosis.as_ref().map(|diagnosis| &diagnosis.command),
        }
    }

    fn observed_location(&self) -> Option<&str> {
        if self.diagnosis_retry_issue.is_some() {
            return None;
        }
        let diagnosis = self.diagnosis.as_ref()?;
        let report = &diagnosis.report;
        if report.operation != "probe" || report.status != "error" {
            return None;
        }
        let aliases = report.data["failure_location"]["last_observed_module_call"]["alias_paths"]
            .as_array()?;
        let (model, modules) = self.inspected_structure.as_ref()?;
        if model.is_empty() || model != &diagnosis.command.model {
            return None;
        }
        let mut matches = modules.iter().filter_map(|module| {
            let path = module["path"].as_str()?;
            aliases
                .iter()
                .any(|alias| alias.as_str() == Some(path))
                .then_some(path)
        });
        let path = matches.next()?;
        matches.next().is_none().then_some(path)
    }

    fn recorded_calls(&self) -> Option<&Vec<Value>> {
        if self.diagnosis_retry_issue.is_some() {
            return None;
        }
        let diagnosis = self.diagnosis.as_ref()?;
        let (model, _) = self.inspected_structure.as_ref()?;
        if model.is_empty()
            || model != &diagnosis.command.model
            || diagnosis.report.operation != "probe"
        {
            return None;
        }
        diagnosis.report.data["calls"].as_array()
    }

    fn call_location<'a>(&'a self, call: &Value) -> Option<&'a str> {
        self.recorded_calls()?;
        let aliases = call["alias_paths"].as_array()?;
        let (_, modules) = self.inspected_structure.as_ref()?;
        let mut matches = modules.iter().filter_map(|module| {
            let path = module["path"].as_str()?;
            aliases
                .iter()
                .any(|alias| alias.as_str() == Some(path))
                .then_some(path)
        });
        let path = matches.next()?;
        matches.next().is_none().then_some(path)
    }

    fn preview_target(&self) -> Option<&str> {
        if self.diagnosis_retry_issue.is_some() {
            return None;
        }
        let diagnosis = self.diagnosis.as_ref()?;
        let report = &diagnosis.report;
        let command = &diagnosis.command;
        let call = &report.data["failure_location"]["last_observed_module_call"];
        if command.action != Action::Probe
            || report.operation != "probe"
            || report.status != "error"
            || call["module_type"].as_str()? != "Linear"
            || mode_mismatch(
                &report.data["observed_mode"],
                command.probe_options.training,
            )
        {
            return None;
        }
        if let Some((model, modules)) = self.inspected_structure.as_ref() {
            if model.is_empty() || model != &command.model {
                return None;
            }
            return preview_alias(call, Some(modules));
        }
        preview_alias(call, None)
    }
    fn record_result(&mut self, command: Command, result: io::Result<Report>) {
        if matches!(command.action, Action::Inspect | Action::Probe) {
            self.call_focus = 0;
            self.call_query.clear();
            self.call_views.clear();
        }
        if matches!(command.action, Action::Inspect | Action::Probe) {
            let batch = if command.action == Action::Probe {
                Some(command.batch.clone())
            } else {
                self.form_context
                    .as_ref()
                    .filter(|(model, _)| {
                        model == &command.model
                            && self.model_context.as_ref().is_some_and(|(name, params)| {
                                name == &command.model
                                    && params == &command.probe_options.model_params
                            })
                    })
                    .and_then(|(_, batch)| batch.clone())
            };
            self.form_context = Some((command.model.clone(), batch));
            self.model_context = Some((
                command.model.clone(),
                command.probe_options.model_params.clone(),
            ));
        }
        if matches!(command.action, Action::Inspect | Action::Probe) {
            self.saved_preview = None;
        }
        if command.action == Action::Inspect
            && self.diagnosis.as_ref().is_some_and(|diagnosis| {
                command.model.is_empty()
                    || diagnosis.command.model != command.model
                    || diagnosis.command.variant != command.variant
                    || diagnosis.command.probe_options.model_params
                        != command.probe_options.model_params
            })
        {
            self.diagnosis = None;
            self.diagnosis_retry_issue = None;
        }
        if command.action == Action::Probe {
            let retain_previous = result.as_ref().is_err_and(|error| {
                matches!(
                    error.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
                )
            }) && !command.model.is_empty()
                && !command.batch.is_empty()
                && self.diagnosis.as_ref().is_some_and(|diagnosis| {
                    diagnosis.command.model == command.model
                        && diagnosis.command.batch == command.batch
                        && diagnosis.command.variant == command.variant
                        && diagnosis.command.probe_options == command.probe_options
                });
            if retain_previous {
                self.diagnosis_retry_issue = Some(
                    if result
                        .as_ref()
                        .is_err_and(|error| error.kind() == io::ErrorKind::TimedOut)
                    {
                        RetryIssue::TimedOut
                    } else {
                        RetryIssue::Cancelled
                    },
                );
            } else {
                let previous = self.diagnosis.as_ref().filter(|diagnosis| {
                    !command.model.is_empty()
                        && !command.batch.is_empty()
                        && diagnosis.command.model == command.model
                        && diagnosis.command.batch == command.batch
                        && diagnosis.command.variant == command.variant
                        && diagnosis.command.probe_options == command.probe_options
                });
                self.diagnosis = result
                    .as_ref()
                    .ok()
                    .filter(|report| report.operation == "probe")
                    .map(|report| RecordedDiagnosis {
                        command: command.clone(),
                        report: report.clone(),
                        batch_comparison: previous
                            .map(|old| compare_batch_ids(&old.report, report)),
                        snapshot: None,
                    });
                self.diagnosis_retry_issue = None;
            }
        }
        if matches!(command.action, Action::Inspect | Action::Probe) {
            if self.inspected_structure.as_ref().is_some_and(|(model, _)| {
                model != &command.model
                    || self.inspected_model_params != command.probe_options.model_params
                    || (command.action == Action::Probe && model.is_empty())
            }) {
                self.inspected_structure = None;
                self.inspected_model_params.clear();
            }
            if command.action == Action::Inspect {
                self.inspected_structure = result
                    .as_ref()
                    .ok()
                    .filter(|report| report.operation == "inspect" && report.status == "success")
                    .and_then(|report| report.data["modules"].as_array())
                    .filter(|modules| !modules.is_empty())
                    .map(|modules| (command.model.clone(), modules.clone()));
                self.inspected_model_params = if self.inspected_structure.is_some() {
                    command.probe_options.model_params.clone()
                } else {
                    BTreeMap::new()
                };
            }
        }
        self.last_result = Some(result);
        self.last_command = Some(command);
        self.report_focus = ReportFocus::Latest;
    }

    fn select(&mut self, action: Action) -> Option<Command> {
        match action {
            Action::ProjectStatus => {
                self.page = Page::ProjectStatus;
                None
            }
            Action::Calls => {
                if self.recorded_calls().is_some_and(|calls| !calls.is_empty()) {
                    self.page = Page::Calls;
                }
                None
            }
            Action::Init => {
                self.page = Page::Initialize;
                None
            }
            Action::CreateClassification => {
                self.page = Page::CreateClassification;
                None
            }
            Action::CreateCustom => {
                self.page = Page::CreateCustom;
                None
            }
            Action::Assembly => {
                self.page = Page::Assembly;
                None
            }
            Action::LayerEdit => {
                self.page = Page::LayerEdit;
                None
            }
            Action::SourceEdit => {
                self.page = Page::SourceEdit;
                None
            }
            Action::DataPreview => {
                self.page = Page::DataPreview;
                None
            }
            Action::Environment => {
                self.page = Page::Environment;
                None
            }
            Action::Components => {
                self.page = Page::Components;
                None
            }
            Action::Supervision => {
                self.page = Page::Supervision;
                None
            }
            Action::SupervisionRules => {
                self.page = Page::SupervisionRules;
                None
            }
            Action::Training => {
                self.page = Page::Training;
                None
            }
            Action::Runs => {
                self.page = Page::Runs;
                None
            }
            Action::OpenProject => {
                self.page = Page::Project {
                    draft: String::new(),
                };
                None
            }
            Action::Inspect | Action::Probe => {
                self.probe_options_draft = None;
                self.model_draft = None;
                self.model_draft_owner = None;
                self.page = Page::Model {
                    action,
                    draft: self
                        .form_context
                        .as_ref()
                        .map(|(model, _)| model.clone())
                        .unwrap_or_default(),
                };
                None
            }
            Action::Doctor => Some(Command {
                action,
                model: String::new(),
                batch: String::new(),
                variant: String::new(),
                probe_options: ProbeOptions::default(),
            }),
            Action::Settings => {
                self.page = Page::Settings;
                None
            }
            Action::LastReport => {
                if self.last_result.is_some() {
                    self.report_focus = ReportFocus::Latest;
                    self.page = Page::Report;
                }
                None
            }
            Action::LastDiagnosis => {
                if self.diagnosis.is_some() {
                    self.report_focus = ReportFocus::Diagnosis;
                    self.page = Page::Report;
                }
                None
            }
            Action::LastPreview => {
                if self.saved_preview.is_some() || self.saved_receipt.is_some() {
                    self.page = Page::SavedPreview;
                }
                None
            }
            Action::Details => {
                if self.detail_items().is_some_and(|items| !items.is_empty()) {
                    self.page = Page::Details;
                }
                None
            }
            Action::Structure => {
                self.page = Page::Structure;
                None
            }
            Action::Preview => {
                if self.preview_target().is_some() {
                    self.page = Page::Preview;
                }
                None
            }
            Action::Locate => {
                if self.observed_location().is_some() {
                    self.page = Page::Locate;
                }
                None
            }
            Action::Quit => None,
        }
    }

    fn detail_items(&self) -> Option<&Vec<Value>> {
        let report = self.displayed_report()?;
        match report.operation.as_str() {
            "inspect" => report.data["modules"].as_array(),
            "probe" => report.data["calls"].as_array(),
            _ => None,
        }
    }

    fn submit(&mut self, value: String) -> Option<Command> {
        match &self.page {
            Page::Model {
                action: Action::Probe,
                ..
            } => {
                let owner = self.model_draft_owner.take();
                let params = self
                    .model_draft
                    .take()
                    .filter(|_| owner.as_deref().is_none_or(|owner| owner == value))
                    .or_else(|| {
                        self.model_context
                            .as_ref()
                            .filter(|(model, _)| model == &value)
                            .map(|(_, params)| params.clone())
                    })
                    .unwrap_or_default();
                let draft = self
                    .form_context
                    .as_ref()
                    .filter(|(model, _)| {
                        model == &value
                            && self.model_context.as_ref().is_some_and(|(name, previous)| {
                                name == &value && *previous == params
                            })
                    })
                    .and_then(|(_, batch)| batch.clone())
                    .unwrap_or_default();
                self.model_draft = Some(params);
                self.model_draft_owner = Some(value.clone());
                self.page = Page::Batch {
                    model: value,
                    draft,
                };
                None
            }
            Page::Model {
                action: Action::Inspect,
                ..
            } => {
                let owner = self.model_draft_owner.take();
                let model_params = self
                    .model_draft
                    .take()
                    .filter(|_| owner.as_deref().is_none_or(|owner| owner == value))
                    .or_else(|| {
                        self.model_context
                            .as_ref()
                            .filter(|(model, _)| model == &value)
                            .map(|(_, params)| params.clone())
                    })
                    .unwrap_or_default();
                Some(Command {
                    action: Action::Inspect,
                    model: value,
                    batch: String::new(),
                    variant: self.variant.clone(),
                    probe_options: ProbeOptions {
                        model_params,
                        ..ProbeOptions::default()
                    },
                })
            }
            Page::Batch { model, .. } => {
                let model_params = self.model_draft.take().unwrap_or_default();
                self.model_draft_owner = None;
                let mut options = self
                    .probe_options_draft
                    .take()
                    .or_else(|| {
                        self.diagnosis
                            .as_ref()
                            .filter(|d| {
                                d.command.model == *model
                                    && d.command.batch == value
                                    && d.command.variant == self.variant
                                    && d.command.probe_options.model_params == model_params
                            })
                            .map(|d| d.command.probe_options.clone())
                    })
                    .unwrap_or_default();
                options.model_params = model_params;
                Some(Command {
                    action: Action::Probe,
                    model: model.clone(),
                    probe_options: options,
                    batch: value,
                    variant: self.variant.clone(),
                })
            }
            _ => None,
        }
    }
}

struct Renderer {
    locale: Locale,
    width: usize,
    style: Style,
    line_mode: bool,
}

impl Renderer {
    fn current(locale: Locale, style: Style, line_mode: bool) -> Self {
        Self {
            locale,
            width: layout::content_width(layout::terminal_width()),
            style,
            line_mode,
        }
    }

    fn menu(&self, page: &MenuPage<'_>, selected: usize) -> String {
        #[cfg(unix)]
        let rows = crate::terminal::menu::rows();
        #[cfg(not(unix))]
        let rows = 24;
        self.menu_with_rows(page, selected, rows)
    }

    fn menu_with_rows(&self, page: &MenuPage<'_>, selected: usize, rows: usize) -> String {
        let locale = self.locale;
        let mut header = String::from("\n");
        self.heading(
            &mut header,
            locale.text(
                "Model-Maven Workbench",
                "Model-Maven 工作台",
                "Model-Maven 工作臺",
            ),
        );
        self.line(
            &mut header,
            &text::truncate(&page.project.display().to_string(), self.width),
        );
        let status = match page.project_state {
            ProjectState::Configured => {
                locale.text("Project configured", "项目已配置", "專案已設定")
            }
            ProjectState::Unconfigured => {
                locale.text("Project not configured", "项目尚未配置", "專案尚未設定")
            }
            ProjectState::NeedsReview => locale.text(
                "Project configuration needs review",
                "项目配置待检查",
                "專案設定待檢查",
            ),
            ProjectState::Unavailable => {
                locale.text("Project status unavailable", "项目状态未知", "專案狀態未知")
            }
        };
        self.line(&mut header, status);
        let next = page.next_action();
        if let Some(binding) = page
            .bindings
            .iter()
            .find(|binding| binding.action == next && page.available(binding.action))
        {
            let line = format!(
                "{}  {}. {}",
                locale.text("Next", "下一步", "下一步"),
                binding.key,
                binding.action.label(locale)
            );
            self.line(&mut header, &text::truncate(&line, self.width));
        }
        self.line(
            &mut header,
            &text::truncate(
                &format!(
                    "{}  {}",
                    locale.text("Variant", "变体", "變體"),
                    if page.variant.is_empty() {
                        locale.text("CLI default", "CLI 默认", "CLI 預設")
                    } else {
                        page.variant
                    }
                ),
                self.width,
            ),
        );
        let unset = locale.text("Not selected", "尚未选择", "尚未選擇");
        let default = locale.text("Project default", "项目默认值", "專案預設值");
        let model = page.context.map(|(model, _)| model.as_str());
        let batch = page.context.and_then(|(_, batch)| batch.as_deref());
        for (label, entry) in [
            (locale.text("Model", "模型", "模型"), model),
            (locale.text("Batch", "批次", "批次"), batch),
        ] {
            let value = match entry {
                None => unset,
                Some("") => default,
                Some(value) => value,
            };
            let line = format!("{label}  {}", text::plain_text(value));
            self.line(&mut header, &text::truncate(&line, self.width));
        }
        if self.style == Style::Default {
            self.line(&mut header, &"─".repeat(self.width.min(48)));
        }
        let bindings: Vec<_> = page
            .bindings
            .iter()
            .filter(|binding| page.available(binding.action))
            .collect();
        let mut guide = String::new();
        if self.style == Style::Story {
            self.line(
                &mut guide,
                &format!(
                    "╭─ {} {}",
                    locale.text("GUIDE", "导航", "導覽"),
                    "─".repeat(self.width.saturating_sub(10).min(40))
                ),
            );
            if let Some(binding) = bindings.get(selected) {
                self.line(
                    &mut guide,
                    &format!("│ {}", binding.action.description(locale)),
                );
            }
            self.line(&mut guide, "╰────────────────────");
        }
        let mut footer = String::from("\n");
        self.line(
            &mut footer,
            if self.line_mode {
                locale.text(
                    "Enter: Next · key + Enter: choose · q: Quit",
                    "回车：下一步 · 按键加回车：选择 · q：退出",
                    "Enter：下一步 · 按鍵加 Enter：選擇 · q：離開",
                )
            } else {
                locale.text(
                    "↑/↓ Move · Tab/Shift+Tab · Enter · q",
                    "↑/↓ 移动 · Tab/Shift+Tab · Enter · q",
                    "↑/↓ 移動 · Tab/Shift+Tab · Enter · q",
                )
            },
        );
        self.line(
            &mut footer,
            locale.text("/ Search commands", "/ 搜索操作", "/ 搜尋操作"),
        );
        // Keep the focused action and scroll indicators visible when the
        // additional session context leaves too little room for the guide.
        let minimum_menu = if bindings.len() > 1 {
            3
        } else {
            bindings.len()
        };
        if !self.line_mode
            && header.lines().count()
                + guide.lines().count()
                + footer.lines().count()
                + minimum_menu
                > rows
        {
            guide.clear();
        }
        let reserved = header.lines().count() + guide.lines().count() + footer.lines().count();
        let available = if self.line_mode {
            bindings.len()
        } else {
            rows.saturating_sub(reserved)
        };
        let (start, end) = if bindings.len() <= available {
            (0, bindings.len())
        } else {
            let visible = available.saturating_sub(2).max(1).min(bindings.len());
            let selected = selected.min(bindings.len().saturating_sub(1));
            let start = selected
                .saturating_add(1)
                .saturating_sub(visible)
                .min(bindings.len().saturating_sub(visible));
            (start, start + visible)
        };
        let mut menu = header;
        if start > 0 {
            self.line(&mut menu, "  ↑ …");
        }
        for (index, binding) in bindings.iter().enumerate().take(end).skip(start) {
            let focus = if (self.line_mode && binding.action == page.next_action())
                || (!self.line_mode && index == selected)
            {
                "> "
            } else {
                "  "
            };
            let label = if binding.action == Action::Structure && self.line_mode {
                locale.text("List inspected modules", "列出已检查模块", "列出已檢查模組")
            } else if binding.action == Action::Structure && !page.has_structure {
                locale.text(
                    "3D structure · Drifting sample",
                    "3D 模型结构 · Drifting 样例",
                    "3D 模型結構 · Drifting 範例",
                )
            } else {
                binding.action.label(locale)
            };
            let row = text::truncate(&format!("{focus}{}. {label}", binding.key), self.width);
            if !self.line_mode
                && index == selected
                && layout::color_enabled(io::stderr().is_terminal())
            {
                menu.push_str("\x1b[1;36m");
                self.line(&mut menu, &row);
                menu.push_str("\x1b[0m");
            } else {
                self.line(&mut menu, &row);
            }
        }
        if end < bindings.len() {
            self.line(&mut menu, "  ↓ …");
        }
        menu.push_str(&guide);
        menu.push_str(&footer);
        menu
    }

    fn prompt(&self, page: &Page) -> Option<&'static str> {
        match page {
            Page::Menu => Some(self.locale.text("Select", "选择操作", "選擇操作")),
            Page::Settings => Some(self.locale.text("Select", "选择操作", "選擇操作")),
            Page::Details => Some(self.locale.text(
                "Item number (b to return)",
                "条目序号（b 返回）",
                "項目序號（b 返回）",
            )),
            Page::Structure => None,
            Page::Preview => None,
            Page::SavedPreview => None,
            Page::Locate => None,
            Page::Project { .. } => Some(if self.line_mode {
                self.locale.text(
                    "Project directory (relative to current project; b back)",
                    "项目目录（相对路径以当前项目为基准；b 返回）",
                    "專案目錄（相對路徑以目前專案為基準；b 返回）",
                )
            } else {
                self.locale.text(
                    "Project directory (relative to current project)",
                    "项目目录（相对路径以当前项目为基准）",
                    "專案目錄（相對路徑以目前專案為基準）",
                )
            }),
            Page::Supervision => None,
            Page::SupervisionRules => None,
            Page::Training | Page::Runs => None,
            Page::LayerEdit => None,
            Page::SourceEdit => None,
            Page::DataPreview => None,
            Page::Environment => None,
            Page::Components => None,
            Page::Initialize | Page::CreateClassification | Page::CreateCustom | Page::Assembly => {
                None
            }
            Page::ProjectStatus => None,
            Page::Calls => None,
            Page::Model { draft, .. } if !draft.is_empty() => Some(if self.line_mode {
                self.locale.text(
                    "Model factory (:model-params; Enter keeps, :default clears, b back)",
                    "模型工厂（:model-params 参数；回车保留，:default 默认，b 返回）",
                    "模型工廠（:model-params 參數；Enter 保留，:default 預設，b 返回）",
                )
            } else {
                self.locale.text(
                    "Model factory (:model-params; Ctrl+U clears)",
                    "模型工厂（:model-params 参数；Ctrl+U 清空）",
                    "模型工廠（:model-params 參數；Ctrl+U 清空）",
                )
            }),
            Page::Model { .. } => Some(if self.line_mode {
                self.locale.text(
                    "Model factory (:model-params; Enter default, b back)",
                    "模型工厂（:model-params 参数；回车默认，b 返回）",
                    "模型工廠（:model-params 參數；Enter 預設，b 返回）",
                )
            } else {
                self.locale.text(
                    "Model factory (:model-params; Enter default)",
                    "模型工厂（:model-params 参数；Enter 默认）",
                    "模型工廠（:model-params 參數；Enter 預設）",
                )
            }),
            Page::Batch { draft, .. } if !draft.is_empty() => Some(if self.line_mode {
                self.locale.text(
                    "Batch factory (:options advanced, :batch-params, :mode, :snapshot; Enter keeps, :default clears, b back)",
                    "批次工厂（:options 高级选项，:batch-params 参数，:mode 模式，:snapshot 快照，回车保留，:default 默认，b 返回）",
                    "批次工廠（:options 進階選項，:batch-params 參數，:mode 模式，:snapshot 快照，Enter 保留，:default 預設，b 返回）",
                )
            } else {
                self.locale.text(
                    "Batch factory (:options advanced; :batch-params, :mode, :snapshot; Ctrl+U clears)",
                    "批次工厂（:options 高级选项；:batch-params 参数；:mode 模式；:snapshot 快照；Ctrl+U 清空）",
                    "批次工廠（:options 進階選項；:batch-params 參數；:mode 模式；:snapshot 快照；Ctrl+U 清空）",
                )
            }),
            Page::Batch { .. } => Some(if self.line_mode {
                self.locale.text(
                    "Batch factory (:options advanced, :batch-params, :mode, :snapshot; Enter default, b back)",
                    "批次工厂（:options 高级选项，:batch-params 参数，:mode 模式，:snapshot 快照，回车默认，b 返回）",
                    "批次工廠（:options 進階選項，:batch-params 參數，:mode 模式，:snapshot 快照，Enter 預設，b 返回）",
                )
            } else {
                self.locale.text(
                    "Batch factory (:options advanced; :batch-params, :mode, :snapshot; Enter default)",
                    "批次工厂（:options 高级选项；:batch-params 参数；:mode 模式；:snapshot 快照；Enter 默认）",
                    "批次工廠（:options 進階選項；:batch-params 參數；:mode 模式；:snapshot 快照；Enter 預設）",
                )
            }),
            Page::Report => None,
        }
    }

    fn form(&self, page: &Page, value: &str, cursor: usize) -> String {
        let mut output = String::new();
        self.heading(
            &mut output,
            self.locale.text(
                "Model-Maven Workbench",
                "Model-Maven 工作台",
                "Model-Maven 工作臺",
            ),
        );
        output.push('\n');
        if let Some(label) = self.prompt(page) {
            self.line(&mut output, label);
        }
        self.line(
            &mut output,
            &text::input_window("> ", value, cursor, self.width),
        );
        output.push('\n');
        self.line(
            &mut output,
            self.locale.text(
                "Enter  Continue    Esc  Back",
                "Enter 继续    Esc 返回",
                "Enter 繼續    Esc 返回",
            ),
        );
        output
    }

    fn settings(
        &self,
        preset: Preset,
        variant: &str,
        language: LanguageChoice,
        selected: usize,
    ) -> String {
        let mut output = String::new();
        self.heading(&mut output, self.locale.text("Settings", "设置", "設定"));
        self.line(
            &mut output,
            self.locale.text(
                "› ─ Default · ▸ ┄ Compact · ◆ ━ Story",
                "› ─ 默认 · ▸ ┄ 紧凑 · ◆ ━ 叙事",
                "› ─ 預設 · ▸ ┄ 緊湊 · ◆ ━ 敘事",
            ),
        );
        let style = match self.style {
            Style::Default => self.locale.text("Default", "默认", "預設"),
            Style::Compact => self.locale.text("Compact", "紧凑", "緊湊"),
            Style::Story => self.locale.text("Story", "叙事", "敘事"),
        };
        let style_cue = match self.style {
            Style::Default => "› ─",
            Style::Compact => "▸ ┄",
            Style::Story => "◆ ━",
        };
        let keys = match preset {
            Preset::Default => self.locale.text("Default", "默认", "預設"),
            Preset::Shortcuts => self.locale.text("Shortcuts", "快捷键", "快速鍵"),
        };
        let language_label = if language.resolve() == self.locale {
            language.label(self.locale).to_owned()
        } else {
            format!(
                "{} · {} {}",
                language.label(self.locale),
                self.locale.text("session", "本次", "本次"),
                self.locale.language_label()
            )
        };
        let entries = [
            format!(
                "1. {}  {style}  {style_cue}",
                self.locale.text("Style", "视觉风格", "視覺風格")
            ),
            format!(
                "2. {}  {keys}",
                self.locale.text("Controls", "操作预设", "操作預設")
            ),
            self.locale
                .text("3. Workbench keys", "3. 工作台键位", "3. 工作台鍵位")
                .to_owned(),
            format!(
                "4. {}  {}",
                self.locale.text("Variant", "变体", "變體"),
                if variant.is_empty() {
                    self.locale.text("CLI default", "CLI 默认", "CLI 預設")
                } else {
                    variant
                }
            ),
            format!(
                "5. {}  {}",
                self.locale.text("Language", "语言", "語言"),
                language_label
            ),
            self.locale
                .text(
                    "r. Reset style and keys",
                    "r. 重置风格和键位",
                    "r. 重設風格與鍵位",
                )
                .to_owned(),
            self.locale.text("b. Back", "b. 返回", "b. 返回").to_owned(),
        ];
        for (index, entry) in entries.iter().enumerate() {
            let marker = if index == selected { "> " } else { "  " };
            let mut row = String::new();
            self.line(&mut row, &format!("{marker}{entry}"));
            if !self.line_mode
                && index == selected
                && layout::color_enabled(io::stderr().is_terminal())
            {
                output.push_str("\x1b[1;36m");
                output.push_str(&row);
                output.push_str("\x1b[0m");
            } else {
                output.push_str(&row);
            }
        }
        output.push('\n');
        self.line(
            &mut output,
            self.locale.text(
                "↑/↓ Move · Tab/Shift+Tab · Enter · b",
                "↑/↓ 移动 · Tab/Shift+Tab · Enter · b",
                "↑/↓ 移動 · Tab/Shift+Tab · Enter · b",
            ),
        );
        output
    }

    fn detail_list(&self, items: &[Value], selected: usize, rows: usize) -> String {
        let mut output = String::new();
        self.heading(
            &mut output,
            self.locale.text("Report items", "报告条目", "報告項目"),
        );
        let visible = rows.saturating_sub(5).max(1);
        let start = selected
            .saturating_sub(visible - 1)
            .min(items.len().saturating_sub(visible));
        for (index, item) in items.iter().enumerate().skip(start).take(visible) {
            let name = item["path"]
                .as_str()
                .or_else(|| {
                    item["alias_paths"]
                        .as_array()
                        .and_then(|paths| paths.iter().find_map(Value::as_str))
                })
                .unwrap_or("—");
            let call = item["invocation_index"]
                .as_u64()
                .map(|n| format!("  #{n}"))
                .unwrap_or_default();
            let module_type = item["module_type"]
                .as_str()
                .or_else(|| item["type"].as_str())
                .map(|kind| format!("  {kind}"))
                .unwrap_or_default();
            let focus = if selected == index { ">" } else { " " };
            self.line(
                &mut output,
                &text::truncate(
                    &format!("{focus} {}. {name}{call}{module_type}", index + 1),
                    self.width,
                ),
            );
        }
        let focus = if selected == items.len() { ">" } else { " " };
        self.line(
            &mut output,
            &format!(
                "{focus} {}",
                self.locale.text("b. Back", "b. 返回", "b. 返回")
            ),
        );
        self.line(
            &mut output,
            self.locale.text(
                "↑/↓ Move · Tab/Shift+Tab · Enter open",
                "↑/↓ 移动 · Tab/Shift+Tab · Enter 查看",
                "↑/↓ 移動 · Tab/Shift+Tab · Enter 檢視",
            ),
        );
        output
    }

    fn call_list(
        &self,
        calls: &[Value],
        nesting: &[CallNesting],
        matches: &[CallSearchResult],
        diagnostics: &[ReportDiagnostic],
        selected: usize,
        rows: usize,
        query: &str,
        limited: bool,
        trace_truncated: bool,
        reused: bool,
    ) -> String {
        let mut output = String::new();
        self.heading(
            &mut output,
            self.locale
                .text("Recorded calls", "已记录调用", "已記錄呼叫"),
        );
        self.line(
            &mut output,
            &text::truncate(
                &format!(
                    "{}  {}  ·  {}/{}",
                    self.locale.text("Filter", "筛选", "篩選"),
                    if query.is_empty() { "—" } else { query },
                    matches.len(),
                    calls.len()
                        + if query.is_empty() {
                            0
                        } else {
                            diagnostics.len()
                        }
                ),
                self.width,
            ),
        );
        if reused {
            self.line(
                &mut output,
                &text::truncate(
                    self.locale.text(
                        "Reused evidence · no new forward",
                        "复用证据 · 本次未执行前向",
                        "重用證據 · 本次未執行前向",
                    ),
                    self.width,
                ),
            );
        }
        if trace_truncated {
            self.line(
                &mut output,
                &text::truncate(
                    self.locale.text(
                        "CLI call trace is truncated",
                        "CLI 调用记录已截断",
                        "CLI 呼叫記錄已截斷",
                    ),
                    self.width,
                ),
            );
        }
        if limited {
            self.line(
                &mut output,
                &text::truncate(
                    self.locale.text(
                        "Filter covers at most 4096 values per call",
                        "每次调用最多筛选 4096 项",
                        "每次呼叫最多篩選 4096 項",
                    ),
                    self.width,
                ),
            );
        }
        let show_nesting = nesting.iter().any(|item| item.depth > 0 || item.incomplete);
        let show_shared = matches.iter().any(|item| match item {
            CallSearchResult::Call(index) => calls
                .get(*index)
                .is_some_and(crate::cli::ambiguous_registration),
            CallSearchResult::Diagnostic(_) => false,
        });
        if show_nesting {
            for line in [
                self.locale.text(
                    "↳ execution nesting · ? unknown · …N",
                    "↳ 执行嵌套 · ? 上级未知 · …N 深度",
                    "↳ 執行巢狀 · ? 上層未知 · …N 深度",
                ),
                self.locale.text(
                    "Nesting does not establish tensor flow",
                    "嵌套不表示张量来源",
                    "巢狀不代表張量來源",
                ),
            ] {
                self.line(&mut output, &text::truncate(line, self.width));
            }
        }
        self.line(
            &mut output,
            &text::truncate(
                self.locale.text(
                    "Snapshot identity not verified against calls",
                    "检查快照身份尚未与调用核对",
                    "檢查快照身分尚未與呼叫核對",
                ),
                self.width,
            ),
        );
        if show_shared {
            self.line(
                &mut output,
                &text::truncate(
                    self.locale.text(
                        "* Shared call · registration unknown",
                        "* 共享调用 · 注册路径未知",
                        "* 共享呼叫 · 註冊路徑未知",
                    ),
                    self.width,
                ),
            );
        }
        let visible = rows
            .saturating_sub(
                7 + usize::from(limited)
                    + usize::from(trace_truncated)
                    + usize::from(reused)
                    + usize::from(show_shared)
                    + 2 * usize::from(show_nesting),
            )
            .max(1);
        let start = selected
            .saturating_sub(visible - 1)
            .min(matches.len().saturating_sub(visible));
        for (position, &result) in matches.iter().enumerate().skip(start).take(visible) {
            let marker = if position == selected { ">" } else { " " };
            let row = match result {
                CallSearchResult::Call(index) => {
                    let item = &calls[index];
                    let name = item["alias_paths"]
                        .as_array()
                        .and_then(|paths| paths.iter().find_map(Value::as_str))
                        .unwrap_or("—");
                    let suffix = item["invocation_index"]
                        .as_u64()
                        .map_or(String::new(), |number| format!("  #{number}"));
                    let level = nesting.get(index).copied().unwrap_or_default();
                    let branch = if level.depth > 0 {
                        format!(
                            "{}{}{}↳ ",
                            if level.incomplete { "? " } else { "" },
                            "  ".repeat(level.depth.saturating_sub(1).min(3)),
                            if level.depth > 4 {
                                format!("…{} ", level.depth)
                            } else {
                                String::new()
                            }
                        )
                    } else if level.incomplete {
                        "? ".to_owned()
                    } else {
                        String::new()
                    };
                    let shared = if crate::cli::ambiguous_registration(item) {
                        "*"
                    } else {
                        ""
                    };
                    let prefix = format!("{marker} {}.{shared} {branch}", index + 1);
                    let budget = self.width.saturating_sub(
                        text::display_width(&prefix) + text::display_width(&suffix),
                    );
                    format!("{prefix}{}{suffix}", path_tail(name, budget))
                }
                CallSearchResult::Diagnostic(index) => {
                    let diagnostic = &diagnostics[index];
                    format!(
                        "{marker} d{}. [{}] {}",
                        index + 1,
                        text::plain_text(&diagnostic.operation),
                        text::plain_text(&diagnostic.item.code)
                    )
                }
            };
            self.line(&mut output, &text::truncate(&row, self.width));
        }
        if matches.is_empty() {
            self.line(
                &mut output,
                self.locale
                    .text("No matching results", "没有匹配的结果", "沒有相符的結果"),
            );
        }
        self.line(
            &mut output,
            &format!(
                "{} {}",
                if selected == matches.len() { ">" } else { " " },
                self.locale.text("b. Back", "b. 返回", "b. 返回")
            ),
        );
        self.line(
            &mut output,
            &format!(
                "{} {}",
                if selected == matches.len() + 1 {
                    ">"
                } else {
                    " "
                },
                self.locale.text(
                    "/. Filter calls and diagnostics",
                    "/. 筛选调用与诊断",
                    "/. 篩選呼叫與診斷"
                )
            ),
        );
        self.line(
            &mut output,
            self.locale.text(
                "↑/↓ move · Enter open · / filter",
                "↑/↓ 移动 · Enter 打开 · / 筛选",
                "↑/↓ 移動 · Enter 開啟 · / 篩選",
            ),
        );
        output
    }

    fn detail(&self, item: &Value) -> String {
        let mut output = String::new();
        self.heading(
            &mut output,
            self.locale.text("Reported details", "报告详情", "報告詳情"),
        );
        self.line(
            &mut output,
            &serde_json::to_string_pretty(item).unwrap_or_default(),
        );
        output
    }

    fn result(&self, result: &io::Result<Report>) -> String {
        match result {
            Ok(report) => self.report(report),
            Err(error) => {
                let mut output = String::new();
                self.line(
                    &mut output,
                    if error.kind() == io::ErrorKind::TimedOut {
                        self.locale
                            .text("CLI operation timed out", "CLI 操作超时", "CLI 操作逾時")
                    } else if error.kind() == io::ErrorKind::Interrupted {
                        self.locale.text(
                            "CLI operation cancelled",
                            "CLI 操作已取消",
                            "CLI 操作已取消",
                        )
                    } else {
                        self.locale
                            .text("Could not run the CLI", "无法执行 CLI", "無法執行 CLI")
                    },
                );
                self.line(&mut output, &error.to_string());
                #[cfg(unix)]
                if error.kind() == io::ErrorKind::TimedOut {
                    self.line(
                        &mut output,
                        self.locale.text(
                            "Retry or adjust MODEL_MAVEN_TUI_CLI_TIMEOUT_MS (0 disables timeout).",
                            "可重试或调整 MODEL_MAVEN_TUI_CLI_TIMEOUT_MS（0 表示不设超时）。",
                            "可重試或調整 MODEL_MAVEN_TUI_CLI_TIMEOUT_MS（0 表示不設逾時）。",
                        ),
                    );
                }
                output
            }
        }
    }

    fn line(&self, output: &mut String, value: &str) {
        for line in text::wrap(value, self.width) {
            output.push_str(&line);
            output.push('\n');
        }
    }

    fn heading(&self, output: &mut String, value: &str) {
        match self.style {
            Style::Default => self.line(output, value),
            Style::Compact => self.line(output, &format!("[ {value} ]")),
            Style::Story => self.line(output, &format!("✦  {value}")),
        }
    }
}

fn read_project_state(cli: &Cli, locale: Locale) -> io::Result<(ProjectState, io::Result<Report>)> {
    {
        let mut stderr = io::stderr().lock();
        writeln!(
            stderr,
            "{}",
            locale.text(
                "Checking project configuration…",
                "正在检查项目配置…",
                "正在檢查專案設定…",
            )
        )?;
        #[cfg(any(unix, windows))]
        if io::stdin().is_terminal() {
            writeln!(
                stderr,
                "{}",
                locale.text(
                    "Ctrl+C cancels this operation",
                    "Ctrl+C 取消本次操作",
                    "Ctrl+C 取消本次操作"
                )
            )?;
        }
        stderr.flush()?;
    }
    let result = cli.execute(&["config", "history"]);
    let state = match &result {
        Ok(report) if report.operation == "config_history" && report.status == "success" => {
            ProjectState::Configured
        }
        Ok(report) if report.operation == "config_history" => match report.diagnostics.first() {
            Some(diagnostic) if diagnostic.code == "PROJECT_NOT_INITIALIZED" => {
                ProjectState::Unconfigured
            }
            Some(diagnostic)
                if matches!(
                    diagnostic.code.as_str(),
                    "PROJECT_CONFIG_INVALID" | "PROJECT_CONFIG_HISTORY_INVALID"
                ) =>
            {
                ProjectState::NeedsReview
            }
            _ => ProjectState::Unavailable,
        },
        _ => ProjectState::Unavailable,
    };
    Ok((state, result))
}

fn resolve_project(current: &Path, input: &str) -> io::Result<PathBuf> {
    let path = Path::new(input);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        current.join(path)
    };
    let path = path.canonicalize()?;
    if !path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a directory",
        ));
    }
    Ok(path)
}

fn same_project_configuration(previous: &io::Result<Report>, current: &io::Result<Report>) -> bool {
    let (Ok(previous), Ok(current)) = (previous, current) else {
        return false;
    };
    if previous.operation != "config_history" || current.operation != "config_history" {
        return false;
    }
    if previous.status == "success" && current.status == "success" {
        return previous.data["current_revision"]
            .as_str()
            .filter(|revision| !revision.is_empty())
            .is_some_and(|revision| current.data["current_revision"].as_str() == Some(revision));
    }
    previous.status == "error"
        && current.status == "error"
        && previous
            .diagnostics
            .first()
            .is_some_and(|d| d.code == "PROJECT_NOT_INITIALIZED")
        && current
            .diagnostics
            .first()
            .is_some_and(|d| d.code == "PROJECT_NOT_INITIALIZED")
}

pub(crate) fn run(cli: &Cli, mut locale: Locale, line_mode: bool) -> io::Result<()> {
    #[cfg(unix)]
    let line_mode = line_mode || !crate::terminal::menu::supported();
    #[cfg(not(unix))]
    let line_mode = {
        let _ = line_mode;
        true
    };
    let mut cli = cli.with_project(cli.project().to_path_buf());
    let mut data_previews = Vec::new();
    let mut trial_tasks: Vec<crate::cli::BackgroundTask> = Vec::new();
    let mut environment_tasks: Vec<EnvironmentTask> = Vec::new();
    let mut component_tasks: Vec<ComponentTask> = Vec::new();
    let mut custom_drafts: HashMap<PathBuf, CustomScaffoldDraft> = HashMap::new();
    let (mut project_state, mut project_check) = read_project_state(&cli, locale)?;
    let Some(handoff) = ({
        #[cfg(unix)]
        let _interrupt = crate::terminal::interrupt::Guard::install();
        if line_mode {
            welcome::show_line(cli.project(), project_state, locale)?
        } else {
            welcome::show(cli.project(), project_state, locale)?
        }
    }) else {
        return Ok(());
    };

    let preferences_path = Preferences::path();
    let mut language = preferences_path
        .as_deref()
        .and_then(|path| Preferences::load_language(path).ok())
        .unwrap_or_default();
    let (preferences, preferences_error) = match preferences_path.as_deref().map(Preferences::load)
    {
        Some(Ok(preferences)) => (preferences, None),
        Some(Err(error)) => (Preferences::default(), Some(error.to_string())),
        None => (
            Preferences::default(),
            Some(
                locale
                    .text(
                        "Preferences directory unavailable",
                        "偏好设置目录不可用",
                        "偏好設定目錄無法使用",
                    )
                    .to_owned(),
            ),
        ),
    };
    let (mut keymaps, keymaps_error) = match preferences_path.as_deref().map(WorkbenchKeymaps::load)
    {
        Some(Ok(keymaps)) => (keymaps, None),
        Some(Err(error)) => (WorkbenchKeymaps::default(), Some(error.to_string())),
        None => (WorkbenchKeymaps::default(), None),
    };
    let mut workbench = Workbench {
        page: Page::Menu,
        style: preferences.style,
        preset: preferences.preset,
        variant: String::new(),
        variant_revision: None,
        last_result: None,
        last_command: None,
        diagnosis: None,
        diagnosis_retry_issue: None,
        saved_preview: None,
        saved_receipt: None,
        report_focus: ReportFocus::Latest,
        form_context: None,
        probe_options_draft: None,
        model_context: None,
        model_draft: None,
        model_draft_owner: None,
        inspected_model_params: BTreeMap::new(),
        call_focus: 0,
        call_query: String::new(),
        call_views: HashMap::new(),
        inspected_structure: None,
    };
    let initial_bindings = keymaps
        .bindings(workbench.preset)
        .expect("validated keymap");
    let initial_page = MenuPage {
        project: cli.project(),
        project_state,
        variant: &workbench.variant,
        bindings: &initial_bindings,
        context: workbench.form_context.as_ref(),
        has_report: false,
        has_diagnosis: false,
        diagnosis_retry_issue: None,
        has_saved_preview: false,
        has_details: false,
        has_structure: false,
        has_preview: false,
        has_location: false,
        has_calls: false,
    };
    let initial_focus = initial_bindings
        .iter()
        .filter(|binding| initial_page.available(binding.action))
        .position(|binding| binding.action == initial_page.next_action())
        .unwrap_or(0);
    handoff.present(
        &Renderer::current(locale, workbench.style, line_mode).menu(&initial_page, initial_focus),
    )?;
    if cli.uses_legacy_transport() {
        let renderer = Renderer::current(locale, workbench.style, line_mode);
        if !show_flow_page(&renderer, locale.text(
            "Historical CLI detected: only inspection and diagnostics are available; upgrade m-model to edit or train. Project status is unavailable. Diagnostics can execute project code.\n",
            "检测到旧版 CLI：仅支持查看和诊断；请升级 m-model 后编辑或训练。项目状态不可用，诊断可能执行项目代码。\n",
            "偵測到舊版 CLI：僅支援檢視和診斷；請升級 m-model 後編輯或訓練。專案狀態不可用，診斷可能執行專案程式碼。\n",
        ))? {
            return Ok(());
        }
    }
    if let Some(error) = preferences_error {
        let renderer = Renderer::current(locale, workbench.style, line_mode);
        let mut output = String::new();
        renderer.line(
            &mut output,
            locale.text(
                "Could not load preferences; using defaults",
                "无法读取偏好设置，已使用默认值",
                "無法讀取偏好設定，已使用預設值",
            ),
        );
        renderer.line(&mut output, &error);
        print!("{output}");
    }

    if let Some(error) = keymaps_error {
        let renderer = Renderer::current(locale, workbench.style, line_mode);
        let mut output = String::new();
        renderer.line(
            &mut output,
            locale.text(
                "Invalid workbench keys; using defaults",
                "工作台键位无效，已使用默认值",
                "工作台鍵位無效，已使用預設值",
            ),
        );
        renderer.line(&mut output, &error);
        print!("{output}");
    }

    let mut menu_focus: Option<Action> = None;
    let mut project_context_reset = false;
    loop {
        let mut environment_changed = false;
        for entry in &mut environment_tasks {
            if entry.mutating && !entry.completion_handled {
                match entry.task.poll() {
                    Ok(false) => {}
                    Ok(true) => {
                        entry.completion_handled = true;
                        environment_changed |= entry.task.project == cli.project();
                    }
                    Err(_) => environment_changed |= entry.task.project == cli.project(),
                }
            }
        }
        for entry in &mut component_tasks {
            if entry.mutating && !entry.completion_handled {
                match entry.task.poll() {
                    Ok(false) => {}
                    Ok(true) => {
                        entry.completion_handled = true;
                        environment_changed |= entry.task.project == cli.project();
                    }
                    Err(_) => environment_changed |= entry.task.project == cli.project(),
                }
            }
        }
        if environment_changed {
            workbench.clear_project_context();
            project_context_reset = true;
        }
        let renderer = Renderer::current(locale, workbench.style, line_mode);
        let mut project_change = None;
        let context = workbench.form_context.clone();
        let selected_variant = workbench.variant.clone();
        let active_bindings = keymaps
            .bindings(workbench.preset)
            .expect("validated keymap");
        let page = MenuPage {
            project: cli.project(),
            project_state,
            variant: &selected_variant,
            bindings: &active_bindings,
            context: context.as_ref(),
            has_report: workbench.last_result.is_some(),
            has_diagnosis: workbench.diagnosis.is_some(),
            diagnosis_retry_issue: workbench.diagnosis_retry_issue,
            has_saved_preview: workbench.saved_preview.is_some()
                || workbench.saved_receipt.is_some(),
            has_details: workbench
                .detail_items()
                .is_some_and(|items| !items.is_empty()),
            has_structure: workbench.inspected_structure.is_some(),
            has_preview: workbench.preview_target().is_some(),
            has_location: workbench.observed_location().is_some(),
            has_calls: workbench
                .recorded_calls()
                .is_some_and(|calls| !calls.is_empty()),
        };
        let command = match &workbench.page {
            Page::Menu => {
                #[cfg(unix)]
                let selected = if !line_mode && crate::terminal::menu::supported() {
                    let bindings: Vec<&Binding> = page
                        .bindings
                        .iter()
                        .filter(|binding| page.available(binding.action))
                        .collect();
                    let keys: Vec<&str> = bindings.iter().map(|binding| binding.key).collect();
                    let initial = bindings
                        .iter()
                        .position(|binding| {
                            binding.action == menu_focus.unwrap_or_else(|| page.next_action())
                        })
                        .unwrap_or(0);
                    match crate::terminal::menu::select_from_with_shortcut(
                        &keys,
                        initial,
                        Some(b'/'),
                        |selected| {
                            Renderer::current(locale, workbench.style, line_mode)
                                .menu(&page, selected)
                        },
                    ) {
                        Ok(Some(index)) if index == keys.len() => {
                            search_workbench_commands(&page, &renderer)?
                        }
                        Ok(Some(index)) => Some(bindings[index].action),
                        Ok(None) => break,
                        Err(_) => {
                            let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())?
                            else {
                                break;
                            };
                            if choice == "/" {
                                search_workbench_commands(&page, &renderer)?
                            } else {
                                page.action(&choice)
                            }
                        }
                    }
                } else {
                    let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                        break;
                    };
                    if choice == "/" {
                        search_workbench_commands(&page, &renderer)?
                    } else {
                        page.action(&choice)
                    }
                };
                #[cfg(not(unix))]
                let selected = {
                    let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                        break;
                    };
                    if choice == "/" {
                        search_workbench_commands(&page, &renderer)?
                    } else {
                        page.action(&choice)
                    }
                };
                match selected {
                    Some(Action::Quit) => break,
                    Some(action) => {
                        if uses_cached_variant_evidence(action, &workbench)
                            && !check_variant_revision(&mut workbench, &cli, &renderer)?
                        {
                            if line_mode {
                                let refreshed = MenuPage {
                                    project: cli.project(),
                                    project_state,
                                    variant: &workbench.variant,
                                    bindings: &active_bindings,
                                    context: None,
                                    has_report: false,
                                    has_diagnosis: false,
                                    diagnosis_retry_issue: None,
                                    has_saved_preview: workbench.saved_receipt.is_some(),
                                    has_details: false,
                                    has_structure: false,
                                    has_preview: false,
                                    has_location: false,
                                    has_calls: false,
                                };
                                print!("{}", renderer.menu(&refreshed, 0));
                            }
                            continue;
                        }
                        menu_focus = Some(action);
                        let command = workbench.select(action);
                        if matches!(workbench.page, Page::Settings) {
                            print!(
                                "{}",
                                renderer.settings(
                                    workbench.preset,
                                    &workbench.variant,
                                    language,
                                    0
                                )
                            );
                        }
                        if matches!(workbench.page, Page::Details) {
                            if let Some(items) = workbench.detail_items() {
                                print!("{}", renderer.detail_list(items, 0, 24));
                            }
                        }
                        command
                    }
                    None => {
                        print!("{}", renderer.menu(&page, 0));
                        None
                    }
                }
            }
            Page::Settings => {
                #[cfg(unix)]
                let choice = if !line_mode && crate::terminal::menu::supported() {
                    let keys = ["1", "2", "3", "4", "5", "r", "b"];
                    match crate::terminal::menu::select(&keys, |selected| {
                        Renderer::current(locale, workbench.style, line_mode).settings(
                            workbench.preset,
                            &workbench.variant,
                            language,
                            selected,
                        )
                    }) {
                        Ok(Some(index)) => keys[index].to_owned(),
                        Ok(None) => break,
                        Err(_) => {
                            let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())?
                            else {
                                break;
                            };
                            choice
                        }
                    }
                } else {
                    let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                        break;
                    };
                    choice
                };
                #[cfg(not(unix))]
                let choice = {
                    let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                        break;
                    };
                    choice
                };
                let selection = choice.to_ascii_lowercase();
                let mut preferences = Preferences {
                    style: workbench.style,
                    preset: workbench.preset,
                };
                match selection.as_str() {
                    "1" => preferences.cycle_style(),
                    "2" => preferences.cycle_preset(),
                    "4" => {
                        if let Some(selected) = choose_variant(&cli, &workbench.variant, &renderer)?
                        {
                            workbench.clear_project_context();
                            workbench.variant = selected.name;
                            workbench.variant_revision = selected.revision;
                            workbench.page = Page::Settings;
                            menu_focus = None;
                        }
                    }
                    "3" => configure_workbench_keys(
                        &mut keymaps,
                        preferences,
                        preferences_path.as_deref(),
                        &renderer,
                    )?,
                    "5" => {
                        if let Some(selected) = choose_language(&renderer)? {
                            language = selected;
                            locale = selected.resolve();
                            let result = preferences_path
                                .as_deref()
                                .ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::NotFound,
                                        "preferences directory unavailable",
                                    )
                                })
                                .and_then(|path| Preferences::save_language(path, selected));
                            if let Err(error) = result {
                                let updated = Renderer::current(locale, workbench.style, line_mode);
                                let mut output = String::new();
                                updated.line(&mut output, locale.text("Language changed for this session, but could not be saved.", "语言已在本次会话切换，但无法保存。", "語言已在本次工作階段切換，但無法儲存。"));
                                updated.line(&mut output, &text::plain_text(&error.to_string()));
                                print!("{output}");
                            }
                        }
                    }
                    "r" => {
                        preferences = Preferences::default();
                        keymaps = WorkbenchKeymaps::default();
                    }
                    "b" | "q" => workbench.page = Page::Menu,
                    _ => {}
                }
                workbench.style = preferences.style;
                workbench.preset = preferences.preset;
                let updated = Renderer::current(locale, workbench.style, line_mode);
                if matches!(selection.as_str(), "1" | "2" | "r") {
                    let result = preferences_path
                        .as_deref()
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::NotFound,
                                "preferences directory unavailable",
                            )
                        })
                        .and_then(|path| {
                            if selection == "r" {
                                keymaps.save(path, preferences)
                            } else {
                                preferences.save(path)
                            }
                        });
                    if let Err(error) = result {
                        let mut output = String::new();
                        updated.line(
                            &mut output,
                            locale.text(
                                "Could not save preferences",
                                "无法保存偏好设置",
                                "無法儲存偏好設定",
                            ),
                        );
                        updated.line(&mut output, &error.to_string());
                        print!("{output}");
                    }
                }
                if matches!(workbench.page, Page::Menu) {
                    let updated_bindings = keymaps
                        .bindings(workbench.preset)
                        .expect("validated keymap");
                    let page = MenuPage {
                        project: cli.project(),
                        project_state,
                        variant: &workbench.variant,
                        bindings: &updated_bindings,
                        context: workbench.form_context.as_ref(),
                        has_report: workbench.last_result.is_some(),
                        has_diagnosis: workbench.diagnosis.is_some(),
                        diagnosis_retry_issue: workbench.diagnosis_retry_issue,
                        has_saved_preview: workbench.saved_preview.is_some()
                            || workbench.saved_receipt.is_some(),
                        has_details: workbench
                            .detail_items()
                            .is_some_and(|items| !items.is_empty()),
                        has_structure: workbench.inspected_structure.is_some(),
                        has_preview: workbench.preview_target().is_some(),
                        has_location: workbench.observed_location().is_some(),
                        has_calls: workbench
                            .recorded_calls()
                            .is_some_and(|calls| !calls.is_empty()),
                    };
                    print!("{}", updated.menu(&page, 0));
                } else {
                    print!(
                        "{}",
                        updated.settings(workbench.preset, &workbench.variant, language, 0)
                    );
                }
                None
            }
            Page::Details => {
                #[cfg(unix)]
                if !line_mode && crate::terminal::menu::supported() {
                    if let Some(items) = workbench.detail_items() {
                        if !browse_details(items, locale, workbench.style)? {
                            break;
                        }
                    }
                    workbench.page = Page::Menu;
                    continue;
                }
                let Some(choice) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                    break;
                };
                if choice.eq_ignore_ascii_case("b") || choice.eq_ignore_ascii_case("q") {
                    workbench.page = Page::Menu;
                    print!("{}", renderer.menu(&page, 0));
                } else if let Some(item) = choice
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| workbench.detail_items().and_then(|items| items.get(index)))
                {
                    print!("{}", renderer.detail(item));
                } else {
                    let mut output = String::new();
                    renderer.line(
                        &mut output,
                        locale.text(
                            "Choose a listed item",
                            "请选择列表中的条目",
                            "請選擇清單中的項目",
                        ),
                    );
                    print!("{output}");
                }
                None
            }
            Page::Structure => {
                let modules = workbench
                    .inspected_structure
                    .as_ref()
                    .map(|(_, modules)| modules);
                if line_mode {
                    let mut content = String::new();
                    renderer.heading(
                        &mut content,
                        locale.text("Inspected modules", "已检查的模块", "已檢查的模組"),
                    );
                    if let Some(modules) = modules.filter(|modules| !modules.is_empty()) {
                        for module in modules {
                            renderer.line(
                                &mut content,
                                &format!(
                                    "{} · {}",
                                    module["path"].as_str().unwrap_or("?"),
                                    module["type"].as_str().unwrap_or("?")
                                ),
                            );
                        }
                    } else {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "No inspected structure available",
                                "没有已检查的结构",
                                "沒有已檢查的結構",
                            ),
                        );
                    }
                    print!("{content}");
                } else {
                    if let Some(selection) = crate::graph::show(modules, locale)? {
                        let graph_model = workbench
                            .inspected_structure
                            .as_ref()
                            .map(|(model, _)| model.clone())
                            .unwrap_or_default();
                        match selection {
                            crate::graph::GraphEdit::Variant(path) => {
                                let name = if workbench.variant.is_empty() {
                                    "working".to_owned()
                                } else {
                                    workbench.variant.clone()
                                };
                                match guide_layer_edit(
                                    &cli,
                                    &renderer,
                                    &name,
                                    Some((&path, &graph_model)),
                                )? {
                                    LayerEditOutcome::Quit => break,
                                    LayerEditOutcome::Cancelled => {}
                                    LayerEditOutcome::Attempted { committed } => {
                                        workbench.clear_project_context();
                                        if committed {
                                            workbench.variant = name;
                                            workbench.variant_revision = None;
                                        }
                                    }
                                }
                            }
                            crate::graph::GraphEdit::Source(path) => {
                                let graph_type = workbench
                                    .inspected_structure
                                    .as_ref()
                                    .and_then(|(_, modules)| {
                                        modules.iter().find(|module| {
                                            module["path"].as_str() == Some(path.as_str())
                                        })
                                    })
                                    .and_then(|module| module["type"].as_str())
                                    .unwrap_or("unknown")
                                    .to_owned();
                                let mut attempted = false;
                                if !guide_source_edit(
                                    &cli,
                                    &renderer,
                                    &mut attempted,
                                    Some((&path, &graph_model, &workbench.variant, &graph_type)),
                                )? {
                                    break;
                                }
                                if attempted {
                                    workbench.clear_project_context();
                                    project_context_reset = true;
                                }
                            }
                        }
                    }
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Preview => {
                let mut attempted_commit = false;
                if let (Some(target), Some(diagnosis)) = (
                    workbench.preview_target().map(str::to_owned),
                    &workbench.diagnosis,
                ) {
                    let evidence = renderer.diagnosis_summary(
                        &diagnosis.report,
                        workbench
                            .inspected_structure
                            .as_ref()
                            .map(|(_, modules)| modules),
                        false,
                        Some(diagnosis.command.probe_options.training),
                    );
                    if !preview_linear(
                        &cli,
                        locale,
                        workbench.style,
                        &target,
                        &diagnosis.command,
                        &evidence,
                        &diagnosis.report,
                        diagnosis.snapshot.as_ref(),
                        workbench
                            .inspected_structure
                            .as_ref()
                            .map(|(_, modules)| modules.as_slice()),
                        &mut workbench.saved_preview,
                        &mut workbench.saved_receipt,
                        line_mode,
                        &mut attempted_commit,
                    )? {
                        break;
                    }
                }
                if attempted_commit {
                    // Even an interrupted or malformed response may follow a write.
                    let receipt = workbench.saved_receipt.take();
                    workbench.clear_project_context();
                    if !workbench.variant.is_empty() {
                        let _ = check_variant_revision(&mut workbench, &cli, &renderer)?;
                    }
                    workbench.saved_receipt = receipt;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    if attempted_commit {
                        let refreshed = MenuPage {
                            project: cli.project(),
                            project_state,
                            variant: &selected_variant,
                            bindings: &active_bindings,
                            context: None,
                            has_report: false,
                            has_diagnosis: false,
                            diagnosis_retry_issue: None,
                            has_saved_preview: workbench.saved_receipt.is_some(),
                            has_details: false,
                            has_structure: false,
                            has_preview: false,
                            has_location: false,
                            has_calls: false,
                        };
                        print!("{}", renderer.menu(&refreshed, 0));
                    } else {
                        print!("{}", renderer.menu(&page, 0));
                    }
                }
                None
            }
            Page::SavedPreview => {
                if workbench.saved_preview.is_some() || workbench.saved_receipt.is_some() {
                    let mut content = String::new();
                    if workbench.saved_preview.is_some() && workbench.saved_receipt.is_some() {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "The preview and receipt below are historical session records.",
                                "下方预览和回执是本次会话的历史记录。",
                                "下方預覽和回執是本次工作階段的歷史記錄。",
                            ),
                        );
                    }
                    if let Some(receipt) = &workbench.saved_receipt {
                        renderer.line(
                        &mut content,
                            locale.text(
                                "Session receipt · historical result; current state has not been revalidated.",
                                "会话回执 · 历史结果；当前状态尚未重新核对。",
                                "工作階段回執 · 歷史結果；目前狀態尚未重新核對。",
                            )
                    );
                        content.push_str(&receipt.content);
                    }
                    if let Some(preview) = &workbench.saved_preview {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "Saved preview from this session; it has not been revalidated.",
                                "本次会话保存的预览；尚未重新验证。",
                                "本次工作階段儲存的預覽；尚未重新驗證。",
                            ),
                        );
                        content.push_str(&preview.content);
                    }
                    #[cfg(unix)]
                    if !line_mode && crate::terminal::menu::supported() {
                        let from_preview = workbench
                            .saved_preview
                            .as_ref()
                            .is_some_and(|record| record.diff.is_some());
                        let mut diff = workbench
                            .saved_preview
                            .as_ref()
                            .and_then(|record| record.diff.clone())
                            .or_else(|| {
                                workbench
                                    .saved_receipt
                                    .as_ref()
                                    .and_then(|record| record.diff.clone())
                            });
                        let shown = view_report_with_diff_state(
                            &content,
                            locale,
                            workbench.style,
                            &mut diff,
                        )?;
                        let record = if from_preview {
                            &mut workbench.saved_preview
                        } else {
                            &mut workbench.saved_receipt
                        };
                        if let Some(record) = record {
                            record.diff = diff;
                        }
                        if !shown {
                            break;
                        }
                    } else {
                        print!("{content}");
                    }
                    #[cfg(not(unix))]
                    print!("{content}");
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Locate => {
                if let (Some(path), Some((_, modules)), Some(diagnosis)) = (
                    workbench.observed_location(),
                    &workbench.inspected_structure,
                    &workbench.diagnosis,
                ) {
                    let report = &diagnosis.report;
                    if line_mode {
                        let mut content = renderer.diagnosis_summary(
                            report,
                            Some(modules),
                            false,
                            Some(diagnosis.command.probe_options.training),
                        );
                        renderer.line(
                            &mut content,
                            &format!(
                                "{}  {path}",
                                locale.text("Matched module", "匹配模块", "符合模組")
                            ),
                        );
                        renderer.line(
                            &mut content,
                            locale.text(
                                "Connections are not shown in line view.",
                                "行式视图不显示连线。",
                                "行式視圖不顯示連線。",
                            ),
                        );
                        print!("{content}");
                    } else {
                        let focused = crate::graph::show_focused(
                            modules,
                            path,
                            &report.data["failure_location"]["last_observed_module_call"],
                            &report.raw_json,
                            true,
                            false,
                            None,
                            None,
                            None,
                            crate::cli::unique_call_registration(
                                &report.data["failure_location"]["last_observed_module_call"],
                                path,
                            ),
                            locale,
                        )?;
                        if let Some(selection) = focused.edit {
                            let graph_model = workbench
                                .inspected_structure
                                .as_ref()
                                .map(|(model, _)| model.clone())
                                .unwrap_or_default();
                            match selection {
                                crate::graph::GraphEdit::Variant(path) => {
                                    let name = if workbench.variant.is_empty() {
                                        "working".to_owned()
                                    } else {
                                        workbench.variant.clone()
                                    };
                                    match guide_layer_edit(
                                        &cli,
                                        &renderer,
                                        &name,
                                        Some((&path, &graph_model)),
                                    )? {
                                        LayerEditOutcome::Quit => break,
                                        LayerEditOutcome::Cancelled => {}
                                        LayerEditOutcome::Attempted { committed } => {
                                            workbench.clear_project_context();
                                            if committed {
                                                workbench.variant = name;
                                                workbench.variant_revision = None;
                                            }
                                        }
                                    }
                                }
                                crate::graph::GraphEdit::Source(path) => {
                                    let graph_type = modules
                                        .iter()
                                        .find(|module| {
                                            module["path"].as_str() == Some(path.as_str())
                                        })
                                        .and_then(|module| module["type"].as_str())
                                        .unwrap_or("unknown")
                                        .to_owned();
                                    let mut attempted = false;
                                    if !guide_source_edit(
                                        &cli,
                                        &renderer,
                                        &mut attempted,
                                        Some((
                                            &path,
                                            &graph_model,
                                            &workbench.variant,
                                            &graph_type,
                                        )),
                                    )? {
                                        break;
                                    }
                                    if attempted {
                                        workbench.clear_project_context();
                                        project_context_reset = true;
                                    }
                                }
                            }
                        }
                    }
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Calls => {
                let style = workbench.style;
                if !browse_recorded_calls(&mut workbench, locale, style, line_mode)? {
                    break;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::ProjectStatus => {
                let mut content = String::new();
                renderer.heading(
                    &mut content,
                    locale.text(
                        "Project and CLI status",
                        "项目与 CLI 状态",
                        "專案與 CLI 狀態",
                    ),
                );
                renderer.line(
                    &mut content,
                    &text::plain_text(&cli.project().display().to_string()),
                );
                if project_context_reset {
                    renderer.line(&mut content, locale.text("Project, components or environment changed or could not be verified; session evidence was cleared.", "项目、组件或环境已改变或无法核对，已清除会话证据。", "專案、元件或環境已變更或無法核對，已清除工作階段證據。"));
                }
                if let Ok(report) = &project_check {
                    if report.operation != "config_history" {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "Unexpected CLI report for project check.",
                                "项目检查收到不匹配的 CLI 报告。",
                                "專案檢查收到不相符的 CLI 報告。",
                            ),
                        );
                        renderer.line(&mut content, &text::plain_text(&report.operation));
                    } else {
                        content.push_str(&renderer.result(&project_check));
                    }
                } else {
                    content.push_str(&renderer.result(&project_check));
                }
                if matches!(project_state, ProjectState::Unavailable) {
                    renderer.line(&mut content, locale.text(
                        "Check the CLI path, then retry; use --cli PATH or MODEL_MAVEN_CLI when starting the TUI.",
                        "检查 CLI 路径后重试；启动 TUI 时可使用 --cli PATH 或 MODEL_MAVEN_CLI。",
                        "檢查 CLI 路徑後重試；啟動 TUI 時可使用 --cli PATH 或 MODEL_MAVEN_CLI。",
                    ));
                }
                #[cfg(unix)]
                if !line_mode && crate::terminal::menu::supported() {
                    if !view_report(&content, locale, workbench.style)? {
                        break;
                    }
                } else {
                    print!("{content}");
                }
                #[cfg(not(unix))]
                print!("{content}");
                #[cfg(unix)]
                let retry = if !line_mode && crate::terminal::menu::supported() {
                    let Some(choice) = crate::terminal::menu::select(&["b", "r"], |selected| {
                        let renderer = Renderer::current(locale, workbench.style, false);
                        let mut output = String::new();
                        renderer.heading(
                            &mut output,
                            locale.text("Project check", "项目检查", "專案檢查"),
                        );
                        renderer.line(
                            &mut output,
                            &format!(
                                "{} {}",
                                if selected == 0 { ">" } else { " " },
                                locale.text("b. Back", "b. 返回", "b. 返回")
                            ),
                        );
                        renderer.line(
                            &mut output,
                            &format!(
                                "{} {}",
                                if selected == 1 { ">" } else { " " },
                                locale.text("r. Retry", "r. 重试", "r. 重試")
                            ),
                        );
                        output
                    })?
                    else {
                        break;
                    };
                    choice == 1
                } else {
                    prompt(locale.text(
                        "r retry · Enter back",
                        "r 重试 · 回车返回",
                        "r 重試 · Enter 返回",
                    ))?
                    .is_some_and(|input| input.trim().eq_ignore_ascii_case("r"))
                };
                #[cfg(not(unix))]
                let retry = prompt(locale.text(
                    "r retry · Enter back",
                    "r 重试 · 回车返回",
                    "r 重試 · Enter 返回",
                ))?
                .is_some_and(|input| input.trim().eq_ignore_ascii_case("r"));
                if retry {
                    let (state, check) = read_project_state(&cli, locale)?;
                    if !same_project_configuration(&project_check, &check)
                        && (workbench.form_context.is_some()
                            || workbench.last_result.is_some()
                            || workbench.inspected_structure.is_some())
                    {
                        workbench.clear_project_context();
                        workbench.page = Page::ProjectStatus;
                        project_context_reset = true;
                    }
                    project_state = state;
                    project_check = check;
                } else {
                    project_context_reset = false;
                    workbench.page = Page::Menu;
                    if line_mode {
                        print!("{}", renderer.menu(&page, 0));
                    }
                }
                None
            }
            Page::Project { draft } => {
                #[cfg(unix)]
                let input = if !line_mode && crate::terminal::menu::supported() {
                    match crate::terminal::menu::input(draft, |value, cursor| {
                        Renderer::current(locale, workbench.style, line_mode).form(
                            &workbench.page,
                            value,
                            cursor,
                        )
                    })? {
                        crate::terminal::menu::Input::Submit(value) => Some(value),
                        crate::terminal::menu::Input::Back => Some(String::new()),
                        crate::terminal::menu::Input::Quit => break,
                    }
                } else {
                    prompt(renderer.prompt(&workbench.page).unwrap())?
                };
                #[cfg(not(unix))]
                let input = prompt(renderer.prompt(&workbench.page).unwrap())?;
                let Some(input) = input else {
                    break;
                };
                if input.trim().is_empty() || (line_mode && input.trim().eq_ignore_ascii_case("b"))
                {
                    workbench.page = Page::Menu;
                    if line_mode {
                        print!("{}", renderer.menu(&page, 0));
                    }
                } else {
                    match resolve_project(cli.project(), &input) {
                        Ok(path) => project_change = Some(path),
                        Err(error) => {
                            let mut message = String::new();
                            renderer.heading(
                                &mut message,
                                locale.text(
                                    "Could not open project",
                                    "无法打开项目",
                                    "無法開啟專案",
                                ),
                            );
                            renderer.line(&mut message, &text::plain_text(&error.to_string()));
                            #[cfg(unix)]
                            if !line_mode && crate::terminal::menu::supported() {
                                if !view_report(&message, locale, workbench.style)? {
                                    break;
                                }
                            } else {
                                print!("{message}");
                            }
                            #[cfg(not(unix))]
                            print!("{message}");
                            workbench.page = Page::Project { draft: input };
                        }
                    }
                }
                None
            }
            Page::LayerEdit => {
                let name = if workbench.variant.is_empty() {
                    "working".to_owned()
                } else {
                    workbench.variant.clone()
                };
                match guide_layer_edit(&cli, &renderer, &name, None)? {
                    LayerEditOutcome::Quit => break,
                    LayerEditOutcome::Cancelled => {}
                    LayerEditOutcome::Attempted { committed } => {
                        workbench.clear_project_context();
                        if committed {
                            workbench.variant = name;
                            workbench.variant_revision = None;
                        }
                    }
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::SourceEdit => {
                let mut attempted = false;
                if !guide_source_edit(&cli, &renderer, &mut attempted, None)? {
                    break;
                }
                if attempted {
                    workbench.clear_project_context();
                    project_context_reset = true;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::DataPreview => {
                let mut mutation_attempted = false;
                if !guide_data_preview(
                    &cli,
                    &renderer,
                    &mut data_previews,
                    &mut environment_tasks,
                    &mut mutation_attempted,
                )? {
                    break;
                }
                if mutation_attempted {
                    workbench.clear_project_context();
                    project_context_reset = true;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Environment => {
                let mut mutation_started = false;
                if !guide_environment(
                    &cli,
                    &renderer,
                    &mut environment_tasks,
                    &mut data_previews,
                    &mut component_tasks,
                    &mut mutation_started,
                )? {
                    break;
                }
                if mutation_started {
                    workbench.clear_project_context();
                    project_context_reset = true;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Components => {
                let mut mutation_started = false;
                if !guide_components(
                    &cli,
                    &renderer,
                    &mut component_tasks,
                    &mut environment_tasks,
                    &mut mutation_started,
                )? {
                    break;
                }
                if mutation_started {
                    workbench.clear_project_context();
                    project_context_reset = true;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Supervision => {
                if let Some(diagnosis) = workbench.diagnosis.as_ref() {
                    if !guide_supervision(&cli, &renderer, diagnosis)? {
                        break;
                    }
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::SupervisionRules => {
                let mut attempted = false;
                if !guide_supervision_rules(&cli, &renderer, &mut attempted)? {
                    break;
                }
                if attempted {
                    workbench.clear_project_context();
                    project_context_reset = true;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Training => {
                if !launch_training(&cli, &renderer, &workbench.variant, &mut trial_tasks)? {
                    break;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Runs => {
                if !manage_runs(&cli, &renderer, &mut trial_tasks)? {
                    break;
                }
                workbench.page = Page::Menu;
                if line_mode {
                    print!("{}", renderer.menu(&page, 0));
                }
                None
            }
            Page::Initialize | Page::CreateClassification | Page::CreateCustom | Page::Assembly => {
                let assembly_action = matches!(workbench.page, Page::Assembly);
                let outcome = if matches!(workbench.page, Page::CreateClassification) {
                    initialize_classification(&cli, &renderer)?
                } else if matches!(workbench.page, Page::CreateCustom) {
                    initialize_custom(
                        &cli,
                        &renderer,
                        custom_drafts
                            .entry(cli.project().to_path_buf())
                            .or_default(),
                    )?
                } else if matches!(workbench.page, Page::Assembly) {
                    configure_assembly(&cli, &renderer, workbench.form_context.as_ref())?
                } else {
                    initialize_project(
                        &cli,
                        locale,
                        workbench.style,
                        line_mode,
                        workbench.form_context.as_ref(),
                    )?
                };
                match outcome {
                    InitOutcome::Cancelled => {
                        workbench.page = Page::Menu;
                        if line_mode {
                            print!("{}", renderer.menu(&page, 0));
                        }
                    }
                    InitOutcome::Quit => break,
                    InitOutcome::Result {
                        command,
                        result,
                        attempted_save,
                    } => {
                        if result.as_ref().is_ok_and(|report| {
                            report.operation == "scaffold"
                                && report.status == "success"
                                && report.data["committed"] == true
                        }) {
                            custom_drafts.remove(cli.project());
                        }
                        if attempted_save {
                            (project_state, project_check) = read_project_state(&cli, locale)?;
                            if (assembly_action
                                && (result.is_err()
                                    || result.as_ref().is_ok_and(|report| {
                                        report
                                            .diagnostics
                                            .iter()
                                            .any(|item| item.code == "PROJECT_CONFIG_CONFLICT")
                                    })))
                                || result.as_ref().is_ok_and(|report| {
                                    (report.operation == "init"
                                        || report.operation == "create"
                                        || report.operation == "scaffold"
                                        || report.operation == "config_set_batch"
                                        || report.operation.starts_with("assembly_"))
                                        && report.status == "success"
                                        && report.data["committed"] == true
                                })
                            {
                                workbench.clear_project_context();
                                if !assembly_action {
                                    workbench.form_context = Some((
                                        command.model.clone(),
                                        Some(command.batch.clone())
                                            .filter(|batch| !batch.is_empty()),
                                    ));
                                    workbench.model_context =
                                        Some((command.model.clone(), BTreeMap::new()));
                                }
                            }
                        }
                        workbench.record_result(command, result);
                        workbench.page = Page::Report;
                    }
                }
                None
            }
            Page::Model { .. } | Page::Batch { .. } => {
                #[cfg(unix)]
                if !line_mode && crate::terminal::menu::supported() {
                    let initial = match &workbench.page {
                        Page::Model { draft, .. } | Page::Batch { draft, .. } => draft.as_str(),
                        _ => "",
                    };
                    match crate::terminal::menu::input(initial, |value, cursor| {
                        Renderer::current(locale, workbench.style, line_mode).form(
                            &workbench.page,
                            value,
                            cursor,
                        )
                    })? {
                        crate::terminal::menu::Input::Submit(value) => {
                            if matches!(&workbench.page, Page::Model { .. })
                                && value.trim().eq_ignore_ascii_case(":model-params")
                            {
                                configure_model_parameters(&mut workbench, &renderer)?;
                                continue;
                            }
                            if matches!(&workbench.page, Page::Batch { .. })
                                && value.trim().eq_ignore_ascii_case(":options")
                            {
                                configure_probe_options(&mut workbench, &renderer)?;
                                continue;
                            }
                            if matches!(&workbench.page, Page::Batch { .. })
                                && value.trim().eq_ignore_ascii_case(":batch-params")
                            {
                                configure_batch_parameters(&mut workbench, &renderer)?;
                                continue;
                            }
                            if matches!(&workbench.page, Page::Batch { .. })
                                && value.trim().eq_ignore_ascii_case(":mode")
                            {
                                configure_execution_mode(&mut workbench, &renderer)?;
                                continue;
                            }
                            if matches!(&workbench.page, Page::Batch { .. })
                                && value.trim().eq_ignore_ascii_case(":snapshot")
                            {
                                configure_input_snapshot(&mut workbench, &renderer)?;
                                continue;
                            }
                            if let Some(command) = workbench.submit(value.trim().to_owned()) {
                                execute_and_record(&mut workbench, command, &cli, &renderer)?;
                                workbench.page = Page::Report;
                            }
                        }
                        crate::terminal::menu::Input::Back => {
                            if matches!(workbench.page, Page::Batch { .. }) {
                                workbench.probe_options_draft = None;
                            }
                            workbench.page = match &workbench.page {
                                Page::Batch { model, .. } => Page::Model {
                                    action: Action::Probe,
                                    draft: model.clone(),
                                },
                                _ => Page::Menu,
                            };
                        }
                        crate::terminal::menu::Input::Quit => break,
                    }
                    continue;
                }
                if line_mode {
                    if let Page::Model { draft, .. } | Page::Batch { draft, .. } = &workbench.page {
                        if !draft.is_empty() {
                            let label = if matches!(&workbench.page, Page::Model { .. }) {
                                locale.text("Current model", "当前模型", "目前模型")
                            } else {
                                locale.text("Current batch", "当前批次", "目前批次")
                            };
                            eprintln!("{}: {}", label, text::plain_text(draft));
                        }
                    }
                }
                let Some(mut value) = prompt(renderer.prompt(&workbench.page).unwrap())? else {
                    break;
                };
                if matches!(&workbench.page, Page::Model { .. })
                    && value.trim().eq_ignore_ascii_case(":model-params")
                {
                    configure_model_parameters(&mut workbench, &renderer)?;
                    continue;
                }
                if matches!(&workbench.page, Page::Batch { .. })
                    && value.trim().eq_ignore_ascii_case(":options")
                {
                    configure_probe_options(&mut workbench, &renderer)?;
                    continue;
                }
                if matches!(&workbench.page, Page::Batch { .. })
                    && value.trim().eq_ignore_ascii_case(":batch-params")
                {
                    configure_batch_parameters(&mut workbench, &renderer)?;
                    continue;
                }
                if matches!(&workbench.page, Page::Batch { .. })
                    && value.trim().eq_ignore_ascii_case(":mode")
                {
                    configure_execution_mode(&mut workbench, &renderer)?;
                    continue;
                }
                if matches!(&workbench.page, Page::Batch { .. })
                    && value.trim().eq_ignore_ascii_case(":snapshot")
                {
                    configure_input_snapshot(&mut workbench, &renderer)?;
                    continue;
                }
                if line_mode && value.trim().eq_ignore_ascii_case("b") {
                    if matches!(workbench.page, Page::Batch { .. }) {
                        workbench.probe_options_draft = None;
                    }
                    workbench.page = match &workbench.page {
                        Page::Batch { model, .. } => Page::Model {
                            action: Action::Probe,
                            draft: model.clone(),
                        },
                        _ => Page::Menu,
                    };
                    if matches!(workbench.page, Page::Menu) {
                        print!("{}", renderer.menu(&page, 0));
                    }
                    None
                } else {
                    if line_mode && value.trim().eq_ignore_ascii_case(":default") {
                        value.clear();
                    } else if line_mode && value.trim().is_empty() {
                        if let Page::Model { draft, .. } | Page::Batch { draft, .. } =
                            &workbench.page
                        {
                            if !draft.is_empty() {
                                value = draft.clone();
                            }
                        }
                    }
                    workbench.submit(value)
                }
            }
            Page::Report => {
                let mut content = String::new();
                if workbench.report_focus == ReportFocus::Diagnosis
                    && workbench.diagnosis_retry_issue.is_some()
                {
                    renderer.heading(
                        &mut content,
                        match workbench.diagnosis_retry_issue {
                            Some(RetryIssue::TimedOut) => locale.text(
                                "Previous diagnosis · latest attempt timed out",
                                "上一次诊断 · 最近一次尝试超时",
                                "上一次診斷 · 最近一次嘗試逾時",
                            ),
                            Some(RetryIssue::Cancelled) => locale.text(
                                "Previous diagnosis · retry cancelled",
                                "上一次诊断 · 最近一次尝试已取消",
                                "上一次診斷 · 最近一次嘗試已取消",
                            ),
                            None => unreachable!(),
                        },
                    );
                    renderer.line(
                        &mut content,
                        locale.text(
                            "Run a new probe before locating or previewing.",
                            "请重新诊断后再定位或预览。",
                            "請重新診斷後再定位或預覽。",
                        ),
                    );
                }
                if let Some(report) = workbench.displayed_report() {
                    if report.operation == "probe" {
                        if let Some(command) = workbench
                            .displayed_command()
                            .filter(|command| command.action == Action::Probe)
                        {
                            let requested = if command.probe_options.training {
                                "train"
                            } else {
                                "eval"
                            };
                            let mode = &report.data["observed_mode"];
                            if mode_mismatch(mode, command.probe_options.training) {
                                renderer.heading(
                                    &mut content,
                                    locale.text(
                                        "Mode mismatch · preview unavailable",
                                        "模式不一致 · 无法预览",
                                        "模式不一致 · 無法預覽",
                                    ),
                                );
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{} {requested} · {} {}",
                                        locale.text("Requested", "请求", "請求"),
                                        locale.text("reported", "报告", "報告"),
                                        mode.as_str().unwrap_or("?")
                                    ),
                                );
                                renderer.line(
                                    &mut content,
                                    locale.text(
                                        "Run a new probe before previewing.",
                                        "请重新诊断后再预览。",
                                        "請重新診斷後再預覽。",
                                    ),
                                );
                            } else if !matches!(mode.as_str(), Some("train" | "eval")) {
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{} {requested} · {}",
                                        locale.text("Requested", "请求", "請求"),
                                        locale.text(
                                            "reported mode unverified",
                                            "报告模式无法核对",
                                            "報告模式無法核對"
                                        )
                                    ),
                                );
                            }
                        }
                    }
                    if report.operation == "probe" && report.status == "error" {
                        let status = locale.text("Probe failed", "前向诊断失败", "前向診斷失敗");
                        let code = report
                            .diagnostics
                            .first()
                            .map(|item| item.code.as_str())
                            .unwrap_or("");
                        renderer.heading(&mut content, &format!("{status}  {code}"));
                        content.push_str(
                            &renderer.diagnosis_summary(
                                report,
                                workbench
                                    .inspected_structure
                                    .as_ref()
                                    .map(|(_, modules)| modules),
                                workbench.diagnosis_retry_issue.is_none(),
                                workbench
                                    .displayed_command()
                                    .filter(|command| command.action == Action::Probe)
                                    .map(|command| command.probe_options.training),
                            ),
                        );
                    }
                }
                if let Some(command) = workbench.displayed_command() {
                    if matches!(command.action, Action::Inspect | Action::Probe) {
                        renderer.line(
                            &mut content,
                            &format!(
                                "{}  {}",
                                locale.text("Model", "模型", "模型"),
                                if command.model.is_empty() {
                                    locale.text("Project default", "项目默认值", "專案預設值")
                                } else {
                                    &command.model
                                }
                            ),
                        );
                    }
                    if matches!(command.action, Action::Inspect | Action::Probe) {
                        renderer.line(
                            &mut content,
                            &format!(
                                "{}  {}",
                                locale.text("Variant", "变体", "變體"),
                                if command.variant.is_empty() {
                                    locale.text(
                                        "CLI default · not pinned",
                                        "CLI 默认 · 未固定",
                                        "CLI 預設 · 未固定",
                                    )
                                } else {
                                    &command.variant
                                }
                            ),
                        );
                    }
                    if command.action == Action::Probe {
                        renderer.line(
                            &mut content,
                            &format!(
                                "{}  {}",
                                locale.text("Batch", "批次", "批次"),
                                if command.batch.is_empty() {
                                    locale.text("Project default", "项目默认值", "專案預設值")
                                } else {
                                    &command.batch
                                }
                            ),
                        );
                    }
                }
                let showing_recorded_probe = match workbench.report_focus {
                    ReportFocus::Diagnosis => true,
                    ReportFocus::Latest => {
                        workbench
                            .last_command
                            .as_ref()
                            .is_some_and(|command| command.action == Action::Probe)
                            && matches!(workbench.last_result.as_ref(), Some(Ok(report)) if report.operation == "probe")
                    }
                };
                if showing_recorded_probe {
                    if workbench.diagnosis.as_ref().is_some_and(|diagnosis| {
                        diagnosis.command.probe_options.capture_snapshot
                            && diagnosis.snapshot.is_none()
                    }) {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "Session snapshot unavailable; replay is disabled.",
                                "会话快照不可用；无法重放。",
                                "工作階段快照無法取得；無法重放。",
                            ),
                        );
                    }
                    if let Some(comparison) = workbench
                        .diagnosis
                        .as_ref()
                        .and_then(|diagnosis| diagnosis.batch_comparison)
                    {
                        let label = match comparison {
                            BatchComparison::Same => locale.text(
                                "Batch ID matches previous probe report",
                                "批次 ID 与上次诊断报告一致",
                                "批次 ID 與上次診斷報告一致",
                            ),
                            BatchComparison::Different => locale.text(
                                "Batch ID differs from previous probe report",
                                "批次 ID 与上次诊断报告不同",
                                "批次 ID 與上次診斷報告不同",
                            ),
                            BatchComparison::Unknown => locale.text(
                                "Batch identity cannot be compared with previous probe",
                                "无法与上次诊断核对批次身份",
                                "無法與上次診斷核對批次身分",
                            ),
                        };
                        renderer.line(&mut content, label);
                    }
                }
                match workbench.report_focus {
                    ReportFocus::Latest => {
                        if let Some(result) = &workbench.last_result {
                            content.push_str(&renderer.result(result));
                        }
                    }
                    ReportFocus::Diagnosis => {
                        if let Some(diagnosis) = &workbench.diagnosis {
                            content.push_str(&renderer.report(&diagnosis.report));
                        }
                    }
                }
                #[cfg(unix)]
                if !line_mode && crate::terminal::menu::supported() {
                    if !view_report(&content, locale, workbench.style)? {
                        break;
                    }
                } else {
                    print!("{content}");
                }
                #[cfg(not(unix))]
                print!("{content}");
                workbench.page = Page::Menu;
                print!("{}", renderer.menu(&page, 0));
                None
            }
        };
        if let Some(command) = command {
            execute_and_record(&mut workbench, command, &cli, &renderer)?;
            workbench.page = Page::Report;
        }
        if let Some(path) = project_change {
            if path != cli.project() {
                cli = cli.with_project(path);
                workbench.clear_project_context();
                workbench.variant.clear();
                workbench.variant_revision = None;
                (project_state, project_check) = read_project_state(&cli, locale)?;
                menu_focus = None;
            } else {
                workbench.page = Page::Menu;
            }
            if line_mode {
                let page = MenuPage {
                    project: cli.project(),
                    project_state,
                    variant: &workbench.variant,
                    bindings: &active_bindings,
                    context: workbench.form_context.as_ref(),
                    has_report: workbench.last_result.is_some(),
                    has_diagnosis: workbench.diagnosis.is_some(),
                    diagnosis_retry_issue: workbench.diagnosis_retry_issue,
                    has_saved_preview: workbench.saved_preview.is_some()
                        || workbench.saved_receipt.is_some(),
                    has_details: workbench
                        .detail_items()
                        .is_some_and(|items| !items.is_empty()),
                    has_structure: workbench.inspected_structure.is_some(),
                    has_preview: workbench.preview_target().is_some(),
                    has_location: workbench.observed_location().is_some(),
                    has_calls: workbench
                        .recorded_calls()
                        .is_some_and(|calls| !calls.is_empty()),
                };
                print!(
                    "{}",
                    Renderer::current(locale, workbench.style, line_mode).menu(&page, 0)
                );
            }
        }
    }
    Ok(())
}

enum InitOutcome {
    Cancelled,
    Quit,
    Result {
        command: Command,
        result: io::Result<Report>,
        attempted_save: bool,
    },
}

fn initialize_project(
    cli: &Cli,
    locale: Locale,
    style: Style,
    line_mode: bool,
    context: Option<&(String, Option<String>)>,
) -> io::Result<InitOutcome> {
    let renderer = Renderer::current(locale, style, line_mode);
    let model_initial = context.map(|(model, _)| model.as_str()).unwrap_or("");
    let batch_initial = context
        .and_then(|(_, batch)| batch.as_deref())
        .unwrap_or("");
    let Some(model) = init_input(
        &renderer,
        locale.text(
            "Model factory (or :preset to create a model)",
            "模型工厂（输入 :preset 创建模型）",
            "模型工廠（輸入 :preset 建立模型）",
        ),
        model_initial,
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    if model.trim() == ":preset" {
        return initialize_classification(cli, &renderer);
    }
    let batch_initial = if model == model_initial {
        batch_initial
    } else {
        ""
    };
    let Some(batch) = init_input(
        &renderer,
        locale.text(
            "Batch factory (optional)",
            "批次工厂（可选）",
            "批次工廠（選填）",
        ),
        batch_initial,
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    let command = Command {
        action: Action::Init,
        model,
        batch,
        variant: String::new(),
        probe_options: ProbeOptions::default(),
    };
    command.announce(&renderer, None, true)?;
    let result = command.execute(cli);
    let expected_config = serde_json::json!({
        "schema_version": 1,
        "model": command.model,
        "batch": if command.batch.is_empty() { None } else { Some(command.batch.as_str()) },
        "batch_binding": null,
        "model_injections": {},
    });
    let valid_preview = result.as_ref().is_ok_and(|report| {
        let config = &report.data["config"];
        report.operation == "init"
            && report.status == "success"
            && report.data["committed"] == false
            && !command.model.is_empty()
            && config == &expected_config
            && report.data["path"].as_str().is_some_and(|path| {
                Path::new(path) == cli.project().join(".model-maven/project.json")
            })
    });
    if !valid_preview {
        let result = match result {
            Ok(report) if report.operation == "init" && report.status == "error" => Ok(report),
            Err(error) => Err(error),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unexpected initialization preview; no save was requested.",
            )),
        };
        return Ok(InitOutcome::Result {
            command,
            result,
            attempted_save: false,
        });
    }
    let report = result.as_ref().unwrap();
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Project setup preview · not saved",
            "项目入口预览 · 尚未保存",
            "專案入口預覽 · 尚未儲存",
        ),
    );
    for (label, value) in [
        (
            locale.text("File", "文件", "檔案"),
            report.data["path"].as_str().unwrap_or("?"),
        ),
        (locale.text("Model", "模型", "模型"), command.model.as_str()),
        (
            locale.text("Batch", "批次", "批次"),
            if command.batch.is_empty() {
                locale.text("None", "无", "無")
            } else {
                &command.batch
            },
        ),
    ] {
        renderer.line(
            &mut content,
            &format!("{label}  {}", text::plain_text(value)),
        );
    }
    renderer.line(
        &mut content,
        locale.text(
            "Preview does not import or validate the factories.",
            "预览不会导入或验证工厂。",
            "預覽不會匯入或驗證工廠。",
        ),
    );
    #[cfg(unix)]
    if !line_mode && crate::terminal::menu::supported() {
        if !view_report(&content, locale, style)? {
            return Ok(InitOutcome::Quit);
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    print!("{content}");

    #[cfg(unix)]
    let confirmed = if !line_mode && crate::terminal::menu::supported() {
        let keys = ["b", "y"];
        let Some(selected) = crate::terminal::menu::select(&keys, |selected| {
            let renderer = Renderer::current(locale, style, false);
            let mut output = String::new();
            renderer.heading(
                &mut output,
                locale.text("Save project defaults?", "保存项目入口？", "儲存專案入口？"),
            );
            renderer.line(
                &mut output,
                locale.text(
                    "This creates .model-maven/project.json.",
                    "这将创建 .model-maven/project.json。",
                    "這將建立 .model-maven/project.json。",
                ),
            );
            let back = locale.text("b. Back", "b. 返回", "b. 返回");
            let save = locale.text("y. Save", "y. 保存", "y. 儲存");
            renderer.line(
                &mut output,
                &format!("{} {back}", if selected == 0 { ">" } else { " " }),
            );
            renderer.line(
                &mut output,
                &format!("{} {save}", if selected == 1 { ">" } else { " " }),
            );
            output
        })?
        else {
            return Ok(InitOutcome::Quit);
        };
        selected == 1
    } else {
        prompt(locale.text(
            "Save project defaults? (y to save, Enter to cancel)",
            "保存项目入口？（输入 y 保存，回车取消）",
            "儲存專案入口？（輸入 y 儲存，Enter 取消）",
        ))?
        .is_some_and(|answer| answer.trim().eq_ignore_ascii_case("y"))
    };
    #[cfg(not(unix))]
    let confirmed = prompt(locale.text(
        "Save project defaults? (y to save, Enter to cancel)",
        "保存项目入口？（输入 y 保存，回车取消）",
        "儲存專案入口？（輸入 y 儲存，Enter 取消）",
    ))?
    .is_some_and(|answer| answer.trim().eq_ignore_ascii_case("y"));
    if !confirmed {
        return Ok(InitOutcome::Cancelled);
    }
    command.announce(&renderer, None, false)?;
    let mut args = vec!["init", "--model", command.model.as_str()];
    if !command.batch.is_empty() {
        args.extend(["--batch", &command.batch]);
    }
    let result = cli.execute(&args);
    Ok(InitOutcome::Result {
        command,
        result,
        attempted_save: true,
    })
}

fn initialize_classification(cli: &Cli, renderer: &Renderer) -> io::Result<InitOutcome> {
    let locale = renderer.locale;
    if let Some(message) = capability_unavailable(cli, locale, &["project.classification_create"]) {
        return Ok(if show_flow_page(renderer, &message)? {
            InitOutcome::Cancelled
        } else {
            InitOutcome::Quit
        });
    }
    let Some(preset) = init_input(
        renderer,
        locale.text(
            "Preset: tiny-cnn or resnet18 (random weights)",
            "预设：tiny-cnn 或 resnet18（随机权重）",
            "預設：tiny-cnn 或 resnet18（隨機權重）",
        ),
        "tiny-cnn",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    if !matches!(preset.as_str(), "tiny-cnn" | "resnet18") {
        println!(
            "{}",
            locale.text(
                "Unknown preset; no files created.",
                "未知预设；没有创建文件。",
                "未知預設；沒有建立檔案。"
            )
        );
        return Ok(InitOutcome::Cancelled);
    }
    let Some(data) = init_input(
        renderer,
        locale.text(
            "Image directory (one subdirectory per class)",
            "图像目录（每类一个子目录）",
            "圖像目錄（每類一個子目錄）",
        ),
        "",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    let command = Command {
        action: Action::Init,
        model: "classification:model".into(),
        batch: "classification:batch".into(),
        variant: String::new(),
        probe_options: ProbeOptions::default(),
    };
    let preview = cli.execute(&["create", "--preset", &preset, "--data", &data]);
    let report = match preview {
        Ok(report)
            if report.operation == "create"
                && report.status == "success"
                && report.data["committed"] == false =>
        {
            report
        }
        Ok(report) if report.status == "error" => {
            return Ok(InitOutcome::Result {
                command,
                result: Ok(report),
                attempted_save: false,
            });
        }
        Ok(_) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unexpected classification preview; no files created",
                )),
                attempted_save: false,
            });
        }
        Err(error) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(error),
                attempted_save: false,
            });
        }
    };
    let Some(revision) = report.data["plan_revision"]
        .as_str()
        .filter(|s| !s.is_empty())
    else {
        return Ok(InitOutcome::Result {
            command,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "classification preview has no revision",
            )),
            attempted_save: false,
        });
    };
    let Some(classes) = report.data["plan"]["classes"].as_array() else {
        return Ok(InitOutcome::Result {
            command,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "classification preview has no classes",
            )),
            attempted_save: false,
        });
    };
    if classes.len() < 2
        || classes.iter().any(|class| class.as_str().is_none())
        || report.data["plan"]["preset"].as_str() != Some(preset.as_str())
        || report.data["plan"]["project_root"].as_str().map(Path::new) != Some(cli.project())
        || report.data["plan"]["files"]
            != serde_json::json!(["classification.py", ".model-maven/project.json"])
    {
        return Ok(InitOutcome::Result {
            command,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "classification preview does not match this project",
            )),
            attempted_save: false,
        });
    }
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Classification preview", "分类项目预览", "分類專案預覽"),
    );
    renderer.line(
        &mut content,
        &format!("Preset: {}", text::plain_text(&preset)),
    );
    renderer.line(
        &mut content,
        &format!(
            "Classes: {}",
            text::plain_text(
                &classes
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        ),
    );
    renderer.line(&mut content, "classification.py; .model-maven/project.json");
    renderer.line(
        &mut content,
        locale.text(
            "Image content is not verified. Existing files are never overwritten.",
            "尚未验证图像内容；不会覆盖已有文件。",
            "尚未驗證圖像內容；不會覆蓋現有檔案。",
        ),
    );
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        if !view_report(&content, locale, renderer.style)? {
            return Ok(InitOutcome::Quit);
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    print!("{content}");
    let Some(answer) = init_input(
        renderer,
        locale.text(
            "Create files? Type y to confirm",
            "创建文件？输入 y 确认",
            "建立檔案？輸入 y 確認",
        ),
        "",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    if !answer.trim().eq_ignore_ascii_case("y") {
        return Ok(InitOutcome::Cancelled);
    }
    let result = cli.execute(&[
        "create",
        "--preset",
        &preset,
        "--data",
        &data,
        "--apply",
        "--if-plan",
        revision,
    ]);
    Ok(InitOutcome::Result {
        command,
        result,
        attempted_save: true,
    })
}

fn capability_unavailable(cli: &Cli, locale: Locale, required: &[&str]) -> Option<String> {
    match cli.check_capabilities(required) {
        Ok(CapabilityCheck::Available) => None,
        Ok(CapabilityCheck::Legacy) => Some(locale.text("This CLI does not provide a capability catalog; this guarded action requires a newer CLI. No requested change was applied.\n", "当前 CLI 没有能力目录；此受保护操作需要较新版本，未执行请求的修改。\n", "目前 CLI 沒有能力目錄；此受保護操作需要較新版本，未執行要求的修改。\n").to_owned()),
        Ok(CapabilityCheck::Missing(missing)) => Some(format!("{}: {}. {}\n", locale.text("CLI does not advertise required capabilities", "CLI 未声明所需能力", "CLI 未宣告所需能力"), text::plain_text(&missing.join(", ")), locale.text("Upgrade the CLI; no requested change was applied.", "请升级 CLI；未执行请求的修改。", "請升級 CLI；未執行要求的修改。"))),
        Err(error) => Some(format!("{}: {}. {}\n", locale.text("Could not verify CLI capabilities", "无法核对 CLI 能力", "無法核對 CLI 能力"), text::plain_text(&error.to_string()), locale.text("No requested change was applied.", "未执行请求的修改。", "未執行要求的修改。"))),
    }
}

fn show_flow_page(renderer: &Renderer, content: &str) -> io::Result<bool> {
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        return view_report(content, renderer.locale, renderer.style);
    }
    print!("{content}");
    Ok(true)
}

fn new_training_request_id(cli: &Cli) -> io::Result<Option<String>> {
    match cli
        .check_capabilities(&["training.request_identity"])
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("Could not verify CLI capabilities: {error}"),
            )
        })? {
        CapabilityCheck::Available => (),
        CapabilityCheck::Missing(_) | CapabilityCheck::Legacy => return Ok(None),
    }
    let directory = tempfile::Builder::new()
        .prefix("tui-")
        .rand_bytes(20)
        .tempdir()?;
    let id = directory
        .path()
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid request identity"))?
        .to_owned();
    crate::pending_start::create(cli.project(), &id)?;
    Ok(Some(id))
}

fn pending_start_notice(cli: &Cli, locale: Locale) -> Option<String> {
    match crate::pending_start::list(cli.project()) {
        Ok(pending) if pending.is_empty() => None,
        Ok(_) => Some(locale.text(
            "An earlier training start has not been acknowledged. Open Training runs, inspect its state, then explicitly forget its pending marker before starting another.\n",
            "之前的训练启动尚未确认。请打开训练记录检查状态，明确清除待确认标记后再启动。\n",
            "先前的訓練啟動尚未確認。請開啟訓練記錄檢查狀態，明確清除待確認標記後再啟動。\n",
        ).to_owned()),
        Err(error) => Some(format!("{}: {}\n", locale.text("Unable to inspect pending training starts", "无法检查待确认的训练启动", "無法檢查待確認的訓練啟動"), text::plain_text(&error.to_string()))),
    }
}

fn lookup_training_request(cli: &Cli, id: &str) -> io::Result<(Vec<Value>, bool)> {
    let report = cli.execute(&["runs", "find-request", id])?;
    if report.operation != "runs_find_request"
        || report.status != "success"
        || report.data["request_id"].as_str() != Some(id)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup returned an invalid report",
        ));
    }
    let complete = report.data["complete"].as_bool().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup has no completeness status",
        )
    })?;
    let unreadable = report.data["unreadable"].as_u64().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup has no unreadable count",
        )
    })?;
    let truncated = report.data["truncated"].as_bool().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup has no truncation status",
        )
    })?;
    if complete != (unreadable == 0 && !truncated) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup completeness is inconsistent",
        ));
    }
    let matches = report.data["matches"].as_array().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "request lookup has no matches")
    })?;
    if matches.len() > 64
        || matches.iter().any(|run| {
            run["request_id"].as_str() != Some(id)
                || run["id"].as_str().is_none_or(|value| value.is_empty())
                || run["operation"].as_str().is_none()
                || run["status"].as_str().is_none()
                || run["status_scope"] != "recorded_not_live"
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request lookup has invalid match records",
        ));
    }
    Ok((
        matches
            .iter()
            .filter(|run| run["operation"] == "train")
            .cloned()
            .collect(),
        complete,
    ))
}

fn show_runs_after_uncertain_start(
    cli: &Cli,
    renderer: &Renderer,
    before: Option<&std::collections::HashSet<String>>,
    request_id: Option<&str>,
) -> io::Result<bool> {
    let mut content = String::new();
    renderer.heading(
        &mut content,
        renderer
            .locale
            .text("Run outcome unknown", "运行结果未知", "執行結果未知"),
    );
    renderer.line(
        &mut content,
        renderer.locale.text(
            "Checking saved runs; no automatic retry.",
            "正在查询运行记录；不会自动重试。",
            "正在查詢執行記錄；不會自動重試。",
        ),
    );
    if let Some(id) = request_id {
        if matches!(
            cli.check_capabilities(&["runs.lookup_request_id"]),
            Ok(CapabilityCheck::Available)
        ) {
            match lookup_training_request(cli, id) {
                Ok((runs, complete)) => {
                    if runs.is_empty() {
                        renderer.line(&mut content, renderer.locale.text("No run matches the request yet; this does not prove it failed. Refresh before retrying.", "暂未找到匹配请求的运行；这不能证明启动失败，重试前请刷新。", "暫未找到相符請求的執行；這不能證明啟動失敗，重試前請重新整理。"));
                    } else {
                        for run in &runs {
                            renderer.line(
                                &mut content,
                                &format!(
                                    "{}: {} · {} ({})",
                                    renderer.locale.text(
                                        "Matching start request",
                                        "匹配的启动请求",
                                        "相符的啟動請求"
                                    ),
                                    text::plain_text(run["id"].as_str().unwrap_or("?")),
                                    text::plain_text(run["status"].as_str().unwrap_or("unknown")),
                                    renderer.locale.text(
                                        "recorded status; refresh to verify",
                                        "记录状态；请刷新核实",
                                        "記錄狀態；請重新整理確認"
                                    )
                                ),
                            );
                        }
                    }
                    if !complete {
                        renderer.line(&mut content, renderer.locale.text("Some run records were unreadable or omitted; other matches may exist.", "部分运行记录无法读取或未列出，可能还有其他匹配项。", "部分執行記錄無法讀取或未列出，可能還有其他相符項目。"));
                    }
                }
                Err(error) => renderer.line(
                    &mut content,
                    &format!(
                        "{}: {}",
                        renderer.locale.text(
                            "Request lookup unavailable",
                            "请求查询不可用",
                            "請求查詢無法使用"
                        ),
                        text::plain_text(&error.to_string())
                    ),
                ),
            }
            return show_flow_page(renderer, &content);
        }
    }
    match cli.execute(&["runs", "list"]) {
        Ok(report) if report.operation == "runs_list" && report.status == "success" => {
            if let Some(runs) = report.data["runs"].as_array() {
                if let Some(request_id) = request_id {
                    let matching: Vec<_> = runs
                        .iter()
                        .filter(|run| {
                            run["request_id"].as_str() == Some(request_id)
                                && run["operation"] == "train"
                        })
                        .collect();
                    if matching.len() == 1 {
                        let run = matching[0];
                        renderer.line(
                            &mut content,
                            &format!(
                                "{}: {} · {}",
                                renderer.locale.text(
                                    "Matching start request",
                                    "匹配的启动请求",
                                    "相符的啟動請求"
                                ),
                                text::plain_text(run["id"].as_str().unwrap_or("?")),
                                text::plain_text(run["status"].as_str().unwrap_or("unknown"))
                            ),
                        );
                    } else if matching.is_empty() {
                        renderer.line(&mut content, renderer.locale.text("No run matches the request yet; this does not prove it failed. Refresh before retrying.", "暂未找到匹配请求的运行；这不能证明启动失败，重试前请刷新。", "暫未找到相符請求的執行；這不能證明啟動失敗，重試前請重新整理。"));
                    } else {
                        renderer.line(&mut content, renderer.locale.text("Multiple runs match this request; inspect them before taking action.", "多个运行匹配此请求，请检查后再操作。", "多個執行符合此請求，請檢查後再操作。"));
                        for run in matching {
                            renderer.line(
                                &mut content,
                                &text::plain_text(run["id"].as_str().unwrap_or("?")),
                            );
                        }
                    }
                    return show_flow_page(renderer, &content);
                }
                if before.is_some() {
                    renderer.line(&mut content, renderer.locale.text("Runs newly observed since the start request (other processes may also create runs):", "启动请求后新观察到的运行（其他进程也可能创建运行）：", "啟動請求後新觀察到的執行（其他程序也可能建立執行）："));
                } else {
                    renderer.line(
                        &mut content,
                        renderer.locale.text(
                            "Earlier run list was unavailable; showing recent runs.",
                            "启动前运行列表不可用；显示最近的运行。",
                            "啟動前執行清單無法使用；顯示最近的執行。",
                        ),
                    );
                }
                let visible: Vec<_> = runs
                    .iter()
                    .filter(|run| {
                        before
                            .is_none_or(|ids| run["id"].as_str().is_none_or(|id| !ids.contains(id)))
                    })
                    .collect();
                if visible.is_empty() {
                    renderer.line(&mut content, renderer.locale.text("No new run visible yet; this does not prove the request failed. Refresh before retrying.", "暂未看到新运行；这不能证明启动失败，重试前请刷新。", "暫未看到新執行；這不能證明啟動失敗，重試前請重新整理。"));
                }
                for run in visible.iter().take(20) {
                    renderer.line(
                        &mut content,
                        &format!(
                            "{} · {}",
                            text::plain_text(run["id"].as_str().unwrap_or("?")),
                            text::plain_text(run["status"].as_str().unwrap_or("unknown"))
                        ),
                    );
                }
                if visible.len() > 20 {
                    renderer.line(
                        &mut content,
                        renderer.locale.text(
                            "More runs exist; open Training runs for the full list.",
                            "还有更多运行；打开训练记录查看完整列表。",
                            "還有更多執行；開啟訓練記錄查看完整清單。",
                        ),
                    );
                }
            } else {
                renderer.line(
                    &mut content,
                    "Run list unavailable; check the project before retrying.",
                );
            }
        }
        _ => renderer.line(
            &mut content,
            "Run list unavailable; check the project before retrying.",
        ),
    }
    show_flow_page(renderer, &content)
}

fn launch_training(
    cli: &Cli,
    renderer: &Renderer,
    variant: &str,
    trials: &mut Vec<crate::cli::BackgroundTask>,
) -> io::Result<bool> {
    let locale = renderer.locale;
    let config = match cli.execute(&["assembly", "show", "--config-only"]) {
        Ok(report) if report.operation == "assembly_show" && report.status == "success" => report,
        other => {
            print!("{}", renderer.result(&other));
            return Ok(true);
        }
    };
    let model = config.data["model"].as_str().unwrap_or("");
    let batch = config.data["batch"].as_str().unwrap_or("");
    let Some(config_revision) = config.data["config_revision"].as_str().filter(|revision| {
        revision.len() == 64
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) else {
        let mut content = String::new();
        renderer.line(
            &mut content,
            locale.text(
                "Project configuration revision unavailable; run not started.",
                "无法核对项目配置修订；未启动运行。",
                "無法核對專案設定修訂；未啟動執行。",
            ),
        );
        return show_flow_page(renderer, &content);
    };
    let variant_report = if variant.is_empty() {
        None
    } else {
        match cli.execute(&["variant", "show", variant]) {
            Ok(report) if report.operation == "variant_show" && report.status == "success"
                && report.data["name"].as_str() == Some(variant)
                && report.data["variant_base"]["name"].as_str() == Some(variant)
                && report.data["variant_base"]["algorithm"] == "canonical-json-sha256-v1"
                && report.data["variant_base"]["token"].as_str().is_some_and(|token| token.len() == 64 && token.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) => Some(report),
            _ => return show_flow_page(renderer, locale.text("Selected variant cannot be guarded by this CLI; inspect it or upgrade before starting.\n", "无法使用当前 CLI 保护所选变体；请检查变体或升级后再启动。\n", "無法使用目前 CLI 保護所選變體；請檢查變體或升級後再啟動。\n")),
        }
    };
    let scaffold_step = model.strip_suffix(":model").and_then(|module| {
        if module.is_empty()
            || !module.split('.').all(|part| {
                !part.is_empty()
                    && part
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            })
        {
            return None;
        }
        let path = cli
            .project()
            .join(format!("{}.py", module.replace('.', "/")))
            .canonicalize()
            .ok()?;
        if !path.starts_with(cli.project()) {
            return None;
        }
        use std::io::Read;
        let mut prefix = String::new();
        fs::File::open(path)
            .ok()?
            .take(64)
            .read_to_string(&mut prefix)
            .ok()?;
        prefix
            .starts_with("# Editable Model-Maven starter.")
            .then(|| format!("{module}:step"))
    });
    let default_step = if model == "classification:model" {
        "classification:step"
    } else if model == "model_starter:model" {
        "model_starter:step"
    } else {
        scaffold_step.as_deref().unwrap_or("")
    };
    let Some(kind) = init_input(
        renderer,
        locale.text(
            "Run type: try (background), try-wait (foreground), or train (detached)",
            "运行类型：try 后台试跑、try-wait 前台试跑或 train 后台训练",
            "執行類型：try 背景試跑、try-wait 前景試跑或 train 背景訓練",
        ),
        "try",
    )?
    else {
        return Ok(true);
    };
    if !matches!(kind.as_str(), "try" | "try-wait" | "train") {
        return Ok(true);
    }
    if kind == "train" {
        if let Some(message) = pending_start_notice(cli, locale) {
            return show_flow_page(renderer, &message);
        }
    }
    if kind == "try"
        && trials
            .iter_mut()
            .filter(|task| task.project == cli.project())
            .any(|task| !matches!(task.poll(), Ok(true)))
    {
        return show_flow_page(renderer, locale.text("A trial is already running or its status is unknown. Open Training runs to inspect it before starting another.\n", "试跑正在运行或状态未知；请先打开训练记录检查，再启动另一次试跑。\n", "試跑正在執行或狀態未知；請先開啟訓練記錄檢查，再啟動另一次試跑。\n"));
    }
    let Some(mode) = init_input(
        renderer,
        locale.text(
            "Training mode: managed-step, manual-step or external-loop",
            "训练模式：managed-step、manual-step 或 external-loop",
            "訓練模式：managed-step、manual-step 或 external-loop",
        ),
        "managed-step",
    )?
    else {
        return Ok(true);
    };
    if !matches!(
        mode.as_str(),
        "managed-step" | "manual-step" | "external-loop"
    ) {
        return Ok(true);
    }
    let external = mode == "external-loop";
    let Some(step) = init_input(
        renderer,
        if external {
            locale.text(
                "Full training loop module:callable",
                "完整训练循环 module:callable",
                "完整訓練迴圈 module:callable",
            )
        } else {
            locale.text(
                "Training step module:callable",
                "训练步骤 module:callable",
                "訓練步驟 module:callable",
            )
        },
        if mode == "managed-step" {
            default_step
        } else {
            ""
        },
    )?
    else {
        return Ok(true);
    };
    if step.is_empty() {
        return Ok(true);
    }
    let Some(raw_params) = init_input(
        renderer,
        locale.text(
            "Step/loop parameters as JSON object",
            "步骤或循环参数（JSON 对象）",
            "步驟或迴圈參數（JSON 物件）",
        ),
        "{}",
    )?
    else {
        return Ok(true);
    };
    let params = match parse_factory_parameters(&raw_params) {
        Ok(params) => params,
        Err(error) => {
            println!("{}", text::plain_text(&error));
            return Ok(true);
        }
    };
    let Some(steps) = init_input(
        renderer,
        locale.text("Number of steps", "训练步数", "訓練步數"),
        "1",
    )?
    else {
        return Ok(true);
    };
    let Some(device) = init_input(
        renderer,
        locale.text(
            "Device: cpu, auto or cuda:N",
            "设备：cpu、auto 或 cuda:N",
            "裝置：cpu、auto 或 cuda:N",
        ),
        "cpu",
    )?
    else {
        return Ok(true);
    };
    let Some(supervision) = init_input(
        renderer,
        locale.text(
            "Apply saved supervision rules? y/n",
            "启用已保存的监督规则？y/n",
            "啟用已儲存的監督規則？y/n",
        ),
        "n",
    )?
    else {
        return Ok(true);
    };
    if !matches!(supervision.as_str(), "y" | "n" | "Y" | "N") {
        return Ok(true);
    }
    let mut required = vec!["training.conditional_config_start"];
    if supervision.eq_ignore_ascii_case("y") {
        required.push("training.conditional_supervision");
    }
    if !variant.is_empty() {
        required.push("training.conditional_variant_start");
    }
    if let Some(message) = capability_unavailable(cli, locale, &required) {
        return show_flow_page(renderer, &message);
    }
    let rule_catalog = if supervision.eq_ignore_ascii_case("y") {
        match cli.execute(&["model", "supervise", "rule", "list"]) {
            Ok(report)
                if report.operation == "semantic_supervise"
                    && report.status == "success"
                    && report.data["rules"].is_array() =>
            {
                Some(report)
            }
            other => {
                let mut content = renderer.result(&other);
                renderer.line(
                    &mut content,
                    locale.text(
                        "Could not verify saved supervision rules; run not started.",
                        "无法核对已保存的监督规则；未启动运行。",
                        "無法核對已儲存的監督規則；未啟動執行。",
                    ),
                );
                return show_flow_page(renderer, &content);
            }
        }
    } else {
        None
    };
    let supervision_revision = if let Some(catalog) = &rule_catalog {
        let revision = catalog.data["state_revision"].as_str().filter(|revision| {
            revision.len() == 64
                && revision
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        if revision.is_none() {
            return show_flow_page(renderer, locale.text("This CLI does not provide a supervision revision; guarded training is unavailable. Upgrade the CLI or start without supervision.\n", "此 CLI 不提供监督修订，无法安全确认监督训练；请升级 CLI 或不启用监督。\n", "此 CLI 不提供監督修訂，無法安全確認監督訓練；請升級 CLI 或不啟用監督。\n"));
        }
        revision
    } else {
        None
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Training plan", "训练计划", "訓練計畫"),
    );
    for (label, value) in [
        ("Model", model),
        (
            "Batch",
            if external {
                "managed by user loop"
            } else {
                batch
            },
        ),
        ("Mode", mode.as_str()),
        (if external { "Loop" } else { "Step" }, step.as_str()),
        ("Steps", steps.as_str()),
        ("Device", device.as_str()),
        (
            "Variant",
            if variant.is_empty() {
                "project default"
            } else {
                variant
            },
        ),
        ("Supervision", supervision.as_str()),
    ] {
        renderer.line(
            &mut content,
            &format!("{label}: {}", text::plain_text(value)),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "Bindings: {}",
            text::plain_text(&config.data["slots"].to_string())
        ),
    );
    if config.data["model_params"].is_object() {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Project model parameters", "项目模型参数", "專案模型參數"),
                text::plain_text(&config.data["model_params"].to_string())
            ),
        );
    }
    if !external {
        renderer.line(
            &mut content,
            &format!(
                "Batch parameters: {}",
                text::plain_text(&config.data["batch_params"].to_string())
            ),
        );
    }
    if let Some(catalog) = &rule_catalog {
        for rule in catalog.data["rules"].as_array().into_iter().flatten() {
            renderer.line(
                &mut content,
                &format!(
                    "Rule: {} · {} · {} · {}",
                    text::plain_text(rule["id"].as_str().unwrap_or("?")),
                    text::plain_text(rule["module_path"].as_str().unwrap_or("?")),
                    text::plain_text(rule["kind"].as_str().unwrap_or("?")),
                    text::plain_text(rule["mode"].as_str().unwrap_or("?"))
                ),
            );
            renderer.line(
                &mut content,
                &format!("  {}", text::plain_text(&rule.to_string())),
            );
        }
        renderer.line(
            &mut content,
            locale.text(
                "Saved rules listed; compatibility is checked when training starts.",
                "列出已保存规则；兼容性在训练启动时检查。",
                "列出已儲存規則；相容性在訓練啟動時檢查。",
            ),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "Parameters: {}",
            text::plain_text(&serde_json::to_string(&params).unwrap_or_default())
        ),
    );
    renderer.line(&mut content, match mode.as_str() {
        "manual-step" => locale.text("User step owns backward and optimizer updates; tracked supervision requires its context protocol.", "用户步骤负责反向和优化器更新；监督追踪需要遵守 context 协议。", "使用者步驟負責反向和最佳化器更新；監督追蹤需要遵守 context 協議。"),
        "external-loop" => locale.text("User loop owns data, optimizer and updates. Progress is user reported; supervision requires its context protocol.", "用户循环负责数据、优化器和更新；进度由用户代码上报，监督需要遵守 context 协议。", "使用者迴圈負責資料、最佳化器和更新；進度由使用者程式碼回報，監督需要遵守 context 協議。"),
        _ => locale.text("Managed step returns loss; the engine performs backward and optimizer update.", "受管步骤返回损失；引擎负责反向与优化器更新。", "受管步驟回傳損失；引擎負責反向與最佳化器更新。"),
    });
    renderer.line(&mut content, locale.text("Configuration, selected variant and enabled supervision revisions are checked. Code and data are not pinned. Supervision is applied only when enabled.", "启动时核对配置、所选变体和已启用监督的修订。源码和数据未固定；监督仅在显式启用时生效。", "啟動時核對設定、所選變體和已啟用監督的修訂。原始碼和資料未固定；監督僅在明確啟用時生效。"));
    renderer.line(&mut content, locale.text("Optional data version: enter 'data VERSION' at confirmation. It is your declaration, not a verified content hash; omit credentials.", "可选数据版本：在确认时输入 data VERSION。该版本由用户声明，不是已验证的内容哈希；请勿包含凭据。", "可選資料版本：在確認時輸入 data VERSION。該版本由使用者聲明，並非已驗證的內容雜湊；請勿包含憑證。"));
    if let Some(report) = &variant_report {
        renderer.line(&mut content, locale.text("The selected variant may override the project model and bindings. Redacted variant snapshot:", "所选变体可能覆盖项目模型和绑定。脱敏后的变体快照：", "所選變體可能覆蓋專案模型和綁定。去識別的變體快照："));
        renderer.line(
            &mut content,
            &text::plain_text(&report.data["variant"].to_string()),
        );
    }
    if kind == "try" {
        renderer.line(&mut content, locale.text("Trial runs in the background for this TUI session. Leaving the screen is safe; exiting requests cancellation. Inspect Training runs and processes if cleanup is uncertain.", "试跑在本次 TUI 会话后台运行，可离开当前页面；退出时请求取消。清理结果不确定时请检查训练记录及进程。", "試跑在本次 TUI 工作階段背景執行，可離開目前頁面；結束時要求取消。清理結果不確定時請檢查訓練記錄及程序。"));
    } else if kind == "try-wait" {
        renderer.line(
            &mut content,
            locale.text(
                "Trial runs in the foreground until it finishes.",
                "试跑在前台执行，完成前暂不可操作。",
                "試跑在前景執行，完成前暫不可操作。",
            ),
        );
    }
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let mut data_fingerprint = String::new();
    loop {
        let Some(confirm) = init_input(
            renderer,
            locale.text(
                "Start this run? Type y · data VERSION sets a data version · clear · b back",
                "启动：y · 设置数据版本：data VERSION · clear 清除 · b 返回",
                "啟動：y · 設定資料版本：data VERSION · clear 清除 · b 返回",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if confirm.eq_ignore_ascii_case("y") {
            if !data_fingerprint.is_empty() {
                if let Some(message) =
                    capability_unavailable(cli, locale, &["training.data_fingerprint"])
                {
                    if !show_flow_page(renderer, &message)? {
                        return Ok(false);
                    }
                    continue;
                }
            }
            break;
        }
        if confirm.eq_ignore_ascii_case("clear") {
            data_fingerprint.clear();
        } else if let Some(value) = confirm.strip_prefix("data ") {
            let value = value.trim();
            if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
                if !show_flow_page(
                    renderer,
                    locale.text(
                        "Data version must contain 1–256 bytes without control characters.\n",
                        "数据版本须为 1–256 字节且不含控制字符。\n",
                        "資料版本須為 1–256 位元組且不含控制字元。\n",
                    ),
                )? {
                    return Ok(false);
                }
                continue;
            }
            data_fingerprint = value.to_owned();
        } else {
            return Ok(true);
        }
        let message = format!(
            "{}: {}\n{}\n",
            locale.text("Declared data version", "声明的数据版本", "聲明的資料版本"),
            if data_fingerprint.is_empty() {
                locale.text("unknown", "未知", "未知").to_owned()
            } else {
                text::plain_text(&data_fingerprint)
            },
            locale.text(
                "This is not an independently verified content hash.",
                "这不是独立验证的内容哈希。",
                "這不是獨立驗證的內容雜湊。"
            )
        );
        if !show_flow_page(renderer, &message)? {
            return Ok(false);
        }
    }
    let mut args = vec![
        if kind == "try-wait" {
            "try"
        } else {
            kind.as_str()
        },
        if external { "--loop" } else { "--step" },
        step.as_str(),
        "--steps",
        steps.as_str(),
        "--device",
        device.as_str(),
        "--training-mode",
        mode.as_str(),
    ];
    args.extend(["--if-config-revision", config_revision]);
    if !data_fingerprint.is_empty() {
        args.extend(["--data-fingerprint", &data_fingerprint]);
    }
    // External loops do not inherit a configured model automatically and own their data.
    if external && variant.is_empty() && !model.is_empty() {
        args.extend(["--model", model]);
    }
    let parameter_values: Vec<String> = params
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    for value in &parameter_values {
        args.extend([
            if external {
                "--loop-param"
            } else {
                "--step-param"
            },
            value,
        ]);
    }
    if kind == "train" {
        args.push("--detach");
    }
    if supervision.eq_ignore_ascii_case("y") {
        args.push("--apply-supervision");
        args.extend([
            "--if-supervision-revision",
            supervision_revision.expect("validated when supervision is enabled"),
        ]);
    }
    if !variant.is_empty() {
        args.extend(["--variant", variant]);
        args.extend([
            "--if-variant-base",
            variant_report
                .as_ref()
                .and_then(|report| report.data["variant_base"]["token"].as_str())
                .expect("validated variant token"),
        ]);
    }
    if kind == "try" {
        if trials.len() >= 8 {
            trials.retain_mut(|task| !matches!(task.poll(), Ok(true)));
        }
        if trials.len() >= 8 {
            return show_flow_page(
                renderer,
                locale.text(
                    "Eight trial tasks are active; inspect or cancel them first.\n",
                    "已有八个试跑任务在运行，请先检查或取消。\n",
                    "已有八個試跑任務在執行，請先檢查或取消。\n",
                ),
            );
        }
        let owned_args: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        return match cli.start_background(&owned_args) {
            Ok(task) => {
                trials.push(task);
                show_flow_page(renderer, locale.text("Trial started in this session; open Training runs to refresh its result. Exiting requests cancellation.\n", "试跑已在本会话启动；打开训练记录刷新结果。退出时请求取消。\n", "試跑已在本工作階段啟動；開啟訓練記錄重新整理結果。結束時要求取消。\n"))
            }
            Err(error) => show_flow_page(
                renderer,
                &format!("{}\n", text::plain_text(&error.to_string())),
            ),
        };
    }
    let request_id = if kind == "train" {
        match new_training_request_id(cli) {
            Ok(id) => id,
            Err(error) => {
                return show_flow_page(
                    renderer,
                    &format!(
                        "{}: {}\n",
                        locale.text(
                            "Unable to prepare a safe start; training was not launched",
                            "无法准备安全启动，未启动训练",
                            "無法準備安全啟動，未啟動訓練"
                        ),
                        text::plain_text(&error.to_string())
                    ),
                );
            }
        }
    } else {
        None
    };
    let before_runs = if request_id.is_some() {
        None
    } else {
        match cli.execute(&["runs", "list"]) {
            Ok(report) if report.operation == "runs_list" && report.status == "success" => {
                report.data["runs"].as_array().and_then(|runs| {
                    runs.iter()
                        .map(|run| {
                            run["id"]
                                .as_str()
                                .filter(|id| !id.is_empty())
                                .map(str::to_owned)
                        })
                        .collect::<Option<std::collections::HashSet<_>>>()
                })
            }
            _ => None,
        }
    };
    if let Some(id) = &request_id {
        args.extend(["--request-id", id]);
    }
    let result = cli.execute(&args);
    match &result {
        Ok(report)
            if report.status == "success"
                && report.operation
                    == if kind == "train" {
                        "train"
                    } else {
                        "training_probe"
                    }
                && report.data["run_id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
                && request_id
                    .as_ref()
                    .is_none_or(|id| report.data["request_id"].as_str() == Some(id)) =>
        {
            let mut text = String::new();
            renderer.heading(
                &mut text,
                locale.text(
                    "Run started or completed",
                    "运行已启动或完成",
                    "執行已啟動或完成",
                ),
            );
            renderer.line(
                &mut text,
                &format!(
                    "Run ID: {}",
                    text::plain_text(report.data["run_id"].as_str().unwrap_or("unavailable"))
                ),
            );
            renderer.line(
                &mut text,
                &format!(
                    "Status: {}",
                    text::plain_text(report.data["status"].as_str().unwrap_or(
                        if kind != "train" {
                            "completed"
                        } else {
                            "unknown"
                        }
                    ))
                ),
            );
            renderer.line(
                &mut text,
                locale.text(
                    "Open Training runs to inspect status; a launch receipt is not completion.",
                    "打开训练记录查看状态；启动回执不代表完成。",
                    "開啟訓練記錄檢視狀態；啟動回執不代表完成。",
                ),
            );
            if !show_flow_page(renderer, &text)? {
                return Ok(false);
            }
            if let Some(id) = &request_id {
                if let Err(error) = crate::pending_start::clear(cli.project(), id) {
                    eprintln!(
                        "{}: {}",
                        locale.text(
                            "Could not clear pending start; inspect Training runs",
                            "无法清除待确认启动，请检查训练记录",
                            "無法清除待確認啟動，請檢查訓練記錄"
                        ),
                        text::plain_text(&error.to_string())
                    );
                }
            }
        }
        Ok(report) if report.status == "error" || report.status == "inconclusive" => {
            if !show_flow_page(renderer, &renderer.result(&result))? {
                return Ok(false);
            }
            if request_id.is_some()
                && report
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code == "RUN_REQUEST_ID_EXISTS")
            {
                if !show_runs_after_uncertain_start(
                    cli,
                    renderer,
                    before_runs.as_ref(),
                    request_id.as_deref(),
                )? {
                    return Ok(false);
                }
            }
        }
        Ok(_) => {
            println!("{}", locale.text("Unexpected run receipt; outcome unknown. Inspect Training runs before retrying.", "运行回执异常，结果未知；重试前查看训练记录。", "執行回執異常，結果未知；重試前檢視訓練記錄。"));
            if !show_runs_after_uncertain_start(
                cli,
                renderer,
                before_runs.as_ref(),
                request_id.as_deref(),
            )? {
                return Ok(false);
            }
        }
        Err(error) => {
            println!("{}", text::plain_text(&error.to_string()));
            println!(
                "{}",
                locale.text(
                    "Run outcome is unknown; inspect Training runs before retrying.",
                    "运行结果未知；重试前请查看训练记录。",
                    "執行結果未知；重試前請檢視訓練記錄。"
                )
            );
            if !show_runs_after_uncertain_start(
                cli,
                renderer,
                before_runs.as_ref(),
                request_id.as_deref(),
            )? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(unix)]
fn watch_run(cli: &Cli, renderer: &Renderer, id: &str) -> io::Result<()> {
    use std::time::{Duration, Instant};
    let terminal = crate::terminal::menu::Terminal::open()?;
    let _interrupt = crate::terminal::interrupt::Guard::install();
    let _screen = crate::terminal::menu::Screen::open()?;
    let mut output = io::stderr().lock();
    let mut task: Option<(crate::cli::BackgroundTask, Instant)> = None;
    let mut next = Instant::now();
    let mut paused = false;
    let mut detail = renderer
        .locale
        .text(
            "Waiting for run status…",
            "正在读取运行状态…",
            "正在讀取執行狀態…",
        )
        .to_owned();
    let mut previous = String::new();
    loop {
        if crate::terminal::interrupt::take() {
            break;
        }
        if task.is_none() && !paused && Instant::now() >= next {
            match cli.start_background(&[
                "runs".into(),
                "attach".into(),
                id.into(),
                "--follow=false".into(),
            ]) {
                Ok(started) => task = Some((started, Instant::now())),
                Err(error) => {
                    detail = format!(
                        "{}: {}",
                        renderer
                            .locale
                            .text("Status unavailable", "状态不可用", "狀態無法使用"),
                        text::plain_text(&error.to_string())
                    );
                    paused = true;
                }
            }
        }
        if let Some((query, started)) = task.as_mut() {
            let polled = query.poll();
            if matches!(polled, Ok(true)) {
                detail = match query.result() {
                    Some(Ok(report))
                        if report.operation == "runs_attach"
                            && report.status == "success"
                            && report.data["run_id"].as_str() == Some(id) =>
                    {
                        let status = text::plain_text(
                            report.data["record"]["status"]
                                .as_str()
                                .unwrap_or("unknown"),
                        );
                        let progress = report.data["progress"]
                            .as_array()
                            .and_then(|items| items.last())
                            .map(|value| text::plain_text(&value.to_string()))
                            .unwrap_or_else(|| {
                                renderer
                                    .locale
                                    .text(
                                        "No progress reported yet.",
                                        "尚未收到进度。",
                                        "尚未收到進度。",
                                    )
                                    .into()
                            });
                        format!(
                            "{}: {status}\n{}: {progress}",
                            renderer.locale.text("Status", "状态", "狀態"),
                            renderer
                                .locale
                                .text("Latest progress", "最新进度", "最新進度")
                        )
                    }
                    Some(Err(error)) => {
                        paused = true;
                        format!(
                            "{}: {}",
                            renderer.locale.text(
                                "Status unavailable",
                                "状态不可用",
                                "狀態無法使用"
                            ),
                            text::plain_text(&error.to_string())
                        )
                    }
                    _ => {
                        paused = true;
                        renderer
                            .locale
                            .text(
                                "Unexpected status report; refresh paused.",
                                "状态报告异常；已暂停刷新。",
                                "狀態報告異常；已暫停重新整理。",
                            )
                            .into()
                    }
                };
                task = None;
                next = Instant::now() + Duration::from_secs(2);
            } else if let Err(error) = polled {
                detail = format!(
                    "{}: {}",
                    renderer
                        .locale
                        .text("Status unavailable", "状态不可用", "狀態無法使用"),
                    text::plain_text(&error.to_string())
                );
                paused = true;
                let _ = query.terminate_read_only();
                task = None;
            } else if started.elapsed() >= Duration::from_secs(10) {
                query.terminate_read_only()?;
                task = None;
                detail = renderer
                    .locale
                    .text(
                        "Status query timed out; refresh paused.",
                        "状态查询超时；已暂停刷新。",
                        "狀態查詢逾時；已暫停重新整理。",
                    )
                    .into();
                paused = true;
            }
        }
        let mut frame = String::new();
        renderer.heading(
            &mut frame,
            renderer
                .locale
                .text("Live run status", "运行实时状态", "執行即時狀態"),
        );
        renderer.line(&mut frame, &format!("Run ID: {}", text::plain_text(id)));
        for line in detail.lines() {
            renderer.line(&mut frame, line);
        }
        renderer.line(
            &mut frame,
            renderer.locale.text(
                "Refresh every 2s · r retry now · b back (does not stop training)",
                "每 2 秒刷新 · r 重试 · b 返回（不会停止训练）",
                "每 2 秒重新整理 · r 重試 · b 返回（不會停止訓練）",
            ),
        );
        if frame != previous {
            write!(output, "\x1b[H\x1b[2J{frame}")?;
            output.flush()?;
            previous = frame;
        }
        match terminal.read(100)? {
            Some(b'b' | b'B' | b'q' | b'Q' | 27 | 4) => break,
            Some(b'r' | b'R') => {
                paused = false;
                next = Instant::now();
            }
            _ => {}
        }
    }
    if let Some((mut query, _)) = task {
        query.terminate_read_only()?;
    }
    Ok(())
}

fn render_run_supervision(renderer: &Renderer, content: &mut String, data: &Value) {
    if data["inputs"]["apply_supervision"].as_bool() != Some(true) {
        return;
    }
    let locale = renderer.locale;
    let supervision = &data["report"]["data"]["supervision"];
    let Some(rules) = supervision["rules"]
        .as_array()
        .filter(|rules| !rules.is_empty())
    else {
        renderer.line(
            content,
            locale.text(
                "Supervision was enabled, but rule measurements are unavailable.",
                "已启用监督，但无法获取规则测量结果。",
                "已啟用監督，但無法取得規則測量結果。",
            ),
        );
        return;
    };
    renderer.line(
        content,
        locale.text(
            "Supervision measurements (only executed, marked ports; coverage is partial):",
            "监督测量（仅限执行到的标记端口，覆盖不完整）：",
            "監督測量（僅限執行到的標記連接埠，覆蓋不完整）：",
        ),
    );
    for rule in rules {
        let count = |key: &str| {
            rule[key]
                .as_u64()
                .map_or_else(|| "?".into(), |v| v.to_string())
        };
        renderer.line(
            content,
            &format!(
                "{}: {} / {} · {} / {} · {}: {} · {}: {} · {}: {}",
                locale.text("Rule", "规则", "規則"),
                text::plain_text(rule["mode"].as_str().unwrap_or("?")),
                text::plain_text(rule["kind"].as_str().unwrap_or("?")),
                text::plain_text(rule["module_path"].as_str().unwrap_or("?")),
                text::plain_text(rule["port"].as_str().unwrap_or("?")),
                locale.text("measured steps", "测量步数", "測量步數"),
                count("steps_observed"),
                locale.text("elements", "元素数", "元素數"),
                count("element_count"),
                locale.text("violations", "违规数", "違規數"),
                count("violation_count"),
            ),
        );
        if rule["steps_observed"].as_u64() == Some(0) {
            renderer.line(
                content,
                locale.text(
                    "No measurable data for this rule; enabled does not mean checked.",
                    "此规则没有可测数据；启用不代表已经检查。",
                    "此規則沒有可測資料；啟用不代表已經檢查。",
                ),
            );
        }
        if rule["sample_coverage"]["status"].as_str() == Some("insufficient")
            || rule["last_sample_localization"]["status"].as_str() == Some("insufficient")
        {
            renderer.line(
                content,
                locale.text(
                    "Sample localization is insufficient for this rule.",
                    "此规则的样本定位信息不足。",
                    "此規則的樣本定位資訊不足。",
                ),
            );
        }
    }
}

fn manage_runs(
    cli: &Cli,
    renderer: &Renderer,
    trials: &mut [crate::cli::BackgroundTask],
) -> io::Result<bool> {
    let locale = renderer.locale;
    loop {
        let pending_result = crate::pending_start::list(cli.project());
        let (report, list_error) = match cli.execute(&["runs", "list"]) {
            Ok(report)
                if report.operation == "runs_list"
                    && report.status == "success"
                    && report.data["runs"].is_array() =>
            {
                (Some(report), None)
            }
            other => (None, Some(renderer.result(&other))),
        };
        let empty_runs = Vec::new();
        let runs = report
            .as_ref()
            .and_then(|report| report.data["runs"].as_array())
            .unwrap_or(&empty_runs);
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Training runs", "训练记录", "訓練記錄"),
        );
        if let Some(error) = &list_error {
            renderer.line(
                &mut content,
                locale.text(
                    "Full run list is unavailable; open a known run ID or review pending requests below.",
                    "完整运行列表不可用；可输入已知运行 ID 或在下方检查待确认请求。",
                    "完整執行清單無法使用；可輸入已知執行 ID 或在下方檢查待確認請求。",
                ),
            );
            renderer.line(&mut content, error);
        }
        let pending = match pending_result {
            Ok(pending) => pending,
            Err(error) => {
                renderer.line(
                    &mut content,
                    &format!(
                        "{}: {}",
                        locale.text(
                            "Unable to inspect pending starts",
                            "无法检查待确认启动",
                            "無法檢查待確認啟動"
                        ),
                        text::plain_text(&error.to_string())
                    ),
                );
                Vec::new()
            }
        };
        let lookup_available = list_error.is_some()
            && matches!(
                cli.check_capabilities(&["runs.lookup_request_id"]),
                Ok(CapabilityCheck::Available)
            );
        for (index, id) in pending.iter().enumerate() {
            let lookup = if list_error.is_some() && lookup_available && index < 16 {
                Some(lookup_training_request(cli, id))
            } else {
                None
            };
            let matching: Vec<&Value> = if let Some(Ok((matches, _))) = &lookup {
                matches.iter().collect()
            } else {
                runs.iter()
                    .filter(|run| {
                        run["request_id"].as_str() == Some(id) && run["operation"] == "train"
                    })
                    .collect()
            };
            renderer.line(
                &mut content,
                &format!(
                    "{} {}: {}",
                    locale.text("Pending start", "待确认启动", "待確認啟動"),
                    index + 1,
                    text::plain_text(id)
                ),
            );
            if list_error.is_some() && !matches!(&lookup, Some(Ok(_))) {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Request lookup is unavailable or omitted; outcome remains unknown.",
                        "请求查询不可用或未执行，结果仍未知。",
                        "請求查詢無法使用或未執行，結果仍未知。",
                    ),
                );
            } else if matching.is_empty() {
                renderer.line(
                    &mut content,
                    locale.text(
                        "No matching run is visible; outcome remains unknown.",
                        "尚无匹配运行，结果仍未知。",
                        "尚無相符執行，結果仍未知。",
                    ),
                );
            } else {
                for run in matching {
                    renderer.line(
                        &mut content,
                        &format!(
                            "{}: {} · {}",
                            locale.text("Matching run", "匹配运行", "相符執行"),
                            text::plain_text(run["id"].as_str().unwrap_or("?")),
                            text::plain_text(run["status"].as_str().unwrap_or("unknown"))
                        ),
                    );
                }
            }
            if let Some(Ok((_, complete))) = &lookup {
                renderer.line(
                    &mut content,
                    locale.text(
                        "The status above is recorded, not live; refresh to verify.",
                        "上述为记录状态，并非实时状态；请刷新核实。",
                        "上述為記錄狀態，並非即時狀態；請重新整理確認。",
                    ),
                );
                if !complete {
                    renderer.line(
                        &mut content,
                        locale.text(
                            "Some run records were unreadable or omitted; other matches may exist.",
                            "部分运行记录无法读取或未列出，可能还有其他匹配项。",
                            "部分執行記錄無法讀取或未列出，可能還有其他相符項目。",
                        ),
                    );
                }
            }
        }
        if !pending.is_empty() {
            renderer.line(&mut content, locale.text("After inspecting, use 'forget NUMBER' to clear a pending marker. This does not stop or delete a run.", "检查后可用 'forget 序号' 清除待确认标记；不会停止或删除运行。", "檢查後可用 'forget 序號' 清除待確認標記；不會停止或刪除執行。"));
        }
        for (index, run) in runs.iter().enumerate() {
            renderer.line(
                &mut content,
                &format!(
                    "{}. {} · {}",
                    index + 1,
                    text::plain_text(run["id"].as_str().unwrap_or("?")),
                    text::plain_text(run["status"].as_str().unwrap_or("unknown"))
                ),
            );
        }
        for (index, task) in trials
            .iter_mut()
            .enumerate()
            .filter(|(_, task)| task.project == cli.project())
        {
            let status = match task.poll() {
                Ok(false) if task.cancellation_requested => "cancellation requested",
                Ok(false) => "running",
                Ok(true) => "finished",
                Err(_) => "status unknown",
            };
            renderer.line(
                &mut content,
                &format!("Trial task {}: {}", index + 1, status),
            );
            if let Some(result) = task.result() {
                match result {
                    Ok(report) if report.operation == "training_probe" => content.push_str(&renderer.result(result)),
                    _ => renderer.line(&mut content, locale.text("Trial result unavailable or invalid; inspect saved runs before retrying.", "试跑结果不可用或无效；重试前请检查已保存运行。", "試跑結果無法使用或無效；重試前請檢查已儲存執行。")),
                }
            }
        }
        renderer.line(&mut content, locale.text("Exiting requests cancellation of session trials. A cancellation request is not confirmation of termination.", "退出 TUI 时会请求取消会话试跑；请求取消不代表已经结束。", "結束 TUI 時會要求取消工作階段試跑；要求取消不代表已經結束。"));
        if runs.is_empty() && list_error.is_none() {
            renderer.line(
                &mut content,
                locale.text("No saved runs.", "没有运行记录。", "沒有執行記錄。"),
            );
        }
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        let Some(choice) = init_input(
            renderer,
            locale.text(
                "Run number, refresh, forget NUMBER, cancel trial NUMBER, run ID, or b back",
                "运行序号、refresh 刷新、forget 序号、cancel trial 序号取消试跑、运行 ID，或 b 返回",
                "執行序號、refresh 重新整理、forget 序號、cancel trial 序號取消試跑、執行 ID，或 b 返回",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if choice == "refresh" {
            continue;
        }
        if let Some(id) = choice
            .strip_prefix("forget ")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .and_then(|n| n.checked_sub(1))
            .and_then(|index| pending.get(index))
        {
            let Some(confirm) = init_input(
                renderer,
                locale.text(
                    "Forget this pending start? The run may still start or continue. Type y",
                    "清除此待确认启动？运行仍可能启动或继续。输入 y",
                    "清除此待確認啟動？執行仍可能啟動或繼續。輸入 y",
                ),
                "",
            )?
            else {
                return Ok(true);
            };
            if confirm.eq_ignore_ascii_case("y") {
                if let Err(error) = crate::pending_start::clear(cli.project(), id) {
                    if !show_flow_page(
                        renderer,
                        &format!("{}\n", text::plain_text(&error.to_string())),
                    )? {
                        return Ok(false);
                    }
                }
            }
            continue;
        }
        if let Some(index) = choice
            .strip_prefix("cancel trial ")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .and_then(|n| n.checked_sub(1))
        {
            if let Some(task) = trials
                .get_mut(index)
                .filter(|task| task.project == cli.project())
            {
                if let Err(error) = task.cancel() {
                    show_flow_page(
                        renderer,
                        &format!("{}\n", text::plain_text(&error.to_string())),
                    )?;
                }
            } else {
                show_flow_page(
                    renderer,
                    locale.text(
                        "No such trial in this project.\n",
                        "当前项目没有该试跑。\n",
                        "目前專案沒有該試跑。\n",
                    ),
                )?;
            }
            continue;
        }
        if choice == "b" || choice.is_empty() {
            return Ok(true);
        }
        let Some(id) = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| runs.get(n))
            .and_then(|r| r["id"].as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                (choice.starts_with("run-")
                    && choice.len() <= 96
                    && choice
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'))
                .then_some(choice.as_str())
            })
        else {
            if !show_flow_page(
                renderer,
                locale.text(
                    "Enter a listed run number or a full run ID.\n",
                    "请输入列表中的运行序号或完整运行 ID。\n",
                    "請輸入清單中的執行序號或完整執行 ID。\n",
                ),
            )? {
                return Ok(false);
            }
            continue;
        };
        let known = runs.iter().find(|run| run["id"].as_str() == Some(id));
        let is_evaluation = if let Some(run) = known {
            run["operation"] == "evaluate"
        } else {
            cli.execute(&["runs", "show", id]).is_ok_and(|report| {
                report.operation == "runs_show"
                    && report.status == "success"
                    && report.data["record"]["id"].as_str() == Some(id)
                    && report.data["record"]["operation"] == "evaluate"
            })
        };
        if is_evaluation {
            if !browse_evaluation(cli, renderer, id)? {
                return Ok(false);
            }
            continue;
        }
        loop {
            let detail = match cli.execute(&["runs", "attach", id, "--follow=false"]) {
                Ok(report)
                    if report.operation == "runs_attach"
                        && report.status == "success"
                        && report.data["run_id"].as_str() == Some(id) =>
                {
                    report
                }
                other => {
                    print!("{}", renderer.result(&other));
                    break;
                }
            };
            let status = detail.data["record"]["status"]
                .as_str()
                .unwrap_or("unknown");
            let mut content = String::new();
            renderer.heading(
                &mut content,
                locale.text("Run status", "运行状态", "執行狀態"),
            );
            renderer.line(&mut content, &format!("Run ID: {}", text::plain_text(id)));
            renderer.line(
                &mut content,
                &format!("Status: {}", text::plain_text(status)),
            );
            if let Some(heartbeat) = detail.data["record"]["heartbeat_status"].as_str() {
                renderer.line(
                    &mut content,
                    &format!("Heartbeat: {}", text::plain_text(heartbeat)),
                );
            }
            let summary = &detail.data["record"]["summary"];
            if let Some(saved) = summary["checkpoint_saved"].as_bool() {
                renderer.line(&mut content, &format!("Checkpoint saved: {saved}"));
            }
            let resumed = &summary["resume_checkpoint"];
            if let Some(digest) = resumed["sha256"].as_str() {
                renderer.line(
                    &mut content,
                    &format!(
                        "{}: {} · {}: {} · SHA-256: {}",
                        locale.text("Resumed checkpoint", "续跑来源检查点", "續跑來源檢查點"),
                        text::plain_text(resumed["file"].as_str().unwrap_or("?")),
                        locale.text("step", "步数", "步數"),
                        resumed["completed_steps"],
                        text::plain_text(digest)
                    ),
                );
                if resumed["loaded_from_verified_copy"] == true {
                    renderer.line(
                        &mut content,
                        locale.text(
                            "Loaded from the confirmed verified copy.",
                            "从确认过的校验副本载入。",
                            "從確認過的驗證副本載入。",
                        ),
                    );
                } else {
                    renderer.line(&mut content, locale.text("The digest was observed before loading; this run did not bind loading to a verified copy.", "该摘要在载入前观察到；此运行未绑定校验副本。", "該摘要在載入前觀察到；此執行未綁定驗證副本。"));
                }
            }
            if let Some(note) = detail.data["record"]["status_note"].as_str() {
                renderer.line(&mut content, &text::plain_text(note));
            }
            if matches!(
                status,
                "completed" | "failed" | "interrupted" | "unknown" | "stopped"
            ) {
                match cli.execute(&["runs", "show", id]) {
                    Ok(report) if report.operation == "runs_show" && report.status == "success" => {
                        render_run_supervision(renderer, &mut content, &report.data);
                        for diagnostic in report.data["report"]["diagnostics"]
                            .as_array()
                            .into_iter()
                            .flatten()
                        {
                            renderer.line(
                                &mut content,
                                &format!(
                                    "[{}] {}",
                                    text::plain_text(
                                        diagnostic["code"].as_str().unwrap_or("unknown")
                                    ),
                                    text::plain_text(diagnostic["message"].as_str().unwrap_or(""))
                                ),
                            );
                            if let Some(hint) = diagnostic["hint"].as_str() {
                                renderer.line(&mut content, &text::plain_text(hint));
                            }
                        }
                        if let Some(cause) = report.data["report"]["data"]["cause_code"].as_str() {
                            renderer
                                .line(&mut content, &format!("Cause: {}", text::plain_text(cause)));
                        }
                        let changes = &report.data["report"]["data"]["runtime_changes"];
                        for (key, label) in [
                            (
                                "python_version",
                                locale.text("Python version", "Python 版本", "Python 版本"),
                            ),
                            (
                                "pytorch_version",
                                locale.text("PyTorch version", "PyTorch 版本", "PyTorch 版本"),
                            ),
                            (
                                "environment_fingerprint_sha256",
                                locale.text(
                                    "Environment inventory SHA-256",
                                    "环境清单 SHA-256",
                                    "環境清單 SHA-256",
                                ),
                            ),
                        ] {
                            if let Some(change) = changes[key].as_object() {
                                let before = change
                                    .get("expected")
                                    .and_then(Value::as_str)
                                    .unwrap_or("?");
                                let after =
                                    change.get("actual").and_then(Value::as_str).unwrap_or("?");
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{}: {}",
                                        label,
                                        locale.text("changed", "已变化", "已變化")
                                    ),
                                );
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{}: {}",
                                        locale.text("Recorded", "原记录", "原記錄"),
                                        text::plain_text(before)
                                    ),
                                );
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{}: {}",
                                        locale.text("Current", "当前", "目前"),
                                        text::plain_text(after)
                                    ),
                                );
                            }
                        }
                    }
                    _ => renderer.line(
                        &mut content,
                        locale.text(
                            "Run diagnostics and supervision details unavailable.",
                            "无法读取运行诊断和监督详情。",
                            "無法讀取執行診斷和監督詳情。",
                        ),
                    ),
                }
            }
            if let Some(last) = detail.data["progress"]
                .as_array()
                .and_then(|items| items.last())
            {
                renderer.line(
                    &mut content,
                    &format!("Latest progress: {}", text::plain_text(&last.to_string())),
                );
            } else {
                renderer.line(
                    &mut content,
                    locale.text(
                        "No progress reported yet.",
                        "尚未收到进度。",
                        "尚未收到進度。",
                    ),
                );
            }
            if !show_flow_page(renderer, &content)? {
                return Ok(false);
            }
            let Some(action) = init_input(
                renderer,
                locale.text(
                    "refresh, watch, stop, resume, compare, evaluate, export or b back",
                    "refresh 刷新、watch 实时查看、stop 停止、resume 续跑、compare 对照、evaluate 评估、export 导出或 b 返回",
                    "refresh 重新整理、watch 即時檢視、stop 停止、resume 續跑、compare 對照、evaluate 評估、export 匯出或 b 返回",
                ),
                "refresh",
            )?
            else {
                break;
            };
            match action.as_str() {
                "refresh" => continue,
                "watch" => {
                    #[cfg(unix)]
                    if !renderer.line_mode && crate::terminal::menu::supported() {
                        watch_run(cli, renderer, id)?;
                        continue;
                    }
                    show_flow_page(
                        renderer,
                        locale.text(
                            "Live watch requires a full-screen terminal; use refresh here.\n",
                            "实时查看需要全屏终端；此处请使用 refresh。\n",
                            "即時檢視需要全螢幕終端；此處請使用 refresh。\n",
                        ),
                    )?;
                }
                "stop" => {
                    let Some(confirm) = init_input(
                        renderer,
                        locale.text(
                            "Request a checkpointed stop? Type y",
                            "请求保存检查点后停止？输入 y",
                            "請求儲存檢查點後停止？輸入 y",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    if confirm.eq_ignore_ascii_case("y") {
                        let result = cli.execute(&["runs", "stop", id]);
                        let mut content = renderer.result(&result);
                        if let Ok(report) = &result {
                            if let Some(message) = report.data["message"].as_str() {
                                renderer.line(&mut content, &text::plain_text(message));
                            }
                        }
                        if !show_flow_page(renderer, &content)? {
                            return Ok(false);
                        }
                    }
                }
                "resume" => {
                    if let Some(message) = pending_start_notice(cli, locale) {
                        if !show_flow_page(renderer, &message)? {
                            return Ok(false);
                        }
                        continue;
                    }
                    if let Some(message) = capability_unavailable(
                        cli,
                        renderer.locale,
                        &["training.conditional_resume_checkpoint"],
                    ) {
                        if !show_flow_page(renderer, &message)? {
                            return Ok(false);
                        }
                        continue;
                    }
                    let checkpoint_report = cli.execute(&["runs", "show", id]);
                    let checkpoint = match &checkpoint_report {
                        Ok(report)
                            if report.operation == "runs_show"
                                && report.status == "success"
                                && report.data["record"]["id"].as_str() == Some(id) =>
                        {
                            &report.data["checkpoint"]
                        }
                        _ => {
                            if !show_flow_page(renderer, &renderer.result(&checkpoint_report))? {
                                return Ok(false);
                            }
                            continue;
                        }
                    };
                    let Some(digest) = checkpoint["sha256"].as_str().filter(|digest| {
                        digest.len() == 64
                            && digest
                                .bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    }) else {
                        if !show_flow_page(renderer, locale.text("Checkpoint identity is unavailable; cannot safely confirm resume.\n", "无法获取检查点身份，不能安全确认续跑。\n", "無法取得檢查點身分，無法安全確認續跑。\n"))? { return Ok(false); }
                        continue;
                    };
                    let mut checkpoint_page = String::new();
                    renderer.heading(
                        &mut checkpoint_page,
                        locale.text("Resume checkpoint", "续跑检查点", "續跑檢查點"),
                    );
                    renderer.line(
                        &mut checkpoint_page,
                        &format!(
                            "{}: {}",
                            locale.text("Run ID", "运行 ID", "執行 ID"),
                            text::plain_text(id)
                        ),
                    );
                    renderer.line(
                        &mut checkpoint_page,
                        &format!(
                            "{}: {}",
                            locale.text("Checkpoint file", "检查点文件", "檢查點檔案"),
                            text::plain_text(checkpoint["file"].as_str().unwrap_or("?"))
                        ),
                    );
                    renderer.line(
                        &mut checkpoint_page,
                        &format!(
                            "{}: {} · SHA-256: {}",
                            locale.text("Completed steps", "已完成步数", "已完成步數"),
                            checkpoint["completed_steps"],
                            digest
                        ),
                    );
                    if let Ok(report) = &checkpoint_report {
                        let recorded = &report.data["report"]["data"];
                        let runtime = &recorded["runtime"];
                        renderer.line(
                            &mut checkpoint_page,
                            &format!(
                                "{}: {} / {}",
                                locale.text(
                                    "Recorded Python / PyTorch",
                                    "原运行 Python / PyTorch",
                                    "原執行 Python / PyTorch"
                                ),
                                text::plain_text(runtime["python_version"].as_str().unwrap_or("?")),
                                text::plain_text(
                                    runtime["pytorch_version"].as_str().unwrap_or("?")
                                )
                            ),
                        );
                        let source_count = recorded["source_manifest"]
                            .as_object()
                            .map(|m| m.len().to_string())
                            .unwrap_or_else(|| "?".into());
                        renderer.line(
                            &mut checkpoint_page,
                            &format!(
                                "{}: {}",
                                locale.text(
                                    "Previously observed source files",
                                    "原运行已观察源码文件数",
                                    "原執行已觀察原始碼檔案數"
                                ),
                                source_count
                            ),
                        );
                        renderer.line(&mut checkpoint_page, locale.text("These are recorded values, not a current comparison. The worker checks recorded source and available environment evidence on load; missing evidence limits verification. Data content may still differ.", "以上是原运行记录，不代表当前比较结果。worker 载入时检查记录的源码及可用环境证据；缺失证据会限制核验，数据内容仍可能变化。", "以上是原執行記錄，不代表目前比較結果。worker 載入時檢查記錄的原始碼及可用環境證據；缺少證據會限制核驗，資料內容仍可能變化。"));
                    }
                    renderer.line(&mut checkpoint_page, locale.text("The worker loads a verified temporary copy. It needs temporary space at least as large as the checkpoint (TMPDIR); the copy is removed after loading. Source, data and environment have separate checks and limitations.", "worker 载入已校验的临时副本；临时目录（TMPDIR）须有至少等于检查点大小的空间，载入后自动删除。源码、数据和环境另有检查与限制。", "worker 載入已驗證的暫存副本；暫存目錄（TMPDIR）須有至少等於檢查點大小的空間，載入後自動刪除。原始碼、資料和環境另有檢查與限制。"));
                    if !show_flow_page(renderer, &checkpoint_page)? {
                        return Ok(false);
                    }
                    let Some(steps) = init_input(
                        renderer,
                        locale.text(
                            "Additional steps from latest checkpoint",
                            "从最新检查点额外运行的步数",
                            "從最新檢查點額外執行的步數",
                        ),
                        "1",
                    )?
                    else {
                        continue;
                    };
                    let Some(reset) = init_input(
                        renderer,
                        locale.text(
                            "Allow unknown callable state to reset? This may repeat data. y/n",
                            "允许未知的自定义状态重置？这可能重复数据。y/n",
                            "允許未知的自訂狀態重設？這可能重複資料。y/n",
                        ),
                        "n",
                    )?
                    else {
                        continue;
                    };
                    if !matches!(reset.as_str(), "y" | "n" | "Y" | "N") {
                        continue;
                    }
                    let Some(confirm) = init_input(
                        renderer,
                        locale.text(
                            "Resume this run in background? Type y",
                            "后台续跑此运行？输入 y",
                            "背景續跑此執行？輸入 y",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    if confirm.eq_ignore_ascii_case("y") {
                        let request_id = match new_training_request_id(cli) {
                            Ok(id) => id,
                            Err(error) => {
                                if !show_flow_page(renderer, &format!("{}: {}\n", locale.text("Unable to prepare a safe start; resume was not launched", "无法准备安全启动，未启动续跑", "無法準備安全啟動，未啟動續跑"), text::plain_text(&error.to_string())))? { return Ok(false); }
                                continue;
                            }
                        };
                        let mut args = vec![
                            "train",
                            "--resume",
                            id,
                            "--additional-steps",
                            &steps,
                            "--if-checkpoint-sha256",
                            digest,
                            "--detach",
                        ];
                        if reset.eq_ignore_ascii_case("y") {
                            args.push("--allow-state-reset");
                        }
                        if let Some(request_id) = &request_id {
                            args.extend(["--request-id", request_id]);
                        }
                        let result = cli.execute(&args).and_then(|report| {
                            if report.status == "error"
                                || (report.status == "success"
                                    && report.operation == "train"
                                    && report.data["detached"] == true
                                    && report.data["run_id"]
                                        .as_str()
                                        .is_some_and(|id| !id.is_empty())
                                    && request_id.as_ref().is_none_or(|id| {
                                        report.data["request_id"].as_str() == Some(id)
                                    }))
                            {
                                Ok(report)
                            } else {
                                Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "unexpected resume receipt",
                                ))
                            }
                        });
                        if result.is_err() {
                            println!(
                                "{}",
                                locale.text(
                                    "Resume outcome unknown; refresh runs before retrying.",
                                    "续跑结果未知；重试前请刷新运行记录。",
                                    "續跑結果未知；重試前請重新整理執行記錄。"
                                )
                            );
                            if !show_runs_after_uncertain_start(
                                cli,
                                renderer,
                                None,
                                request_id.as_deref(),
                            )? {
                                return Ok(false);
                            }
                        }
                        if result.as_ref().is_ok_and(|report| {
                            report
                                .diagnostics
                                .iter()
                                .any(|diagnostic| diagnostic.code == "RUN_REQUEST_ID_EXISTS")
                        }) && !show_runs_after_uncertain_start(
                            cli,
                            renderer,
                            None,
                            request_id.as_deref(),
                        )? {
                            return Ok(false);
                        }
                        let mut content = renderer.result(&result);
                        if let Ok(report) = &result {
                            if let Some(id) = report.data["run_id"].as_str() {
                                renderer.line(
                                    &mut content,
                                    &format!("Run ID: {}", text::plain_text(id)),
                                );
                            }
                        }
                        if result.is_ok() && !show_flow_page(renderer, &content)? {
                            return Ok(false);
                        }
                        if result
                            .as_ref()
                            .is_ok_and(|report| report.status == "success")
                        {
                            if let Some(request_id) = &request_id {
                                if let Err(error) =
                                    crate::pending_start::clear(cli.project(), request_id)
                                {
                                    eprintln!(
                                        "{}: {}",
                                        locale.text(
                                            "Could not clear pending start; inspect Training runs",
                                            "无法清除待确认启动，请检查训练记录",
                                            "無法清除待確認啟動，請檢查訓練記錄"
                                        ),
                                        text::plain_text(&error.to_string())
                                    );
                                }
                            }
                        }
                        break;
                    }
                }
                "compare" => {
                    if !compare_run(cli, renderer, id, runs)? {
                        return Ok(false);
                    }
                }
                "evaluate" => {
                    if !evaluate_run(cli, renderer, id)? {
                        return Ok(false);
                    }
                }
                "export" => {
                    if !export_run(cli, renderer, id)? {
                        return Ok(false);
                    }
                }
                _ => break,
            }
        }
    }
}

fn compare_run(cli: &Cli, renderer: &Renderer, id: &str, runs: &[Value]) -> io::Result<bool> {
    let locale = renderer.locale;
    let Some(choice) = init_input(
        renderer,
        locale.text(
            "Compare with run number from the list",
            "选择列表中要对照的运行序号",
            "選擇清單中要對照的執行序號",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(other) = choice
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| runs.get(n))
        .and_then(|run| run["id"].as_str())
    else {
        return Ok(true);
    };
    let result = cli.execute(&["runs", "compare", id, other]);
    let report = match &result {
        Ok(report)
            if report.operation == "runs_compare"
                && report.status == "success"
                && report.data["left_run_id"].as_str() == Some(id)
                && report.data["right_run_id"].as_str() == Some(other)
                && report.data["dimensions"].is_array() =>
        {
            report
        }
        Ok(report) if report.status == "error" => {
            return show_flow_page(renderer, &renderer.result(&result));
        }
        _ => {
            return show_flow_page(
                renderer,
                "Invalid comparison report; no comparison was recorded.\n",
            );
        }
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Run comparison", "运行对照", "執行對照"),
    );
    renderer.line(
        &mut content,
        &format!("{} ↔ {}", text::plain_text(id), text::plain_text(other)),
    );
    for dimension in report.data["dimensions"].as_array().into_iter().flatten() {
        let name = dimension["name"].as_str().unwrap_or("unknown");
        let reported_status = dimension["status"].as_str().unwrap_or("unknown");
        // Older CLIs used "same" for matching user declarations; that did not
        // independently verify data contents.
        let status = if name == "data" && reported_status == "same" {
            "unknown"
        } else {
            reported_status
        };
        renderer.line(
            &mut content,
            &format!("{}: {}", text::plain_text(name), text::plain_text(status)),
        );
        if let Some(reason) = dimension["reason"].as_str() {
            renderer.line(&mut content, &text::plain_text(reason));
        }
        if name == "data" {
            let settings_status = dimension["settings_status"]
                .as_str()
                .unwrap_or(reported_status);
            let declared_status = dimension["declared_version_status"]
                .as_str()
                .unwrap_or("unknown");
            renderer.line(
                &mut content,
                &format!(
                    "{}: {} · {}: {}",
                    locale.text("Data settings", "数据设置", "資料設定"),
                    text::plain_text(settings_status),
                    locale.text(
                        "Declared version comparison",
                        "声明版本对比",
                        "宣告版本比較"
                    ),
                    text::plain_text(declared_status)
                ),
            );
            renderer.line(
                &mut content,
                locale.text(
                    "Data content identity is not independently verified.",
                    "数据内容身份未经独立验证。",
                    "資料內容身分未經獨立驗證。",
                ),
            );
            let declared = |side: &str| {
                dimension[side]["data_fingerprint"]
                    .as_str()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or("unknown")
            };
            renderer.line(
                &mut content,
                &format!(
                    "{}: {} ↔ {}",
                    locale.text("Declared data versions", "声明的数据版本", "聲明的資料版本"),
                    text::plain_text(declared("left")),
                    text::plain_text(declared("right"))
                ),
            );
        }
        if name == "metrics" {
            renderer.line(
                &mut content,
                &format!(
                    "  left: {}",
                    text::plain_text(&dimension["left"].to_string())
                ),
            );
            renderer.line(
                &mut content,
                &format!(
                    "  right: {}",
                    text::plain_text(&dimension["right"].to_string())
                ),
            );
        }
    }
    renderer.line(
        &mut content,
        locale.text(
            "Unknown means evidence is missing; differences do not establish causation.",
            "unknown 表示证据不足；差异不等于因果关系。",
            "unknown 表示證據不足；差異不等於因果關係。",
        ),
    );
    show_flow_page(renderer, &content)
}

fn browse_evaluation(cli: &Cli, renderer: &Renderer, id: &str) -> io::Result<bool> {
    let locale = renderer.locale;
    let result = cli.execute(&["runs", "show", id]);
    let report = match &result {
        Ok(report)
            if report.operation == "runs_show"
                && report.status == "success"
                && report.data["record"]["id"].as_str() == Some(id)
                && report.data["record"]["operation"] == "evaluate" =>
        {
            report
        }
        _ => return show_flow_page(renderer, &renderer.result(&result)),
    };
    let data = &report.data["report"]["data"];
    let previews = &data["evaluation"]["sample_previews"];
    let Some(samples) = previews["samples"].as_array() else {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Saved evaluation", "已保存评估", "已儲存評估"),
        );
        renderer.line(&mut content, &format!("Run: {}", text::plain_text(id)));
        renderer.line(
            &mut content,
            &format!(
                "Metrics: {}",
                text::plain_text(&data["evaluation"]["metrics"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            locale.text(
                "Prediction previews unavailable or not reported.",
                "预测预览不可用或未报告。",
                "預測預覽無法使用或未回報。",
            ),
        );
        return show_flow_page(renderer, &content);
    };
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        return browse_evaluation_terminal(renderer, id, data, samples);
    }
    let mut offset = 0usize;
    const PAGE_SIZE: usize = 8;
    loop {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text(
                "Saved evaluation predictions",
                "已保存评估预测",
                "已儲存評估預測",
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Run: {} · Source run: {}",
                text::plain_text(id),
                text::plain_text(data["source_run_id"].as_str().unwrap_or("unknown"))
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Metrics: {}",
                text::plain_text(&data["evaluation"]["metrics"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Checkpoint: {}",
                text::plain_text(&data["checkpoint"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Samples evaluated: {}",
                data["evaluation"]["samples_evaluated"]
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Data identity: {}",
                text::plain_text(&data["provenance"]["data_identity"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Preview state: {} · Reported: {}",
                text::plain_text(previews["state"].as_str().unwrap_or("unknown")),
                samples.len()
            ),
        );
        for (index, sample) in samples.iter().enumerate().skip(offset).take(PAGE_SIZE) {
            renderer.line(
                &mut content,
                &format!(
                    "{}. {}",
                    index + 1,
                    text::plain_text(sample["sample_id"].as_str().unwrap_or("unknown"))
                ),
            );
        }
        if samples.is_empty() {
            renderer.line(
                &mut content,
                locale.text(
                    "No prediction previews were reported.",
                    "没有报告预测预览。",
                    "沒有回報預測預覽。",
                ),
            );
        }
        renderer.line(
            &mut content,
            locale.text(
                "Only explicitly reported previews are shown; they are not the complete dataset.",
                "仅显示显式报告的预览，不代表完整数据集。",
                "僅顯示明確回報的預覽，不代表完整資料集。",
            ),
        );
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        let Some(choice) = init_input(
            renderer,
            locale.text(
                "Sample number, next, prev or b back",
                "样本序号、next 下一页、prev 上一页或 b 返回",
                "樣本序號、next 下一頁、prev 上一頁或 b 返回",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        match choice.as_str() {
            "next" => {
                if offset + PAGE_SIZE < samples.len() {
                    offset += PAGE_SIZE;
                }
            }
            "prev" => {
                offset = offset.saturating_sub(PAGE_SIZE);
            }
            _ => {
                if let Some(sample) = choice
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| samples.get(index))
                {
                    let mut detail = String::new();
                    renderer.heading(
                        &mut detail,
                        locale.text("Prediction detail", "预测详情", "預測詳情"),
                    );
                    renderer.line(&mut detail, &text::plain_text(&sample.to_string()));
                    if !show_flow_page(renderer, &detail)? {
                        return Ok(false);
                    }
                }
            }
        }
    }
}

#[cfg(unix)]
fn browse_evaluation_terminal(
    renderer: &Renderer,
    id: &str,
    data: &Value,
    samples: &[Value],
) -> io::Result<bool> {
    let locale = renderer.locale;
    let mut offset = 0usize;
    let mut focus = 0usize;
    const PAGE_SIZE: usize = 8;
    loop {
        let page = &samples[offset..samples.len().min(offset + PAGE_SIZE)];
        let mut keys: Vec<String> = (1..=page.len()).map(|index| index.to_string()).collect();
        keys.extend(["p".into(), "n".into(), "i".into(), "b".into()]);
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let Some(selected) =
            crate::terminal::menu::select_from(&key_refs, focus.min(keys.len() - 1), |selected| {
                let renderer = Renderer::current(locale, renderer.style, false);
                let mut content = String::new();
                renderer.heading(
                    &mut content,
                    locale.text(
                        "Saved evaluation predictions",
                        "已保存评估预测",
                        "已儲存評估預測",
                    ),
                );
                let line = |content: &mut String, value: String| {
                    renderer.line(
                        content,
                        &text::truncate(
                            &text::plain_text(&value).replace('\n', " "),
                            renderer.width,
                        ),
                    )
                };
                line(
                    &mut content,
                    format!("{}: {id}", locale.text("Evaluation", "评估", "評估")),
                );
                line(
                    &mut content,
                    format!(
                        "{}: {}",
                        locale.text("Source run", "源运行", "來源執行"),
                        data["source_run_id"].as_str().unwrap_or("unknown")
                    ),
                );
                line(
                    &mut content,
                    format!(
                        "{}: {}",
                        locale.text("Declared data", "声明数据版本", "聲明資料版本"),
                        data["provenance"]["data_identity"]["data_fingerprint"]
                            .as_str()
                            .unwrap_or("unknown")
                    ),
                );
                line(
                    &mut content,
                    format!(
                        "{}: {}",
                        locale.text("Metrics", "指标", "指標"),
                        data["evaluation"]["metrics"]
                    ),
                );
                line(
                    &mut content,
                    format!(
                        "{} {} / {} · {} {}",
                        locale.text("Page", "页", "頁"),
                        offset / PAGE_SIZE + 1,
                        samples.len().saturating_sub(1) / PAGE_SIZE + 1,
                        samples.len(),
                        locale.text("reported previews", "条报告预览", "筆回報預覽")
                    ),
                );
                for (index, sample) in page.iter().enumerate() {
                    let marker = if selected == index { ">" } else { " " };
                    line(
                        &mut content,
                        format!(
                            "{marker} {}. {}",
                            index + 1,
                            sample["sample_id"].as_str().unwrap_or("unknown")
                        ),
                    );
                }
                if page.is_empty() {
                    renderer.line(
                        &mut content,
                        locale.text(
                            "No prediction previews reported.",
                            "没有报告预测预览。",
                            "沒有回報預測預覽。",
                        ),
                    );
                }
                for (index, label) in [
                    locale.text("p Previous", "p 上一页", "p 上一頁"),
                    locale.text("n Next", "n 下一页", "n 下一頁"),
                    locale.text("i Provenance", "i 来源", "i 來源"),
                    locale.text("b Back", "b 返回", "b 返回"),
                ]
                .iter()
                .enumerate()
                {
                    let marker = if selected == page.len() + index {
                        ">"
                    } else {
                        " "
                    };
                    line(&mut content, format!("{marker} {label}"));
                }
                renderer.line(
                    &mut content,
                    locale.text(
                        "↑/↓ select · Enter · 1–8 open · previews are partial",
                        "↑/↓ 选择 · Enter · 1–8 打开 · 仅为部分预览",
                        "↑/↓ 選擇 · Enter · 1–8 開啟 · 僅為部分預覽",
                    ),
                );
                content
            })?
        else {
            return Ok(false);
        };
        if selected < page.len() {
            focus = selected;
            let mut detail = String::new();
            renderer.heading(
                &mut detail,
                locale.text("Prediction detail", "预测详情", "預測詳情"),
            );
            renderer.line(
                &mut detail,
                &format!(
                    "{}: {}",
                    locale.text("Evaluation", "评估", "評估"),
                    text::plain_text(id)
                ),
            );
            renderer.line(
                &mut detail,
                &format!(
                    "{}: {}",
                    locale.text("Sample ID", "样本 ID", "樣本 ID"),
                    text::plain_text(page[selected]["sample_id"].as_str().unwrap_or("unknown"))
                ),
            );
            renderer.line(
                &mut detail,
                &format!(
                    "{}: {}",
                    locale.text("Prediction", "预测", "預測"),
                    text::plain_text(&page[selected]["prediction"].to_string())
                ),
            );
            if let Some(target) = page[selected].get("target") {
                renderer.line(
                    &mut detail,
                    &format!(
                        "{}: {}",
                        locale.text("Target", "目标", "目標"),
                        text::plain_text(&target.to_string())
                    ),
                );
            }
            if !view_report(&detail, locale, renderer.style)? {
                return Ok(false);
            }
        } else {
            match selected - page.len() {
                0 => {
                    offset = offset.saturating_sub(PAGE_SIZE);
                    focus = 0;
                }
                1 if offset + PAGE_SIZE < samples.len() => {
                    offset += PAGE_SIZE;
                    focus = 0;
                }
                2 => {
                    let mut provenance = String::new();
                    renderer.heading(
                        &mut provenance,
                        locale.text("Evaluation provenance", "评估来源", "評估來源"),
                    );
                    for (label, value) in [
                        (locale.text("Evaluation", "评估", "評估"), id),
                        (
                            locale.text("Source run", "源运行", "來源執行"),
                            data["source_run_id"].as_str().unwrap_or("unknown"),
                        ),
                        (
                            locale.text(
                                "Declared data version",
                                "声明的数据版本",
                                "聲明的資料版本",
                            ),
                            data["provenance"]["data_identity"]["data_fingerprint"]
                                .as_str()
                                .unwrap_or("unknown"),
                        ),
                        (
                            locale.text("Split", "数据切分", "資料切分"),
                            data["provenance"]["data_identity"]["split"]
                                .as_str()
                                .unwrap_or("unknown"),
                        ),
                    ] {
                        renderer.line(
                            &mut provenance,
                            &format!("{label}: {}", text::plain_text(value)),
                        );
                    }
                    renderer.line(
                        &mut provenance,
                        locale.text(
                            "Declared data versions are not independently verified content hashes.",
                            "声明的数据版本不是独立验证的内容哈希。",
                            "聲明的資料版本不是獨立驗證的內容雜湊。",
                        ),
                    );
                    for (label, value) in [
                        ("Checkpoint", data["checkpoint"].clone()),
                        ("Data identity", data["provenance"]["data_identity"].clone()),
                        ("Metrics", data["evaluation"]["metrics"].clone()),
                        (
                            "Samples evaluated",
                            data["evaluation"]["samples_evaluated"].clone(),
                        ),
                        (
                            "Preview metadata",
                            json!({"state":data["evaluation"]["sample_previews"]["state"], "limit":data["evaluation"]["sample_previews"]["limit"], "reported":samples.len()}),
                        ),
                    ] {
                        renderer.line(
                            &mut provenance,
                            &format!("{label}: {}", text::plain_text(&value.to_string())),
                        );
                    }
                    renderer.line(
                        &mut provenance,
                        locale.text(
                            "Reported previews are not the complete dataset.",
                            "报告的预览不代表完整数据集。",
                            "回報的預覽不代表完整資料集。",
                        ),
                    );
                    if !view_report(&provenance, locale, renderer.style)? {
                        return Ok(false);
                    }
                }
                3 => return Ok(true),
                _ => {}
            }
        }
    }
}

fn evaluate_run(cli: &Cli, renderer: &Renderer, id: &str) -> io::Result<bool> {
    let locale = renderer.locale;
    if let Some(message) =
        capability_unavailable(cli, locale, &["evaluation.conditional_checkpoint"])
    {
        return show_flow_page(renderer, &message);
    }
    let Some(evaluator) = init_input(
        renderer,
        locale.text(
            "Evaluator module:callable",
            "评估函数 module:callable",
            "評估函式 module:callable",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(data) = init_input(
        renderer,
        locale.text(
            "Evaluation data factory module:callable",
            "评估数据工厂 module:callable",
            "評估資料工廠 module:callable",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(split) = init_input(
        renderer,
        locale.text("Data split", "数据切分", "資料切分"),
        "test",
    )?
    else {
        return Ok(true);
    };
    let Some(fingerprint) = init_input(
        renderer,
        locale.text(
            "Data version / fingerprint (optional)",
            "数据版本或指纹（可选）",
            "資料版本或指紋（選填）",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(sample_limit) = init_input(
        renderer,
        locale.text(
            "Maximum evaluation samples (empty for no limit)",
            "最大评估样本数（留空表示不限）",
            "最大評估樣本數（留空表示不限）",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !sample_limit.is_empty() && !sample_limit.parse::<u64>().is_ok_and(|value| value > 0) {
        println!(
            "{}",
            locale.text(
                "Sample limit must be a positive integer.",
                "样本上限必须是正整数。",
                "樣本上限必須是正整數。"
            )
        );
        return Ok(true);
    }
    let Some(preview_limit) = init_input(
        renderer,
        locale.text(
            "Prediction previews: 0–32 (0 disables)",
            "预测预览条数：0–32（0 表示关闭）",
            "預測預覽筆數：0–32（0 表示關閉）",
        ),
        "3",
    )?
    else {
        return Ok(true);
    };
    if !preview_limit.parse::<u8>().is_ok_and(|value| value <= 32) {
        println!(
            "{}",
            locale.text(
                "Preview count must be an integer from 0 to 32.",
                "预览条数必须是 0 到 32 的整数。",
                "預覽筆數必須是 0 到 32 的整數。"
            )
        );
        return Ok(true);
    }
    let Some(raw_params) = init_input(
        renderer,
        locale.text(
            "Evaluator parameters as JSON object",
            "评估函数参数（JSON 对象）",
            "評估函式參數（JSON 物件）",
        ),
        "{}",
    )?
    else {
        return Ok(true);
    };
    let evaluator_params = match parse_factory_parameters(&raw_params) {
        Ok(params) => params,
        Err(error) => {
            println!("{}", text::plain_text(&error));
            return Ok(true);
        }
    };
    let Some(raw_data_params) = init_input(
        renderer,
        locale.text(
            "Data factory parameters as JSON object",
            "数据工厂参数（JSON 对象）",
            "資料工廠參數（JSON 物件）",
        ),
        "{}",
    )?
    else {
        return Ok(true);
    };
    let data_params = match parse_factory_parameters(&raw_data_params) {
        Ok(params) => params,
        Err(error) => {
            println!("{}", text::plain_text(&error));
            return Ok(true);
        }
    };
    let checkpoint_report = cli.execute(&["runs", "show", id]);
    let checkpoint = match &checkpoint_report {
        Ok(report)
            if report.operation == "runs_show"
                && report.status == "success"
                && report.data["record"]["id"].as_str() == Some(id) =>
        {
            &report.data["checkpoint"]
        }
        _ => return show_flow_page(renderer, &renderer.result(&checkpoint_report)),
    };
    let Some(digest) = checkpoint["sha256"].as_str().filter(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }) else {
        return show_flow_page(
            renderer,
            locale.text(
                "Checkpoint identity is unavailable; cannot safely confirm evaluation.\n",
                "无法获取检查点身份，不能安全确认评估。\n",
                "無法取得檢查點身分，無法安全確認評估。\n",
            ),
        );
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Evaluation plan", "评估计划", "評估計畫"),
    );
    for (label, value) in [
        ("Run", id),
        ("Checkpoint", "latest"),
        ("Evaluator", evaluator.as_str()),
        ("Data", data.as_str()),
        ("Split", split.as_str()),
        (
            "Data identity",
            if fingerprint.is_empty() {
                "unknown"
            } else {
                fingerprint.as_str()
            },
        ),
    ] {
        renderer.line(
            &mut content,
            &format!("{label}: {}", text::plain_text(value)),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "Checkpoint file: {} · step: {} · SHA-256: {}",
            text::plain_text(checkpoint["file"].as_str().unwrap_or("?")),
            checkpoint["completed_steps"],
            digest
        ),
    );
    renderer.line(&mut content, locale.text("Evaluation loads a verified temporary copy; TMPDIR needs space at least equal to the checkpoint.", "评估载入已校验的临时副本；TMPDIR 需要至少等于检查点大小的空间。", "評估載入已驗證的暫存副本；TMPDIR 需要至少等於檢查點大小的空間。"));
    renderer.line(
        &mut content,
        &format!(
            "Sample limit: {}",
            if sample_limit.is_empty() {
                "unspecified"
            } else {
                &sample_limit
            }
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Prediction previews requested: {}",
            text::plain_text(&preview_limit)
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Evaluator parameters: {}",
            text::plain_text(&serde_json::to_string(&evaluator_params).unwrap_or_default())
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Data factory parameters: {}",
            text::plain_text(&serde_json::to_string(&data_params).unwrap_or_default())
        ),
    );
    renderer.line(&mut content, locale.text("The evaluator controls the actual sample count and must explicitly report preview records. Evaluation runs in the foreground.", "实际样本数由评估器决定，预测预览须由评估器显式报告；评估在前台执行。", "實際樣本數由評估器決定，預測預覽須由評估器明確回報；評估在前景執行。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let Some(confirm) = init_input(
        renderer,
        locale.text(
            "Run evaluation? Type y",
            "开始评估？输入 y",
            "開始評估？輸入 y",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !confirm.eq_ignore_ascii_case("y") {
        return Ok(true);
    }
    let mut args = vec![
        "evaluate",
        "--run",
        id,
        "--evaluator",
        &evaluator,
        "--data",
        &data,
        "--split",
        &split,
        "--if-checkpoint-sha256",
        digest,
    ];
    if !sample_limit.is_empty() {
        args.extend(["--samples", &sample_limit]);
    }
    if preview_limit != "0" {
        args.extend(["--preview-samples", &preview_limit]);
    }
    let evaluator_values: Vec<String> = evaluator_params
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    let data_values: Vec<String> = data_params
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    for value in &evaluator_values {
        args.extend(["--param", value]);
    }
    for value in &data_values {
        args.extend(["--data-param", value]);
    }
    if !fingerprint.is_empty() {
        args.extend(["--data-fingerprint", &fingerprint]);
    }
    let result = cli.execute(&args);
    let report = match &result {
        Ok(report)
            if report.operation == "evaluate"
                && report.status == "success"
                && report.data["source_run_id"].as_str() == Some(id)
                && report.data["checkpoint"]["sha256"].as_str() == Some(digest)
                && report.data["evaluation"]["metrics"].is_object() =>
        {
            report
        }
        Ok(report) if report.status == "error" || report.status == "inconclusive" => {
            return show_flow_page(renderer, &renderer.result(&result));
        }
        _ => return show_runs_after_uncertain_start(cli, renderer, None, None),
    };
    let identity = &report.data["provenance"]["data_identity"];
    let fingerprint_expected = (!fingerprint.is_empty()).then(|| fingerprint.trim());
    let sample_expected = if sample_limit.is_empty() {
        None
    } else {
        sample_limit.parse::<u64>().ok()
    };
    let identity_keys = identity.as_object();
    let evaluation_keys = report.data["evaluation"].as_object();
    let identity_matches = report.data["evaluation"]["entrypoint"].as_str()
        == Some(evaluator.as_str())
        && report.data["evaluation"]["split"].as_str() == Some(split.trim())
        && identity["entrypoint"].as_str() == Some(data.as_str())
        && identity["split"].as_str() == Some(split.trim())
        && serde_json::to_value(&data_params).ok().as_ref() == Some(&identity["parameters"])
        && identity_keys.is_some_and(|keys| {
            keys.contains_key("data_fingerprint") && keys.contains_key("sample_limit")
        })
        && identity["data_fingerprint"] == json!(fingerprint_expected)
        && identity["sample_limit"] == json!(sample_expected)
        && evaluation_keys.is_some_and(|keys| keys.contains_key("sample_limit"))
        && report.data["evaluation"]["sample_limit"] == json!(sample_expected);
    if !identity_matches {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text(
                "Evaluation receipt unverified",
                "评估回执无法核对",
                "評估回執無法核對",
            ),
        );
        renderer.line(&mut content, locale.text("CLI reported success, but evaluator or data identity is missing or differs from the request. Inspect the saved run before retrying; metrics are not attributed to this request.", "CLI 报告成功，但评估器或数据身份缺失或与请求不符。重试前请检查已保存运行；不将指标归属于本次请求。", "CLI 回報成功，但評估器或資料身分缺失或與要求不符。重試前請檢查已儲存執行；不將指標歸屬於本次要求。"));
        if let Some(run_id) = report.data["run_id"].as_str() {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text("Reported run ID", "报告的运行 ID", "回報的執行 ID"),
                    text::plain_text(run_id)
                ),
            );
        }
        return show_flow_page(renderer, &content);
    }
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Evaluation result", "评估结果", "評估結果"),
    );
    if !evaluator_params.is_empty() {
        renderer.line(
            &mut content,
            locale.text(
                "The receipt does not independently verify evaluator parameters.",
                "回执不能独立核对评估器参数。",
                "回執無法獨立核對評估器參數。",
            ),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "Run ID: {}",
            text::plain_text(report.data["run_id"].as_str().unwrap_or("unknown"))
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Metrics: {}",
            text::plain_text(&report.data["evaluation"]["metrics"].to_string())
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Samples: {}",
            report.data["evaluation"]["samples_evaluated"]
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Data identity: {}",
            text::plain_text(&report.data["provenance"]["data_identity"].to_string())
        ),
    );
    if report.data["provenance"]["data_identity"]["data_fingerprint"]
        .as_str()
        .is_none()
    {
        renderer.line(
            &mut content,
            locale.text(
                "Data fingerprint unknown; matching settings do not prove identical samples.",
                "数据指纹未知；设置相同不代表样本相同。",
                "資料指紋未知；設定相同不代表樣本相同。",
            ),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "Predictions: {}",
            text::plain_text(&report.data["evaluation"]["sample_previews"].to_string())
        ),
    );
    show_flow_page(renderer, &content)
}

fn export_run(cli: &Cli, renderer: &Renderer, id: &str) -> io::Result<bool> {
    let locale = renderer.locale;
    if let Some(message) = capability_unavailable(cli, locale, &["export.conditional_checkpoint"]) {
        return show_flow_page(renderer, &message);
    }
    let Some(destination) = init_input(
        renderer,
        locale.text(
            "New export directory outside the project",
            "项目目录之外的新导出目录",
            "專案目錄之外的新匯出目錄",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let checkpoint_report = cli.execute(&["runs", "show", id]);
    let checkpoint = match &checkpoint_report {
        Ok(report)
            if report.operation == "runs_show"
                && report.status == "success"
                && report.data["record"]["id"].as_str() == Some(id) =>
        {
            &report.data["checkpoint"]
        }
        _ => return show_flow_page(renderer, &renderer.result(&checkpoint_report)),
    };
    let Some(digest) = checkpoint["sha256"].as_str().filter(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }) else {
        return show_flow_page(
            renderer,
            locale.text(
                "Checkpoint identity is unavailable; cannot safely confirm export.\n",
                "无法获取检查点身份，不能安全确认导出。\n",
                "無法取得檢查點身分，無法安全確認匯出。\n",
            ),
        );
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Export plan", "导出计划", "匯出計畫"),
    );
    renderer.line(&mut content, &format!("Run: {}", text::plain_text(id)));
    renderer.line(
        &mut content,
        &format!(
            "Checkpoint file: {} · step: {} · SHA-256: {}",
            text::plain_text(checkpoint["file"].as_str().unwrap_or("?")),
            checkpoint["completed_steps"],
            digest
        ),
    );
    renderer.line(
        &mut content,
        &format!("Destination: {}", text::plain_text(&destination)),
    );
    renderer.line(&mut content, locale.text("Destination must be outside the project. Includes verified code and checkpoint; datasets and external resources are not included. Existing destinations are not overwritten.", "目标须位于项目目录之外。包含已校验代码和检查点；不包含数据集和外部资源。不会覆盖已有目标。", "目標須位於專案目錄之外。包含已校驗程式碼和檢查點；不包含資料集和外部資源。不會覆蓋現有目標。"));
    renderer.line(&mut content, locale.text("The checkpoint can contain training state or sensitive parameters. The exported README describes limited continuation, optional new checkpoints and their requirements.", "检查点可能包含训练状态或敏感参数。导出包 README 说明有限续跑、可选的新检查点及其条件。", "檢查點可能包含訓練狀態或敏感參數。匯出套件 README 說明有限續跑、可選的新檢查點及其條件。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let Some(confirm) = init_input(
        renderer,
        locale.text(
            "Export this run? Type y",
            "导出此运行？输入 y",
            "匯出此執行？輸入 y",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !confirm.eq_ignore_ascii_case("y") {
        return Ok(true);
    }
    let result = cli.execute(&[
        "export",
        &destination,
        "--run",
        id,
        "--if-checkpoint-sha256",
        digest,
    ]);
    match &result {
        Ok(report)
            if report.operation == "export"
                && report.status == "success"
                && report.data["run_id"].as_str() == Some(id)
                && report.data["checkpoint_sha256"].as_str() == Some(digest)
                && report.data["path"].is_string() =>
        {
            let mut content = String::new();
            renderer.heading(
                &mut content,
                locale.text("Export completed", "导出完成", "匯出完成"),
            );
            renderer.line(
                &mut content,
                &format!(
                    "Destination: {}",
                    text::plain_text(report.data["path"].as_str().unwrap_or("?"))
                ),
            );
            renderer.line(&mut content, locale.text("Review manifest.json and README.md; restore with a compatible Python environment.", "请查看 manifest.json 和 README.md，并使用兼容 Python 环境恢复。", "請查看 manifest.json 和 README.md，並使用相容 Python 環境還原。"));
            show_flow_page(renderer, &content)
        }
        Ok(report) if report.status == "error" => {
            show_flow_page(renderer, &renderer.result(&result))
        }
        _ => show_flow_page(
            renderer,
            locale.text(
                "Export outcome unknown; inspect the destination before retrying.\n",
                "导出结果未知；重试前请检查目标目录。\n",
                "匯出結果未知；重試前請檢查目標目錄。\n",
            ),
        ),
    }
}

enum LayerEditOutcome {
    Quit,
    Cancelled,
    Attempted { committed: bool },
}

fn render_candidate_state_effects(renderer: &Renderer, content: &mut String, effects: &Value) {
    let locale = renderer.locale;
    let Some(effects) = effects.as_array() else {
        renderer.line(
            content,
            locale.text(
                "Candidate state effects unavailable.",
                "候选状态影响不可用。",
                "候選狀態影響無法使用。",
            ),
        );
        return;
    };
    for effect in effects {
        let action = effect["action"].as_str().unwrap_or("");
        let path = effect["path"].as_str().unwrap_or("?");
        let target = if action == "add" {
            effect["name"]
                .as_str()
                .map(|name| format!("{path}.{name}"))
                .unwrap_or_else(|| path.to_owned())
        } else {
            path.to_owned()
        };
        for (key, label) in [
            (
                "preserved",
                locale.text(
                    "Candidate copied state",
                    "候选中复制的状态",
                    "候選中複製的狀態",
                ),
            ),
            (
                "initialized",
                locale.text(
                    "Candidate initialized state",
                    "候选中初始化的状态",
                    "候選中初始化的狀態",
                ),
            ),
            (
                "old_state_not_copied",
                locale.text(
                    "Previous module state not copied",
                    "未复制的原模块状态",
                    "未複製的原模組狀態",
                ),
            ),
        ] {
            if let Some(items) = effect[key].as_array() {
                renderer.line(
                    content,
                    &format!(
                        "{} · {}: {}",
                        label,
                        text::plain_text(&target),
                        text::plain_text(&Value::Array(items.clone()).to_string())
                    ),
                );
            } else if key == "old_state_not_copied"
                && matches!(action, "set" | "replace" | "remove")
            {
                renderer.line(
                    content,
                    &format!(
                        "{} · {}: {}",
                        label,
                        text::plain_text(&target),
                        locale.text("unknown", "未知", "未知")
                    ),
                );
            }
        }
    }
    renderer.line(content, locale.text("These describe the in-memory candidate, not a saved training checkpoint or its migration. State may remain in other shared modules.", "这些信息描述内存候选，不代表已保存训练检查点或其迁移结果；状态也可能仍存在于其他共享模块。", "這些資訊描述記憶體候選，不代表已儲存訓練檢查點或其遷移結果；狀態也可能仍存在於其他共享模組。"));
}

fn guide_source_edit(
    cli: &Cli,
    renderer: &Renderer,
    attempted: &mut bool,
    selected: Option<(&str, &str, &str, &str)>,
) -> io::Result<bool> {
    let mut temporary = None;
    let result = guide_source_edit_inner(cli, renderer, attempted, selected, &mut temporary);
    if let Some(directory) = temporary {
        let path = directory.path().to_path_buf();
        let cleanup = directory.close();
        let message = match &cleanup {
            Ok(()) => format!(
                "{}: {}",
                renderer.locale.text(
                    "Temporary input snapshot removed",
                    "临时输入快照已清理",
                    "暫存輸入快照已清理"
                ),
                text::plain_text(&path.display().to_string())
            ),
            Err(error) => format!(
                "{}: {} · {}",
                renderer.locale.text(
                    "Snapshot cleanup failed; remove this directory manually",
                    "快照清理失败，请手动删除目录",
                    "快照清理失敗，請手動刪除目錄"
                ),
                text::plain_text(&path.display().to_string()),
                text::plain_text(&error.to_string())
            ),
        };
        if matches!(result, Ok(true)) {
            if !show_flow_page(renderer, &format!("{message}\n"))? {
                return Ok(false);
            }
        } else {
            eprintln!("{message}");
        }
    }
    result
}

fn guide_source_edit_inner(
    cli: &Cli,
    renderer: &Renderer,
    attempted: &mut bool,
    selected: Option<(&str, &str, &str, &str)>,
    temporary: &mut Option<tempfile::TempDir>,
) -> io::Result<bool> {
    let locale = renderer.locale;
    let config = cli.execute(&["assembly", "show", "--config-only"]);
    let (model, batch) = match &config {
        Ok(report) if report.operation == "assembly_show" && report.status == "success" => {
            match (report.data["model"].as_str(), report.data["batch"].as_str()) {
                (Some(model), Some(batch)) if !model.is_empty() && !batch.is_empty() => {
                    (model, batch)
                }
                _ => {
                    return show_flow_page(
                        renderer,
                        locale.text(
                            "A project model and batch are required for source editing.\n",
                            "源码编辑需要项目模型和批次入口。\n",
                            "原始碼編輯需要專案模型和批次入口。\n",
                        ),
                    );
                }
            }
        }
        _ => return show_flow_page(renderer, &renderer.result(&config)),
    };
    let config_revision = match &config {
        Ok(report) => report.data["config_revision"].as_str().filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }),
        _ => None,
    };
    let mut intro = String::new();
    renderer.heading(
        &mut intro,
        locale.text(
            "Edit project Python source",
            "编辑项目 Python 源码",
            "編輯專案 Python 原始碼",
        ),
    );
    renderer.line(
        &mut intro,
        &format!(
            "{}: {} · {}: {}",
            locale.text("Model", "模型", "模型"),
            text::plain_text(model),
            locale.text("Batch", "批次", "批次"),
            text::plain_text(batch)
        ),
    );
    renderer.line(&mut intro, locale.text("Edits project files, not a variant. Only supported, unambiguous constructor forms are available. Existing checkpoints are not migrated.", "此操作修改项目文件，而非变体，仅支持无歧义的构造形式；不会迁移已有检查点。", "此操作修改專案檔案，而非變體，僅支援無歧義的建構形式；不會遷移現有檢查點。"));
    renderer.line(&mut intro, locale.text("For a changing batch, append +snapshot to the action (e.g. set+snapshot) to explicitly capture and replay one input.", "批次会变化时，可在操作后加 +snapshot（如 set+snapshot），显式捕获并重放一次输入。", "批次會變化時，可在操作後加 +snapshot（如 set+snapshot），明確擷取並重放一次輸入。"));
    if let Some((path, graph_model, graph_variant, graph_type)) = selected {
        renderer.line(
            &mut intro,
            &format!(
                "{}: {} · {}: {} · {}: {}",
                locale.text("Graph selection", "图中选择", "圖中選擇"),
                text::plain_text(path),
                locale.text("Graph model", "图所检查的模型", "圖所檢查的模型"),
                text::plain_text(graph_model),
                locale.text("Currently selected variant", "当前所选变体", "目前所選變體"),
                if graph_variant.is_empty() {
                    "(none)".to_owned()
                } else {
                    text::plain_text(graph_variant)
                }
            ),
        );
        renderer.line(&mut intro, locale.text("The source target is the project model above. The graph may describe a different model or variant; the target is checked again below.", "源码目标是上方项目模型。图可能对应不同模型或变体；下方将重新核对目标。", "原始碼目標是上方專案模型。圖可能對應不同模型或變體；下方將重新核對目標。"));
        renderer.line(
            &mut intro,
            &format!(
                "{}: {}",
                locale.text("Graph module type", "图中模块类型", "圖中模組類型"),
                text::plain_text(graph_type)
            ),
        );
    }
    if !show_flow_page(renderer, &intro)? {
        return Ok(false);
    }
    let Some(action) = init_input(
        renderer,
        locale.text(
            "add, set, replace, remove, migrate-starter (+snapshot optional), undo, redo, or b back",
            "add、set、replace、remove、migrate-starter（可加 +snapshot）、undo、redo，或 b 返回",
            "add、set、replace、remove、migrate-starter（可加 +snapshot）、undo、redo，或 b 返回",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let capture_snapshot = action.ends_with("+snapshot");
    let action = action
        .strip_suffix("+snapshot")
        .unwrap_or(&action)
        .to_owned();
    if (capture_snapshot && matches!(action.as_str(), "undo" | "redo"))
        || !matches!(
            action.as_str(),
            "add" | "set" | "replace" | "remove" | "migrate-starter" | "undo" | "redo"
        )
    {
        return Ok(true);
    }
    if !matches!(action.as_str(), "undo" | "redo") {
        if let Some(message) = capability_unavailable(
            cli,
            locale,
            &[
                "edit.conditional_source_plan",
                "edit.conditional_source_input",
                "edit.conditional_source_config",
                "edit.conditional_source_project_manifest",
            ],
        ) {
            return show_flow_page(renderer, &message);
        }
    }
    if action == "migrate-starter" {
        if let Some(message) = capability_unavailable(cli, locale, &["edit.starter_migration"]) {
            return show_flow_page(renderer, &message);
        }
    }
    if matches!(action.as_str(), "undo" | "redo") {
        if !show_flow_page(renderer, locale.text("Source history affects the entire last source transaction. It checks recorded file hashes but has no candidate forward preview.\n", "源码历史操作影响最近一次完整源码事务；它检查已记录文件摘要，但没有候选前向预览。\n", "原始碼歷史操作影響最近一次完整原始碼事務；它檢查已記錄檔案摘要，但沒有候選前向預覽。\n"))? { return Ok(false); }
        if !init_input(
            renderer,
            locale.text(
                "Type apply to change source history",
                "输入 apply 修改源码历史",
                "輸入 apply 修改原始碼歷史",
            ),
            "",
        )?
        .is_some_and(|v| v == "apply")
        {
            return Ok(true);
        }
        *attempted = true;
        let result = cli.execute(&["layer", &action, "--target", "source"]);
        let mut content = renderer.result(&result);
        if result.as_ref().is_ok_and(|r| {
            r.operation == "edit"
                && r.status == "success"
                && r.data["committed"] == true
                && r.data["target"] == "source"
        }) {
            let recheck = cli.execute(&["model", "probe", "--model", model]);
            renderer.line(
                &mut content,
                locale.text(
                    "Saved source history · separate forward recheck",
                    "源码历史已保存 · 独立前向重验",
                    "原始碼歷史已儲存 · 獨立前向重驗",
                ),
            );
            content.push_str(&renderer.result(&recheck));
        } else {
            renderer.line(
                &mut content,
                locale.text(
                    "Inspect source history before retrying if the outcome is unknown.",
                    "若结果未知，请检查源码历史后再重试。",
                    "若結果未知，請檢查原始碼歷史後再重試。",
                ),
            );
        }
        return show_flow_page(renderer, &content);
    }
    let Some(config_revision) = config_revision else {
        return show_flow_page(renderer, locale.text("Project configuration revision is unavailable; source editing cannot be guarded.\n", "无法获取项目配置修订，不能安全执行源码编辑。\n", "無法取得專案設定修訂，無法安全執行原始碼編輯。\n"));
    };
    let list = cli.execute(&["layer", "list", "--model", model]);
    let modules = match &list {
        Ok(report) if report.operation == "layers" && report.status == "success" => {
            match report.data["modules"].as_array() {
                Some(modules) => modules,
                None => return show_flow_page(renderer, "Invalid Layer list report.\n"),
            }
        }
        _ => return show_flow_page(renderer, &renderer.result(&list)),
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Source layer targets", "源码层目标", "原始碼層目標"),
    );
    for (index, module) in modules.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!(
                "{}. {} · {} · source-set={} · source-insert={}",
                index + 1,
                text::plain_text(module["path"].as_str().unwrap_or("?")),
                text::plain_text(module["type"].as_str().unwrap_or("?")),
                module["can_source_set"],
                module["can_source_insert"]
            ),
        );
    }
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let module = if action == "migrate-starter" {
        let Some(module) = modules.iter().find(|module| module["path"] == "model") else {
            return show_flow_page(
                renderer,
                locale.text(
                    "The source model root is unavailable.\n",
                    "源码模型根节点不可用。\n",
                    "原始碼模型根節點不可用。\n",
                ),
            );
        };
        module
    } else if let Some((path, _, _, graph_type)) = selected {
        let mut matches = modules
            .iter()
            .filter(|module| module["path"].as_str() == Some(path));
        let Some(module) = matches.next().filter(|_| matches.next().is_none()) else {
            return show_flow_page(renderer, locale.text("The graph path is absent or ambiguous in the current source model; inspect again.\n", "图中路径在当前源码模型中不存在或不唯一，请重新检查。\n", "圖中路徑在目前原始碼模型中不存在或不唯一，請重新檢查。\n"));
        };
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text(
                "Graph selection in current source model",
                "图中选择在当前源码模型中的对应项",
                "圖中選擇在目前原始碼模型中的對應項",
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "{} · {} · {}",
                text::plain_text(model),
                text::plain_text(path),
                text::plain_text(module["type"].as_str().unwrap_or("?"))
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Graph module type", "图中模块类型", "圖中模組類型"),
                text::plain_text(graph_type)
            ),
        );
        if module["type"].as_str() != Some(graph_type) {
            renderer.line(&mut content, locale.text("The graph and current source target have different types; confirm intentionally.", "图与当前源码目标类型不同，请确认这是预期操作。", "圖與目前原始碼目標類型不同，請確認這是預期操作。"));
        }
        renderer.line(&mut content, locale.text("The graph and target were inspected separately; confirm they correspond before editing.", "图与目标分别检查；编辑前请确认两者对应。", "圖與目標分別檢查；編輯前請確認兩者對應。"));
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        if !init_input(
            renderer,
            locale.text("Continue? Type y", "继续？输入 y", "繼續？輸入 y"),
            "",
        )?
        .is_some_and(|answer| answer.eq_ignore_ascii_case("y"))
        {
            return Ok(true);
        }
        module
    } else {
        let Some(choice) = init_input(
            renderer,
            locale.text("Layer number", "层序号", "層序號"),
            "",
        )?
        else {
            return Ok(true);
        };
        let Some(module) = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| modules.get(n))
        else {
            return Ok(true);
        };
        module
    };
    let Some(path) = module["path"].as_str() else {
        return Ok(true);
    };
    if (action == "set" && module["can_source_set"] != true)
        || (action == "add" && module["can_source_insert"] != true)
    {
        let reason = module["source_set_block_reason"]
            .as_str()
            .unwrap_or("This source action is not advertised for the selected layer.");
        return show_flow_page(renderer, &format!("{}\n", text::plain_text(reason)));
    }
    let mut args = vec!["layer".to_owned(), action.clone()];
    if action == "add" {
        args.extend(["--parent".into(), path.into()]);
    } else if action != "migrate-starter" {
        args.push(path.into());
    }
    let mut component_params = None;
    let mut builtin_kind = None;
    if matches!(action.as_str(), "add" | "replace") {
        let Some(kind) = init_input(
            renderer,
            locale.text(
                "Layer type or component:ID@VERSION",
                "层类型或 component:ID@VERSION",
                "層類型或 component:ID@VERSION",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if let Some(component) = kind.strip_prefix("component:") {
            let Some((reference, params)) = guide_component_parameters(cli, renderer, component)?
            else {
                return Ok(true);
            };
            args.extend(["--component".into(), reference]);
            component_params = Some(params);
        } else {
            builtin_kind = Some(kind.clone());
            args.extend(["--type".into(), kind]);
        }
    }
    if action == "add" {
        let Some(name) = init_input(
            renderer,
            locale.text(
                "Child name (empty for ModuleList)",
                "子层名（ModuleList 留空）",
                "子層名（ModuleList 留空）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if !name.is_empty() {
            args.extend(["--name".into(), name]);
        }
        let Some(after) = init_input(
            renderer,
            locale.text(
                "Insert after child (empty to append)",
                "插入到哪个子项之后（留空追加）",
                "插入至哪個子項之後（留空附加）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if !after.is_empty() {
            args.extend(["--after".into(), after]);
        }
    }
    if !matches!(action.as_str(), "remove" | "migrate-starter") {
        let params = if let Some(params) = component_params {
            params
        } else if action == "set"
            && module["editable_params"]
                .as_object()
                .is_some_and(|fields| !fields.is_empty())
        {
            let Some(params) = guide_layer_set_params(renderer, module)? else {
                return Ok(true);
            };
            params
        } else {
            let Some(raw) = init_input(
                renderer,
                locale.text(
                    "Layer parameters as JSON object or :form",
                    "层参数 JSON 对象或 :form 表单",
                    "層參數 JSON 物件或 :form 表單",
                ),
                "{}",
            )?
            else {
                return Ok(true);
            };
            if raw == ":form" {
                let Some(kind) = builtin_kind.as_deref() else {
                    return Ok(true);
                };
                let Some(params) = guide_builtin_parameters(cli, renderer, kind)? else {
                    return Ok(true);
                };
                params
            } else {
                match parse_factory_parameters(&raw) {
                    Ok(params) => params,
                    Err(error) => {
                        show_flow_page(
                            renderer,
                            &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                        )?;
                        return Ok(true);
                    }
                }
            }
        };
        for (key, value) in params {
            args.extend(["--param".into(), format!("{key}={value}")]);
        }
    }
    args.extend([
        "--target".into(),
        "source".into(),
        "--model".into(),
        model.into(),
        "--require-forward".into(),
        "--if-source-config".into(),
        config_revision.into(),
    ]);
    let captured = if capture_snapshot {
        match cli.supports_input_snapshot() {
            Ok(true) => (),
            Ok(false) => {
                return show_flow_page(
                    renderer,
                    locale.text(
                        "This CLI does not support input snapshot capture and replay.\n",
                        "当前 CLI 不支持输入快照捕获和重放。\n",
                        "目前 CLI 不支援輸入快照擷取與重放。\n",
                    ),
                );
            }
            Err(error) => {
                return show_flow_page(
                    renderer,
                    &format!(
                        "{}: {}\n",
                        locale.text(
                            "Unable to verify snapshot support",
                            "无法确认快照支持",
                            "無法確認快照支援"
                        ),
                        text::plain_text(&error.to_string())
                    ),
                );
            }
        }
        let directory = tempfile::Builder::new()
            .prefix("model-maven-source-input-")
            .tempdir()?;
        let snapshot = directory.path().join("input.snapshot");
        let Some(snapshot_path) = snapshot.to_str().map(str::to_owned) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "temporary snapshot path is not UTF-8",
            ));
        };
        *temporary = Some(directory);
        let mut notice = String::new();
        renderer.heading(
            &mut notice,
            locale.text("Capturing input snapshot", "捕获输入快照", "擷取輸入快照"),
        );
        renderer.line(
            &mut notice,
            &format!(
                "{}: {}",
                locale.text("Temporary file", "临时文件", "暫存檔案"),
                text::plain_text(&snapshot_path)
            ),
        );
        renderer.line(&mut notice, locale.text("This private temporary file contains input values. It is removed after this flow; abnormal termination may leave it behind. Targets are not captured.", "该私有临时文件包含输入值，流程结束后清理；异常退出可能留下文件。目标不会被捕获。", "此私有暫存檔包含輸入值，流程結束後清理；異常結束可能留下檔案。目標不會被擷取。"));
        if !show_flow_page(renderer, &notice)? {
            return Ok(false);
        }
        let result = cli.execute(&[
            "model",
            "probe",
            "--model",
            model,
            "--capture-input-to",
            &snapshot_path,
        ]);
        let report = match &result {
            Ok(report) if report.operation == "probe" => report,
            _ => return show_flow_page(renderer, &renderer.result(&result)),
        };
        if report.data["config_revision"].as_str() != Some(config_revision) {
            return show_flow_page(renderer, locale.text("Project configuration changed during input capture; no source edit was saved.\n", "输入捕获时项目配置发生变化；源码未修改。\n", "輸入擷取時專案設定發生變化；原始碼未修改。\n"));
        }
        let content = &report.data["batch_content"];
        let digest = content["digest"].as_str().filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        let Some(digest) = digest.filter(|_| {
            content["status"] == "available"
                && content["algorithm"] == "sha256-forward-input-v1"
                && report.data["batch_snapshot"]["status"] == "captured"
                && report.data["batch_snapshot"]["format"]
                    == "model-maven-forward-input-snapshot-v1"
                && snapshot.is_file()
        }) else {
            return show_flow_page(renderer, &renderer.result(&result));
        };
        if report.status != "success" {
            renderer.line(&mut notice, locale.text("The original forward failed; its captured input can still be used to validate the candidate.", "原模型前向失败；仍可用捕获的输入验证候选。", "原模型前向失敗；仍可用擷取的輸入驗證候選。"));
            if !show_flow_page(renderer, &notice)? {
                return Ok(false);
            }
        }
        Some((snapshot_path, digest.to_owned()))
    } else {
        None
    };
    let mut preview_args = args.clone();
    if let Some((path, digest)) = &captured {
        preview_args.extend([
            "--replay-input-from".into(),
            path.clone(),
            "--if-input-content".into(),
            digest.clone(),
        ]);
    }
    preview_args.push("--dry-run".into());
    let preview = cli.execute(&preview_args.iter().map(String::as_str).collect::<Vec<_>>());
    let report = match &preview {
        Ok(report)
            if report.operation == "edit"
                && report.status == "success"
                && report.data["target"] == "source"
                && report.data["committed"] == false
                && report.data["validation"]["construction"] == "passed"
                && report.data["validation"]["syntax"] == "passed"
                && report.data["validation"]["forward"] == "passed" =>
        {
            report
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&preview))?;
            return Ok(true);
        }
    };
    let content = &report.data["validation"]["batch_content"];
    let input_digest = content["digest"].as_str().filter(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    let Some(input_digest) = input_digest.filter(|digest| {
        report.data["config_revision"].as_str() == Some(config_revision)
            && report.data["validation"]["forward_evidence"] == "executed"
            && report.data["validation"]["mode"] == "eval"
            && content["status"] == "available"
            && content["algorithm"] == "sha256-forward-input-v1"
            && captured.as_ref().is_none_or(|(_, expected)| {
                *digest == expected
                    && report.data["validation"]["batch_snapshot"]["status"] == "replayed"
                    && report.data["validation"]["batch_snapshot"]["format"]
                        == "model-maven-forward-input-snapshot-v1"
            })
    }) else {
        return show_flow_page(renderer, locale.text("A fresh candidate forward with a verifiable input digest is required before source changes can be saved. Upgrade the CLI or choose a supported batch.\n", "保存源码前需要可核对输入摘要的实际候选前向；请升级 CLI 或选择受支持的批次。\n", "儲存原始碼前需要可核對輸入摘要的實際候選前向；請升級 CLI 或選擇受支援的批次。\n"));
    };
    let source_evidence = &report.data["source_manifest_evidence"];
    let source_digest = source_evidence["digest"].as_str().filter(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    let Some(source_digest) = source_digest.filter(|_| {
        source_evidence["algorithm"] == "canonical-json-sha256-v1"
            && source_evidence["scope"] == "worker-recorded-project-sources"
    }) else {
        return show_flow_page(renderer, locale.text("Observed project source evidence is unavailable; source editing cannot be guarded.\n", "无法获取已观察的项目源码证据，不能安全提交源码编辑。\n", "無法取得已觀察的專案原始碼證據，無法安全提交原始碼編輯。\n"));
    };
    let plan = &report.data["source_plan"];
    let digest = plan["digest"].as_str().filter(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    let Some(digest) = digest.filter(|_| {
        plan["algorithm"] == "canonical-json-sha256-v1"
            && plan["scope"] == "planned-source-file-patches-v1"
    }) else {
        return show_flow_page(
            renderer,
            locale.text(
                "Source plan evidence unavailable; no source change can be committed.\n",
                "源码计划证据不可用，无法提交源码修改。\n",
                "原始碼計畫證據不可用，無法提交原始碼修改。\n",
            ),
        );
    };
    let Some(diff) = report.data["source_patch"]["unified_diff"]
        .as_str()
        .filter(|diff| !diff.is_empty())
    else {
        return show_flow_page(
            renderer,
            locale.text(
                "Source diff unavailable; no source change can be committed.\n",
                "源码差异不可用，无法提交源码修改。\n",
                "原始碼差異不可用，無法提交原始碼修改。\n",
            ),
        );
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Source preview · not saved",
            "源码预览 · 尚未保存",
            "原始碼預覽 · 尚未儲存",
        ),
    );
    if report.data["shared_state_evidence"]["scope"] == "in_memory_candidate_object_storage_v2"
        && report.data["shared_state_evidence"]["groups"].is_array()
    {
        renderer.line(&mut content, locale.text("The in-memory candidate is checked for shared objects and inspectable strided storage. After saving, a fresh model is checked; mismatch triggers rollback.", "内存候选会检查共享对象和可检查的稠密存储。保存后重新构造模型并核对；不一致时回滚。", "記憶體候選會檢查共享物件和可檢查的稠密儲存。儲存後重新建構模型並核對；不一致時回滾。"));
        let unavailable = report.data["shared_state_evidence"]["groups"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|group| group["kind"] == "storage_unavailable")
            .count();
        if unavailable > 0 {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text(
                        "Storage sharing could not be checked for these state entries",
                        "无法检查存储共享的状态项",
                        "無法檢查儲存共享的狀態項"
                    ),
                    unavailable
                ),
            );
        }
    } else if report.data["shared_state_evidence"]["scope"]
        == "in_memory_candidate_after_validation"
        && report.data["shared_state_evidence"]["groups"].is_array()
    {
        renderer.line(
            &mut content,
            locale.text(
                "This CLI checks object identity only; storage sharing is unknown.",
                "当前 CLI 仅核对对象身份；存储共享情况未知。",
                "目前 CLI 僅核對物件身分；儲存共享情況未知。",
            ),
        );
    } else {
        renderer.line(&mut content, locale.text("This CLI did not report shared-state evidence; source sharing has not been checked.", "当前 CLI 未提供共享状态证据；无法核对源码共享关系。", "目前 CLI 未提供共享狀態證據；無法核對原始碼共享關係。"));
    }
    for effect in report.data["edit_effects"].as_array().into_iter().flatten() {
        if let Some(entries) = effect["unverified_old_state_storage"]
            .as_array()
            .filter(|items| !items.is_empty())
        {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text(
                        "Original state with unknown storage sharing",
                        "原状态的存储共享未知",
                        "原狀態的儲存共享未知"
                    ),
                    text::plain_text(&Value::Array(entries.clone()).to_string())
                ),
            );
        }
        for (key, label) in [
            (
                "candidate_external_state_sharing",
                locale.text(
                    "Candidate preserves sharing",
                    "候选保留共享",
                    "候選保留共享",
                ),
            ),
            (
                "removed_candidate_external_state_sharing",
                locale.text("Candidate removes sharing", "候选移除共享", "候選移除共享"),
            ),
        ] {
            for item in effect[key].as_array().into_iter().flatten() {
                renderer.line(
                    &mut content,
                    &format!(
                        "{}{}: {} · {}",
                        label,
                        if item["sharing"] == "storage" {
                            locale.text(" (storage view)", "（存储视图）", "（儲存視圖）")
                        } else {
                            ""
                        },
                        text::plain_text(item["name"].as_str().unwrap_or("?")),
                        text::plain_text(&item["external_paths"].to_string())
                    ),
                );
            }
        }
    }
    renderer.line(&mut content, &text::plain_text(diff));
    render_candidate_state_effects(renderer, &mut content, &report.data["edit_effects"]);
    if action == "migrate-starter" {
        renderer.line(&mut content, locale.text("The old default head width stays fixed when routes change. Checkpoints are not migrated, and variants pinned to the old source may need recreation. Review the code and candidate forward.", "修改接线时旧默认预测头的输入宽度不会自动调整；检查点不会迁移，固定旧源码的变体可能需要重建。请检查代码与候选前向。", "修改接線時舊預設預測頭的輸入寬度不會自動調整；檢查點不會遷移，固定舊原始碼的變體可能需要重建。請檢查程式碼與候選前向。"));
    }
    renderer.line(
        &mut content,
        &format!(
            "Effects: {}",
            text::plain_text(&report.data["edit_effects"].to_string())
        ),
    );
    if matches!(action.as_str(), "add" | "replace" | "remove") {
        renderer.line(&mut content, locale.text("Structural source edits may append a container rebuild; earlier constructors can remain and still execute. Review the diff and constructor side effects.", "结构源码编辑可能追加容器重建；原有构造代码可能保留并继续执行，请检查差异和构造器副作用。", "結構原始碼編輯可能附加容器重建；原有建構程式碼可能保留並繼續執行，請檢查差異和建構器副作用。"));
    }
    renderer.line(&mut content, locale.text("Candidate forward passed. Commit checks the patch, observed project sources, configuration and forward input, then reconstructs and checks again. Targets, external libraries and runtime are not pinned.", "候选前向已通过。提交时核对计划、已观察的项目源码、配置及前向输入，重新构造时再次核对；目标、外部库和运行环境未固定。", "候選前向已通過。提交時核對計畫和前向輸入摘要，重新建構時再次核對；其他檔案、目標、設定和執行環境未固定。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    if !init_input(
        renderer,
        locale.text(
            "Type apply to modify source files",
            "输入 apply 修改源码文件",
            "輸入 apply 修改原始碼檔案",
        ),
        "",
    )?
    .is_some_and(|v| v == "apply")
    {
        return Ok(true);
    }
    args.extend([
        "--if-source-plan".into(),
        digest.into(),
        "--if-source-manifest".into(),
        source_digest.into(),
        "--if-input-content".into(),
        input_digest.into(),
    ]);
    if let Some((path, _)) = &captured {
        args.extend(["--replay-input-from".into(), path.clone()]);
    }
    *attempted = true;
    let result = cli.execute(&args.iter().map(String::as_str).collect::<Vec<_>>()).and_then(|report| {
        if report.status != "success" { return Ok(report); }
        let check = &report.data["source_validation"];
        if report.operation == "edit" && report.data["committed"] == true && report.data["target"] == "source"
            && report.data["source_plan"]["digest"].as_str() == Some(digest)
            && check["operation"] == "probe" && check["status"] == "passed"
            && check["batch_content"]["status"] == "available"
            && check["batch_content"]["algorithm"] == "sha256-forward-input-v1"
            && check["batch_content"]["digest"].as_str() == Some(input_digest)
            && report.data["config_revision"].as_str() == Some(config_revision)
            && check["source_config_revision"].as_str() == Some(config_revision)
            && check["source_manifest_digest"].as_str() == Some(source_digest)
            && check["source_manifest_status"] == "passed"
            && captured.as_ref().is_none_or(|_| check["batch_snapshot"]["status"] == "replayed" && check["batch_snapshot"]["format"] == "model-maven-forward-input-snapshot-v1") {
            Ok(report)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "Source save or input verification outcome is unknown; inspect project files and source history before retrying"))
        }
    });
    let mut content = renderer.result(&result);
    if let Ok(report) = &result {
        if report.operation == "edit"
            && report.status == "success"
            && report.data["committed"] == true
            && report.data["target"] == "source"
        {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text("Post-save verification", "保存后验证", "儲存後驗證"),
                    text::plain_text(&report.data["source_validation"].to_string())
                ),
            );
        } else {
            renderer.line(
                &mut content,
                locale.text(
                    "Review the diagnostic and source history before retrying.",
                    "重试前请检查诊断和源码历史。",
                    "重試前請檢查診斷和原始碼歷史。",
                ),
            );
        }
    } else {
        renderer.line(
            &mut content,
            locale.text(
                "Save outcome unknown; inspect source and history before retrying.",
                "保存结果未知；重试前请检查源码和历史。",
                "儲存結果未知；重試前請檢查原始碼和歷史。",
            ),
        );
    }
    show_flow_page(renderer, &content)
}

fn guide_layer_edit(
    cli: &Cli,
    renderer: &Renderer,
    variant: &str,
    selected_path: Option<(&str, &str)>,
) -> io::Result<LayerEditOutcome> {
    let locale = renderer.locale;
    let variants = cli.execute(&["variant", "list"]);
    let existing = match &variants {
        Ok(report)
            if report.operation == "variant_list"
                && report.status == "success"
                && report.data["variants"].is_array() =>
        {
            report.data["variants"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["name"].as_str() == Some(variant))
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&variants))?;
            return Ok(LayerEditOutcome::Cancelled);
        }
    };
    if !existing && variant != "working" {
        show_flow_page(renderer, "Selected variant does not exist.\n")?;
        return Ok(LayerEditOutcome::Cancelled);
    }
    let initial_inputs = if !existing {
        match cli.execute(&["assembly", "show", "--config-only"]) {
            Ok(report) if report.operation == "assembly_show" && report.status == "success" => {
                match (report.data["model"].as_str(), report.data["batch"].as_str()) {
                    (Some(model), Some(batch)) if !model.is_empty() && !batch.is_empty() => {
                        Some((model.to_owned(), batch.to_owned()))
                    }
                    _ => {
                        show_flow_page(
                            renderer,
                            "Project model and batch are required for a new variant.\n",
                        )?;
                        return Ok(LayerEditOutcome::Cancelled);
                    }
                }
            }
            _ => {
                show_flow_page(
                    renderer,
                    "Project configuration unavailable; no edit was started.\n",
                )?;
                return Ok(LayerEditOutcome::Cancelled);
            }
        }
    } else {
        None
    };
    let target_model = initial_inputs
        .as_ref()
        .map(|(model, _)| model.as_str())
        .or_else(|| {
            variants.as_ref().ok()?.data["variants"]
                .as_array()?
                .iter()
                .find(|item| item["name"].as_str() == Some(variant))?["model"]
                .as_str()
        })
        .filter(|model| !model.is_empty());
    let mut list_args = vec!["layer", "list"];
    if existing {
        list_args.extend(["--variant", variant]);
    }
    let list = cli.execute(&list_args);
    let modules = match &list {
        Ok(report) if report.operation == "layers" && report.status == "success" => {
            match report.data["modules"].as_array() {
                Some(modules) => modules,
                None => {
                    show_flow_page(renderer, "Invalid Layer list report.\n")?;
                    return Ok(LayerEditOutcome::Cancelled);
                }
            }
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&list))?;
            return Ok(LayerEditOutcome::Cancelled);
        }
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Model layers", "模型层", "模型層"),
    );
    for (index, module) in modules.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!(
                "{}. {} · {}",
                index + 1,
                text::plain_text(module["path"].as_str().unwrap_or("?")),
                text::plain_text(module["type"].as_str().unwrap_or("?"))
            ),
        );
    }
    renderer.line(&mut content, locale.text("Editing a variant; the inspected structure is not pinned to the preview. Python source is unchanged.", "编辑变体；当前结构列表未与预览绑定。Python 源码不会改写。", "編輯變體；目前結構清單未與預覽綁定。Python 原始碼不會改寫。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(LayerEditOutcome::Quit);
    }
    let module = if let Some((path, graph_model)) = selected_path {
        let mut matches = modules
            .iter()
            .filter(|module| module["path"].as_str() == Some(path));
        let Some(module) = matches.next().filter(|_| matches.next().is_none()) else {
            show_flow_page(renderer, locale.text("The selected graph path is absent or ambiguous in the current edit target. Inspect the target again.\n", "图中所选路径在当前编辑目标中不存在或不唯一，请重新检查目标。\n", "圖中所選路徑在目前編輯目標中不存在或不唯一，請重新檢查目標。\n"))?;
            return Ok(LayerEditOutcome::Cancelled);
        };
        let mut selected = String::new();
        renderer.heading(
            &mut selected,
            locale.text(
                "Graph selection in current variant",
                "图中选择在当前变体中的对应项",
                "圖中選擇在目前變體中的對應項",
            ),
        );
        renderer.line(
            &mut selected,
            &format!(
                "{} · {} · {}",
                text::plain_text(variant),
                text::plain_text(path),
                text::plain_text(module["type"].as_str().unwrap_or("?"))
            ),
        );
        renderer.line(
            &mut selected,
            &format!(
                "{}: {}",
                locale.text("Graph model", "图所检查的模型", "圖所檢查的模型"),
                text::plain_text(graph_model)
            ),
        );
        renderer.line(
            &mut selected,
            &format!(
                "{}: {}",
                locale.text("Edit target model", "编辑目标模型", "編輯目標模型"),
                text::plain_text(target_model.unwrap_or("unknown"))
            ),
        );
        renderer.line(
            &mut selected,
            locale.text(
                "The graph and target are separately inspected; confirm their correspondence.",
                "图与目标分别检查，请确认两者对应。",
                "圖與目標分別檢查，請確認兩者對應。",
            ),
        );
        if !show_flow_page(renderer, &selected)? {
            return Ok(LayerEditOutcome::Quit);
        }
        if !init_input(
            renderer,
            locale.text("Continue? Type y", "继续？输入 y", "繼續？輸入 y"),
            "",
        )?
        .is_some_and(|answer| answer.eq_ignore_ascii_case("y"))
        {
            return Ok(LayerEditOutcome::Cancelled);
        }
        module
    } else {
        let Some(choice) = init_input(
            renderer,
            locale.text("Layer number", "层序号", "層序號"),
            "",
        )?
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        let Some(module) = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| modules.get(n))
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        module
    };
    let Some(path) = module["path"].as_str() else {
        return Ok(LayerEditOutcome::Cancelled);
    };
    let Some(action) = init_input(
        renderer,
        locale.text(
            "Action: add, set, replace, remove, undo or redo",
            "操作：add、set、replace、remove、undo 或 redo",
            "操作：add、set、replace、remove、undo 或 redo",
        ),
        "",
    )?
    else {
        return Ok(LayerEditOutcome::Cancelled);
    };
    if !matches!(
        action.as_str(),
        "add" | "set" | "replace" | "remove" | "undo" | "redo"
    ) {
        return Ok(LayerEditOutcome::Cancelled);
    }
    if matches!(action.as_str(), "undo" | "redo") && !existing {
        show_flow_page(renderer, "No variant history exists to undo or redo.\n")?;
        return Ok(LayerEditOutcome::Cancelled);
    }
    if (action == "add" && module["can_insert"] != true)
        || (action == "set" && module["can_set"] != true)
    {
        show_flow_page(
            renderer,
            locale.text(
                "This action is not advertised for this layer.\n",
                "此层未声明支持该操作。\n",
                "此層未宣告支援該操作。\n",
            ),
        )?;
        return Ok(LayerEditOutcome::Cancelled);
    }
    let mut owned = vec!["layer".to_owned(), action.clone()];
    if action == "add" {
        owned.extend(["--parent".into(), path.into()]);
    } else if !matches!(action.as_str(), "undo" | "redo") {
        owned.push(path.into());
    }
    let mut component_params = None;
    let mut builtin_kind = None;
    if action == "add" || action == "replace" {
        let Some(kind) = init_input(
            renderer,
            locale.text(
                "Layer type or component:ID@VERSION",
                "层类型或 component:ID@VERSION",
                "層類型或 component:ID@VERSION",
            ),
            "",
        )?
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        if let Some(component) = kind.strip_prefix("component:") {
            let Some((reference, params)) = guide_component_parameters(cli, renderer, component)?
            else {
                return Ok(LayerEditOutcome::Cancelled);
            };
            owned.extend(["--component".into(), reference]);
            component_params = Some(params);
        } else {
            builtin_kind = Some(kind.clone());
            owned.extend(["--type".into(), kind]);
        }
    }
    if action == "add" {
        let Some(name) = init_input(
            renderer,
            locale.text(
                "Child name (empty for ModuleList)",
                "子层名（ModuleList 留空）",
                "子層名（ModuleList 留空）",
            ),
            "",
        )?
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        if !name.is_empty() {
            owned.extend(["--name".into(), name]);
        }
        let Some(after) = init_input(
            renderer,
            locale.text(
                "Insert after child (empty to append)",
                "插入到哪个子项之后（留空追加）",
                "插入至哪個子項之後（留空附加）",
            ),
            "",
        )?
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        if !after.is_empty() {
            owned.extend(["--after".into(), after]);
        }
    }
    if action == "set" {
        let Some(params) = guide_layer_set_params(renderer, module)? else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        for (name, value) in params {
            owned.extend(["--param".into(), format!("{name}={value}")]);
        }
    } else if let Some(params) = component_params {
        for (name, value) in params {
            owned.extend(["--param".into(), format!("{name}={value}")]);
        }
    } else if !matches!(action.as_str(), "remove" | "undo" | "redo") {
        let Some(raw) = init_input(
            renderer,
            locale.text(
                "Layer parameters as JSON object or :form",
                "层参数 JSON 对象或 :form 表单",
                "層參數 JSON 物件或 :form 表單",
            ),
            "{}",
        )?
        else {
            return Ok(LayerEditOutcome::Cancelled);
        };
        let params = if raw == ":form" {
            let Some(kind) = builtin_kind.as_deref() else {
                return Ok(LayerEditOutcome::Cancelled);
            };
            let Some(params) = guide_builtin_parameters(cli, renderer, kind)? else {
                return Ok(LayerEditOutcome::Cancelled);
            };
            params
        } else {
            match parse_factory_parameters(&raw) {
                Ok(params) => params,
                Err(error) => {
                    show_flow_page(
                        renderer,
                        &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                    )?;
                    return Ok(LayerEditOutcome::Cancelled);
                }
            }
        };
        for (name, value) in params {
            owned.extend(["--param".into(), format!("{name}={value}")]);
        }
    }
    if let Some((model, batch)) = &initial_inputs {
        owned.extend([
            "--model".into(),
            model.clone(),
            "--batch".into(),
            batch.clone(),
        ]);
    }
    owned.extend([
        "--variant".into(),
        variant.into(),
        "--require-forward".into(),
    ]);
    let mut preview_args = owned.clone();
    preview_args.push("--dry-run".into());
    let preview = cli.execute(&preview_args.iter().map(String::as_str).collect::<Vec<_>>());
    let report = match &preview {
        Ok(report)
            if report.operation == "edit"
                && report.status == "success"
                && report.data["committed"] == false =>
        {
            report
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&preview))?;
            if preview
                .as_ref()
                .is_ok_and(|report| report.status == "success")
            {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Unexpected preview response; no changes saved.\n",
                        "预览回执不符合预期；未保存任何更改。\n",
                        "預覽回執不符合預期；未儲存任何變更。\n",
                    ),
                )?;
            }
            return Ok(LayerEditOutcome::Cancelled);
        }
    };
    let base = &report.data["variant_base"];
    let token = base["token"].as_str().unwrap_or("");
    let input = &report.data["validation"]["batch_content"];
    let digest = input["digest"].as_str().unwrap_or("");
    let hex = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if base["name"] != variant
        || base["algorithm"] != "canonical-json-sha256-v1"
        || !(token == "absent" || hex(token))
        || report.data["variant_name"] != variant
        || report.data["validation"]["forward"] != "passed"
        || report.data["validation"]["forward_evidence"] != "executed"
        || report.data["validation"]["mode"] != "eval"
        || input["status"] != "available"
        || input["algorithm"] != "sha256-forward-input-v1"
        || !hex(digest)
    {
        show_flow_page(renderer, locale.text("Preview lacks a verified variant base or actual input digest; commit unavailable.\n", "预览缺少可核对的变体基准或实际输入指纹；无法提交。\n", "預覽缺少可核對的變體基準或實際輸入指紋；無法提交。\n"))?;
        return Ok(LayerEditOutcome::Cancelled);
    }
    let source = &report.data["source_manifest_evidence"];
    let source_digest = if source.is_null() {
        None
    } else if source["algorithm"] == "canonical-json-sha256-v1"
        && source["scope"] == "worker-recorded-project-sources-and-component-manifests"
        && source["digest"].as_str().is_some_and(hex)
    {
        source["digest"].as_str()
    } else {
        show_flow_page(
            renderer,
            locale.text(
                "Malformed source evidence; commit unavailable.\n",
                "源码证据格式异常，无法提交。\n",
                "原始碼證據格式異常，無法提交。\n",
            ),
        )?;
        return Ok(LayerEditOutcome::Cancelled);
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Layer preview · not saved",
            "层编辑预览 · 尚未保存",
            "層編輯預覽 · 尚未儲存",
        ),
    );
    if matches!(action.as_str(), "undo" | "redo") {
        renderer.line(
            &mut content,
            locale.text(
                "History action applies to the whole variant, not the selected layer.",
                "历史操作针对整个变体，而非所选层。",
                "歷史操作針對整個變體，而非所選層。",
            ),
        );
    }
    renderer.line(
        &mut content,
        &format!(
            "{} · {} · {}",
            text::plain_text(variant),
            text::plain_text(&action),
            text::plain_text(path)
        ),
    );
    for effect in report.data["edit_effects"].as_array().into_iter().flatten() {
        if let Some(entries) = effect["unverified_old_state_storage"]
            .as_array()
            .filter(|items| !items.is_empty())
        {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text(
                        "Original state with unknown storage sharing",
                        "原状态的存储共享未知",
                        "原狀態的儲存共享未知"
                    ),
                    text::plain_text(&Value::Array(entries.clone()).to_string())
                ),
            );
        }
        for shared in effect["preserved_external_state_sharing"]
            .as_array()
            .into_iter()
            .flatten()
        {
            renderer.line(
                &mut content,
                &format!(
                    "{}{}: {} · {}",
                    locale.text("Shared state retained", "保留共享状态", "保留共享狀態"),
                    if shared["sharing"] == "storage" {
                        locale.text(" (storage view)", "（存储视图）", "（儲存視圖）")
                    } else {
                        ""
                    },
                    text::plain_text(shared["name"].as_str().unwrap_or("?")),
                    text::plain_text(&shared["external_paths"].to_string())
                ),
            );
        }
    }
    renderer.line(
        &mut content,
        &format!(
            "Effects: {}",
            text::plain_text(&report.data["edit_effects"].to_string())
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Validation: {}",
            text::plain_text(&report.data["validation"].to_string())
        ),
    );
    render_candidate_state_effects(renderer, &mut content, &report.data["edit_effects"]);
    if let Some(effects) = report.data["edit_effects"].as_array() {
        for effect in effects {
            if let Some(values) = effect["affected_aliases"].as_array() {
                renderer.line(
                    &mut content,
                    &format!(
                        "{}: {}",
                        locale.text(
                            "Affected registrations",
                            "受影响的注册路径",
                            "受影響的註冊路徑"
                        ),
                        text::plain_text(&Value::Array(values.clone()).to_string())
                    ),
                );
            }
        }
    }
    if source_digest.is_some() {
        renderer.line(&mut content, locale.text("Commit rechecks the variant, input content and worker-recorded project source manifest. External libraries, weights, configuration, runtime and changes after the check are not pinned.", "提交时核对变体、输入内容及 worker 记录的项目源码清单。外部库、权重、项目配置、运行环境及检查后的变化未固定。", "提交時核對變體、輸入內容及 worker 記錄的專案原始碼清單。外部程式庫、權重、專案設定、執行環境及檢查後的變化未固定。"));
    } else {
        renderer.line(&mut content, locale.text("Commit rechecks variant and input; source evidence is unavailable. Project configuration and runtime are not pinned.", "提交时核对变体和输入；源码证据不可用。项目配置及运行环境未固定。", "提交時核對變體和輸入；原始碼證據不可用。專案設定及執行環境未固定。"));
    }
    if !show_flow_page(renderer, &content)? {
        return Ok(LayerEditOutcome::Quit);
    }
    let Some(answer) = init_input(
        renderer,
        locale.text(
            "Save this variant edit? Type y",
            "保存此变体编辑？输入 y",
            "儲存此變體編輯？輸入 y",
        ),
        "",
    )?
    else {
        return Ok(LayerEditOutcome::Cancelled);
    };
    if !answer.eq_ignore_ascii_case("y") {
        return Ok(LayerEditOutcome::Cancelled);
    }
    owned.extend([
        "--if-variant-base".into(),
        token.into(),
        "--if-input-content".into(),
        digest.into(),
    ]);
    if let Some(source_digest) = source_digest {
        owned.extend(["--if-source-manifest".into(), source_digest.into()]);
    }
    let result = cli.execute(&owned.iter().map(String::as_str).collect::<Vec<_>>());
    let committed = result.as_ref().is_ok_and(|report| {
        report.operation == "edit"
            && report.status == "success"
            && report.data["committed"] == true
            && report.data["variant_name"] == variant
    });
    let mut content = renderer.result(&result);
    if committed {
        let report = result.as_ref().unwrap();
        let committed_token = report.data["variant_committed"]["token"]
            .as_str()
            .filter(|token| hex(token));
        if let Some(committed_token) = committed_token {
            let recheck = cli.execute(&[
                "model",
                "probe",
                "--variant",
                variant,
                "--if-variant-base",
                committed_token,
                "--if-input-content",
                digest,
            ]);
            renderer.line(
                &mut content,
                locale.text(
                    "Saved edit · independent recheck",
                    "已保存编辑 · 独立重验",
                    "已儲存編輯 · 獨立重驗",
                ),
            );
            content.push_str(&renderer.result(&recheck));
        } else {
            renderer.line(
                &mut content,
                locale.text(
                    "Saved; recheck unavailable because receipt lacks the new variant token.",
                    "已保存；回执缺少新变体令牌，无法重验。",
                    "已儲存；回執缺少新變體令牌，無法重驗。",
                ),
            );
        }
    } else if result.is_err()
        || result
            .as_ref()
            .is_ok_and(|report| report.status == "success")
    {
        renderer.line(
            &mut content,
            locale.text(
                "Save outcome unknown; inspect the variant before retrying.",
                "保存结果未知；重试前请检查变体。",
                "儲存結果未知；重試前請檢查變體。",
            ),
        );
    }
    if !show_flow_page(renderer, &content)? {
        return Ok(LayerEditOutcome::Quit);
    }
    Ok(LayerEditOutcome::Attempted { committed })
}

struct ComponentTask {
    task: crate::cli::BackgroundTask,
    operation: String,
    mutating: bool,
    completion_handled: bool,
    selection_handled: bool,
    uninstall_reference: Option<String>,
    uninstall_revision: Option<String>,
}

fn guide_components(
    cli: &Cli,
    renderer: &Renderer,
    tasks: &mut Vec<ComponentTask>,
    environments: &mut [EnvironmentTask],
    mutation_started: &mut bool,
) -> io::Result<bool> {
    let locale = renderer.locale;
    let mut selected: Option<String> = None;
    const COMPONENTS_PER_PAGE: usize = 20;
    let mut component_page = 0usize;
    loop {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Project components", "项目组件", "專案元件"),
        );
        renderer.line(&mut content, locale.text("List/show/check read component metadata; check does not import component code. Discovery registers found manifests. Tests run component code and may save verification records.", "list/show/check 读取组件元数据；check 不导入组件代码。discover 会登记发现的清单；test 会运行组件代码，并可能保存验证记录。", "list/show/check 讀取元件中繼資料；check 不匯入元件程式碼。discover 會登記發現的清單；test 會執行元件程式碼，並可能儲存驗證紀錄。"));
        renderer.line(&mut content, locale.text("Selected versions do not pin component source. Operations run in the background; exiting stops them and may leave partial effects. Inspect before retrying.", "所选版本不固定组件源码。操作在后台运行；退出会停止操作，可能留下部分影响，重试前请检查。", "所選版本不固定元件原始碼。操作在背景執行；結束會停止操作，可能留下部分影響，重試前請檢查。"));
        renderer.line(&mut content, locale.text("The list is a snapshot; run list again after registering or discovering components.", "列表是快照；登记或发现组件后请重新执行 list。", "清單是快照；登記或發現元件後請重新執行 list。"));
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Selected", "已选择", "已選擇"),
                text::plain_text(selected.as_deref().unwrap_or("—"))
            ),
        );
        let mut active = false;
        for (index, entry) in tasks
            .iter_mut()
            .enumerate()
            .filter(|(_, entry)| entry.task.project == cli.project())
        {
            match entry.task.poll() {
                Ok(done) => {
                    active |= !done;
                    renderer.line(
                        &mut content,
                        &format!(
                            "{}. {} · {}",
                            index + 1,
                            text::plain_text(&entry.operation),
                            if done {
                                locale.text("finished", "已结束", "已結束")
                            } else {
                                locale.text("running", "运行中", "執行中")
                            }
                        ),
                    );
                    if let Some(result) = entry.task.result() {
                        content.push_str(&renderer.result(result));
                        let uninstall = entry.operation == "component_uninstall_preview"
                            || entry.operation == "component_uninstall_apply";
                        if entry.operation == "component_uninstall_apply"
                            && !result.as_ref().is_ok_and(|report| {
                                report.operation == "component_uninstall"
                                    && report.status == "success"
                                    && report.data["applied"] == true
                                    && report.data["plan_revision"].as_str()
                                        == entry.uninstall_revision.as_deref()
                                    && report.data["component_id"]
                                        .as_str()
                                        .zip(report.data["component_version"].as_str())
                                        .map(|(id, version)| format!("{id}@{version}"))
                                        == entry.uninstall_reference
                            })
                        {
                            renderer.line(&mut content, locale.text("Removal result not verified; inspect the component list before retrying.", "移除结果尚未核实；重试前请检查组件列表。", "移除結果尚未核實；重試前請檢查元件清單。"));
                        }
                        if entry.operation == "component_install"
                            && !result.as_ref().is_ok_and(|report| {
                                report.operation == "component_install"
                                    && report.status == "success"
                                    && report.data["installation"]["environment_modified"] == false
                                    && report.data["installation"]["components"]
                                        .as_array()
                                        .is_some_and(|items| !items.is_empty())
                            })
                        {
                            renderer.line(&mut content, locale.text("Installation result not verified; inspect project extension storage and component list before retrying.", "安装结果尚未核实；重试前请检查项目扩展存储和组件列表。", "安裝結果尚未核實；重試前請檢查專案擴充儲存與元件清單。"));
                        }
                        if let Ok(report) = result {
                            if entry.operation == "component_uninstall_preview"
                                && report.operation == "component_uninstall"
                                && report.status == "success"
                                && report.data["applied"] == false
                                && !entry.selection_handled
                            {
                                renderer.line(
                                    &mut content,
                                    locale.text(
                                        "Removal preview ready; use review to inspect and confirm.",
                                        "移除预览已就绪；使用 review 查看并确认。",
                                        "移除預覽已就緒；使用 review 檢視並確認。",
                                    ),
                                );
                            }
                            if entry.operation == "component_install"
                                && report.operation == "component_install"
                                && report.status == "success"
                                && report.data["installation"]["environment_modified"] == false
                                && !entry.selection_handled
                            {
                                let components =
                                    report.data["installation"]["components"].as_array();
                                if let Some(component) = components
                                    .filter(|items| items.len() == 1)
                                    .and_then(|items| items.first())
                                {
                                    if let Some((id, version)) = component["component_id"]
                                        .as_str()
                                        .zip(component["version"].as_str())
                                        .filter(|(id, version)| {
                                            !id.is_empty() && !version.is_empty()
                                        })
                                    {
                                        let reference = format!("{id}@{version}");
                                        renderer.line(
                                            &mut content,
                                            &format!(
                                                "{}: {}",
                                                locale.text(
                                                    "Installed and selected",
                                                    "已安装并选择",
                                                    "已安裝並選擇"
                                                ),
                                                text::plain_text(&reference)
                                            ),
                                        );
                                        selected = Some(reference);
                                    }
                                }
                                entry.selection_handled = true;
                            }
                            if entry.operation == "component_add"
                                && report.operation == "component_add"
                                && report.status == "success"
                                && !entry.selection_handled
                            {
                                let manifest = &report.data["component"]["manifest"];
                                if let (Some(id), Some(version)) = (
                                    manifest["component_id"].as_str().filter(|v| !v.is_empty()),
                                    manifest["component_version"]
                                        .as_str()
                                        .filter(|v| !v.is_empty()),
                                ) {
                                    let reference = format!("{id}@{version}");
                                    renderer.line(
                                        &mut content,
                                        &format!(
                                            "{}: {} · {}",
                                            locale.text(
                                                "Registered and selected",
                                                "已登记并选择",
                                                "已登記並選擇"
                                            ),
                                            text::plain_text(&reference),
                                            locale.text(
                                                "use test to verify it",
                                                "可使用 test 验证",
                                                "可使用 test 驗證"
                                            )
                                        ),
                                    );
                                    selected = Some(reference);
                                }
                                entry.selection_handled = true;
                            }
                            if (report.operation == entry.operation
                                || (uninstall && report.operation == "component_uninstall"))
                                && entry.operation != "component_list"
                            {
                                let raw = report.data.to_string();
                                let shown: String = raw.chars().take(12000).collect();
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "Details: {}{}",
                                        text::plain_text(&shown),
                                        if raw.chars().count() > 12000 {
                                            "… [truncated]"
                                        } else {
                                            ""
                                        }
                                    ),
                                );
                            } else if report.operation != entry.operation {
                                renderer.line(
                                    &mut content,
                                    locale.text(
                                        "Unexpected operation in receipt.",
                                        "回执操作不匹配。",
                                        "回執操作不相符。",
                                    ),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    active = true;
                    renderer.line(
                        &mut content,
                        &format!(
                            "Status unavailable: {}",
                            text::plain_text(&error.to_string())
                        ),
                    );
                }
            }
        }
        let latest_list = tasks
            .iter()
            .rev()
            .find(|entry| {
                entry.task.project == cli.project() && entry.operation == "component_list"
            })
            .and_then(|entry| {
                entry
                    .task
                    .result()
                    .and_then(|result| result.as_ref().ok())
                    .filter(|report| {
                        report.operation == "component_list" && report.status == "success"
                    })
                    .and_then(|report| report.data["components"].as_array())
            });
        if let Some(components) = latest_list {
            let page_count = components.len().max(1).div_ceil(COMPONENTS_PER_PAGE);
            component_page = component_page.min(page_count - 1);
            let start = component_page * COMPONENTS_PER_PAGE;
            renderer.line(
                &mut content,
                &format!(
                    "{} · {}/{} · {} {}",
                    locale.text("Latest component list", "最近的组件列表", "最近的元件清單"),
                    component_page + 1,
                    page_count,
                    components.len(),
                    locale.text("components", "个组件", "個元件")
                ),
            );
            for (index, item) in components
                .iter()
                .enumerate()
                .skip(start)
                .take(COMPONENTS_PER_PAGE)
            {
                let manifest = &item["manifest"];
                renderer.line(
                    &mut content,
                    &format!(
                        "{}. {}@{} · {}",
                        index + 1,
                        text::plain_text(manifest["component_id"].as_str().unwrap_or("?")),
                        text::plain_text(manifest["component_version"].as_str().unwrap_or("?")),
                        text::plain_text(manifest["kind"].as_str().unwrap_or("?"))
                    ),
                );
            }
        }
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        let Some(action) = init_input(renderer, locale.text("list, search, select NUMBER, n/p page, show, check, test, discover, add, scaffold, install, uninstall, review, refresh, or b back", "list 列表、search 搜索、select 序号、n/p 翻页、show 查看、check 检查、test 测试、discover 发现、add 登记、scaffold 创建扩展、install 安装、uninstall 预览移除、review 审核、refresh 刷新，或 b 返回", "list 清單、search 搜尋、select 序號、n/p 翻頁、show 檢視、check 檢查、test 測試、discover 發現、add 登記、scaffold 建立擴充、install 安裝、uninstall 預覽移除、review 審核、refresh 重新整理，或 b 返回"), "")? else { return Ok(true); };
        if action == "refresh" {
            continue;
        }
        if action == "n" || action == "p" {
            if action == "n"
                && latest_list
                    .is_some_and(|list| (component_page + 1) * COMPONENTS_PER_PAGE < list.len())
            {
                component_page += 1;
            } else if action == "p" {
                component_page = component_page.saturating_sub(1);
            }
            continue;
        }
        if let Some(number) = action.strip_prefix("select ") {
            let candidate = number
                .trim()
                .parse::<usize>()
                .ok()
                .and_then(|n| n.checked_sub(1))
                .filter(|n| {
                    *n >= component_page * COMPONENTS_PER_PAGE
                        && *n < (component_page + 1) * COMPONENTS_PER_PAGE
                })
                .and_then(|index| latest_list.and_then(|list| list.get(index)))
                .and_then(|item| {
                    let manifest = &item["manifest"];
                    let id = manifest["component_id"]
                        .as_str()
                        .filter(|s| !s.is_empty())?;
                    let version = manifest["component_version"]
                        .as_str()
                        .filter(|s| !s.is_empty())?;
                    Some(format!("{id}@{version}"))
                });
            if let Some(candidate) = candidate {
                selected = Some(candidate);
            } else {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Select a number shown on the current page.\n",
                        "请选择当前页显示的序号。\n",
                        "請選擇目前頁面顯示的序號。\n",
                    ),
                )?;
            }
            continue;
        }
        if !matches!(
            action.as_str(),
            "list"
                | "search"
                | "show"
                | "check"
                | "test"
                | "discover"
                | "add"
                | "scaffold"
                | "install"
                | "uninstall"
                | "review"
        ) {
            if action == "b" {
                return Ok(true);
            }
            show_flow_page(
                renderer,
                locale.text(
                    "Unknown action; choose an option from the menu.\n",
                    "未知操作，请选择菜单中的选项。\n",
                    "未知操作，請選擇選單中的選項。\n",
                ),
            )?;
            continue;
        }
        if active {
            show_flow_page(
                renderer,
                locale.text(
                    "A component operation is still running; refresh before starting another.\n",
                    "组件操作仍在运行，请刷新后再启动其他操作。\n",
                    "元件操作仍在執行，請重新整理後再啟動其他操作。\n",
                ),
            )?;
            continue;
        }
        if environments
            .iter_mut()
            .filter(|entry| entry.task.project == cli.project())
            .any(|entry| !matches!(entry.task.poll(), Ok(true)))
        {
            show_flow_page(renderer, locale.text("An environment operation is active or unknown; wait before using components.\n", "环境操作正在运行或状态未知；请等待完成后再使用组件。\n", "環境操作正在執行或狀態未知；請等待完成後再使用元件。\n"))?;
            continue;
        }
        if tasks.len() >= 8 && action != "review" {
            tasks.retain_mut(|entry| {
                (entry.operation == "component_uninstall_preview" && !entry.selection_handled)
                    || !matches!(entry.task.poll(), Ok(true))
            });
            if tasks.len() >= 8 {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Too many active operations or unreviewed plans; review or cancel a plan first.\n",
                        "运行中的操作或待审核计划过多；请先审核或取消计划。\n",
                        "執行中的操作或待審核計畫過多；請先審核或取消計畫。\n",
                    ),
                )?;
                continue;
            }
        }
        if action == "uninstall" {
            let Some(reference) = selected.clone() else {
                show_flow_page(
                    renderer,
                    locale.text(
                        "List and select a component first.\n",
                        "请先列出并选择组件。\n",
                        "請先列出並選擇元件。\n",
                    ),
                )?;
                continue;
            };
            if let Some(message) =
                capability_unavailable(cli, locale, &["component.conditional_uninstall"])
            {
                if !show_flow_page(renderer, &message)? {
                    return Ok(false);
                }
                continue;
            }
            let args = vec![
                "component".to_owned(),
                "uninstall".into(),
                reference.clone(),
            ];
            match cli.start_background(&args) {
                Ok(task) => {
                    // A newer request supersedes any earlier unreviewed plan for this project.
                    for entry in tasks.iter_mut().filter(|entry| {
                        entry.task.project == cli.project()
                            && entry.operation == "component_uninstall_preview"
                    }) {
                        entry.selection_handled = true;
                    }
                    tasks.push(ComponentTask {
                        task,
                        operation: "component_uninstall_preview".into(),
                        mutating: false,
                        completion_handled: false,
                        selection_handled: false,
                        uninstall_reference: Some(reference),
                        uninstall_revision: None,
                    });
                }
                Err(error) => {
                    if !show_flow_page(
                        renderer,
                        &format!("{}\n", text::plain_text(&error.to_string())),
                    )? {
                        return Ok(false);
                    }
                }
            }
            continue;
        }
        if action == "review" {
            let candidate = tasks.iter().enumerate().rev().find(|(_, entry)| {
                entry.task.project == cli.project()
                    && entry.operation == "component_uninstall_preview"
                    && !entry.selection_handled
            });
            let Some((preview_index, entry)) = candidate else {
                show_flow_page(
                    renderer,
                    locale.text(
                        "No unreviewed removal plan. Start uninstall and refresh.\n",
                        "没有待审核的移除计划；请先启动 uninstall 并刷新。\n",
                        "沒有待審核的移除計畫；請先啟動 uninstall 並重新整理。\n",
                    ),
                )?;
                continue;
            };
            let Some(report) = entry
                .task
                .result()
                .and_then(|result| result.as_ref().ok())
                .filter(|report| {
                    report.operation == "component_uninstall"
                        && report.status == "success"
                        && report.data["applied"] == false
                })
            else {
                tasks[preview_index].selection_handled = true;
                show_flow_page(
                    renderer,
                    locale.text(
                        "Removal preview unavailable or failed; nothing removed.\n",
                        "移除预览不可用或失败；未移除组件。\n",
                        "移除預覽無法使用或失敗；未移除元件。\n",
                    ),
                )?;
                continue;
            };
            let plan = &report.data;
            let reference = entry.uninstall_reference.as_deref().unwrap_or("");
            let reported = plan["component_id"]
                .as_str()
                .zip(plan["component_version"].as_str())
                .map(|(id, version)| format!("{id}@{version}"));
            let revision = plan["plan_revision"].as_str().filter(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            });
            let Some((revision, force)) = revision
                .zip(plan["requires_force"].as_bool())
                .filter(|_| reported.as_deref() == Some(reference))
            else {
                tasks[preview_index].selection_handled = true;
                show_flow_page(
                    renderer,
                    locale.text(
                        "Removal plan cannot be verified; nothing removed.\n",
                        "无法核对移除计划；未移除组件。\n",
                        "無法核對移除計畫；未移除元件。\n",
                    ),
                )?;
                continue;
            };
            let revision = revision.to_owned();
            let reference = reference.to_owned();
            let mut content = String::new();
            renderer.heading(
                &mut content,
                locale.text("Component removal preview", "组件移除预览", "元件移除預覽"),
            );
            renderer.line(&mut content, &text::plain_text(&plan.to_string()));
            renderer.line(&mut content, locale.text("This removes the registration and possibly its unused project-local package. Source, configuration and variants are not rewritten; references can stop working.", "这会移除登记及可能不再使用的项目本地包。不会改写源码、配置和变体；现有引用可能失效。", "這會移除登記及可能不再使用的專案本地套件。不會改寫原始碼、設定和變體；現有參照可能失效。"));
            if !show_flow_page(renderer, &content)? {
                return Ok(false);
            }
            let Some(answer) = init_input(
                renderer,
                if force {
                    locale.text(
                        "Type FORCE to accept the references and remove",
                        "存在引用；输入 FORCE 接受影响并移除",
                        "存在參照；輸入 FORCE 接受影響並移除",
                    )
                } else {
                    locale.text(
                        "Type the exact component ID@version to remove",
                        "输入完整组件 ID@版本确认移除",
                        "輸入完整元件 ID@版本確認移除",
                    )
                },
                "",
            )?
            else {
                tasks[preview_index].selection_handled = true;
                continue;
            };
            if (force && answer != "FORCE") || (!force && answer != reference) {
                tasks[preview_index].selection_handled = true;
                continue;
            }
            let mut commit = vec![
                "component".to_owned(),
                "uninstall".into(),
                reference.clone(),
                "--apply".into(),
                "--if-plan".into(),
                revision.clone(),
            ];
            if force {
                commit.push("--force".into());
            }
            match cli.start_background(&commit) {
                Ok(task) => {
                    *mutation_started = true;
                    tasks[preview_index].selection_handled = true;
                    selected = None;
                    tasks.retain(|entry| {
                        !(entry.task.project == cli.project()
                            && (entry.operation == "component_list"
                                || (entry.operation == "component_uninstall_preview"
                                    && entry.selection_handled)))
                    });
                    tasks.push(ComponentTask {
                        task,
                        operation: "component_uninstall_apply".into(),
                        mutating: true,
                        completion_handled: false,
                        selection_handled: false,
                        uninstall_reference: Some(reference),
                        uninstall_revision: Some(revision),
                    });
                }
                Err(error) => {
                    if !show_flow_page(
                        renderer,
                        &format!("{}\n", text::plain_text(&error.to_string())),
                    )? {
                        return Ok(false);
                    }
                }
            }
            continue;
        }
        let mut args = vec!["component".to_owned()];
        let expected = match action.as_str() {
            "scaffold" | "install" => {
                if let Some(message) =
                    capability_unavailable(cli, locale, &["component.project_package_lifecycle"])
                {
                    if !show_flow_page(renderer, &message)? {
                        return Ok(false);
                    }
                    continue;
                }
                if action == "scaffold" {
                    let Some(id) = init_input(
                        renderer,
                        locale.text(
                            "Component ID (namespace/name)",
                            "组件 ID（命名空间/名称）",
                            "元件 ID（命名空間/名稱）",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    let Some(kind) = init_input(
                        renderer,
                        locale.text(
                            "Extension kind: layer or layer-edit-provider",
                            "扩展类型：layer 或 layer-edit-provider",
                            "擴充類型：layer 或 layer-edit-provider",
                        ),
                        "layer",
                    )?
                    else {
                        continue;
                    };
                    let Some(version) = init_input(
                        renderer,
                        locale.text("Component version", "组件版本", "元件版本"),
                        "0.1.0",
                    )?
                    else {
                        continue;
                    };
                    let Some(output) = init_input(
                        renderer,
                        locale.text(
                            "New project-relative output directory",
                            "项目内的新输出目录",
                            "專案內的新輸出目錄",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    if id.is_empty()
                        || output.is_empty()
                        || !matches!(kind.as_str(), "layer" | "layer-edit-provider")
                    {
                        show_flow_page(
                            renderer,
                            locale.text(
                                "Invalid scaffold fields; nothing created.\n",
                                "骨架字段无效；未创建文件。\n",
                                "骨架欄位無效；未建立檔案。\n",
                            ),
                        )?;
                        continue;
                    }
                    args.extend([
                        "scaffold".into(),
                        id,
                        "--kind".into(),
                        kind,
                        "--version".into(),
                        version,
                        "--output".into(),
                        output,
                    ]);
                    "component_scaffold"
                } else {
                    let Some(path) = init_input(
                        renderer,
                        locale.text(
                            "Project-relative extension package directory",
                            "项目内扩展包目录",
                            "專案內擴充套件目錄",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    let Some(isolation) = init_input(
                        renderer,
                        locale.text(
                            "Build isolation: yes or no",
                            "隔离构建：yes 或 no",
                            "隔離建置：yes 或 no",
                        ),
                        "yes",
                    )?
                    else {
                        continue;
                    };
                    if path.is_empty() || !matches!(isolation.as_str(), "yes" | "no") {
                        show_flow_page(
                            renderer,
                            locale.text(
                                "Invalid install fields; nothing installed.\n",
                                "安装字段无效；未安装组件。\n",
                                "安裝欄位無效；未安裝元件。\n",
                            ),
                        )?;
                        continue;
                    }
                    args.extend(["install".into(), path]);
                    if isolation == "no" {
                        args.push("--no-build-isolation".into());
                    }
                    "component_install"
                }
            }
            "list" => {
                args.push("list".into());
                "component_list"
            }
            "search" => {
                let Some(query) = init_input(
                    renderer,
                    locale.text(
                        "Search component ID or description",
                        "搜索组件 ID 或说明",
                        "搜尋元件 ID 或說明",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                args.extend(["list".into(), "--query".into(), query]);
                "component_list"
            }
            "add" => {
                let Some(id) = init_input(
                    renderer,
                    locale.text(
                        "Component ID (namespace/name)",
                        "组件 ID（命名空间/名称）",
                        "元件 ID（命名空間/名稱）",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                let Some(kind) = init_input(renderer, locale.text("Kind: model, layer, dataset, transform, loss, metric, trainer, layer-edit-provider", "类型：model、layer、dataset、transform、loss、metric、trainer、layer-edit-provider", "類型：model、layer、dataset、transform、loss、metric、trainer、layer-edit-provider"), "layer")? else { continue; };
                let Some(entrypoint) = init_input(
                    renderer,
                    locale.text(
                        "Python factory module:callable",
                        "Python 工厂 module:callable",
                        "Python 工廠 module:callable",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                let Some(version) = init_input(
                    renderer,
                    locale.text("Component version", "组件版本", "元件版本"),
                    "0.1.0",
                )?
                else {
                    continue;
                };
                let Some(description) = init_input(
                    renderer,
                    locale.text("Description (optional)", "说明（可选）", "說明（可選）"),
                    "",
                )?
                else {
                    continue;
                };
                args.extend([
                    "add".into(),
                    id,
                    "--kind".into(),
                    kind,
                    "--entrypoint".into(),
                    entrypoint,
                    "--version".into(),
                    version,
                    "--description".into(),
                    description,
                ]);
                let Some(advanced) = init_input(
                    renderer,
                    locale.text(
                        "Configure schema, dependencies or fixture? (y/N)",
                        "配置参数 Schema、依赖或测试 fixture？（y/N）",
                        "設定參數 Schema、依賴或測試 fixture？（y/N）",
                    ),
                    "n",
                )?
                else {
                    continue;
                };
                if advanced.eq_ignore_ascii_case("y") {
                    let fields = [
                        (
                            "--params-schema",
                            locale.text(
                                "Parameter schema JSON object (optional)",
                                "参数 Schema JSON 对象（可选）",
                                "參數 Schema JSON 物件（可選）",
                            ),
                            true,
                        ),
                        (
                            "--capabilities",
                            locale.text(
                                "Capabilities JSON object (optional)",
                                "能力 JSON 对象（可选）",
                                "能力 JSON 物件（可選）",
                            ),
                            true,
                        ),
                        (
                            "--dependencies",
                            locale.text(
                                "Dependencies JSON array (optional)",
                                "依赖 JSON 数组（可选）",
                                "依賴 JSON 陣列（可選）",
                            ),
                            false,
                        ),
                    ];
                    let mut invalid = false;
                    for (flag, prompt, object) in fields {
                        let Some(raw) = init_input(renderer, prompt, "")? else {
                            invalid = true;
                            break;
                        };
                        if raw.trim().is_empty() {
                            continue;
                        }
                        let valid = serde_json::from_str::<serde_json::Value>(&raw)
                            .is_ok_and(|v| if object { v.is_object() } else { v.is_array() });
                        if !valid {
                            show_flow_page(
                                renderer,
                                locale.text(
                                    "Invalid JSON type; no component was registered.\n",
                                    "JSON 类型无效；尚未登记组件。\n",
                                    "JSON 類型無效；尚未登記元件。\n",
                                ),
                            )?;
                            invalid = true;
                            break;
                        }
                        args.extend([flag.into(), raw]);
                    }
                    if invalid {
                        continue;
                    }
                    let Some(fixture) = init_input(
                        renderer,
                        locale.text(
                            "Conformance fixture module:callable (optional)",
                            "符合性 fixture module:callable（可选）",
                            "符合性 fixture module:callable（可選）",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    if !fixture.is_empty() {
                        args.extend(["--conformance-fixture".into(), fixture]);
                    }
                }
                "component_add"
            }
            "show" | "check" | "test" => {
                let Some(reference) = selected.clone() else {
                    show_flow_page(
                        renderer,
                        locale.text(
                            "List and select a component first.\n",
                            "请先列出并选择组件。\n",
                            "請先列出並選擇元件。\n",
                        ),
                    )?;
                    continue;
                };
                args.extend([action.clone(), reference]);
                if action == "test" {
                    let Some(raw) = init_input(
                        renderer,
                        locale.text(
                            "Component parameters as JSON object",
                            "组件参数 JSON 对象",
                            "元件參數 JSON 物件",
                        ),
                        "{}",
                    )?
                    else {
                        continue;
                    };
                    let params = match parse_factory_parameters(&raw) {
                        Ok(params) => params,
                        Err(error) => {
                            show_flow_page(
                                renderer,
                                &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                            )?;
                            continue;
                        }
                    };
                    for (name, value) in params {
                        args.extend(["--param".into(), format!("{name}={value}")]);
                    }
                    let Some(batch) = init_input(
                        renderer,
                        locale.text(
                            "Optional batch factory module:callable",
                            "可选批次工厂 module:callable",
                            "可選批次工廠 module:callable",
                        ),
                        "",
                    )?
                    else {
                        continue;
                    };
                    if !batch.is_empty() {
                        args.extend(["--batch".into(), batch]);
                    }
                }
                match action.as_str() {
                    "show" => "component_show",
                    "check" => "component_check",
                    _ => "component_test",
                }
            }
            "discover" => {
                args.push("discover".into());
                let Some(distribution) = init_input(
                    renderer,
                    locale.text(
                        "Distribution filter (empty scans selected environment)",
                        "发行包过滤（留空扫描所选环境）",
                        "發行套件篩選（留空掃描所選環境）",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                if !distribution.is_empty() {
                    args.extend(["--distribution".into(), distribution]);
                }
                let Some(path) = init_input(
                    renderer,
                    locale.text(
                        "Optional additional package directory",
                        "可选附加包目录",
                        "可選附加套件目錄",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                if !path.is_empty() {
                    args.extend(["--path".into(), path]);
                }
                "component_discover"
            }
            _ => unreachable!(),
        };
        if matches!(
            action.as_str(),
            "discover" | "test" | "add" | "scaffold" | "install"
        ) {
            let mut plan = String::new();
            renderer.heading(
                &mut plan,
                locale.text(
                    "Confirm component operation",
                    "确认组件操作",
                    "確認元件操作",
                ),
            );
            renderer.line(
                &mut plan,
                &format!(
                    "{}: {}",
                    locale.text("Project", "项目", "專案"),
                    text::plain_text(&cli.project().display().to_string())
                ),
            );
            renderer.line(
                &mut plan,
                &format!(
                    "{}: {}",
                    locale.text("Arguments", "参数", "參數"),
                    text::plain_text(&serde_json::to_string(&args).unwrap_or_default())
                ),
            );
            renderer.line(&mut plan, locale.text("Registration and discovery save component metadata; testing executes component code and may record verification. Registration does not install dependencies, copy code or prove the factory works.", "登记和发现会保存组件元数据；测试会执行组件代码并可能记录验证结果。登记不会安装依赖、复制代码或证明工厂可运行。", "登記和發現會儲存元件中繼資料；測試會執行元件程式碼並可能記錄驗證結果。登記不會安裝依賴、複製程式碼或證明工廠可執行。"));
            if action == "scaffold" {
                renderer.line(&mut plan, locale.text("Creates a new editable package directory; an existing target is not overwritten.", "创建可编辑的新扩展包目录；不会覆盖已有目标。", "建立可編輯的新擴充套件目錄；不會覆蓋現有目標。"));
            }
            if action == "install" {
                renderer.line(&mut plan, locale.text("Build executes package code; isolated builds may download build dependencies. The wheel is installed into project-local storage and registered; the selected Python environment is not modified. A failure may leave partial project files.", "构建会执行包代码；隔离构建可能下载构建依赖。wheel 安装到项目本地存储并登记，不修改所选 Python 环境；失败可能留下部分项目文件。", "建置會執行套件程式碼；隔離建置可能下載建置相依套件。wheel 安裝到專案本地儲存並登記，不修改所選 Python 環境；失敗可能留下部分專案檔案。"));
            }
            if !show_flow_page(renderer, &plan)? {
                return Ok(false);
            }
            if !init_input(
                renderer,
                locale.text(
                    "Type apply to continue",
                    "输入 apply 继续",
                    "輸入 apply 繼續",
                ),
                "",
            )?
            .is_some_and(|s| s == "apply")
            {
                continue;
            }
        }
        match cli.start_background(&args) {
            Ok(task) => {
                if expected == "component_list" {
                    component_page = 0;
                }
                let mutating = matches!(
                    action.as_str(),
                    "discover" | "test" | "add" | "scaffold" | "install"
                );
                *mutation_started |= mutating;
                tasks.push(ComponentTask {
                    task,
                    operation: expected.into(),
                    mutating,
                    completion_handled: false,
                    selection_handled: false,
                    uninstall_reference: None,
                    uninstall_revision: None,
                });
            }
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!(
                        "Could not start: {}\n",
                        text::plain_text(&error.to_string())
                    ),
                )?;
            }
        }
    }
}

struct EnvironmentTask {
    task: crate::cli::BackgroundTask,
    operation: String,
    mutating: bool,
    completion_handled: bool,
}

fn guide_environment(
    cli: &Cli,
    renderer: &Renderer,
    tasks: &mut Vec<EnvironmentTask>,
    previews: &mut [crate::cli::BackgroundTask],
    components: &mut [ComponentTask],
    mutation_started: &mut bool,
) -> io::Result<bool> {
    let locale = renderer.locale;
    loop {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Project environment", "项目环境", "專案環境"),
        );
        renderer.line(&mut content, locale.text("Checks read the selected environment; imports execute library code. Setup can download packages and change the project environment and lock. No operation selects a trainer or framework automatically.", "检查读取所选环境；导入探测会执行库代码。setup 可能下载依赖并修改项目环境和依赖锁。任何操作都不会自动选择训练器或框架。", "檢查讀取所選環境；匯入探測會執行程式庫程式碼。setup 可能下載依賴並修改專案環境和依賴鎖。任何操作都不會自動選擇訓練器或框架。"));
        renderer.line(&mut content, locale.text("Operations run in the background. Exiting the TUI stops active operations; interrupted setup may leave partial files, so inspect the environment before retrying.", "操作在后台运行。退出 TUI 会停止正在运行的操作；中断 setup 可能留下部分文件，重试前请检查环境。", "操作在背景執行。結束 TUI 會停止執行中的操作；中斷 setup 可能留下部分檔案，重試前請檢查環境。"));
        let mut active = false;
        for (index, entry) in tasks
            .iter_mut()
            .enumerate()
            .filter(|(_, entry)| entry.task.project == cli.project())
        {
            match entry.task.poll() {
                Ok(done) => {
                    active |= !done;
                    renderer.line(
                        &mut content,
                        &format!(
                            "{}. {} · {}",
                            index + 1,
                            text::plain_text(&entry.operation),
                            if done {
                                locale.text("finished", "已结束", "已結束")
                            } else {
                                locale.text("running", "运行中", "執行中")
                            }
                        ),
                    );
                    if let Some(result) = entry.task.result() {
                        content.push_str(&renderer.result(result));
                        if let Ok(report) = result {
                            if report.operation == entry.operation {
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "Environment: {}",
                                        text::plain_text(&report.data.to_string())
                                    ),
                                );
                            } else {
                                renderer.line(
                                    &mut content,
                                    locale.text(
                                        "Unexpected operation; inspect before retrying.",
                                        "回执操作不匹配，请检查后再重试。",
                                        "回執操作不相符，請檢查後再重試。",
                                    ),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    active = true;
                    renderer.line(
                        &mut content,
                        &format!(
                            "Status unavailable: {}",
                            text::plain_text(&error.to_string())
                        ),
                    );
                }
            }
        }
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        let Some(action) = init_input(renderer, locale.text("check, libraries, probe, configure, setup, refresh, or b back", "check 检查、libraries 库、probe 探测、configure 配置、setup 准备、refresh 刷新，或 b 返回", "check 檢查、libraries 程式庫、probe 探測、configure 設定、setup 準備、refresh 重新整理，或 b 返回"), "")? else { return Ok(true); };
        if action == "refresh" {
            continue;
        }
        if !matches!(
            action.as_str(),
            "check" | "libraries" | "probe" | "configure" | "setup"
        ) {
            return Ok(true);
        }
        if active {
            show_flow_page(renderer, locale.text("An environment operation is still running in this project; refresh before starting another.\n", "此项目仍有环境操作运行中；启动其他操作前请刷新。\n", "此專案仍有環境操作執行中；啟動其他操作前請重新整理。\n"))?;
            continue;
        }
        if matches!(action.as_str(), "configure" | "setup")
            && previews
                .iter_mut()
                .filter(|task| task.project == cli.project())
                .any(|task| !matches!(task.poll(), Ok(true)))
        {
            show_flow_page(renderer, locale.text("A data preview is active or its status is unknown; finish it before changing the environment.\n", "数据预览正在运行或状态未知；请先结束预览再更改环境。\n", "資料預覽正在執行或狀態未知；請先結束預覽再更改環境。\n"))?;
            continue;
        }
        if matches!(action.as_str(), "configure" | "setup")
            && components
                .iter_mut()
                .filter(|entry| entry.task.project == cli.project())
                .any(|entry| !matches!(entry.task.poll(), Ok(true)))
        {
            show_flow_page(renderer, locale.text("A component operation is active or its status is unknown; wait before changing the environment.\n", "组件操作正在运行或状态未知；请等待完成后再更改环境。\n", "元件操作正在執行或狀態未知；請等待完成後再更改環境。\n"))?;
            continue;
        }
        if tasks.len() >= 8 {
            tasks.retain_mut(|entry| !matches!(entry.task.poll(), Ok(true)));
            if tasks.len() >= 8 {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Too many active environment operations.\n",
                        "运行中的环境操作过多。\n",
                        "執行中的環境操作過多。\n",
                    ),
                )?;
                continue;
            }
        }
        let mut args = vec!["env".to_owned(), action.clone()];
        let expected = match action.as_str() {
            "check" => "environment_check",
            "libraries" => "environment_library_inventory",
            "probe" => {
                let Some(library) = init_input(
                    renderer,
                    locale.text("Library ID to import", "要导入的库 ID", "要匯入的程式庫 ID"),
                    "",
                )?
                else {
                    continue;
                };
                if library.is_empty() {
                    continue;
                }
                args.extend(["--library".into(), library]);
                let Some(runtime) = init_input(
                    renderer,
                    locale.text(
                        "Run supported CPU smoke? Type y",
                        "运行支持的 CPU 冒烟测试？输入 y",
                        "執行支援的 CPU 冒煙測試？輸入 y",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                if runtime.eq_ignore_ascii_case("y") {
                    args.push("--check-runtime".into());
                }
                "environment_library_probe"
            }
            "configure" => {
                let Some(mode) = init_input(
                    renderer,
                    locale.text(
                        "New environment policy: isolate or inherit",
                        "新环境策略：isolate 隔离或 inherit 继承",
                        "新環境策略：isolate 隔離或 inherit 繼承",
                    ),
                    "isolate",
                )?
                else {
                    continue;
                };
                match mode.as_str() {
                    "isolate" => args.push("--isolate-system-site-packages".into()),
                    "inherit" => args.push("--system-site-packages".into()),
                    _ => continue,
                }
                let Some(venv) = init_input(
                    renderer,
                    locale.text(
                        "New environment path (empty uses default)",
                        "新环境路径（留空使用默认）",
                        "新環境路徑（留空使用預設）",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                if !venv.is_empty() {
                    args.extend(["--venv".into(), venv]);
                }
                "environment_configure"
            }
            "setup" => {
                let Some(requirements) = init_input(
                    renderer,
                    locale.text("Requirements file", "依赖文件", "依賴檔案"),
                    "requirements.txt",
                )?
                else {
                    continue;
                };
                let Some(venv) = init_input(
                    renderer,
                    locale.text(
                        "Environment path (empty uses selected/default)",
                        "环境路径（留空使用已选或默认路径）",
                        "環境路徑（留空使用已選或預設路徑）",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                let Some(update) = init_input(
                    renderer,
                    locale.text(
                        "Re-resolve and update existing lock? Type y",
                        "重新解析并更新已有依赖锁？输入 y",
                        "重新解析並更新現有依賴鎖？輸入 y",
                    ),
                    "",
                )?
                else {
                    continue;
                };
                args.extend(["--requirements".into(), requirements]);
                if !venv.is_empty() {
                    args.extend(["--venv".into(), venv]);
                }
                if update.eq_ignore_ascii_case("y") {
                    args.push("--update-lock".into());
                }
                "environment_setup"
            }
            _ => unreachable!(),
        };
        if matches!(action.as_str(), "configure" | "setup") {
            let mut plan = String::new();
            renderer.heading(
                &mut plan,
                locale.text("Confirm environment change", "确认环境变更", "確認環境變更"),
            );
            renderer.line(
                &mut plan,
                &format!(
                    "{}: {}",
                    locale.text("Project", "项目", "專案"),
                    text::plain_text(&cli.project().display().to_string())
                ),
            );
            renderer.line(
                &mut plan,
                &format!(
                    "{}: {}",
                    locale.text("Arguments", "参数", "參數"),
                    text::plain_text(&serde_json::to_string(&args).unwrap_or_default())
                ),
            );
            renderer.line(&mut plan, locale.text("Setup may resolve and download the declared requirements and create or modify the selected virtual environment and lock. Configure writes environment policy. No trainer is selected.", "setup 可能解析、下载所声明的依赖并创建或修改所选虚拟环境及依赖锁；configure 写入环境策略。不会选择训练器。", "setup 可能解析、下載所宣告的依賴並建立或修改所選虛擬環境及依賴鎖；configure 寫入環境策略。不會選擇訓練器。"));
            if !show_flow_page(renderer, &plan)? {
                return Ok(false);
            }
            if !init_input(
                renderer,
                locale.text(
                    "Type apply to continue",
                    "输入 apply 继续",
                    "輸入 apply 繼續",
                ),
                "",
            )?
            .is_some_and(|s| s == "apply")
            {
                continue;
            }
        }
        match cli.start_background(&args) {
            Ok(task) => {
                let mutating = matches!(action.as_str(), "configure" | "setup");
                *mutation_started |= mutating;
                tasks.push(EnvironmentTask {
                    task,
                    operation: expected.into(),
                    mutating,
                    completion_handled: false,
                });
            }
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!(
                        "Could not start: {}\n",
                        text::plain_text(&error.to_string())
                    ),
                )?;
            }
        }
    }
}

fn guide_augmentation(
    cli: &Cli,
    renderer: &Renderer,
    mutation_attempted: &mut bool,
) -> io::Result<bool> {
    let locale = renderer.locale;
    let Some(split) = init_input(
        renderer,
        locale.text(
            "Augmentation split: train or eval",
            "增强切分：train 或 eval",
            "增強切分：train 或 eval",
        ),
        "train",
    )?
    else {
        return Ok(true);
    };
    if !matches!(split.as_str(), "train" | "eval") {
        return Ok(true);
    }
    let list = cli.execute(&["augment", "list", "--split", &split]);
    let report = match &list {
        Ok(report)
            if report.operation == "augment_list"
                && report.status == "success"
                && report.data["steps"].is_array() =>
        {
            report
        }
        _ => return show_flow_page(renderer, &renderer.result(&list)),
    };
    let revision = report.data["config_revision"].as_str().filter(|revision| {
        revision.len() == 64
            && revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    let Some(revision) = revision else {
        return show_flow_page(
            renderer,
            locale.text(
                "Configuration revision unavailable; editing is disabled.\n",
                "配置版本不可用，无法编辑。\n",
                "設定版本不可用，無法編輯。\n",
            ),
        );
    };
    let steps = report.data["steps"].as_array().unwrap();
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Current augmentation", "当前数据增强", "目前資料增強"),
    );
    for (index, step) in steps.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!("{}. {}", index + 1, text::plain_text(&step.to_string())),
        );
    }
    if steps.is_empty() {
        renderer.line(
            &mut content,
            locale.text(
                "No steps configured.",
                "尚未配置增强步骤。",
                "尚未設定增強步驟。",
            ),
        );
    }
    renderer.line(&mut content, locale.text("The order shown is the saved pipeline. Preview checks configuration only; run a data preview to observe actual behavior.", "显示的是已保存的执行顺序。编辑预览只检查配置；请运行数据预览观察实际效果。", "顯示的是已儲存的執行順序。編輯預覽只檢查設定；請執行資料預覽觀察實際效果。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let Some(action) = init_input(
        renderer,
        locale.text(
            "add, set, move, remove, or b back",
            "add 添加、set 修改、move 移动、remove 删除，或 b 返回",
            "add 新增、set 修改、move 移動、remove 刪除，或 b 返回",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !matches!(action.as_str(), "add" | "set" | "move" | "remove") {
        return Ok(true);
    }
    if let Some(message) =
        capability_unavailable(cli, locale, &["augmentation.conditional_config_edit"])
    {
        return show_flow_page(renderer, &message);
    }
    let mut args = vec!["augment".to_owned(), action.clone()];
    if action == "add" {
        let Some(kind) = init_input(renderer, locale.text("Type: Resize, RandomResizedCrop, ToTensor, Normalize, RandomHorizontalFlip, custom", "类型：Resize、RandomResizedCrop、ToTensor、Normalize、RandomHorizontalFlip、custom", "類型：Resize、RandomResizedCrop、ToTensor、Normalize、RandomHorizontalFlip、custom"), "")? else { return Ok(true); };
        let Some(id) = init_input(renderer, locale.text("Step ID", "步骤 ID", "步驟 ID"), "")?
        else {
            return Ok(true);
        };
        args.extend([kind.clone(), "--id".into(), id]);
        if kind == "custom" {
            let Some(entrypoint) = init_input(
                renderer,
                locale.text(
                    "Transform factory module:callable",
                    "变换工厂 module:callable",
                    "變換工廠 module:callable",
                ),
                "",
            )?
            else {
                return Ok(true);
            };
            args.extend(["--entrypoint".into(), entrypoint]);
        }
    } else {
        let Some(choice) = init_input(
            renderer,
            locale.text("Step number", "步骤序号", "步驟序號"),
            "",
        )?
        else {
            return Ok(true);
        };
        let Some(id) = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|index| steps.get(index))
            .and_then(|step| step["id"].as_str())
            .filter(|id| !id.is_empty())
        else {
            return Ok(true);
        };
        args.push(id.to_owned());
    }
    if action == "add" || action == "set" {
        let Some(target) = init_input(
            renderer,
            locale.text(
                "Batch target path (e.g. args.0; empty leaves unset/unchanged)",
                "批次目标路径（如 args.0；留空不设置或不更改）",
                "批次目標路徑（如 args.0；留空不設定或不變更）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if !target.is_empty() {
            args.extend(["--target".into(), target]);
        }
        let Some(raw) = init_input(
            renderer,
            locale.text(
                "Transform parameters as JSON object",
                "变换参数 JSON 对象",
                "變換參數 JSON 物件",
            ),
            "{}",
        )?
        else {
            return Ok(true);
        };
        let params = match parse_factory_parameters(&raw) {
            Ok(params) => params,
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?;
                return Ok(true);
            }
        };
        for (name, value) in params {
            args.extend(["--param".into(), format!("{name}={value}")]);
        }
    }
    if action == "move" {
        let Some(before) = init_input(
            renderer,
            locale.text(
                "Move before step ID (empty moves to end)",
                "移到哪个步骤 ID 之前（留空移到末尾）",
                "移到哪個步驟 ID 之前（留空移到末尾）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if !before.is_empty() {
            args.extend(["--before".into(), before]);
        }
    }
    args.extend([
        "--split".into(),
        split.clone(),
        "--if-config-revision".into(),
        revision.into(),
    ]);
    let mut preview_args = args.clone();
    preview_args.push("--dry-run".into());
    let preview = cli.execute(&preview_args.iter().map(String::as_str).collect::<Vec<_>>());
    let expected_operation = format!("augment_{action}");
    let report = match &preview {
        Ok(report)
            if report.operation == expected_operation
                && report.status == "success"
                && report.data["committed"] == false
                && report.data["base_config_revision"].as_str() == Some(revision)
                && report.data["split"].as_str() == Some(&split)
                && report.data["steps"].is_array()
                && report.data["validation"]["configuration"] == "passed"
                && report.data["validation"]["runtime"] == "not_run" =>
        {
            report
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&preview))?;
            show_flow_page(
                renderer,
                locale.text(
                    "No changes saved; refresh the pipeline before retrying.\n",
                    "未保存更改；重试前请刷新流水线。\n",
                    "未儲存變更；重試前請重新整理流水線。\n",
                ),
            )?;
            return Ok(true);
        }
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Augmentation preview · not saved",
            "增强预览 · 尚未保存",
            "增強預覽 · 尚未儲存",
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "Before: {}",
            text::plain_text(&Value::Array(steps.clone()).to_string())
        ),
    );
    renderer.line(
        &mut content,
        &format!(
            "After: {}",
            text::plain_text(&report.data["steps"].to_string())
        ),
    );
    renderer.line(&mut content, locale.text("Configuration validated; transform code and data were not run. Commit checks the project configuration revision.", "已校验配置；尚未执行变换代码或数据。提交时核对项目配置版本。", "已檢查設定；尚未執行變換程式碼或資料。提交時核對專案設定版本。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    if !init_input(
        renderer,
        locale.text(
            "Save augmentation? Type y",
            "保存增强？输入 y",
            "儲存增強？輸入 y",
        ),
        "",
    )?
    .is_some_and(|answer| answer.eq_ignore_ascii_case("y"))
    {
        return Ok(true);
    }
    *mutation_attempted = true;
    let result = cli.execute(&args.iter().map(String::as_str).collect::<Vec<_>>());
    let mut content = renderer.result(&result);
    if !result.as_ref().is_ok_and(|report| {
        report.operation == expected_operation
            && report.status == "success"
            && report.data["committed"] == true
            && report.data["base_config_revision"].as_str() == Some(revision)
    }) {
        renderer.line(&mut content, locale.text("If the result is interrupted or malformed, inspect the saved pipeline before retrying.", "若结果中断或格式异常，重试前请检查已保存流水线。", "若結果中斷或格式異常，重試前請檢查已儲存流水線。"));
    } else {
        renderer.line(
            &mut content,
            locale.text(
                "Saved. Run a data preview to check the actual transformed sample.",
                "已保存。运行数据预览以检查实际变换后的样本。",
                "已儲存。執行資料預覽以檢查實際變換後的樣本。",
            ),
        );
    }
    show_flow_page(renderer, &content)
}

fn guide_data_preview(
    cli: &Cli,
    renderer: &Renderer,
    tasks: &mut Vec<crate::cli::BackgroundTask>,
    environment_tasks: &mut [EnvironmentTask],
    mutation_attempted: &mut bool,
) -> io::Result<bool> {
    let locale = renderer.locale;
    loop {
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Data previews", "数据预览", "資料預覽"),
        );
        renderer.line(&mut content, locale.text("Preview runs in the background and applies the saved transforms for the selected split. Leaving this screen does not stop it; exiting the TUI stops its active previews.", "预览在后台运行，并应用所选切分已保存的增强。离开此页面不会停止预览；退出 TUI 会停止其运行中的预览。", "預覽在背景執行，並套用所選切分已儲存的增強。離開此頁面不會停止預覽；結束 TUI 會停止其執行中的預覽。"));
        renderer.line(&mut content, locale.text("Reports reflect their original request, not necessarily the current configuration. The session retains up to eight previews; finished receipts may be removed when starting more.", "报告反映发起时的请求，不一定对应当前配置。会话最多保留八个预览；继续启动时可能移除已结束的回执。", "報告反映發起時的要求，不一定對應目前設定。工作階段最多保留八個預覽；繼續啟動時可能移除已結束的回執。"));
        renderer.line(&mut content, locale.text("A seed selects Dataset indices and augmentation randomness; a batch factory controls its own samples.", "种子控制 Dataset 索引选择和增强随机性；批次工厂自行决定样本。", "種子控制 Dataset 索引選擇及增強隨機性；批次工廠自行決定樣本。"));
        for (index, task) in tasks
            .iter_mut()
            .enumerate()
            .filter(|(_, task)| task.project == cli.project())
        {
            if let Err(error) = task.poll() {
                renderer.line(
                    &mut content,
                    &format!("{}. {}", index + 1, text::plain_text(&error.to_string())),
                );
                continue;
            }
            renderer.line(
                &mut content,
                &format!(
                    "{}. {}",
                    index + 1,
                    if task.result().is_some() {
                        locale.text("finished", "已结束", "已結束")
                    } else if task.cancellation_requested {
                        locale.text("cancellation requested", "已请求取消", "已要求取消")
                    } else {
                        locale.text("running", "运行中", "執行中")
                    }
                ),
            );
            if let Some(result) = task.result() {
                content.push_str(&renderer.result(result));
                if let Ok(report) = result {
                    if report.operation == "preview" {
                        if report.status == "success" && report.data["published"] == true {
                            let scope = &report.data["scope"];
                            let coverage = &scope["data_observation"]["coverage"];
                            renderer.line(
                                &mut content,
                                &format!(
                                    "{}: {} / {} · {} {} · {}",
                                    locale.text("Samples", "样本", "樣本"),
                                    scope["actual_samples"],
                                    scope["requested_samples"],
                                    locale.text("seed", "种子", "種子"),
                                    scope["seed"],
                                    text::plain_text(
                                        scope["sampling_algorithm"].as_str().unwrap_or("unknown")
                                    )
                                ),
                            );
                            if coverage.is_object() {
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{}: {} · {}",
                                        locale.text("Coverage", "覆盖范围", "涵蓋範圍"),
                                        text::plain_text(
                                            coverage["kind"].as_str().unwrap_or("unknown")
                                        ),
                                        if coverage["complete"] == true {
                                            locale.text("complete", "完整", "完整")
                                        } else {
                                            locale.text(
                                                "partial or unknown",
                                                "部分或未知",
                                                "部分或未知",
                                            )
                                        }
                                    ),
                                );
                            }
                            if coverage["kind"] == "bounded-window-sample" {
                                renderer.line(
                                    &mut content,
                                    &format!(
                                        "{}: {} / {} · {}: {}",
                                        locale.text(
                                            "Stream items examined",
                                            "已读取流样本",
                                            "已讀取串流樣本"
                                        ),
                                        coverage["examined_items"],
                                        coverage["scan_limit"],
                                        locale.text("selected offsets", "抽样位置", "抽樣位置"),
                                        coverage["selected_offsets"]
                                    ),
                                );
                                renderer.line(&mut content, locale.text("Selection repeats only if the stream prefix and order are unchanged; a factory accepting a seed does not guarantee this.", "仅当流的前缀及顺序不变时抽样位置可复现；工厂接受种子并不能保证数据流不变。", "僅當串流前綴及順序不變時抽樣位置可重現；工廠接受種子並不能保證資料流不變。"));
                            }
                            renderer.line(&mut content, locale.text("A preview is an observation of selected samples, not proof of dataset quality.", "预览仅观察所选样本，不能证明数据集整体质量。", "預覽僅觀察所選樣本，不能證明資料集整體品質。"));
                            if let Some(artifacts) = report.data["artifacts"].as_array() {
                                for artifact in artifacts {
                                    if let (Some(kind), Some(uri)) =
                                        (artifact["kind"].as_str(), artifact["uri"].as_str())
                                    {
                                        renderer.line(
                                            &mut content,
                                            &format!(
                                                "{}: {} · {}",
                                                locale.text("Artifact", "预览文件", "預覽檔案"),
                                                text::plain_text(kind),
                                                text::plain_text(uri)
                                            ),
                                        );
                                    }
                                }
                                renderer.line(&mut content, locale.text("Open the project-relative preview document locally; image previews may contain sample data.", "可在本地打开项目内的预览文档；图像预览可能包含样本数据。", "可在本機開啟專案內的預覽文件；圖像預覽可能包含樣本資料。"));
                            }
                        }
                        renderer.line(
                            &mut content,
                            &format!("Preview: {}", text::plain_text(&report.data.to_string())),
                        );
                    } else {
                        renderer.line(
                            &mut content,
                            locale.text(
                                "Unexpected operation; this is not a preview receipt.",
                                "回执操作不匹配；不能视为预览结果。",
                                "回執操作不相符；不能視為預覽結果。",
                            ),
                        );
                    }
                }
            }
        }
        if !show_flow_page(renderer, &content)? {
            return Ok(false);
        }
        let Some(action) = init_input(
            renderer,
            locale.text(
                "start, refresh, cancel NUMBER, augment, or b back",
                "start 启动、refresh 刷新、cancel 序号、augment 增强，或 b 返回",
                "start 啟動、refresh 重新整理、cancel 序號、augment 增強，或 b 返回",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if action == "refresh" {
            continue;
        }
        if let Some(index) = action
            .strip_prefix("cancel ")
            .and_then(|n| n.trim().parse::<usize>().ok())
            .and_then(|n| n.checked_sub(1))
        {
            if let Some(task) = tasks
                .get_mut(index)
                .filter(|task| task.project == cli.project())
            {
                if let Err(error) = task.cancel() {
                    show_flow_page(
                        renderer,
                        &format!(
                            "Cancellation failed: {}\n",
                            text::plain_text(&error.to_string())
                        ),
                    )?;
                }
            } else {
                show_flow_page(
                    renderer,
                    locale.text(
                        "No such preview in this project.\n",
                        "当前项目没有该预览。\n",
                        "目前專案沒有該預覽。\n",
                    ),
                )?;
            }
            continue;
        }
        if action == "augment" {
            if tasks
                .iter_mut()
                .filter(|task| task.project == cli.project())
                .any(|task| !matches!(task.poll(), Ok(true)))
            {
                show_flow_page(renderer, locale.text("A preview is active or its status is unknown; finish it before editing augmentation.\n", "预览正在运行或状态未知；请先结束再编辑增强。\n", "預覽正在執行或狀態未知；請先結束再編輯增強。\n"))?;
                continue;
            }
            if !guide_augmentation(cli, renderer, mutation_attempted)? {
                return Ok(false);
            }
            continue;
        }
        if action != "start" {
            return Ok(true);
        }
        if environment_tasks
            .iter_mut()
            .filter(|entry| entry.task.project == cli.project())
            .any(|entry| !matches!(entry.task.poll(), Ok(true)))
        {
            show_flow_page(renderer, locale.text("An environment operation is active or its status is unknown; wait before starting a preview.\n", "环境操作正在运行或状态未知；请等待完成后再启动预览。\n", "環境操作正在執行或狀態未知；請等待完成後再啟動預覽。\n"))?;
            continue;
        }
        if tasks.len() >= 8 {
            tasks.retain_mut(|task| !matches!(task.poll(), Ok(true)));
            if tasks.len() >= 8 {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Eight previews are still active; finish or cancel one first.\n",
                        "已有八个预览运行中，请先等待或取消一个。\n",
                        "已有八個預覽執行中，請先等待或取消一個。\n",
                    ),
                )?;
                continue;
            }
        }
        let Some(kind) = init_input(
            renderer,
            locale.text(
                "Source: batch or dataset",
                "来源：batch 或 dataset",
                "來源：batch 或 dataset",
            ),
            "batch",
        )?
        else {
            continue;
        };
        if !matches!(kind.as_str(), "batch" | "dataset") {
            continue;
        }
        let saved_config = if kind == "batch" {
            cli.execute(&["assembly", "show", "--config-only"])
                .ok()
                .filter(|report| report.operation == "assembly_show" && report.status == "success")
        } else {
            None
        };
        let saved_batch = saved_config
            .as_ref()
            .and_then(|report| report.data["batch"].as_str())
            .filter(|value| !value.is_empty());
        if saved_batch.is_some() {
            println!("{}", locale.text("Suggested factory comes from project configuration; matching non-redacted parameters are suggested below. Redacted values must be entered manually.", "建议工厂来自项目配置；匹配且未脱敏的参数会在下方建议，脱敏值须手动输入。", "建議工廠來自專案設定；相符且未去識別的參數會在下方建議，去識別值須手動輸入。"));
        }
        let Some(factory) = init_input(
            renderer,
            locale.text(
                "Factory module:callable",
                "工厂 module:callable",
                "工廠 module:callable",
            ),
            saved_batch.unwrap_or(""),
        )?
        else {
            continue;
        };
        if factory.is_empty() {
            continue;
        }
        let Some(split) = init_input(
            renderer,
            locale.text("Split", "数据切分", "資料切分"),
            "train",
        )?
        else {
            continue;
        };
        let Some(samples) = init_input(
            renderer,
            locale.text(
                "Maximum samples (1-64), optional @seed and dataset /scan limit (e.g. 12@42/1000)",
                "最大样本数（1-64），可用 @种子及流式数据集 /扫描上限（如 12@42/1000）",
                "最大樣本數（1-64），可用 @種子及串流資料集 /掃描上限（如 12@42/1000）",
            ),
            "12",
        )?
        else {
            continue;
        };
        let mut window_parts = samples.split('/');
        let selection = window_parts.next().unwrap_or("");
        let scan_limit = window_parts.next().map(str::trim);
        let mut parts = selection.split('@');
        let count = parts.next().unwrap_or("").trim();
        let seed = parts.next().map(str::trim);
        if !count.parse::<u8>().is_ok_and(|n| (1..=64).contains(&n))
            || seed.is_some_and(|value| value.parse::<u64>().is_err())
            || parts.next().is_some()
            || window_parts.next().is_some()
            || scan_limit.is_some_and(|value| {
                kind != "dataset"
                    || !value
                        .parse::<u32>()
                        .is_ok_and(|n| n >= count.parse::<u32>().unwrap_or(0) && n <= 100_000)
            })
        {
            show_flow_page(
                renderer,
                locale.text(
                    "Use count 1-64, optional @seed and dataset /scan limit between count and 100000, e.g. 12@42/1000.\n",
                    "请输入 1 到 64 的样本数、可选 @种子和数据集 /扫描上限（不小于样本数且不超过 100000），例如 12@42/1000。\n",
                    "請輸入 1 到 64 的樣本數、可選 @種子和資料集 /掃描上限（不小於樣本數且不超過 100000），例如 12@42/1000。\n",
                ),
            )?;
            continue;
        }
        let suggested_params = saved_config
            .as_ref()
            .filter(|report| {
                report.data["batch"].as_str() == Some(factory.as_str())
                    && !report.data["redacted_fields"]
                        .as_array()
                        .is_some_and(|fields| {
                            fields
                                .iter()
                                .filter_map(Value::as_str)
                                .any(|path| path.starts_with("$.assembly.batch_params"))
                        })
            })
            .and_then(|report| report.data["batch_params"].as_object())
            .map(|params| Value::Object(params.clone()).to_string())
            .unwrap_or_else(|| "{}".into());
        let Some(raw) = init_input(
            renderer,
            locale.text(
                "Factory parameters as JSON object",
                "工厂参数 JSON 对象",
                "工廠參數 JSON 物件",
            ),
            &suggested_params,
        )?
        else {
            continue;
        };
        let params = match parse_factory_parameters(&raw) {
            Ok(value) => value,
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?;
                continue;
            }
        };
        let Some(mode) = init_input(
            renderer,
            locale.text(
                "Renderer: auto, summary, or custom",
                "预览方式：auto 自动、summary 摘要或 custom 自定义",
                "預覽方式：auto 自動、summary 摘要或 custom 自訂",
            ),
            "auto",
        )?
        else {
            continue;
        };
        if !matches!(mode.as_str(), "auto" | "summary" | "custom") {
            continue;
        }
        let mut args = vec![
            format!("--{kind}"),
            factory,
            "--split".into(),
            split,
            "--samples".into(),
            count.to_owned(),
            "--renderer".into(),
            mode.clone(),
        ];
        if let Some(seed) = seed {
            args.extend(["--seed".into(), seed.to_owned()]);
        }
        if let Some(limit) = scan_limit {
            args.extend(["--scan-limit".into(), limit.to_owned()]);
        }
        if mode == "custom" {
            let Some(entrypoint) = init_input(
                renderer,
                locale.text(
                    "Renderer factory module:callable",
                    "渲染器工厂 module:callable",
                    "渲染器工廠 module:callable",
                ),
                "",
            )?
            else {
                continue;
            };
            let Some(version) = init_input(
                renderer,
                locale.text(
                    "Renderer version (update when code changes)",
                    "渲染器版本（代码变更时更新）",
                    "渲染器版本（程式碼變更時更新）",
                ),
                "1",
            )?
            else {
                continue;
            };
            if entrypoint.is_empty() || version.is_empty() {
                continue;
            }
            args.extend([
                "--renderer-entrypoint".into(),
                entrypoint,
                "--renderer-version".into(),
                version,
            ]);
        }
        for (name, value) in params {
            args.extend(["--param".into(), format!("{name}={value}")]);
        }
        match cli.start_preview(&args) {
            Ok(task) => tasks.push(task),
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!(
                        "Could not start preview: {}\n",
                        text::plain_text(&error.to_string())
                    ),
                )?;
            }
        }
    }
}

fn component_field_error(schema: &Value, value: &Value) -> Option<&'static str> {
    let Some(kind) = schema["type"].as_str() else {
        return Some("field type is unavailable");
    };
    let matches = match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "json" => true,
        _ => return Some("unsupported field type"),
    };
    if !matches {
        return Some("value has the wrong type");
    }
    if schema["enum"]
        .as_array()
        .is_some_and(|choices| !choices.contains(value))
    {
        return Some("value is not in the allowed choices");
    }
    if let Some(number) = value.as_f64() {
        if schema["minimum"]
            .as_f64()
            .is_some_and(|bound| number < bound)
        {
            return Some("value is below the minimum");
        }
        if schema["maximum"]
            .as_f64()
            .is_some_and(|bound| number > bound)
        {
            return Some("value is above the maximum");
        }
    }
    None
}

fn field_error_message(locale: Locale, error: &'static str) -> &'static str {
    match error {
        "field type is unavailable" => locale.text(error, "字段类型不可用", "欄位型別不可用"),
        "unsupported field type" => locale.text(error, "不支持的字段类型", "不支援的欄位型別"),
        "value has the wrong type" => locale.text(error, "值的类型不匹配", "值的型別不相符"),
        "value is not in the allowed choices" => {
            locale.text(error, "值不在允许的选项中", "值不在允許的選項中")
        }
        "value is below the minimum" => locale.text(error, "值小于下限", "值小於下限"),
        "value is above the maximum" => locale.text(error, "值大于上限", "值大於上限"),
        _ => error,
    }
}

fn guide_builtin_parameters(
    cli: &Cli,
    renderer: &Renderer,
    kind: &str,
) -> io::Result<Option<BTreeMap<String, Value>>> {
    let locale = renderer.locale;
    let result = cli.execute(&["layer", "catalog"]);
    let report = match &result {
        Ok(report)
            if report.operation == "layer_catalog"
                && report.status == "success"
                && report.data["catalog_version"] == 1
                && report.data["scope"] == "common-pytorch-constructor-fields" =>
        {
            report
        }
        _ => {
            show_flow_page(renderer, &renderer.result(&result))?;
            return Ok(None);
        }
    };
    let Some(layer) = report.data["layers"].as_array().and_then(|layers| {
        layers
            .iter()
            .find(|layer| layer["name"].as_str() == Some(kind))
    }) else {
        show_flow_page(
            renderer,
            locale.text(
                "No built-in field guidance for this type; use JSON or a registered component.\n",
                "此类型没有内置字段引导；请使用 JSON 或已登记组件。\n",
                "此類型沒有內建欄位引導；請使用 JSON 或已登記元件。\n",
            ),
        )?;
        return Ok(None);
    };
    let Some(fields) = layer["fields"].as_array() else {
        show_flow_page(
            renderer,
            locale.text(
                "Invalid Layer catalog.\n",
                "Layer 目录格式无效。\n",
                "Layer 目錄格式無效。\n",
            ),
        )?;
        return Ok(None);
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Built-in Layer fields",
            "内置 Layer 字段",
            "內建 Layer 欄位",
        ),
    );
    renderer.line(&mut content, &text::plain_text(kind));
    for field in fields {
        renderer.line(&mut content, &text::plain_text(&field.to_string()));
    }
    renderer.line(&mut content, locale.text("Common fields only. Leave optional fields empty to use PyTorch defaults; actual constructor and input compatibility are checked in the candidate preview.", "仅列出常用字段。可选字段留空使用 PyTorch 默认值；实际构造及输入兼容性由候选预览检查。", "僅列出常用欄位。可選欄位留空使用 PyTorch 預設值；實際建構及輸入相容性由候選預覽檢查。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(None);
    }
    let mut params = BTreeMap::new();
    for field in fields {
        let Some(name) = field["name"].as_str().filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        let Some(kind) = field["type"].as_str() else {
            return Ok(None);
        };
        let required = field["required"] == true;
        loop {
            let label = format!(
                "{} {} ({}, {}):",
                locale.text("Value for", "输入", "輸入"),
                text::plain_text(name),
                text::plain_text(kind),
                if required {
                    locale.text("required", "必填", "必填")
                } else {
                    locale.text("empty skips", "留空跳过", "留空略過")
                }
            );
            let Some(raw) = init_input(renderer, &label, "")? else {
                return Ok(None);
            };
            if raw.is_empty() || raw == ":skip" {
                if required {
                    show_flow_page(
                        renderer,
                        locale.text(
                            "This field is required.\n",
                            "此字段必填。\n",
                            "此欄位必填。\n",
                        ),
                    )?;
                    continue;
                }
                break;
            }
            let value = match serde_json::from_str::<Value>(&raw) {
                Ok(value) => value,
                Err(_) if kind == "string" => Value::String(raw),
                Err(error) => {
                    show_flow_page(
                        renderer,
                        &format!(
                            "Invalid JSON value: {}\n",
                            text::plain_text(&error.to_string())
                        ),
                    )?;
                    continue;
                }
            };
            if let Some(error) = component_field_error(field, &value) {
                show_flow_page(
                    renderer,
                    &format!(
                        "{}: {}\n",
                        locale.text("Invalid value", "值无效", "值無效"),
                        text::plain_text(field_error_message(locale, error))
                    ),
                )?;
                continue;
            }
            params.insert(name.to_owned(), value);
            break;
        }
    }
    Ok(Some(params))
}

fn guide_component_parameters(
    cli: &Cli,
    renderer: &Renderer,
    requested: &str,
) -> io::Result<Option<(String, BTreeMap<String, Value>)>> {
    let locale = renderer.locale;
    let result = cli.execute(&["component", "show", requested]);
    let report = match &result {
        Ok(report) if report.operation == "component_show" && report.status == "success" => report,
        _ => {
            show_flow_page(renderer, &renderer.result(&result))?;
            return Ok(None);
        }
    };
    let manifest = &report.data["component"]["manifest"];
    let (Some(id), Some(version), Some(properties)) = (
        manifest["component_id"].as_str().filter(|s| !s.is_empty()),
        manifest["component_version"]
            .as_str()
            .filter(|s| !s.is_empty()),
        manifest["params_schema"]["properties"].as_object(),
    ) else {
        show_flow_page(
            renderer,
            locale.text(
                "Component metadata or Schema is unavailable.\n",
                "组件元数据或 Schema 不可用。\n",
                "元件中繼資料或 Schema 不可用。\n",
            ),
        )?;
        return Ok(None);
    };
    if manifest["kind"] != "layer" || manifest["params_schema"]["type"] != "object" {
        show_flow_page(
            renderer,
            locale.text(
                "The selected component is not a Layer with an object parameter Schema.\n",
                "所选组件不是具有对象参数 Schema 的 Layer。\n",
                "所選元件不是具有物件參數 Schema 的 Layer。\n",
            ),
        )?;
        return Ok(None);
    }
    let reference = format!("{id}@{version}");
    let schema = &manifest["params_schema"];
    let required: Vec<&str> = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Component parameter Schema",
            "组件参数 Schema",
            "元件參數 Schema",
        ),
    );
    renderer.line(&mut content, &text::plain_text(&reference));
    for (name, field) in properties {
        renderer.line(
            &mut content,
            &format!(
                "{}{}: {}",
                text::plain_text(name),
                if required.contains(&name.as_str()) {
                    " *"
                } else {
                    ""
                },
                text::plain_text(&field.to_string())
            ),
        );
    }
    if properties.is_empty() {
        renderer.line(
            &mut content,
            locale.text("No named parameters.", "没有具名参数。", "沒有具名參數。"),
        );
    }
    renderer.line(&mut content, locale.text("Field mode checks simple types, choices and bounds; nested values and the complete Schema are checked by the backend during preview. Metadata may change before preview.", "逐字段模式检查基本类型、选项和范围；嵌套值及完整 Schema 由后端在预览时校验。元数据可能在预览前变化。", "逐欄位模式檢查基本型別、選項和範圍；巢狀值及完整 Schema 由後端在預覽時驗證。中繼資料可能在預覽前變更。"));
    renderer.line(&mut content, locale.text("Optional fields may be left empty or set to :skip; the backend applies declared defaults. Strings accept plain text or JSON quotes; arrays and objects use JSON.", "可选字段留空或输入 :skip；声明的默认值由后端填充。字符串可直接输入或使用 JSON 引号；数组和对象使用 JSON。", "可選欄位留空或輸入 :skip；宣告的預設值由後端填入。字串可直接輸入或使用 JSON 引號；陣列和物件使用 JSON。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(None);
    }
    let Some(mode) = init_input(
        renderer,
        locale.text(
            "Parameters: fields or json",
            "参数方式：fields 逐字段或 json",
            "參數方式：fields 逐欄位或 json",
        ),
        "fields",
    )?
    else {
        return Ok(None);
    };
    if mode == "json" {
        let Some(raw) = init_input(
            renderer,
            locale.text(
                "Component parameters as JSON object",
                "组件参数 JSON 对象",
                "元件參數 JSON 物件",
            ),
            "{}",
        )?
        else {
            return Ok(None);
        };
        return match parse_factory_parameters(&raw) {
            Ok(params) => Ok(Some((reference, params))),
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?;
                Ok(None)
            }
        };
    }
    if mode != "fields" {
        return Ok(None);
    }
    let mut params = BTreeMap::new();
    for (name, field) in properties {
        let mandatory = required.contains(&name.as_str()) && field.get("default").is_none();
        loop {
            let label = format!(
                "{} {} ({}, {}):",
                locale.text("Value for", "输入", "輸入"),
                text::plain_text(name),
                text::plain_text(field["type"].as_str().unwrap_or("?")),
                if mandatory {
                    locale.text("required", "必填", "必填")
                } else {
                    locale.text("empty skips", "留空跳过", "留空略過")
                }
            );
            let Some(raw) = init_input(renderer, &label, "")? else {
                return Ok(None);
            };
            if raw.is_empty() || raw == ":skip" {
                if mandatory {
                    show_flow_page(
                        renderer,
                        locale.text(
                            "This field is required.\n",
                            "此字段必填。\n",
                            "此欄位必填。\n",
                        ),
                    )?;
                    continue;
                }
                break;
            }
            let value = match serde_json::from_str::<Value>(&raw) {
                Ok(value) => value,
                Err(_) if field["type"] == "string" => Value::String(raw),
                Err(error) => {
                    show_flow_page(
                        renderer,
                        &format!(
                            "Invalid JSON value: {}\n",
                            text::plain_text(&error.to_string())
                        ),
                    )?;
                    continue;
                }
            };
            if let Some(error) = component_field_error(field, &value) {
                let message = field_error_message(locale, error);
                show_flow_page(
                    renderer,
                    &format!(
                        "{}: {}\n",
                        locale.text("Invalid value", "值无效", "值無效"),
                        text::plain_text(message)
                    ),
                )?;
                continue;
            }
            params.insert(name.to_owned(), value);
            break;
        }
    }
    if schema["additionalProperties"] != false {
        let Some(raw) = init_input(
            renderer,
            locale.text(
                "Additional parameters as JSON object",
                "额外参数 JSON 对象",
                "額外參數 JSON 物件",
            ),
            "{}",
        )?
        else {
            return Ok(None);
        };
        let extra = match parse_factory_parameters(&raw) {
            Ok(params) => params,
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?;
                return Ok(None);
            }
        };
        if extra.keys().any(|name| properties.contains_key(name)) {
            show_flow_page(
                renderer,
                locale.text(
                    "Additional parameters cannot replace named fields.\n",
                    "额外参数不能覆盖具名字段。\n",
                    "額外參數不能覆蓋具名欄位。\n",
                ),
            )?;
            return Ok(None);
        }
        params.extend(extra);
    }
    Ok(Some((reference, params)))
}

fn guide_layer_set_params(
    renderer: &Renderer,
    module: &Value,
) -> io::Result<Option<BTreeMap<String, Value>>> {
    let locale = renderer.locale;
    let Some(fields) = module["editable_params"]
        .as_object()
        .filter(|fields| !fields.is_empty())
    else {
        show_flow_page(
            renderer,
            locale.text(
                "No editable parameters were reported.\n",
                "未报告可编辑参数。\n",
                "未回報可編輯參數。\n",
            ),
        )?;
        return Ok(None);
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Editable layer parameters",
            "可编辑的层参数",
            "可編輯的層參數",
        ),
    );
    for (index, (name, value)) in fields.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!(
                "{}. {} = {}",
                index + 1,
                text::plain_text(name),
                text::plain_text(&value.to_string())
            ),
        );
    }
    renderer.line(&mut content, locale.text("Choose field numbers or names, then :done. Use :list to view fields again. Values use JSON syntax; a JSON object is also accepted.", "逐个输入字段序号或名称，完成后输入 :done；:list 再次查看字段。字段值使用 JSON 格式，也可直接输入 JSON 对象。", "逐一輸入欄位序號或名稱，完成後輸入 :done；:list 再次查看欄位。欄位值使用 JSON 格式，也可直接輸入 JSON 物件。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(None);
    }
    let mut changes = BTreeMap::new();
    loop {
        let Some(choice) = init_input(
            renderer,
            locale.text(
                "Layer parameters: field, :list, :done, or JSON object",
                "层参数：字段、:list、:done 或 JSON 对象",
                "層參數：欄位、:list、:done 或 JSON 物件",
            ),
            "",
        )?
        else {
            return Ok(None);
        };
        if choice == ":list" {
            if !show_flow_page(renderer, &content)? {
                return Ok(None);
            }
            continue;
        }
        if choice == ":done" {
            if changes.is_empty() {
                show_flow_page(
                    renderer,
                    locale.text(
                        "No parameter changes selected.\n",
                        "没有选择参数更改。\n",
                        "沒有選擇參數變更。\n",
                    ),
                )?;
                return Ok(None);
            }
            return Ok(Some(changes));
        }
        if choice.trim_start().starts_with('{') {
            match parse_factory_parameters(&choice) {
                Ok(params) if !params.is_empty() => {
                    changes.extend(params);
                    return Ok(Some(changes));
                }
                Ok(_) => show_flow_page(
                    renderer,
                    locale.text(
                        "Enter at least one parameter.\n",
                        "请至少输入一个参数。\n",
                        "請至少輸入一個參數。\n",
                    ),
                )?,
                Err(error) => show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?,
            };
            continue;
        }
        let selected = choice
            .parse::<usize>()
            .ok()
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| fields.keys().nth(index))
            .cloned()
            .unwrap_or(choice);
        let Some(current) = fields.get(&selected) else {
            show_flow_page(
                renderer,
                locale.text(
                    "Unknown field; use one of the reported names.\n",
                    "字段不存在；请使用列表中的名称。\n",
                    "欄位不存在；請使用清單中的名稱。\n",
                ),
            )?;
            continue;
        };
        let previous = changes.get(&selected).unwrap_or(current);
        let label = format!(
            "{} {}",
            locale.text("JSON value for", "JSON 值：", "JSON 值："),
            text::plain_text(&selected)
        );
        let Some(raw) = init_input(renderer, &label, &previous.to_string())? else {
            return Ok(None);
        };
        match serde_json::from_str::<Value>(&raw) {
            Ok(value) => {
                changes.insert(selected, value);
            }
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!(
                        "Invalid JSON value: {}\n",
                        text::plain_text(&error.to_string())
                    ),
                )?;
            }
        }
    }
}

fn guide_supervision_rules(
    cli: &Cli,
    renderer: &Renderer,
    attempted: &mut bool,
) -> io::Result<bool> {
    let locale = renderer.locale;
    let listed = cli.execute(&["model", "supervise", "rule", "list"]);
    let report = match &listed {
        Ok(report)
            if report.operation == "semantic_supervise"
                && report.status == "success"
                && report.data["rules"].is_array() =>
        {
            report
        }
        _ => return show_flow_page(renderer, &renderer.result(&listed)),
    };
    let rules = report.data["rules"].as_array().unwrap();
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Saved supervision rules",
            "已保存监督规则",
            "已儲存監督規則",
        ),
    );
    for (index, rule) in rules.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!("{}. {}", index + 1, text::plain_text(&rule.to_string())),
        );
    }
    if rules.is_empty() {
        renderer.line(
            &mut content,
            locale.text(
                "No saved rules.",
                "没有已保存的规则。",
                "沒有已儲存的規則。",
            ),
        );
    }
    renderer.line(&mut content, locale.text("Rules affect training only when explicitly enabled. Removal requires the selected rule to remain unchanged; inspect saved rules before retrying an uncertain result.", "规则仅在训练时明确启用才会生效。仅当所选规则未变化时才会移除；结果不确定时请先检查。", "規則僅在訓練時明確啟用才會生效。僅當所選規則未變更時才會移除；結果不確定時請先檢查。"));
    if !show_flow_page(renderer, &content)? {
        return Ok(false);
    }
    let Some(choice) = init_input(
        renderer,
        locale.text(
            "Rule number to remove, or b back",
            "输入要移除的规则序号，或 b 返回",
            "輸入要移除的規則序號，或 b 返回",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(rule) = choice
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| rules.get(n))
    else {
        return Ok(true);
    };
    let Some(id) = rule["id"].as_str().filter(|id| !id.is_empty()) else {
        return show_flow_page(
            renderer,
            locale.text(
                "Rule has no valid ID; nothing removed.\n",
                "规则没有有效 ID，未移除。\n",
                "規則沒有有效 ID，未移除。\n",
            ),
        );
    };
    let mut detail = String::new();
    renderer.heading(
        &mut detail,
        locale.text("Confirm rule removal", "确认移除规则", "確認移除規則"),
    );
    renderer.line(&mut detail, &text::plain_text(&rule.to_string()));
    if !show_flow_page(renderer, &detail)? {
        return Ok(false);
    }
    if !init_input(
        renderer,
        locale.text(
            "Type remove to continue",
            "输入 remove 继续",
            "輸入 remove 繼續",
        ),
        "",
    )?
    .is_some_and(|answer| answer == "remove")
    {
        return Ok(true);
    }
    let current = cli.execute(&["model", "supervise", "rule", "list"]);
    let unchanged = current.as_ref().is_ok_and(|report| {
        report.operation == "semantic_supervise"
            && report.status == "success"
            && report.data["rules"].as_array().is_some_and(|items| {
                let mut matching = items.iter().filter(|item| item["id"].as_str() == Some(id));
                matching.next() == Some(rule) && matching.next().is_none()
            })
    });
    if !unchanged {
        return show_flow_page(
            renderer,
            locale.text(
                "Rule changed or cannot be verified; nothing removed. Refresh before retrying.\n",
                "规则已变化或无法核对；未移除，请刷新后重试。\n",
                "規則已變更或無法核對；未移除，請重新整理後重試。\n",
            ),
        );
    }
    *attempted = true;
    let expected_rule = rule.to_string();
    let result = cli.execute(&[
        "model",
        "supervise",
        "rule",
        "remove",
        "--id",
        id,
        "--expected-rule",
        &expected_rule,
    ]);
    let mut content = renderer.result(&result);
    if !result.as_ref().is_ok_and(|report| {
        report.operation == "semantic_supervise"
            && report.status == "success"
            && report.data["removed"].as_str() == Some(id)
    }) {
        renderer.line(&mut content, locale.text("If the result is interrupted or malformed, inspect the saved rules before retrying.", "若结果中断或格式异常，请检查已保存规则后再重试。", "若結果中斷或格式異常，請檢查已儲存規則後再重試。"));
    }
    show_flow_page(renderer, &content)
}

fn guide_supervision(
    cli: &Cli,
    renderer: &Renderer,
    diagnosis: &RecordedDiagnosis,
) -> io::Result<bool> {
    let locale = renderer.locale;
    if cli.uses_legacy_transport() {
        return show_flow_page(
            renderer,
            locale.text(
                "Historical CLI cannot edit supervision; upgrade m-model. No changes were made.\n",
                "旧版 CLI 无法编辑监督，请升级 m-model；未作任何修改。\n",
                "舊版 CLI 無法編輯監督，請升級 m-model；未作任何修改。\n",
            ),
        );
    }
    let candidates: Vec<(&Value, &str, Vec<&str>)> = diagnosis.report.data["calls"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|call| {
            let source = call["implementation"]["source_file"].as_str()?;
            if source.is_empty() || source.starts_with('@') {
                return None;
            }
            let path = call["paths"].as_array()?.first()?.as_str()?;
            let ports: Vec<&str> = call["input_output_paths"]
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .filter(|port| port.starts_with("outputs"))
                .collect();
            (!ports.is_empty()).then_some((call, path, ports))
        })
        .collect();
    if candidates.is_empty() {
        println!(
            "{}",
            locale.text(
                "No observed project-source output ports available; probe the model first.",
                "没有可用的项目源码输出端口，请先运行前向诊断。",
                "沒有可用的專案原始碼輸出連接埠，請先執行前向診斷。"
            )
        );
        return Ok(true);
    }
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text(
            "Observed project-source calls",
            "已观察的项目源码调用",
            "已觀察的專案原始碼呼叫",
        ),
    );
    for (index, (call, path, ports)) in candidates.iter().enumerate() {
        renderer.line(
            &mut content,
            &format!(
                "{}. {} · {} · {}",
                index + 1,
                text::plain_text(path),
                text::plain_text(call["module_type"].as_str().unwrap_or("?")),
                text::plain_text(&ports.join(", "))
            ),
        );
    }
    renderer.line(&mut content, locale.text("A mark binds a module path and port, including repeated calls. Saving runs a new forward; inputs and source may differ from this observation.", "标记绑定模块路径和端口，涵盖重复调用。保存会重新前向，输入和源码可能与此观察不同。", "標記綁定模組路徑和連接埠，涵蓋重複呼叫。儲存會重新前向，輸入和原始碼可能與此觀察不同。"));
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        if !view_report(&content, locale, renderer.style)? {
            return Ok(false);
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    print!("{content}");
    let Some(number) = init_input(
        renderer,
        locale.text("Call number", "调用序号", "呼叫序號"),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some((_, path, ports)) = number
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| candidates.get(n))
    else {
        return Ok(true);
    };
    let Some(port) = init_input(
        renderer,
        locale.text(
            "Observed output port",
            "已观察的输出端口",
            "已觀察的輸出連接埠",
        ),
        ports[0],
    )?
    else {
        return Ok(true);
    };
    if !ports.contains(&port.as_str()) {
        println!(
            "{}",
            locale.text(
                "Port was not observed; nothing saved.",
                "端口未被观察；没有保存。",
                "連接埠未被觀察；沒有儲存。"
            )
        );
        return Ok(true);
    }
    let Some(role) = init_input(
        renderer,
        locale.text("Semantic role", "语义角色", "語義角色"),
        "",
    )?
    else {
        return Ok(true);
    };
    let Some(meaning) = init_input(renderer, locale.text("Meaning", "语义说明", "語義說明"), "")?
    else {
        return Ok(true);
    };
    let Some(confirm) = init_input(
        renderer,
        locale.text(
            "Save mark after a new forward? Type y",
            "重新前向后保存标记？输入 y",
            "重新前向後儲存標記？輸入 y",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !confirm.trim().eq_ignore_ascii_case("y") {
        return Ok(true);
    }
    let observed = &diagnosis.command;
    let mut context = Vec::<String>::new();
    if !observed.model.is_empty() {
        context.extend(["--model".into(), observed.model.clone()]);
    }
    if !observed.batch.is_empty() {
        context.extend(["--batch".into(), observed.batch.clone()]);
    }
    if !observed.variant.is_empty() {
        context.extend(["--variant".into(), observed.variant.clone()]);
    }
    if observed.probe_options.training {
        context.push("--training".into());
    }
    if !observed.probe_options.binding.is_empty() {
        context.extend([
            "--batch-binding".into(),
            observed.probe_options.binding.clone(),
        ]);
    }
    for (name, value) in &observed.probe_options.model_params {
        context.extend(["--model-param".into(), format!("{name}={value}")]);
    }
    for (name, value) in &observed.probe_options.batch_params {
        context.extend(["--batch-param".into(), format!("{name}={value}")]);
    }
    let mut mark_args = vec![
        "model",
        "mark",
        "--path",
        *path,
        "--port",
        &port,
        "--role",
        &role,
        "--meaning",
        &meaning,
    ];
    mark_args.extend(context.iter().map(String::as_str));
    let mark = match cli.execute(&mark_args) {
        Ok(report) if report.operation == "semantic_mark" && report.status == "success" => report,
        other => {
            print!("{}", renderer.result(&other));
            if other.is_err()
                || other
                    .as_ref()
                    .is_ok_and(|report| report.status == "success")
            {
                println!(
                    "{}",
                    locale.text(
                        "Mark save outcome is unknown; inspect saved marks before retrying.",
                        "标记保存结果未知；重试前请检查已保存标记。",
                        "標記儲存結果未知；重試前請檢查已儲存標記。"
                    )
                );
            }
            return Ok(true);
        }
    };
    let Some(mark_id) = mark.data["mark"]["id"].as_str().filter(|id| !id.is_empty()) else {
        println!(
            "{}",
            locale.text(
                "Mark receipt is incomplete; inspect saved marks before retrying.",
                "标记回执不完整；重试前请检查已保存标记。",
                "標記回執不完整；重試前請檢查已儲存標記。"
            )
        );
        return Ok(true);
    };
    println!(
        "{} {}",
        locale.text("Mark saved:", "已保存标记：", "已儲存標記："),
        text::plain_text(mark_id)
    );
    let Some(kind) = init_input(
        renderer,
        locale.text(
            "Rule: finite, range, batch-alignment, custom-validator or custom-penalty (b to finish)",
            "规则：finite、range、batch-alignment、custom-validator 或 custom-penalty（b 结束）",
            "規則：finite、range、batch-alignment、custom-validator 或 custom-penalty（b 結束）",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !matches!(
        kind.as_str(),
        "finite" | "range" | "batch-alignment" | "custom-validator" | "custom-penalty"
    ) {
        return Ok(true);
    }
    let Some(mode) = init_input(
        renderer,
        locale.text(
            "Mode: observe, require or penalty",
            "模式：observe、require 或 penalty",
            "模式：observe、require 或 penalty",
        ),
        "observe",
    )?
    else {
        return Ok(true);
    };
    if !matches!(mode.as_str(), "observe" | "require" | "penalty") {
        return Ok(true);
    }
    if (mode == "penalty"
        && matches!(
            kind.as_str(),
            "finite" | "batch-alignment" | "custom-validator"
        ))
        || (kind == "custom-penalty" && mode != "penalty")
    {
        println!(
            "{}",
            locale.text(
                "This rule kind does not support the chosen mode.",
                "该规则类型不支持所选模式。",
                "該規則類型不支援所選模式。"
            )
        );
        return Ok(true);
    }
    let mut rule_args: Vec<String> = [
        "model",
        "supervise",
        "rule",
        "add",
        "--mark",
        mark_id,
        "--kind",
        &kind,
        "--mode",
        &mode,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    if kind == "range" {
        let Some(min) = init_input(
            renderer,
            locale.text(
                "Minimum (empty for none)",
                "下界（留空表示无）",
                "下界（留空表示無）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        let Some(max) = init_input(
            renderer,
            locale.text(
                "Maximum (empty for none)",
                "上界（留空表示无）",
                "上界（留空表示無）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if !min.is_empty() {
            rule_args.extend(["--min".into(), min]);
        }
        if !max.is_empty() {
            rule_args.extend(["--max".into(), max]);
        }
    }
    if kind == "batch-alignment" {
        let Some(target) = init_input(
            renderer,
            locale.text(
                "Target batch path (e.g. batch.args[0])",
                "目标批次路径（例如 batch.args[0]）",
                "目標批次路徑（例如 batch.args[0]）",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if target.is_empty() {
            return Ok(true);
        }
        rule_args.extend(["--target-path".into(), target]);
    }
    if matches!(
        kind.as_str(),
        "finite" | "range" | "batch-alignment" | "custom-validator"
    ) {
        if matches!(kind.as_str(), "finite" | "range") {
            println!("{}", locale.text(
                "Specifying an axis requires stable sample IDs for every batch; otherwise this numeric rule is unknown and require/penalty will stop the step. Leave it empty for a tensor-wide rule without per-sample localization.",
                "指定样本轴后，每个批次都需要稳定样本 ID；否则数值规则结果未知，require/penalty 会停止训练步。留空可检查整个张量，但不能定位到样本。",
                "指定樣本軸後，每個批次都需要穩定樣本 ID；否則數值規則結果未知，require/penalty 會停止訓練步。留空可檢查整個張量，但不能定位到樣本。"
            ));
        }
        let default_axis = if kind == "batch-alignment" { "0" } else { "" };
        let Some(axis) = init_input(
            renderer,
            locale.text(
                "Sample axis 0–16 (empty to leave unspecified)",
                "样本轴 0–16（留空表示未指定）",
                "樣本軸 0–16（留空表示未指定）",
            ),
            default_axis,
        )?
        else {
            return Ok(true);
        };
        if !axis.is_empty() {
            if !axis.parse::<u8>().is_ok_and(|v| v <= 16) {
                println!(
                    "{}",
                    locale.text(
                        "Sample axis must be an integer from 0 to 16.",
                        "样本轴必须是 0 到 16 的整数。",
                        "樣本軸必須是 0 到 16 的整數。"
                    )
                );
                return Ok(true);
            }
            rule_args.extend(["--axis".into(), axis]);
        }
    }
    if matches!(kind.as_str(), "custom-validator" | "custom-penalty") {
        println!("{}", locale.text("The plugin is project code and runs in the project environment during preview and training; it is not sandboxed.", "插件是项目代码，预览和训练时在项目环境执行；它不在沙箱中运行。", "插件是專案程式碼，預覽和訓練時在專案環境執行；它不在沙箱中執行。"));
        let Some(plugin) = init_input(
            renderer,
            locale.text(
                "Plugin module:callable",
                "插件 module:callable",
                "外掛 module:callable",
            ),
            "",
        )?
        else {
            return Ok(true);
        };
        if plugin.is_empty() {
            return Ok(true);
        }
        rule_args.extend(["--plugin".into(), plugin]);
        let Some(params) = init_input(
            renderer,
            locale.text(
                "Plugin parameters as JSON object",
                "插件参数（JSON 对象）",
                "外掛參數（JSON 物件）",
            ),
            "{}",
        )?
        else {
            return Ok(true);
        };
        let Ok(serde_json::Value::Object(params)) =
            serde_json::from_str::<serde_json::Value>(&params)
        else {
            println!(
                "{}",
                locale.text(
                    "Parameters must be a JSON object.",
                    "参数必须是 JSON 对象。",
                    "參數必須是 JSON 物件。"
                )
            );
            return Ok(true);
        };
        for (name, value) in params {
            rule_args.extend(["--plugin-param".into(), format!("{name}={value}")]);
        }
    }
    if mode == "penalty" {
        let Some(weight) = init_input(
            renderer,
            locale.text("Penalty weight", "惩罚权重", "懲罰權重"),
            "",
        )?
        else {
            return Ok(true);
        };
        rule_args.extend(["--weight".into(), weight]);
    }
    println!("{}", locale.text("Rules do not affect training unless explicitly enabled; require can reject a step and penalty changes the objective.", "规则只有显式启用才影响训练；require 可拒绝更新，penalty 会改变训练目标。", "規則只有明確啟用才影響訓練；require 可拒絕更新，penalty 會改變訓練目標。"));
    let Some(confirm) = init_input(
        renderer,
        locale.text(
            "Rule affects training only when enabled; require rejects, penalty changes loss. Save? y",
            "规则须在训练中启用；require 拒绝更新，penalty 改变损失。保存？输入 y",
            "規則須在訓練中啟用；require 拒絕更新，penalty 改變損失。儲存？輸入 y",
        ),
        "",
    )?
    else {
        return Ok(true);
    };
    if !confirm.trim().eq_ignore_ascii_case("y") {
        return Ok(true);
    }
    let rule = cli.execute(&rule_args.iter().map(String::as_str).collect::<Vec<_>>());
    if !rule
        .as_ref()
        .is_ok_and(|r| r.operation == "semantic_supervise" && r.status == "success")
    {
        print!("{}", renderer.result(&rule));
        if rule.is_err() || rule.as_ref().is_ok_and(|report| report.status == "success") {
            println!(
                "{}",
                locale.text(
                    "Rule save outcome is unknown; inspect saved rules before retrying.",
                    "规则保存结果未知；重试前请检查已保存规则。",
                    "規則儲存結果未知；重試前請檢查已儲存規則。"
                )
            );
        }
        return Ok(true);
    }
    let mut preview_args = vec!["model", "supervise", "preview"];
    preview_args.extend(context.iter().map(String::as_str));
    let preview = cli.execute(&preview_args);
    match preview {
        Ok(report)
            if report.operation == "semantic_supervise"
                && report.status == "success"
                && report.data["objective_applied"] == false
                && report.data["results"].is_array() =>
        {
            let mut content = String::new();
            renderer.heading(
                &mut content,
                locale.text(
                    "Rule preview · no parameter update",
                    "规则预览 · 未更新参数",
                    "規則預覽 · 未更新參數",
                ),
            );
            renderer.line(
                &mut content,
                &format!("Objective applied: {}", report.data["objective_applied"]),
            );
            for result in report.data["results"].as_array().into_iter().flatten() {
                renderer.line(
                    &mut content,
                    &format!(
                        "{} · {} · {}",
                        text::plain_text(result["kind"].as_str().unwrap_or("?")),
                        text::plain_text(result["mode"].as_str().unwrap_or("?")),
                        text::plain_text(result["status"].as_str().unwrap_or("unknown"))
                    ),
                );
                renderer.line(
                    &mut content,
                    &format!("Violations: {}", result["violation_count"]),
                );
            }
            if let Some(coverage) = report.data["coverage"].as_str() {
                renderer.line(&mut content, &text::plain_text(coverage));
            }
            renderer.line(
                &mut content,
                &format!(
                    "Sample localization: {}",
                    text::plain_text(
                        report.data["sample_localization"]["status"]
                            .as_str()
                            .unwrap_or("unknown")
                    )
                ),
            );
            #[cfg(unix)]
            if !renderer.line_mode && crate::terminal::menu::supported() {
                if !view_report(&content, locale, renderer.style)? {
                    return Ok(false);
                }
            } else {
                print!("{content}");
            }
            #[cfg(not(unix))]
            print!("{content}");
        }
        Ok(report) if report.status == "success" => {
            println!("{}", locale.text("Unexpected supervision preview; rule was saved but results could not be verified.", "监督预览报告不完整；规则已保存，但结果无法确认。", "監督預覽報告不完整；規則已儲存，但結果無法確認。"));
        }
        other => print!("{}", renderer.result(&other)),
    }
    Ok(true)
}

fn configure_assembly(
    cli: &Cli,
    renderer: &Renderer,
    context: Option<&(String, Option<String>)>,
) -> io::Result<InitOutcome> {
    let locale = renderer.locale;
    let command = Command {
        action: Action::Init,
        model: context.map(|c| c.0.clone()).unwrap_or_default(),
        batch: context.and_then(|c| c.1.clone()).unwrap_or_default(),
        variant: String::new(),
        probe_options: ProbeOptions::default(),
    };
    let show = match cli.execute(&["assembly", "show", "--config-only"]) {
        Ok(report) if report.operation == "assembly_show" && report.status == "success" => report,
        Ok(report) if report.status == "error" => {
            return Ok(InitOutcome::Result {
                command,
                result: Ok(report),
                attempted_save: false,
            });
        }
        Ok(_) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unexpected assembly report",
                )),
                attempted_save: false,
            });
        }
        Err(error) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(error),
                attempted_save: false,
            });
        }
    };
    let Some(slots) = show.data["slots"].as_array() else {
        return Ok(InitOutcome::Result {
            command,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "assembly report has no slot list",
            )),
            attempted_save: false,
        });
    };
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Current assembly", "当前装配", "目前裝配"),
    );
    renderer.line(
        &mut content,
        locale.text(
            "Saved bindings (unbound factory slots are not listed):",
            "已保存的绑定（未绑定接缝不会列出）：",
            "已儲存的綁定（未綁定接縫不會列出）：",
        ),
    );
    renderer.line(
        &mut content,
        locale.text(
            "Configuration only; current model has not been loaded or verified.",
            "仅显示配置，尚未加载或验证当前模型。",
            "僅顯示設定，尚未載入或驗證目前模型。",
        ),
    );
    for slot in slots {
        renderer.line(
            &mut content,
            &format!(
                "{}  {}",
                text::plain_text(slot["slot"].as_str().unwrap_or("?")),
                text::plain_text(&slot["binding"].to_string())
            ),
        );
    }
    if let Some(routes) = show.data["model_params"]["input_routes"].as_object() {
        for (slot, input) in routes {
            renderer.line(
                &mut content,
                &format!(
                    "{} ← {}",
                    text::plain_text(slot),
                    text::plain_text(input.as_str().unwrap_or("?"))
                ),
            );
        }
    }
    if let Some(modes) = show.data["model_params"]["encoder_output_modes"].as_object() {
        renderer.line(
            &mut content,
            locale.text(
                "Encoder output overrides (defaults are defined in project source):",
                "编码器输出覆盖（默认值由项目源码定义）：",
                "編碼器輸出覆蓋（預設值由專案原始碼定義）：",
            ),
        );
        for (slot, mode) in modes {
            renderer.line(
                &mut content,
                &format!(
                    "{} · {}",
                    text::plain_text(slot),
                    text::plain_text(mode.as_str().unwrap_or("?"))
                ),
            );
        }
    }
    renderer.line(
        &mut content,
        locale.text(
            "Edits project defaults; existing variants keep their own snapshots.",
            "编辑项目默认装配；已有变体保留各自快照。",
            "編輯專案預設裝配；現有變體保留各自快照。",
        ),
    );
    if let Some(factory) = show.data["model"]
        .as_str()
        .filter(|value| !value.is_empty())
    {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text(
                    "Python model factory",
                    "Python 模型工厂入口",
                    "Python 模型工廠入口"
                ),
                text::plain_text(factory),
            ),
        );
        renderer.line(&mut content, locale.text("For connections or control flow outside the supported slots, open the factory's Python module in your editor; then return to inspect and probe.", "接缝范围之外的连接或控制流，请在编辑器中打开该工厂的 Python 模块修改，然后返回检查与前向。", "接縫範圍之外的連接或控制流程，請在編輯器中開啟該工廠的 Python 模組修改，然後返回檢查與前向。"));
    }
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        if !view_report(&content, locale, renderer.style)? {
            return Ok(InitOutcome::Quit);
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    print!("{content}");
    let Some(action) = init_input(
        renderer,
        locale.text(
            "Action: add, set, remove, patch, route, encoder-output (+snapshot optional), batch or generate",
            "操作：add、set、remove、patch、route、encoder-output（可加 +snapshot）、batch 或 generate",
            "操作：add、set、remove、patch、route、encoder-output（可加 +snapshot）、batch 或 generate",
        ),
        "add",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    if action == "patch" || action == "patch+snapshot" {
        let capture_snapshot = action == "patch+snapshot";
        let mut required = vec!["assembly.atomic_patch", "assembly.atomic_patch_evidence"];
        if capture_snapshot {
            required.push("assembly.atomic_patch_snapshot");
        }
        if let Some(message) = capability_unavailable(cli, locale, &required) {
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
            return Ok(InitOutcome::Cancelled);
        }
        let Some(sets) = init_input(
            renderer,
            locale.text(
                "Bindings separated by semicolons; use @output.slot=mode and @route.slot=input",
                "绑定以分号分隔；用 @output.接缝=方式、@route.接缝=输入",
                "繫結以分號分隔；用 @output.接縫=方式、@route.接縫=輸入",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let Some(removes) = init_input(
            renderer,
            locale.text(
                "Seams to remove, separated by commas (optional)",
                "要移除的接缝，以逗号分隔（可选）",
                "要移除的接縫，以逗號分隔（可選）",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let Some(raw_params) = init_input(
            renderer,
            locale.text(
                "Registered component parameters as JSON object, e.g. {\"slot.width\":4}",
                "已登记组件参数 JSON 对象，如 {\"slot.width\":4}",
                "已登記元件參數 JSON 物件，如 {\"slot.width\":4}",
            ),
            "{}",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let params = match parse_factory_parameters(&raw_params) {
            Ok(params) => params,
            Err(error) => {
                show_flow_page(
                    renderer,
                    &format!("Invalid parameters: {}\n", text::plain_text(&error)),
                )?;
                return Ok(InitOutcome::Cancelled);
            }
        };
        let mut args = vec!["assembly".to_owned(), "patch".to_owned()];
        let mut encoder_outputs = Vec::new();
        let mut route_changes = Vec::new();
        for value in sets
            .split(';')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if let Some(mode) = value.strip_prefix("@output.") {
                encoder_outputs.push(mode.to_owned());
                args.extend(["--encoder-output".into(), mode.into()]);
            } else if let Some(route) = value.strip_prefix("@route.") {
                route_changes.push(route.to_owned());
                args.extend(["--route".into(), route.into()]);
            } else {
                args.extend(["--set".into(), value.into()]);
            }
        }
        if !encoder_outputs.is_empty() {
            if let Some(message) =
                capability_unavailable(cli, locale, &["assembly.atomic_patch_encoder_output"])
            {
                if !show_flow_page(renderer, &message)? {
                    return Ok(InitOutcome::Quit);
                }
                return Ok(InitOutcome::Cancelled);
            }
        }
        if !route_changes.is_empty() {
            if let Some(message) =
                capability_unavailable(cli, locale, &["assembly.atomic_patch_route"])
            {
                if !show_flow_page(renderer, &message)? {
                    return Ok(InitOutcome::Quit);
                }
                return Ok(InitOutcome::Cancelled);
            }
        }
        for value in removes
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            args.extend(["--remove".into(), value.into()]);
        }
        for (name, value) in params {
            args.extend(["--param".into(), format!("{name}={value}")]);
        }
        let temporary = if capture_snapshot {
            let directory = tempfile::Builder::new()
                .prefix("model-maven-assembly-")
                .tempdir()?;
            let path = directory.path().join("input.snapshot");
            let Some(path) = path.to_str() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "temporary snapshot path is not UTF-8",
                ));
            };
            let mut notice = String::new();
            renderer.heading(
                &mut notice,
                locale.text("Capturing input snapshot", "捕获输入快照", "擷取輸入快照"),
            );
            renderer.line(
                &mut notice,
                &format!(
                    "{}: {}",
                    locale.text("Temporary file", "临时文件", "暫存檔案"),
                    text::plain_text(path)
                ),
            );
            renderer.line(&mut notice, locale.text("The private file contains input values, not targets. Cleanup is attempted when this flow ends; abnormal termination may leave it behind.", "私有文件包含输入值，不包含目标；流程结束时尝试清理，异常退出可能留下文件。", "私有檔案包含輸入值，不包含目標；流程結束時嘗試清理，異常結束可能留下檔案。"));
            if !show_flow_page(renderer, &notice)? {
                return Ok(InitOutcome::Quit);
            }
            Some(directory)
        } else {
            None
        };
        let snapshot_path = temporary.as_ref().map(|directory| {
            directory
                .path()
                .join("input.snapshot")
                .to_string_lossy()
                .into_owned()
        });
        let mut preview_args = args.clone();
        if let Some(path) = &snapshot_path {
            preview_args.extend(["--capture-input-to".into(), path.clone()]);
        }
        let preview = match cli
            .execute(&preview_args.iter().map(String::as_str).collect::<Vec<_>>())
        {
            Ok(report)
                if report.operation == "assembly_patch"
                    && report.status == "success"
                    && report.data["committed"] == false
                    && (encoder_outputs.is_empty()
                        || (report.data["encoder_outputs"]
                            == serde_json::json!(encoder_outputs)
                            && encoder_outputs.iter().all(|entry| {
                                entry.split_once('=').is_some_and(|(slot, mode)| {
                                    report.data["encoder_output_modes"][slot] == mode
                                })
                            })))
                    && (route_changes.is_empty()
                        || (report.data["route_changes"] == serde_json::json!(route_changes)
                            && route_changes.iter().all(|entry| {
                                entry.split_once('=').is_some_and(|(slot, input)| {
                                    report.data["input_routes"][slot] == input
                                })
                            })))
                    && report.data["validation"]["forward"] == "passed" =>
            {
                report
            }
            Ok(report) if report.status == "error" => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Ok(report),
                    attempted_save: false,
                });
            }
            other => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Batch assembly preview unavailable: {}; nothing saved",
                            renderer.result(&other)
                        ),
                    )),
                    attempted_save: false,
                });
            }
        };
        let Some(revision) = preview.data["base_config_revision"]
            .as_str()
            .filter(|value| {
                value.len() == 64
                    && value
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
        else {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Assembly preview has no valid config revision",
                )),
                attempted_save: false,
            });
        };
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text(
                "Batch assembly preview · not saved",
                "批量装配预览 · 尚未保存",
                "批次裝配預覽 · 尚未儲存",
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Bindings", "绑定", "繫結"),
                text::plain_text(&preview.data["model_injections"].to_string())
            ),
        );
        if !encoder_outputs.is_empty() {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text(
                        "Encoder output overrides",
                        "编码器输出覆盖",
                        "編碼器輸出覆蓋"
                    ),
                    text::plain_text(&preview.data["encoder_output_modes"].to_string())
                ),
            );
        }
        if !route_changes.is_empty() {
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    locale.text("Input route overrides", "输入接线覆盖", "輸入接線覆蓋"),
                    text::plain_text(&preview.data["input_routes"].to_string())
                ),
            );
        }
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Validation", "验证", "驗證"),
                text::plain_text(&preview.data["validation"].to_string())
            ),
        );
        let valid_digest = |value: &Value| {
            value
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
                .map(str::to_owned)
        };
        let source_digest = valid_digest(&preview.data["source_evidence"]["digest"]);
        let input_digest = if preview.data["input_evidence"]["status"] == "available"
            && preview.data["input_evidence"]["algorithm"] == "sha256-forward-input-v1"
        {
            valid_digest(&preview.data["input_evidence"]["digest"])
        } else {
            None
        };
        if capture_snapshot
            && (input_digest.is_none() || preview.data["input_snapshot"]["status"] != "captured")
        {
            show_flow_page(renderer, locale.text("Input snapshot could not be captured and verified; no assembly changes saved.\n", "无法捕获并校验输入快照；装配未保存。\n", "無法擷取並驗證輸入快照；裝配未儲存。\n"))?;
            return Ok(InitOutcome::Cancelled);
        }
        renderer.line(&mut content, locale.text("One full forward passed. Commit checks configuration and, when available, observed project sources; external libraries, runtime, targets and changes after the checks are not pinned.", "完整前向已通过；提交核对配置及可用的已观察项目源码，外部库、运行环境、目标和检查后的变化未固定。", "完整前向已通過；提交核對設定及可用的已觀察專案原始碼，外部函式庫、執行環境、目標和檢查後的變化未固定。"));
        if source_digest.is_none() {
            renderer.line(
                &mut content,
                locale.text(
                    "Project source evidence unavailable; it will not be pinned.",
                    "项目源码证据不可用；源码不会被固定。",
                    "專案原始碼證據不可用；原始碼不會被固定。",
                ),
            );
        }
        if input_digest.is_none() {
            renderer.line(
                &mut content,
                locale.text(
                    "Input content evidence unavailable; input cannot be pinned.",
                    "输入内容证据不可用，无法固定输入。",
                    "輸入內容證據不可用，無法固定輸入。",
                ),
            );
        }
        if capture_snapshot {
            renderer.line(&mut content, locale.text("Commit replays the captured input; targets and future batches are not included.", "提交将重放捕获的输入；不包含目标或后续批次。", "提交將重播擷取的輸入；不包含目標或後續批次。"));
        }
        if !show_flow_page(renderer, &content)? {
            return Ok(InitOutcome::Quit);
        }
        let bind_input = if capture_snapshot {
            true
        } else if input_digest.is_some() {
            let Some(choice) = init_input(
                renderer,
                locale.text(
                    "Require the same observed input on commit? (y/N)",
                    "提交时要求相同的已观察输入？（y/N）",
                    "提交時要求相同的已觀察輸入？（y/N）",
                ),
                "n",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            if !choice.eq_ignore_ascii_case("y") && !choice.eq_ignore_ascii_case("n") {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Choose y or n; no assembly changes saved.\n",
                        "请选择 y 或 n；装配未保存。\n",
                        "請選擇 y 或 n；裝配未儲存。\n",
                    ),
                )?;
                return Ok(InitOutcome::Cancelled);
            }
            choice.eq_ignore_ascii_case("y")
        } else {
            false
        };
        if !init_input(
            renderer,
            locale.text(
                "Save all assembly changes? Type y",
                "保存全部装配更改？输入 y",
                "儲存全部裝配變更？輸入 y",
            ),
            "",
        )?
        .is_some_and(|value| value.eq_ignore_ascii_case("y"))
        {
            return Ok(InitOutcome::Cancelled);
        }
        args.extend([
            "--apply".into(),
            "--if-config-revision".into(),
            revision.into(),
        ]);
        if let Some(digest) = source_digest {
            args.extend(["--if-source-revision".into(), digest.into()]);
        }
        if bind_input {
            args.extend(["--if-input-sha256".into(), input_digest.unwrap().into()]);
        }
        if let Some(path) = snapshot_path {
            args.extend(["--replay-input-from".into(), path]);
        }
        let result = cli
            .execute(&args.iter().map(String::as_str).collect::<Vec<_>>())
            .and_then(|report| {
                if report.status == "error"
                    || (report.operation == "assembly_patch"
                        && report.status == "success"
                        && report.data["committed"] == true
                        && (encoder_outputs.is_empty()
                            || (report.data["encoder_outputs"]
                                == serde_json::json!(encoder_outputs)
                                && encoder_outputs.iter().all(|entry| {
                                    entry.split_once('=').is_some_and(|(slot, mode)| {
                                        report.data["encoder_output_modes"][slot] == mode
                                    })
                                })))
                        && (route_changes.is_empty()
                            || (report.data["route_changes"] == serde_json::json!(route_changes)
                                && route_changes.iter().all(|entry| {
                                    entry.split_once('=').is_some_and(|(slot, input)| {
                                        report.data["input_routes"][slot] == input
                                    })
                                })))
                        && report.data["config_revision"].as_str().is_some())
                {
                    Ok(report)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unexpected batch assembly receipt; inspect project before retrying",
                    ))
                }
            });
        if let Some(directory) = temporary {
            let message = match directory.close() {
                Ok(()) => locale
                    .text(
                        "Temporary input snapshot removed.\n",
                        "临时输入快照已清理。\n",
                        "暫存輸入快照已清理。\n",
                    )
                    .to_owned(),
                Err(error) => format!(
                    "{}: {}\n",
                    locale.text(
                        "Could not remove temporary input snapshot",
                        "无法清理临时输入快照",
                        "無法清理暫存輸入快照"
                    ),
                    text::plain_text(&error.to_string())
                ),
            };
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
        }
        return Ok(InitOutcome::Result {
            command,
            result,
            attempted_save: true,
        });
    }
    if matches!(
        action.as_str(),
        "route" | "encoder-output" | "route+snapshot" | "encoder-output+snapshot"
    ) {
        let capture_snapshot = action.ends_with("+snapshot");
        let output_mode = action.starts_with("encoder-output");
        if capture_snapshot {
            if let Some(message) =
                capability_unavailable(cli, locale, &["project.scaffold_setting_snapshot"])
            {
                if !show_flow_page(renderer, &message)? {
                    return Ok(InitOutcome::Quit);
                }
                return Ok(InitOutcome::Cancelled);
            }
        }
        if output_mode {
            if let Some(message) = capability_unavailable(
                cli,
                locale,
                &[
                    "project.scaffold_encoder_output_edit",
                    "project.scaffold_encoder_output_evidence",
                ],
            ) {
                if !show_flow_page(renderer, &message)? {
                    return Ok(InitOutcome::Quit);
                }
                return Ok(InitOutcome::Cancelled);
            }
            println!("{}", locale.text("Requires a starter that accepts encoder_output_modes. Older starters need a manual source update. Each preview executes the project model and batch.", "需要支持 encoder_output_modes 的新骨架；旧骨架需要手动更新源码。每次预览都会执行项目模型和批次。", "需要支援 encoder_output_modes 的新骨架；舊骨架需要手動更新原始碼。每次預覽都會執行專案模型和批次。"));
        } else if let Some(message) =
            capability_unavailable(cli, locale, &["project.scaffold_route_evidence"])
        {
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
            return Ok(InitOutcome::Cancelled);
        }
        let (verb, operation, field, collection) = if output_mode {
            (
                "encoder-output",
                "assembly_encoder_output",
                "mode",
                "encoder_output_modes",
            )
        } else {
            ("route", "assembly_route", "input", "routes")
        };
        let Some(slot) = init_input(
            renderer,
            locale.text(
                "Encoder seam, e.g. image_encoder",
                "编码器接缝，例如 image_encoder",
                "編碼器接縫，例如 image_encoder",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let Some(input) = init_input_value(
            renderer,
            if output_mode {
                locale.text(
                    "Output mode: native or flatten (:back cancels)",
                    "输出方式：native 或 flatten（:back 取消）",
                    "輸出方式：native 或 flatten（:back 取消）",
                )
            } else {
                locale.text(
                    "Input name (use :back to cancel)",
                    "输入名（:back 取消）",
                    "輸入名（:back 取消）",
                )
            },
            "",
            true,
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let temporary = if capture_snapshot {
            let directory = tempfile::Builder::new()
                .prefix("model-maven-setting-")
                .tempdir()?;
            let path = directory.path().join("input.snapshot");
            let mut notice = String::new();
            renderer.heading(
                &mut notice,
                locale.text("Capturing input snapshot", "捕获输入快照", "擷取輸入快照"),
            );
            renderer.line(
                &mut notice,
                &format!(
                    "{}: {}",
                    locale.text("Temporary file", "临时文件", "暫存檔案"),
                    text::plain_text(&path.to_string_lossy())
                ),
            );
            renderer.line(&mut notice, locale.text("The private file contains input values, not targets. Cleanup is attempted when this flow ends; abnormal termination may leave it behind.", "私有文件包含输入值，不包含目标；流程结束时尝试清理，异常退出可能留下文件。", "私有檔案包含輸入值，不包含目標；流程結束時嘗試清理，異常結束可能留下檔案。"));
            if !show_flow_page(renderer, &notice)? {
                return Ok(InitOutcome::Quit);
            }
            Some(directory)
        } else {
            None
        };
        let snapshot_path = temporary
            .as_ref()
            .map(|directory| {
                directory
                    .path()
                    .join("input.snapshot")
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "temporary snapshot path is not UTF-8",
                        )
                    })
            })
            .transpose()?;
        let outcome = (|| -> io::Result<InitOutcome> {
            let mut preview_args = vec!["assembly", verb, &slot, &input, "--dry-run"];
            if let Some(path) = snapshot_path.as_deref() {
                preview_args.extend(["--capture-input-to", path]);
            }
            let preview = match cli.execute(&preview_args) {
                Ok(report)
                    if report.operation == operation
                        && report.status == "success"
                        && report.data["committed"] == false
                        && report.data["slot"] == slot
                        && report.data[field] == input
                        && report.data["validation"]["forward"] == "passed" =>
                {
                    report
                }
                Ok(report) if report.status == "error" => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Ok(report),
                        attempted_save: false,
                    });
                }
                Ok(_) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unexpected route preview",
                        )),
                        attempted_save: false,
                    });
                }
                Err(error) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(error),
                        attempted_save: false,
                    });
                }
            };
            let Some(revision) = preview.data["base_config_revision"].as_str().filter(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }) else {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "route preview lacks configuration revision",
                    )),
                    attempted_save: false,
                });
            };
            let mut content = String::new();
            let valid_digest = |value: &Value| {
                value
                    .as_str()
                    .filter(|s| {
                        s.len() == 64
                            && s.bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    })
                    .map(str::to_owned)
            };
            let source_digest = if preview.data["source_evidence"]["algorithm"]
                == "canonical-json-sha256-v1"
                && preview.data["source_evidence"]["scope"] == "worker-recorded-project-sources"
            {
                valid_digest(&preview.data["source_evidence"]["digest"])
            } else {
                None
            };
            let input_digest = if preview.data["input_evidence"]["status"] == "available"
                && preview.data["input_evidence"]["algorithm"] == "sha256-forward-input-v1"
            {
                valid_digest(&preview.data["input_evidence"]["digest"])
            } else {
                None
            };
            if capture_snapshot
                && (input_digest.is_none()
                    || preview.data["input_snapshot"]["status"] != "captured"
                    || preview.data["input_snapshot"]["format"]
                        != "model-maven-forward-input-snapshot-v1")
            {
                show_flow_page(
                    renderer,
                    locale.text(
                        "Input snapshot could not be captured and verified; no changes saved.\n",
                        "无法捕获并校验输入快照；未保存更改。\n",
                        "無法擷取並驗證輸入快照；未儲存變更。\n",
                    ),
                )?;
                return Ok(InitOutcome::Cancelled);
            }
            renderer.heading(
                &mut content,
                if output_mode {
                    locale.text(
                        "Encoder output preview · not saved",
                        "编码器输出预览 · 尚未保存",
                        "編碼器輸出預覽 · 尚未儲存",
                    )
                } else {
                    locale.text(
                        "Input route preview · not saved",
                        "输入接线预览 · 尚未保存",
                        "輸入接線預覽 · 尚未儲存",
                    )
                },
            );
            renderer.line(
                &mut content,
                &format!("{} ← {}", text::plain_text(&slot), text::plain_text(&input)),
            );
            renderer.line(
                &mut content,
                &format!(
                    "{}: {}",
                    if output_mode {
                        locale.text(
                            "Encoder output overrides",
                            "编码器输出覆盖",
                            "編碼器輸出覆蓋",
                        )
                    } else {
                        locale.text("Saved route overrides", "接线覆盖", "接線覆蓋")
                    },
                    text::plain_text(&preview.data[collection].to_string())
                ),
            );
            renderer.line(&mut content, locale.text("Representative forward passed; configuration and available observed project sources will be checked. External code, runtime and targets are not pinned.", "代表批次前向通过；提交时核对配置及可用的已观察项目源码。外部代码、运行环境和目标未固定。", "代表批次前向通過；提交時核對設定及可用的已觀察專案原始碼。外部程式碼、執行環境和目標未固定。"));
            if source_digest.is_none() {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Project source evidence unavailable; sources cannot be pinned.",
                        "项目源码证据不可用；无法固定源码。",
                        "專案原始碼證據不可用；無法固定原始碼。",
                    ),
                );
            }
            if input_digest.is_none() {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Input content evidence unavailable; input cannot be pinned.",
                        "输入内容证据不可用；无法固定输入。",
                        "輸入內容證據不可用；無法固定輸入。",
                    ),
                );
            }
            if capture_snapshot {
                renderer.line(&mut content, locale.text("Commit replays the captured input; targets and future batches are not included.", "提交将重放捕获的输入；不包含目标或后续批次。", "提交將重播擷取的輸入；不包含目標或後續批次。"));
            }
            if !show_flow_page(renderer, &content)? {
                return Ok(InitOutcome::Quit);
            }
            let bind_input = if capture_snapshot {
                true
            } else if input_digest.is_some() {
                let Some(choice) = init_input(
                    renderer,
                    locale.text(
                        "Require the same input content at save? y/n",
                        "保存时要求输入内容一致？y/n",
                        "儲存時要求輸入內容一致？y/n",
                    ),
                    "n",
                )?
                else {
                    return Ok(InitOutcome::Cancelled);
                };
                if !matches!(choice.as_str(), "y" | "Y" | "n" | "N") {
                    return Ok(InitOutcome::Cancelled);
                }
                choice.eq_ignore_ascii_case("y")
            } else {
                false
            };
            if !init_input(
                renderer,
                if output_mode {
                    locale.text(
                        "Save encoder output mode? Type y",
                        "保存编码器输出方式？输入 y",
                        "儲存編碼器輸出方式？輸入 y",
                    )
                } else {
                    locale.text("Save route? Type y", "保存接线？输入 y", "儲存接線？輸入 y")
                },
                "",
            )?
            .is_some_and(|value| value.eq_ignore_ascii_case("y"))
            {
                return Ok(InitOutcome::Cancelled);
            }
            let mut commit_args = vec![
                "assembly",
                verb,
                &slot,
                &input,
                "--if-config-revision",
                revision,
            ];
            if let Some(digest) = source_digest.as_deref() {
                commit_args.extend(["--if-source-revision", digest]);
            }
            if bind_input {
                commit_args.extend([
                    "--if-input-sha256",
                    input_digest.as_deref().expect("checked"),
                ]);
            }
            if let Some(path) = snapshot_path.as_deref() {
                commit_args.extend(["--replay-input-from", path]);
            }
            let result = cli.execute(&commit_args).and_then(|report| {
                if report.status == "error"
                    || (report.operation == operation
                        && report.status == "success"
                        && report.data["committed"] == true
                        && report.data["slot"] == slot
                        && report.data[field] == input
                        && report.data["validation"]["forward"] == "passed"
                        && (!capture_snapshot
                            || (report.data["input_snapshot"]["status"] == "replayed"
                                && report.data["input_evidence"]["digest"].as_str()
                                    == input_digest.as_deref()))
                        && report.data["config_revision"].as_str().is_some())
                {
                    Ok(report)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "route save outcome unknown; inspect project before retrying",
                    ))
                }
            });
            Ok(InitOutcome::Result {
                command,
                result,
                attempted_save: true,
            })
        })();
        if let Some(directory) = temporary {
            let message = match directory.close() {
                Ok(()) => locale
                    .text(
                        "Temporary input snapshot removed.\n",
                        "临时输入快照已清理。\n",
                        "暫存輸入快照已清理。\n",
                    )
                    .to_owned(),
                Err(error) => format!(
                    "{}: {}\n",
                    locale.text(
                        "Could not remove temporary input snapshot",
                        "无法清理临时输入快照",
                        "無法清理暫存輸入快照"
                    ),
                    text::plain_text(&error.to_string())
                ),
            };
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
        }
        return outcome;
    }
    if action == "batch" {
        if let Some(message) =
            capability_unavailable(cli, locale, &["project.conditional_batch_edit"])
        {
            return Ok(if show_flow_page(renderer, &message)? {
                InitOutcome::Cancelled
            } else {
                InitOutcome::Quit
            });
        }
        let Some(batch) = init_input(
            renderer,
            locale.text(
                "New batch factory module:callable, or :csv to create a numeric CSV adapter",
                "新的批次工厂 module:callable，或 :csv 创建数值 CSV 适配器",
                "新的批次工廠 module:callable，或 :csv 建立數值 CSV 適配器",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let (batch, binding, raw_params) = if batch == ":csv" {
            let Some(path) = init_input(
                renderer,
                locale.text(
                    "Numeric CSV file path",
                    "数值 CSV 文件路径",
                    "數值 CSV 檔案路徑",
                ),
                "",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            let Some(inputs) = init_input(
                renderer,
                locale.text(
                    "Input JSON: e.g. {\"features\":[\"x1\",\"x2\"]}; typed fields have columns/shape/dtype",
                    "输入列 JSON，如 {\"features\":[\"x1\",\"x2\"]}；类型字段含 columns/shape/dtype",
                    "輸入欄位 JSON，如 {\"features\":[\"x1\",\"x2\"]}；型別欄位含 columns/shape/dtype",
                ),
                "",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            let Some(targets) = init_input(
                renderer,
                locale.text(
                    "Target JSON: list, named output object, or single target wrapped in \"$tensor\"",
                    "目标列 JSON 数组、具名输出对象，或以 \"$tensor\" 包装单个目标",
                    "目標欄位 JSON 陣列、具名輸出物件，或以 \"$tensor\" 包裝單個目標",
                ),
                "",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            let Some(size) = init_input(
                renderer,
                locale.text("Batch size (1-256)", "批大小（1-256）", "批次大小（1-256）"),
                "2",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            let parsed = (
                serde_json::from_str::<Value>(&inputs),
                serde_json::from_str::<Value>(&targets),
                size.parse::<u16>(),
            );
            let (inputs, targets, size) = match parsed {
                (Ok(inputs), Ok(targets), Ok(size))
                    if inputs.is_object()
                        && (targets.is_array() || targets.is_object())
                        && (1..=256).contains(&size)
                        && !path.is_empty() =>
                {
                    (inputs, targets, size)
                }
                _ => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "CSV path, input/target JSON, or batch size is invalid",
                        )),
                        attempted_save: false,
                    });
                }
            };
            let typed = inputs
                .as_object()
                .is_some_and(|items| items.values().any(Value::is_object))
                || targets.as_object().is_some_and(|items| {
                    items.contains_key("$tensor") || items.values().any(Value::is_object)
                });
            let mut required = vec!["data.csv_batch_template"];
            if typed {
                required.push("data.csv_batch_typed");
            }
            if let Some(message) = capability_unavailable(cli, locale, &required) {
                if !show_flow_page(renderer, &message)? {
                    return Ok(InitOutcome::Quit);
                }
                return Ok(InitOutcome::Cancelled);
            }
            let raw_params = serde_json::json!({"path":path,"inputs":inputs,"targets":targets,"batch_size":size}).to_string();
            let preview = match cli.execute(&["data", "csv-template"]) {
                Ok(report)
                    if report.operation == "data_csv_template"
                        && report.status == "success"
                        && report.data["committed"] == false =>
                {
                    report
                }
                Ok(report) if report.status != "success" => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Ok(report),
                        attempted_save: false,
                    });
                }
                Ok(_) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Unexpected CSV template preview; nothing created",
                        )),
                        attempted_save: false,
                    });
                }
                Err(error) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(error),
                        attempted_save: false,
                    });
                }
            };
            let revision = preview.data["plan_revision"]
                .as_str()
                .filter(|value| value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()));
            let plan = &preview.data["plan"];
            let Some(revision) = revision.filter(|_| {
                plan["project_root"].as_str().map(Path::new) == Some(cli.project())
                    && plan["file"] == "model_csv.py"
                    && plan["entrypoint"] == "model_csv:batch"
                    && plan["source"].as_str().is_some()
            }) else {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid CSV template preview; nothing created",
                    )),
                    attempted_save: false,
                });
            };
            let mut content = String::new();
            renderer.heading(
                &mut content,
                locale.text("CSV adapter preview", "CSV 适配器预览", "CSV 適配器預覽"),
            );
            renderer.line(&mut content, locale.text("Create model_csv.py; typed fixed-shape columns, rows cycle in file order. The file is editable; data remains in place.", "创建 model_csv.py；读取具类型的固定形状列，按文件顺序循环。文件可编辑，数据留在原位置。", "建立 model_csv.py；讀取具型別的固定形狀欄位，按檔案順序循環。檔案可編輯，資料留在原位置。"));
            renderer.line(
                &mut content,
                &format!("Parameters: {}", text::plain_text(&raw_params)),
            );
            renderer.line(&mut content, locale.text("Before using the adapter, verify the selected split and target columns. This template does not shuffle or split data.", "使用前请确认所选数据切分和目标列；此模板不会打乱或切分数据。", "使用前請確認所選資料切分和目標欄位；此範本不會打亂或切分資料。"));
            for line in plan["source"].as_str().unwrap_or("").lines() {
                renderer.line(&mut content, &format!("│{}", text::plain_text(line)));
            }
            if !show_flow_page(renderer, &content)? {
                return Ok(InitOutcome::Quit);
            }
            if !init_input(
                renderer,
                locale.text(
                    "Create CSV adapter? Type y",
                    "创建 CSV 适配器？输入 y",
                    "建立 CSV 適配器？輸入 y",
                ),
                "",
            )?
            .is_some_and(|value| value.eq_ignore_ascii_case("y"))
            {
                return Ok(InitOutcome::Cancelled);
            }
            let created = cli.execute(&["data", "csv-template", "--apply", "--if-plan", revision]);
            match created {
                Ok(report)
                    if report.operation == "data_csv_template"
                        && report.status == "success"
                        && report.data["committed"] == true
                        && report.data["plan_revision"] == revision
                        && report.data["entrypoint"] == "model_csv:batch" =>
                {
                    ()
                }
                Ok(report) if report.status != "success" => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Ok(report),
                        attempted_save: true,
                    });
                }
                Ok(_) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "CSV template creation outcome unknown; inspect model_csv.py before retrying",
                        )),
                        attempted_save: true,
                    });
                }
                Err(error) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            error.kind(),
                            format!("{error}; inspect model_csv.py before retrying"),
                        )),
                        attempted_save: true,
                    });
                }
            }
            (
                "model_csv:batch".to_owned(),
                "envelope".to_owned(),
                raw_params,
            )
        } else {
            let Some(binding) = init_input(
                renderer,
                locale.text(
                    "Batch binding: envelope, single, args or kwargs",
                    "批次绑定：envelope、single、args 或 kwargs",
                    "批次繫結：envelope、single、args 或 kwargs",
                ),
                "envelope",
            )?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            if !matches!(binding.as_str(), "envelope" | "single" | "args" | "kwargs") {
                return Ok(InitOutcome::Cancelled);
            }
            let Some(raw_params) = init_input(
                renderer,
                locale.text("Batch factory parameters as JSON object (saved locally; do not enter credentials)", "批次工厂参数（JSON 对象，保存在本地；请勿填写凭据）", "批次工廠參數（JSON 物件，儲存在本機；請勿填寫憑據）"),
                "{}",
            )? else { return Ok(InitOutcome::Cancelled); };
            (batch, binding, raw_params)
        };
        let params = match parse_factory_parameters(&raw_params) {
            Ok(params) => params,
            Err(error) => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
                    attempted_save: false,
                });
            }
        };
        let param_values: Vec<String> = params
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        let mut args = vec!["config", "set-batch", "--batch", batch.as_str()];
        args.extend(["--batch-binding", binding.as_str()]);
        for value in &param_values {
            args.extend(["--batch-param", value]);
        }
        let preview = match cli.execute(&args) {
            Ok(report)
                if report.operation == "config_set_batch"
                    && report.status == "success"
                    && report.data["committed"] == false
                    && report.data["batch"].as_str() == Some(batch.as_str())
                    && report.data["batch_binding"].as_str() == Some(binding.as_str()) =>
            {
                report
            }
            Ok(report) if report.status != "success" => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Ok(report),
                    attempted_save: false,
                });
            }
            Err(error) => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(error),
                    attempted_save: false,
                });
            }
            _ => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid batch preview receipt; nothing saved",
                    )),
                    attempted_save: false,
                });
            }
        };
        let Some(revision) = preview.data["from_revision"]
            .as_str()
            .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Batch preview revision unavailable",
                )),
                attempted_save: false,
            });
        };
        let mut probe_args = vec!["model", "probe", "--batch", batch.as_str()];
        probe_args.extend(["--batch-binding", binding.as_str()]);
        for value in &param_values {
            probe_args.extend(["--batch-param", value]);
        }
        let probe = cli.execute(&probe_args);
        match probe {
            Ok(report) if report.operation == "probe" && report.status == "success" => {}
            Ok(report) if report.status != "success" => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Ok(report),
                    attempted_save: false,
                });
            }
            Err(error) => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(error),
                    attempted_save: false,
                });
            }
            _ => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid forward receipt; nothing saved",
                    )),
                    attempted_save: false,
                });
            }
        }
        let current = cli.execute(&["assembly", "show", "--config-only"]);
        if !current.as_ref().is_ok_and(|report| {
            report.operation == "assembly_show"
                && report.status == "success"
                && report.data["config_revision"].as_str() == Some(revision)
        }) {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Configuration changed during batch verification; nothing saved",
                )),
                attempted_save: false,
            });
        }
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text("Batch change preview", "批次更改预览", "批次變更預覽"),
        );
        renderer.line(
            &mut content,
            &format!(
                "Previous: {} · {}",
                text::plain_text(&preview.data["previous_batch"].to_string()),
                text::plain_text(&preview.data["previous_batch_binding"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "New: {} · {}",
                text::plain_text(&batch),
                text::plain_text(&binding)
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Parameters: {}",
                text::plain_text(&preview.data["batch_params"].to_string())
            ),
        );
        renderer.line(&mut content, locale.text("One real forward passed; targets and training loss were not verified. Run a trial. Parameters are stored locally; do not save credentials. Data, source and runtime are not pinned.", "一次真实前向已通过；目标和训练损失尚未验证，请执行试跑。参数将保存在本地，请勿保存凭据；数据、源码和环境未固定。", "一次實際前向已通過；目標和訓練損失尚未驗證，請執行試跑。參數將儲存在本機，請勿儲存憑據；資料、原始碼和環境未固定。"));
        if !show_flow_page(renderer, &content)? {
            return Ok(InitOutcome::Quit);
        }
        if !init_input(
            renderer,
            locale.text("Save batch? Type y", "保存批次？输入 y", "儲存批次？輸入 y"),
            "",
        )?
        .is_some_and(|v| v.eq_ignore_ascii_case("y"))
        {
            return Ok(InitOutcome::Cancelled);
        }
        args.extend(["--apply", "--if-current", revision]);
        let result = cli.execute(&args).and_then(|report| {
            if report.status == "error"
                || (report.operation == "config_set_batch"
                    && report.status == "success"
                    && (report.data["committed"] == true || report.data["unchanged"] == true)
                    && report.data["batch"].as_str() == Some(batch.as_str()))
            {
                Ok(report)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Batch save outcome unknown; inspect project configuration before retrying",
                ))
            }
        });
        return Ok(InitOutcome::Result {
            command,
            result,
            attempted_save: true,
        });
    }
    if action == "generate" {
        if let Some(message) =
            capability_unavailable(cli, locale, &["assembly.generated_source_verify"])
        {
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
            return Ok(InitOutcome::Cancelled);
        }
        let Some(output) = init_input(
            renderer,
            locale.text(
                "New project-relative Python file",
                "项目内新 Python 文件路径",
                "專案內新 Python 檔案路徑",
            ),
            "generated/model_assembly.py",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let Some(revision) = show.data["config_revision"]
            .as_str()
            .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Assembly configuration revision unavailable; nothing generated",
                )),
                attempted_save: false,
            });
        };
        let mut plan = String::new();
        renderer.heading(
            &mut plan,
            locale.text(
                "Generate assembly adapter",
                "生成装配适配器",
                "產生裝配適配器",
            ),
        );
        renderer.line(&mut plan, &format!("Output: {}", text::plain_text(&output)));
        renderer.line(&mut plan, locale.text("Reconstructs the current model, then imports the generated code in another worker and runs a forward if a batch is configured. This executes project factories and the batch again. No trained weights are included; project modules, dependencies and data remain required. Existing files are not overwritten.", "先重建当前模型，再在另一 worker 导入生成代码；有批次时执行前向。这会再次运行项目工厂和批次。生成物不包含训练权重，仍依赖项目模块、环境和数据；不会覆盖已有文件。", "先重建目前模型，再於另一 worker 匯入產生程式碼；有批次時執行前向。這會再次執行專案工廠與批次。產物不含訓練權重，仍依賴專案模組、環境及資料；不會覆蓋現有檔案。"));
        if !show_flow_page(renderer, &plan)? {
            return Ok(InitOutcome::Quit);
        }
        if !init_input(
            renderer,
            locale.text(
                "Generate file? Type y",
                "生成文件？输入 y",
                "產生檔案？輸入 y",
            ),
            "",
        )?
        .is_some_and(|v| v.eq_ignore_ascii_case("y"))
        {
            return Ok(InitOutcome::Cancelled);
        }
        let result = cli
            .execute(&[
                "assembly",
                "generate",
                "--output",
                &output,
                "--if-config-revision",
                revision,
            ])
            .and_then(|report| {
                if report.status == "error"
                    || (report.operation == "assembly_generate"
                        && report.status == "success"
                        && report.data["output"].as_str() == Some(output.as_str())
                        && report.data["validated"]["construction"] == "passed"
                        && report.data["generated_validation"]["construction"] == "passed"
                        && matches!(
                            report.data["generated_validation"]["forward"].as_str(),
                            Some("passed" | "not_run")
                        ))
                {
                    Ok(report)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Generation outcome unknown; inspect the target file before retrying",
                    ))
                }
            })
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("{error}; inspect the target file before retrying"),
                )
            });
        return Ok(InitOutcome::Result {
            command,
            result,
            attempted_save: false,
        });
    }
    let capture_snapshot = matches!(
        action.as_str(),
        "add+snapshot" | "set+snapshot" | "remove+snapshot"
    );
    let action = action
        .strip_suffix("+snapshot")
        .unwrap_or(&action)
        .to_owned();
    if !matches!(action.as_str(), "add" | "set" | "remove") {
        return Ok(InitOutcome::Cancelled);
    }
    if let Some(message) =
        capability_unavailable(cli, locale, &["assembly.conditional_config_edit"])
    {
        return Ok(if show_flow_page(renderer, &message)? {
            InitOutcome::Cancelled
        } else {
            InitOutcome::Quit
        });
    }
    if capture_snapshot {
        if let Some(message) = capability_unavailable(
            cli,
            locale,
            &[
                "assembly.single_input_snapshot",
                "assembly.single_input_evidence",
            ],
        ) {
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
            return Ok(InitOutcome::Cancelled);
        }
    }
    let Some(slot) = init_input(
        renderer,
        locale.text("Factory seam name", "工厂接缝名", "工廠接縫名"),
        "",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    let mut args = vec!["assembly".to_owned(), action.clone(), slot.clone()];
    if action != "remove" {
        let Some(mut binding) = init_input(
            renderer,
            locale.text(
                "Python factory module:callable, component:ID@VERSION or :catalog",
                "Python 工厂 module:callable、component:ID@VERSION 或 :catalog 组件目录",
                "Python 工廠 module:callable、component:ID@VERSION 或 :catalog 元件目錄",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        if binding == ":catalog" {
            let catalog = cli.execute(&["component", "list"]);
            let report = match &catalog {
                Ok(report)
                    if report.operation == "component_list" && report.status == "success" =>
                {
                    report
                }
                _ => {
                    show_flow_page(renderer, &renderer.result(&catalog))?;
                    return Ok(InitOutcome::Cancelled);
                }
            };
            let Some(items) = report.data["components"].as_array() else {
                show_flow_page(renderer, "Component catalog is unavailable.\n")?;
                return Ok(InitOutcome::Cancelled);
            };
            let choices: Vec<String> = items
                .iter()
                .filter_map(|item| {
                    let manifest = &item["manifest"];
                    if manifest["kind"] != "layer" {
                        return None;
                    }
                    let id = manifest["component_id"]
                        .as_str()
                        .filter(|s| !s.is_empty())?;
                    let version = manifest["component_version"]
                        .as_str()
                        .filter(|s| !s.is_empty())?;
                    Some(format!("{id}@{version}"))
                })
                .collect();
            let mut query = String::new();
            let mut page = 0usize;
            let reference = loop {
                let filtered: Vec<&String> = choices
                    .iter()
                    .filter(|reference| reference.to_lowercase().contains(&query.to_lowercase()))
                    .collect();
                let pages = filtered.len().saturating_sub(1) / 32 + 1;
                page = page.min(pages - 1);
                let mut content = String::new();
                renderer.heading(
                    &mut content,
                    locale.text(
                        "Registered Layer components",
                        "已登记 Layer 组件",
                        "已登記 Layer 元件",
                    ),
                );
                renderer.line(
                    &mut content,
                    &format!(
                        "{}/{} · {} {}",
                        page + 1,
                        pages,
                        filtered.len(),
                        locale.text("matches", "项匹配", "項符合")
                    ),
                );
                for (index, reference) in filtered.iter().enumerate().skip(page * 32).take(32) {
                    renderer.line(
                        &mut content,
                        &format!("{}. {}", index + 1, text::plain_text(reference)),
                    );
                }
                renderer.line(&mut content, locale.text("n/p: next/previous · /text: search · /: clear · b: cancel. Choosing a component does not install it.", "n/p：下一页/上一页 · /文字：搜索 · /：清空 · b：取消。选择组件不会安装依赖。", "n/p：下一頁/上一頁 · /文字：搜尋 · /：清空 · b：取消。選擇元件不會安裝依賴。"));
                if !show_flow_page(renderer, &content)? {
                    return Ok(InitOutcome::Quit);
                }
                let Some(choice) = init_input(
                    renderer,
                    locale.text(
                        "Component number or navigation",
                        "组件序号或导航",
                        "元件序號或導覽",
                    ),
                    "",
                )?
                else {
                    return Ok(InitOutcome::Cancelled);
                };
                if let Some(search) = choice.strip_prefix('/') {
                    query = search.trim().to_owned();
                    page = 0;
                } else if choice.eq_ignore_ascii_case("n") {
                    page = (page + 1).min(pages - 1);
                } else if choice.eq_ignore_ascii_case("p") {
                    page = page.saturating_sub(1);
                } else if let Some(reference) = choice
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| n.checked_sub(1))
                    .and_then(|n| filtered.get(n))
                {
                    break (*reference).clone();
                }
            };
            binding = format!("component:{reference}");
        }
        if let Some(component) = binding.strip_prefix("component:") {
            let Some((reference, params)) = guide_component_parameters(cli, renderer, component)?
            else {
                return Ok(InitOutcome::Cancelled);
            };
            args.extend(["--component".into(), reference]);
            for (key, value) in params {
                args.extend(["--param".into(), format!("{key}={value}")]);
            }
        } else {
            if binding.is_empty() {
                return Ok(InitOutcome::Cancelled);
            }
            args.extend(["--factory".into(), binding]);
        }
    }
    let temporary = if capture_snapshot {
        let directory = tempfile::Builder::new()
            .prefix("model-maven-single-assembly-")
            .tempdir()?;
        let path = directory.path().join("input.snapshot");
        let notice = format!("{}\n{}: {}\n{}\n", locale.text("Capturing input snapshot", "捕获输入快照", "擷取輸入快照"), locale.text("Temporary file", "临时文件", "暫存檔案"), text::plain_text(&path.to_string_lossy()), locale.text("The private file contains input values, not targets. Cleanup is attempted when this flow ends; abnormal termination may leave it behind.", "私有文件包含输入值，不包含目标；流程结束时尝试清理，异常退出可能留下文件。", "私有檔案包含輸入值，不包含目標；流程結束時嘗試清理，異常結束可能留下檔案。"));
        if !show_flow_page(renderer, &notice)? {
            return Ok(InitOutcome::Quit);
        }
        Some(directory)
    } else {
        None
    };
    let snapshot_path = temporary.as_ref().map(|directory| {
        directory
            .path()
            .join("input.snapshot")
            .to_string_lossy()
            .into_owned()
    });
    let outcome = (|| -> io::Result<InitOutcome> {
        args.push("--dry-run".into());
        let mut preview_args = args.clone();
        if let Some(path) = &snapshot_path {
            preview_args.extend(["--capture-input-to".into(), path.clone()]);
        }
        let preview =
            match cli.execute(&preview_args.iter().map(String::as_str).collect::<Vec<_>>()) {
                Ok(report)
                    if report.operation == format!("assembly_{action}")
                        && report.status == "success"
                        && report.data["committed"] == false
                        && report.data["slot"].as_str() == Some(slot.as_str()) =>
                {
                    report
                }
                Ok(report) if report.status == "error" => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Ok(report),
                        attempted_save: false,
                    });
                }
                Ok(_) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Unexpected assembly preview; no changes saved",
                        )),
                        attempted_save: false,
                    });
                }
                Err(error) => {
                    return Ok(InitOutcome::Result {
                        command,
                        result: Err(error),
                        attempted_save: false,
                    });
                }
            };
        let Some(revision) = preview.data["base_config_revision"]
            .as_str()
            .filter(|value| value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Assembly preview has no valid config revision",
                )),
                attempted_save: false,
            });
        };
        let (source_supported, input_supported) = match cli.check_capabilities(&[
            "assembly.single_source_evidence",
            "assembly.single_input_evidence",
        ]) {
            Ok(crate::cli::CapabilityCheck::Available) => (true, true),
            Ok(crate::cli::CapabilityCheck::Missing(missing)) => (
                !missing
                    .iter()
                    .any(|item| item == "assembly.single_source_evidence"),
                !missing
                    .iter()
                    .any(|item| item == "assembly.single_input_evidence"),
            ),
            Ok(crate::cli::CapabilityCheck::Legacy) => (false, false),
            Err(error) => {
                return Ok(InitOutcome::Result {
                    command,
                    result: Err(io::Error::new(
                        error.kind(),
                        format!("CLI capability discovery failed; nothing saved: {error}"),
                    )),
                    attempted_save: false,
                });
            }
        };
        let source_digest = if source_supported
            && preview.data["source_evidence"]["algorithm"] == "canonical-json-sha256-v1"
        {
            preview.data["source_evidence"]["digest"]
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
        } else {
            None
        };
        let input_digest = if input_supported
            && preview.data["input_evidence"]["status"] == "available"
            && preview.data["input_evidence"]["algorithm"] == "sha256-forward-input-v1"
        {
            preview.data["input_evidence"]["digest"]
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
        } else {
            None
        };
        if capture_snapshot
            && (input_digest.is_none() || preview.data["input_snapshot"]["status"] != "captured")
        {
            show_flow_page(
                renderer,
                locale.text(
                    "Input snapshot could not be captured and verified; nothing saved.\n",
                    "无法捕获并校验输入快照；没有保存。\n",
                    "無法擷取並驗證輸入快照；沒有儲存。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        }
        let mut content = String::new();
        renderer.heading(
            &mut content,
            locale.text(
                "Assembly preview · not saved",
                "装配预览 · 尚未保存",
                "裝配預覽 · 尚未儲存",
            ),
        );
        renderer.line(
            &mut content,
            &format!("{}: {}", text::plain_text(&action), text::plain_text(&slot)),
        );
        if input_digest.is_some() {
            renderer.line(&mut content, locale.text("Choose input at confirmation to require the same observed forward input; dynamic batches may differ.", "确认时可输入 input，要求提交使用相同的已观察前向输入；动态批次可能不同。", "確認時可輸入 input，要求提交使用相同的已觀察前向輸入；動態批次可能不同。"));
        }
        if capture_snapshot {
            renderer.line(
                &mut content,
                locale.text(
                    "Save replays the captured input; targets and future batches are not included.",
                    "保存将重放捕获的输入；不包含目标或后续批次。",
                    "儲存將重播擷取的輸入；不包含目標或後續批次。",
                ),
            );
        }
        renderer.line(
            &mut content,
            &format!(
                "Bindings: {}",
                text::plain_text(&preview.data["model_injections"].to_string())
            ),
        );
        renderer.line(
            &mut content,
            &format!(
                "Validation: {}",
                text::plain_text(&preview.data["validation"].to_string())
            ),
        );
        renderer.line(
        &mut content,
        locale.text(
            if source_digest.is_some() { "Commit rechecks configuration and observed project sources; input, external libraries and changes after the check are not pinned." } else { "Commit rechecks configuration; source evidence is unavailable and will not be pinned. Input and external resources are not pinned." },
            if source_digest.is_some() { "提交时复核配置和已观察项目源码；输入、外部库及检查后的变化未固定。" } else { "提交时复核配置；源码证据不可用，源码、输入和外部资源未固定。" },
            if source_digest.is_some() { "提交時複核設定和已觀察專案原始碼；輸入、外部函式庫及檢查後的變化未固定。" } else { "提交時複核設定；原始碼證據不可用，原始碼、輸入和外部資源未固定。" },
        ),
    );
        #[cfg(unix)]
        if !renderer.line_mode && crate::terminal::menu::supported() {
            if !view_report(&content, locale, renderer.style)? {
                return Ok(InitOutcome::Quit);
            }
        } else {
            print!("{content}");
        }
        #[cfg(not(unix))]
        print!("{content}");
        let Some(answer) = init_input(
            renderer,
            locale.text(
                "Save assembly? Type y; input requires the same observed input",
                "保存装配？输入 y；输入 input 要求相同的已观察输入",
                "儲存裝配？輸入 y；輸入 input 要求相同的已觀察輸入",
            ),
            "",
        )?
        else {
            return Ok(InitOutcome::Cancelled);
        };
        let bind_input = capture_snapshot || answer.trim().eq_ignore_ascii_case("input");
        if !answer.trim().eq_ignore_ascii_case("y") && !answer.trim().eq_ignore_ascii_case("input")
        {
            return Ok(InitOutcome::Cancelled);
        }
        if bind_input && input_digest.is_none() {
            show_flow_page(
                renderer,
                locale.text(
                    "Input evidence unavailable; nothing saved.\n",
                    "输入证据不可用；没有保存。\n",
                    "輸入證據不可用；沒有儲存。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        }
        args.pop();
        args.extend(["--if-config-revision".into(), revision.into()]);
        if let Some(digest) = source_digest {
            args.extend(["--if-source-manifest".into(), digest.into()]);
        }
        if bind_input {
            args.extend([
                "--if-input-content".into(),
                input_digest.expect("checked above").into(),
            ]);
        }
        if let Some(path) = snapshot_path.as_deref() {
            args.extend(["--replay-input-from".into(), path.into()]);
        }
        let result = cli.execute(&args.iter().map(String::as_str).collect::<Vec<_>>()).and_then(|report| {
        if report.status == "error" || (report.operation == format!("assembly_{action}")
            && report.status == "success" && report.data["committed"] == true
            && report.data["slot"].as_str() == Some(slot.as_str())
            && (!capture_snapshot || (report.data["input_snapshot"]["status"] == "replayed" && report.data["input_evidence"]["digest"].as_str() == input_digest))
            && report.data["config_revision"].as_str().is_some()) {
            Ok(report)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected assembly commit receipt"))
        }
    }).map_err(|error| io::Error::new(error.kind(), format!("{error}; assembly save outcome is unknown; check project assembly before retrying")));
        Ok(InitOutcome::Result {
            command,
            result,
            attempted_save: true,
        })
    })();
    if let Some(directory) = temporary {
        let message = match directory.close() {
            Ok(()) => locale
                .text(
                    "Temporary input snapshot removed.\n",
                    "临时输入快照已清理。\n",
                    "暫存輸入快照已清理。\n",
                )
                .to_owned(),
            Err(error) => format!(
                "{}: {}\n",
                locale.text(
                    "Could not remove temporary input snapshot",
                    "无法清理临时输入快照",
                    "無法清理暫存輸入快照"
                ),
                text::plain_text(&error.to_string())
            ),
        };
        if !show_flow_page(renderer, &message)? {
            return Ok(InitOutcome::Quit);
        }
    }
    outcome
}

fn initialize_custom(
    cli: &Cli,
    renderer: &Renderer,
    draft: &mut CustomScaffoldDraft,
) -> io::Result<InitOutcome> {
    let locale = renderer.locale;
    if let Some(message) = capability_unavailable(cli, locale, &["project.custom_scaffold"]) {
        return Ok(if show_flow_page(renderer, &message)? {
            InitOutcome::Cancelled
        } else {
            InitOutcome::Quit
        });
    }
    let store = match if draft.unsaved {
        Err(io::Error::other(
            "unsaved in-memory changes; preserving this session",
        ))
    } else {
        DraftStore::open(cli.project()).and_then(|store| {
            if let Some(saved) = store.load()? {
                *draft = saved;
            }
            Ok(store)
        })
    } {
        Ok(store) => {
            eprintln!(
                "{}",
                locale.text(
                    "Accepted fields are saved to .model-maven/scaffold-draft.json for recovery; do not enter credentials.",
                    "已确认的字段会保存到 .model-maven/scaffold-draft.json 供下次恢复，请勿填写凭据。",
                    "已確認的欄位會儲存到 .model-maven/scaffold-draft.json 供下次恢復，請勿填入憑據。"
                )
            );
            Some(store)
        }
        Err(error) => {
            eprintln!(
                "{}: {error}",
                locale.text(
                    "Draft recovery unavailable; this session remains in memory",
                    "草稿恢复不可用；本次仅保存在内存中",
                    "草稿恢復不可用；本次僅保存在記憶體中"
                )
            );
            None
        }
    };
    let Some(specs) = init_input(
        renderer,
        locale.text(
            "Inputs and options, e.g. x=?5,4;@native=x_encoder or @seam.x_encoder=layers:make",
            "输入和选项，例如 x=?5,4;@native=x_encoder 或 @seam.x_encoder=layers:make",
            "輸入和選項，例如 x=?5,4;@native=x_encoder 或 @seam.x_encoder=layers:make",
        ),
        &draft.specs,
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    draft.specs = specs.clone();
    if let Some(store) = &store {
        if let Err(error) = store.save(draft) {
            draft.unsaved = true;
            eprintln!(
                "{}: {error}",
                locale.text(
                    "Draft save failed; this session remains in memory",
                    "草稿保存失败；本次仅保存在内存中",
                    "草稿儲存失敗；本次僅保存在記憶體中"
                )
            );
        } else {
            draft.unsaved = false;
        }
    } else {
        draft.unsaved = true;
    }
    let Some(output) = init_input(
        renderer,
        locale.text(
            "Output features: size, ce:classes, or named heads e.g. score=1:mse@0.5;classes=3:ce",
            "输出：特征数、ce:类别数，或具名头如 score=1:mse@0.5;classes=3:ce",
            "輸出：特徵數、ce:類別數，或具名頭如 score=1:mse@0.5;classes=3:ce",
        ),
        &draft.output,
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    draft.output = output.clone();
    if let Some(store) = &store {
        if let Err(error) = store.save(draft) {
            draft.unsaved = true;
            eprintln!(
                "{}: {error}",
                locale.text(
                    "Draft save failed; this session remains in memory",
                    "草稿保存失败；本次仅保存在内存中",
                    "草稿儲存失敗；本次僅保存在記憶體中"
                )
            );
        } else {
            draft.unsaved = false;
        }
    } else {
        draft.unsaved = true;
    }
    let entries: Vec<&str> = specs
        .split(';')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    let specs: Vec<&str> = entries
        .iter()
        .copied()
        .filter(|entry| !entry.starts_with('@'))
        .collect();
    let modules: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@module="))
        .collect();
    if modules.len() > 1 || modules.first().is_some_and(|module| module.is_empty()) {
        show_flow_page(
            renderer,
            locale.text(
                "Specify at most one nonempty @module path.\n",
                "请最多指定一个非空的 @module 路径。\n",
                "請最多指定一個非空的 @module 路徑。\n",
            ),
        )?;
        return Ok(InitOutcome::Cancelled);
    }
    let module = modules.first().copied().unwrap_or("model_starter");
    let model_entry = format!("{module}:model");
    let batches: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@batch="))
        .collect();
    let bindings: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@binding="))
        .collect();
    let batch_params: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@batchparam."))
        .collect();
    let seams: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@seam."))
        .collect();
    let seam_params: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@seamparam."))
        .collect();
    let native_encoders: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("@native="))
        .collect();
    let unique_native: std::collections::HashSet<_> = native_encoders.iter().copied().collect();
    if unique_native.len() != native_encoders.len()
        || native_encoders.iter().any(|slot| slot.is_empty())
    {
        show_flow_page(
            renderer,
            locale.text(
                "Use unique nonempty @native=ENCODER entries.\n",
                "请使用非空且不重复的 @native=编码器接缝。\n",
                "請使用非空且不重複的 @native=編碼器接縫。\n",
            ),
        )?;
        return Ok(InitOutcome::Cancelled);
    }
    let mut expected_seams = BTreeMap::new();
    for seam in &seams {
        let Some((slot, binding)) = seam.split_once('=') else {
            show_flow_page(
                renderer,
                locale.text(
                    "Bindings use @seam.SLOT=module:factory or component:ID@VERSION.\n",
                    "绑定格式为 @seam.接缝=模块:工厂 或 component:ID@版本。\n",
                    "繫結格式為 @seam.接縫=模組:工廠 或 component:ID@版本。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        if slot.is_empty() || binding.is_empty() || expected_seams.insert(slot, binding).is_some() {
            show_flow_page(
                renderer,
                locale.text(
                    "Seam names and bindings must be nonempty and unique.\n",
                    "接缝名及绑定不能为空或重复。\n",
                    "接縫名稱及繫結不可為空或重複。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        }
    }
    let mut expected_seam_params = BTreeMap::new();
    for param in &seam_params {
        let Some((key, encoded)) = param.split_once('=') else {
            show_flow_page(
                renderer,
                locale.text(
                    "Component parameters use @seamparam.SLOT.NAME=JSON.\n",
                    "组件参数格式为 @seamparam.接缝.名称=JSON。\n",
                    "元件參數格式為 @seamparam.接縫.名稱=JSON。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        let Some((slot, name)) = key.split_once('.') else {
            show_flow_page(
                renderer,
                locale.text(
                    "Component parameter needs SLOT.NAME.\n",
                    "组件参数需要接缝.名称。\n",
                    "元件參數需要接縫.名稱。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        let Ok(value) = serde_json::from_str::<Value>(encoded) else {
            show_flow_page(
                renderer,
                locale.text(
                    "Component parameter must contain valid JSON.\n",
                    "组件参数值须为有效 JSON。\n",
                    "元件參數值須為有效 JSON。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        if name.is_empty()
            || !expected_seams.contains_key(slot)
            || expected_seam_params.insert((slot, name), value).is_some()
        {
            show_flow_page(
                renderer,
                locale.text(
                    "Component parameters need a selected seam and unique nonempty names.\n",
                    "组件参数须对应所选接缝，名称不能为空或重复。\n",
                    "元件參數須對應所選接縫，名稱不可為空或重複。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        }
    }
    if batches.len() > 1
        || batches.first().is_some_and(|batch| batch.is_empty())
        || bindings.len() > 1
        || (!bindings.is_empty() && batches.is_empty())
        || (!batch_params.is_empty() && batches.is_empty())
        || bindings
            .first()
            .is_some_and(|binding| !matches!(*binding, "envelope" | "single" | "args" | "kwargs"))
    {
        show_flow_page(renderer, locale.text("Use one nonempty @batch=module:callable and optional @binding=envelope|single|args|kwargs.\n", "请指定一个非空 @batch=模块:函数，并可选 @binding=envelope|single|args|kwargs。\n", "請指定一個非空 @batch=模組:函式，並可選 @binding=envelope|single|args|kwargs。\n"))?;
        return Ok(InitOutcome::Cancelled);
    }
    let batch_entry = batches
        .first()
        .map(|batch| (*batch).to_owned())
        .unwrap_or_else(|| format!("{module}:batch"));
    let mut expected_params = BTreeMap::new();
    for param in &batch_params {
        let Some((name, encoded)) = param.split_once('=') else {
            show_flow_page(
                renderer,
                locale.text(
                    "Batch parameters use @batchparam.NAME=JSON.\n",
                    "批次参数格式为 @batchparam.NAME=JSON。\n",
                    "批次參數格式為 @batchparam.NAME=JSON。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        let Ok(value) = serde_json::from_str::<Value>(encoded) else {
            show_flow_page(
                renderer,
                locale.text(
                    "Batch parameter must contain a valid JSON value.\n",
                    "批次参数值须为有效 JSON。\n",
                    "批次參數值須為有效 JSON。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        };
        if name.is_empty() || expected_params.insert(name, value).is_some() {
            show_flow_page(
                renderer,
                locale.text(
                    "Batch parameter names must be nonempty and unique.\n",
                    "批次参数名不能为空或重复。\n",
                    "批次參數名稱不可為空或重複。\n",
                ),
            )?;
            return Ok(InitOutcome::Cancelled);
        }
    }
    let source_file = format!("{}.py", module.replace('.', "/"));
    let routes: Vec<&str> = entries
        .iter()
        .filter(|entry| {
            !entry.starts_with("@module=")
                && !entry.starts_with("@batch=")
                && !entry.starts_with("@binding=")
                && !entry.starts_with("@batchparam.")
                && !entry.starts_with("@seam.")
                && !entry.starts_with("@seamparam.")
                && !entry.starts_with("@native=")
        })
        .filter_map(|entry| entry.strip_prefix('@'))
        .collect();
    let command = Command {
        action: Action::Init,
        model: model_entry.clone(),
        batch: batch_entry.clone(),
        variant: String::new(),
        probe_options: ProbeOptions::default(),
    };
    let mut args = vec!["scaffold"];
    if !modules.is_empty() {
        args.extend(["--module", module]);
    }
    if let Some(batch) = batches.first() {
        args.extend(["--batch", batch]);
    }
    if let Some(binding) = bindings.first() {
        args.extend(["--batch-binding", binding]);
    }
    for param in &batch_params {
        args.extend(["--batch-param", param]);
    }
    for seam in &seams {
        args.extend(["--inject", seam]);
    }
    for param in &seam_params {
        args.extend(["--inject-param", param]);
    }
    for input in &specs {
        args.extend(["--input", input]);
    }
    for route in &routes {
        args.extend(["--route", route]);
    }
    for slot in &native_encoders {
        args.extend(["--native-encoder", slot]);
    }
    let named_outputs: Vec<&str> = if output.contains('=') {
        output
            .split(';')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .collect()
    } else {
        Vec::new()
    };
    let single_classification = if named_outputs.is_empty() {
        output.trim().strip_prefix("ce:").map(str::trim)
    } else {
        None
    };
    if !named_outputs.is_empty() {
        for item in &named_outputs {
            args.extend(["--output", item]);
        }
    } else if let Some(classes) = single_classification {
        args.extend(["--output-dim", classes, "--loss", "cross_entropy"]);
    } else if !output.trim().is_empty() {
        args.extend(["--output-dim", output.trim()]);
    }
    let mut required = Vec::new();
    if !routes.is_empty() {
        required.push("project.scaffold_input_routes");
    }
    if !native_encoders.is_empty() {
        required.push("project.scaffold_native_encoder_output");
    }
    if !modules.is_empty() {
        required.push("project.scaffold_custom_module");
    }
    if !batches.is_empty() {
        required.push("project.scaffold_external_batch");
    }
    if !batch_params.is_empty() {
        required.push("project.scaffold_external_batch_params");
    }
    if !seams.is_empty() {
        required.push("project.scaffold_initial_bindings");
    }
    if !named_outputs.is_empty() {
        required.push("project.scaffold_named_outputs");
    }
    if single_classification.is_some()
        || named_outputs.iter().any(|item| {
            item.split_once('=')
                .is_some_and(|(_, descriptor)| descriptor.contains(':'))
        })
    {
        required.push("project.scaffold_objectives");
    }
    if named_outputs.iter().any(|item| item.contains('@')) {
        required.push("project.scaffold_objective_weights");
    }
    if !required.is_empty() {
        if let Some(message) = capability_unavailable(cli, locale, &required) {
            if !show_flow_page(renderer, &message)? {
                return Ok(InitOutcome::Quit);
            }
            return Ok(InitOutcome::Cancelled);
        }
    }
    let report = match cli.execute(&args) {
        Ok(report)
            if report.operation == "scaffold"
                && report.status == "success"
                && report.data["committed"] == false =>
        {
            report
        }
        Ok(report) if report.status == "error" => {
            return Ok(InitOutcome::Result {
                command,
                result: Ok(report),
                attempted_save: false,
            });
        }
        Ok(_) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unexpected scaffold preview; no files created",
                )),
                attempted_save: false,
            });
        }
        Err(error) => {
            return Ok(InitOutcome::Result {
                command,
                result: Err(error),
                attempted_save: false,
            });
        }
    };
    let plan = &report.data["plan"];
    let revision = report.data["plan_revision"]
        .as_str()
        .filter(|s| !s.is_empty());
    let valid_inputs = plan["inputs"].as_array().is_some_and(|inputs| {
        inputs.len() == specs.len()
            && inputs.iter().zip(&specs).all(|(actual, requested)| {
                let Some((typed_name, dims)) = requested.split_once('=') else {
                    return false;
                };
                let (name, dtype) = typed_name
                    .split_once(':')
                    .unwrap_or((typed_name, "float32"));
                let native = unique_native.contains(format!("{name}_encoder").as_str());
                let mut dynamic_axes = Vec::new();
                let parsed: Option<Vec<u64>> = dims
                    .split(',')
                    .enumerate()
                    .map(|(axis, value)| {
                        if let Some(value) = value.strip_prefix('?') {
                            dynamic_axes.push(axis);
                            value.parse().ok()
                        } else {
                            value.parse().ok()
                        }
                    })
                    .collect();
                actual["name"].as_str() == Some(name)
                    && parsed.is_some_and(|dims| actual["shape"] == serde_json::json!(dims))
                    && (actual["dynamic_axes"] == serde_json::json!(dynamic_axes)
                        || (dynamic_axes.is_empty() && actual["dynamic_axes"].is_null()))
                    && (if native {
                        actual["encoder_output"] == "native"
                            && actual["dynamic_reduction"] == "none_native_encoder_output"
                    } else {
                        (actual["encoder_output"].is_null()
                            || actual["encoder_output"] == "flatten")
                            && (dynamic_axes.is_empty()
                                || actual["dynamic_reduction"] == "mean_after_encoder")
                    })
                    && actual["dtype"] == dtype
            })
    });
    let valid_routes = if routes.is_empty() && plan["routes"].is_null() {
        true // Older CLIs can still create the default one-to-one starter.
    } else {
        let mut expected = Vec::new();
        for spec in &specs {
            let Some((typed_name, _)) = spec.split_once('=') else {
                return Ok(InitOutcome::Cancelled);
            };
            let name = typed_name
                .split_once(':')
                .map_or(typed_name, |(name, _)| name);
            expected.push((format!("{name}_encoder"), name.to_owned()));
        }
        let mut seen = std::collections::HashSet::new();
        let mut valid = true;
        for route in &routes {
            if let Some((slot, input)) = route.split_once('=') {
                if !seen.insert(slot) {
                    valid = false;
                }
                if let Some((_, source)) = expected.iter_mut().find(|(name, _)| name == slot) {
                    *source = input.to_owned();
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
        }
        let actual = expected
            .iter()
            .map(|(slot, source)| serde_json::json!({"encoder_slot":slot, "input":source}))
            .collect::<Vec<_>>();
        valid && plan["routes"] == serde_json::json!(actual)
    };
    let valid_outputs = if named_outputs.is_empty() {
        if let Some(classes) = single_classification {
            classes.parse::<u64>().ok() == plan["output"]["features"].as_u64()
                && plan["output"]["objective"] == "cross_entropy"
        } else {
            (output.trim().is_empty()
                || output.trim().parse::<u64>().ok() == plan["output"]["features"].as_u64())
                && (plan["output"]["objective"].is_null() || plan["output"]["objective"] == "mse")
        }
    } else {
        plan["output"]["kind"] == "named"
            && plan["output"]["heads"].as_array().is_some_and(|heads| {
                heads.len() == named_outputs.len()
                    && heads.iter().zip(&named_outputs).all(|(actual, requested)| {
                        let Some((name, descriptor)) = requested.split_once('=') else {
                            return false;
                        };
                        let (descriptor, requested_weight) =
                            descriptor.split_once('@').unwrap_or((descriptor, "1"));
                        let weight = match requested_weight.parse::<f64>() {
                            Ok(weight) if weight.is_finite() && weight > 0.0 => weight,
                            _ => return false,
                        };
                        let (width, loss) =
                            descriptor.split_once(':').unwrap_or((descriptor, "mse"));
                        let objective = match loss {
                            "mse" => "mse",
                            "ce" | "cross_entropy" => "cross_entropy",
                            _ => return false,
                        };
                        actual["name"].as_str() == Some(name)
                            && width.parse::<u64>().ok() == actual["features"].as_u64()
                            && actual["dtype"] == "float32"
                            && (actual["weight"].as_f64() == Some(weight)
                                || (weight == 1.0 && actual["weight"].is_null()))
                            && (actual["objective"] == objective
                                || (objective == "mse" && actual["objective"].is_null()))
                    })
            })
    };
    let redacted: Vec<&str> = report.data["redacted_fields"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let actual_params = plan["config"]["batch_params"].as_object();
    let valid_batch_params = if expected_params.is_empty() {
        actual_params.is_none_or(|params| params.is_empty())
    } else {
        actual_params.is_some_and(|params| {
            params.len() == expected_params.len()
                && expected_params.iter().all(|(name, expected)| {
                    let Some(actual) = params.get(*name) else {
                        return false;
                    };
                    let path = format!("$.plan.config.batch_params.{name}");
                    let obscured = redacted.iter().any(|item| {
                        *item == path
                            || item.strip_prefix(&path).is_some_and(|suffix| {
                                suffix.starts_with('.') || suffix.starts_with('[')
                            })
                    });
                    obscured || actual == expected
                })
        })
    };
    let valid_seams = (expected_seams.is_empty() && plan["config"]["model_injections"].is_null())
        || plan["config"]["model_injections"]
            .as_object()
            .is_some_and(|actual| {
                actual.len() == expected_seams.len()
                    && expected_seams.iter().all(|(slot, binding)| {
                        let Some(value) = actual.get(*slot) else {
                            return false;
                        };
                        if let Some(reference) = binding.strip_prefix("component:") {
                            let Some((id, version)) = reference.rsplit_once('@') else {
                                return false;
                            };
                            value["component_ref"]["id"].as_str() == Some(id)
                                && value["component_ref"]["version"].as_str() == Some(version)
                                && value["params"].is_object()
                        } else {
                            value.as_str() == Some(*binding)
                        }
                    })
                    && expected_seam_params.iter().all(|((slot, name), expected)| {
                        let Some(value) = actual
                            .get(*slot)
                            .and_then(|binding| binding["params"].get(*name))
                        else {
                            return false;
                        };
                        let path = format!("$.plan.config.model_injections.{slot}.params.{name}");
                        let obscured = redacted.iter().any(|item| {
                            *item == path
                                || item.strip_prefix(&path).is_some_and(|suffix| {
                                    suffix.starts_with('.') || suffix.starts_with('[')
                                })
                        });
                        obscured || value == expected
                    })
            });
    if revision.is_none()
        || !valid_outputs
        || !valid_inputs
        || !valid_routes
        || !valid_batch_params
        || !valid_seams
        || plan["project_root"].as_str().map(Path::new) != Some(cli.project())
        || plan["files"] != serde_json::json!([source_file, ".model-maven/project.json"])
        || plan["config"]["model"] != model_entry
        || plan["config"]["batch"] != batch_entry
        || plan["batch_source"]
            != if batches.is_empty() {
                "synthetic"
            } else {
                "external"
            }
        || (if batches.is_empty() {
            plan["config"]["batch_binding"] != "envelope"
        } else if let Some(binding) = bindings.first() {
            plan["config"]["batch_binding"] != *binding
        } else {
            !plan["config"]["batch_binding"].is_null()
        })
        || plan["source"].as_str().is_none()
    {
        return Ok(InitOutcome::Result {
            command,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "scaffold preview does not match this project or inputs",
            )),
            attempted_save: false,
        });
    }
    let mut content = String::new();
    renderer.heading(
        &mut content,
        locale.text("Custom model preview", "自定义模型预览", "自訂模型預覽"),
    );
    if !expected_seams.is_empty() {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Initial seam bindings", "初始接缝绑定", "初始接縫繫結"),
                text::plain_text(&plan["config"]["model_injections"].to_string())
            ),
        );
        renderer.line(&mut content, locale.text("Factories have not run yet; a forward must verify their compatibility. Bindings and parameters are stored in project configuration; do not enter credentials.", "工厂尚未运行，需要前向验证兼容性。绑定和参数保存在项目配置中，请勿填写凭据。", "工廠尚未執行，需要前向驗證相容性。繫結及參數儲存在專案設定中，請勿填入憑據。"));
    }
    for input in &specs {
        renderer.line(&mut content, &text::plain_text(input));
    }
    if !native_encoders.is_empty() {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text(
                    "Native encoder outputs (no automatic reduction or flattening)",
                    "保留编码器原始输出（不自动归约或展平）",
                    "保留編碼器原始輸出（不自動歸約或展平）"
                ),
                text::plain_text(&native_encoders.join(", "))
            ),
        );
        renderer.line(&mut content, locale.text("Provide compatible fusion and head components and verify a real forward; defaults may not match native shapes.", "请提供兼容的融合和预测头并验证真实前向；默认组件可能不适配原始形状。", "請提供相容的融合和預測頭並驗證真實前向；預設元件可能不適配原始形狀。"));
    }
    for route in plan["routes"].as_array().into_iter().flatten() {
        renderer.line(
            &mut content,
            &format!(
                "{} ← {}",
                text::plain_text(route["encoder_slot"].as_str().unwrap_or("?")),
                text::plain_text(route["input"].as_str().unwrap_or("?"))
            ),
        );
    }
    if let Some(routes) = plan["routes"].as_array() {
        for input in plan["inputs"].as_array().into_iter().flatten() {
            if let Some(name) = input["name"].as_str() {
                if !routes
                    .iter()
                    .any(|route| route["input"].as_str() == Some(name))
                {
                    renderer.line(
                        &mut content,
                        &format!(
                            "{}: {}",
                            locale.text("Unconnected input", "未连接的输入", "未連接的輸入"),
                            text::plain_text(name)
                        ),
                    );
                }
            }
        }
    }
    if !expected_params.is_empty() {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Batch parameters", "批次参数", "批次參數"),
                text::plain_text(&plan["config"]["batch_params"].to_string())
            ),
        );
        renderer.line(&mut content, locale.text("Parameters are stored in the local project configuration and draft; do not enter credentials.", "参数会保存在本地项目配置和草稿中，请勿填写凭据。", "參數會儲存在本地專案設定與草稿中，請勿填入憑據。"));
    }
    if named_outputs.is_empty() {
        renderer.line(
            &mut content,
            &format!("Output features: {}", plan["output"]["features"]),
        );
    } else {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Named outputs", "具名输出", "具名輸出"),
                text::plain_text(&plan["output"]["heads"].to_string())
            ),
        );
        renderer.line(&mut content, locale.text("The optimized total uses the displayed task weights; reported task losses are unweighted. MSE uses float targets and cross entropy uses integer class IDs; ensure the selected batch supplies suitable targets.", "优化总损失使用显示的任务权重；单项报告为未加权损失。MSE 使用浮点目标，交叉熵使用整数类别 ID；请确认所选批次提供合适目标。", "最佳化總損失使用顯示的任務權重；單項報告為未加權損失。MSE 使用浮點目標，交叉熵使用整數類別 ID；請確認所選批次提供合適目標。"));
    }
    if single_classification.is_some() {
        renderer.line(&mut content, locale.text("Cross entropy expects logits and integer class IDs in [0, classes); synthetic labels are placeholders.", "交叉熵使用 logits 和 [0, 类别数) 的整数标签；合成标签只是占位。", "交叉熵使用 logits 和 [0, 類別數) 的整數標籤；合成標籤只是佔位。"));
    }
    if batches.is_empty() {
        renderer.line(
            &mut content,
            locale.text(
                "Synthetic batch and placeholder loss; replace them for real training.",
                "批次和损失均为占位示例；真实训练前请替换。",
                "批次和損失均為佔位範例；實際訓練前請替換。",
            ),
        );
    } else {
        renderer.line(
            &mut content,
            &format!(
                "{}: {}",
                locale.text("Selected batch factory", "选择的批次工厂", "選擇的批次工廠"),
                text::plain_text(&batch_entry)
            ),
        );
        renderer.line(&mut content, locale.text("The selected batch factory has not run yet. The generated file still contains a synthetic example; verify real inputs and targets with Probe and a trial.", "所选批次工厂尚未运行；生成文件仍包含合成样例，请用前向诊断和试跑核对真实输入及目标。", "所選批次工廠尚未執行；產生檔案仍包含合成範例，請用前向診斷和試跑核對實際輸入及目標。"));
    }
    renderer.line(&mut content, locale.text("Supported input dtypes: float32 (default), float64, float16, int64, bool. Default fusion converts encoder outputs to float32; a custom fusion controls its own contract.", "支持的输入类型：float32（默认）、float64、float16、int64、bool。默认融合将编码器输出转换为 float32；自定义融合自行定义契约。", "支援的輸入型別：float32（預設）、float64、float16、int64、bool。預設融合將編碼器輸出轉換為 float32；自訂融合自行定義契約。"));
    renderer.line(&mut content, locale.text("?N marks a variable axis with example length N. The default model averages that axis after its encoder; edit the starter for other behavior.", "?N 标记可变轴，N 为样例长度。默认模型在编码器之后沿该轴取均值；其他行为可修改生成代码。", "?N 標記可變軸，N 為樣例長度。預設模型在編碼器後沿該軸取平均；其他行為可修改產生的程式碼。"));
    renderer.line(
        &mut content,
        &format!(
            "{}; .model-maven/project.json",
            text::plain_text(&source_file)
        ),
    );
    for line in text::plain_text(plan["source"].as_str().unwrap_or("")).lines() {
        renderer.line(&mut content, &format!("│{line}"));
    }
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        if !view_report(&content, locale, renderer.style)? {
            return Ok(InitOutcome::Quit);
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    print!("{content}");
    let Some(answer) = init_input(
        renderer,
        locale.text(
            "Create files? Type y to confirm",
            "创建文件？输入 y 确认",
            "建立檔案？輸入 y 確認",
        ),
        "",
    )?
    else {
        return Ok(InitOutcome::Cancelled);
    };
    if !answer.trim().eq_ignore_ascii_case("y") {
        return Ok(InitOutcome::Cancelled);
    }
    args.extend(["--apply", "--if-plan", revision.unwrap()]);
    let result = cli
        .execute(&args)
        .and_then(|report| {
            if report.status == "error"
                || (report.operation == "scaffold"
                    && report.status == "success"
                    && report.data["committed"] == true
                    && report.data["plan_revision"].as_str() == revision
                    && report.data["model"] == model_entry
                    && report.data["batch"] == batch_entry)
            {
                Ok(report)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected scaffold creation receipt; outcome unknown",
                ))
            }
        })
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("{error}; inspect project files before retrying"),
            )
        });
    if result.as_ref().is_ok_and(|report| {
        report.operation == "scaffold"
            && report.status == "success"
            && report.data["committed"] == true
    }) {
        if let Some(store) = &store {
            if let Err(error) = store.clear() {
                eprintln!(
                    "{}: {error}",
                    locale.text(
                        "Draft cleanup failed; inspect the project before restoring it",
                        "草稿清理失败；恢复前请检查项目",
                        "草稿清理失敗；恢復前請檢查專案"
                    )
                );
            }
        }
    }
    Ok(InitOutcome::Result {
        command,
        result,
        attempted_save: true,
    })
}

fn choose_language(renderer: &Renderer) -> io::Result<Option<LanguageChoice>> {
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        let keys = ["0", "1", "2", "3", "b"];
        let labels = [
            renderer
                .locale
                .text("Follow system", "跟随系统", "跟隨系統"),
            "English",
            "简体中文",
            "繁體中文",
            renderer.locale.text("Back", "返回", "返回"),
        ];
        let selected = crate::terminal::menu::select(&keys, |selected| {
            let mut output = String::new();
            renderer.heading(
                &mut output,
                renderer.locale.text("Language", "语言", "語言"),
            );
            for (index, label) in labels.iter().enumerate() {
                renderer.line(
                    &mut output,
                    &format!("{} {label}", if index == selected { ">" } else { " " }),
                );
            }
            renderer.line(
                &mut output,
                renderer.locale.text(
                    "↑/↓ Move · Enter select · Esc back",
                    "↑/↓ 移动 · Enter 选择 · Esc 返回",
                    "↑/↓ 移動 · Enter 選擇 · Esc 返回",
                ),
            );
            output
        })?;
        return Ok(selected.and_then(|index| LanguageChoice::parse(keys[index])));
    }
    let Some(value) = init_input(
        renderer,
        renderer.locale.text(
            "Language: 0 system, 1 English, 2 简体中文, 3 繁體中文 (b back)",
            "语言：0 跟随系统，1 English，2 简体中文，3 繁體中文（b 返回）",
            "語言：0 跟隨系統，1 English，2 简体中文，3 繁體中文（b 返回）",
        ),
        "",
    )?
    else {
        return Ok(None);
    };
    let selection = LanguageChoice::parse(&value);
    if selection.is_none() {
        show_flow_page(
            renderer,
            renderer.locale.text(
                "Unknown language; selection unchanged.\n",
                "未知语言，选择未改变。\n",
                "未知語言，選擇未變更。\n",
            ),
        )?;
    }
    Ok(selection)
}

fn init_input(renderer: &Renderer, label: &str, initial: &str) -> io::Result<Option<String>> {
    init_input_value(renderer, label, initial, false)
}

fn init_input_value(
    renderer: &Renderer,
    label: &str,
    initial: &str,
    allow_b: bool,
) -> io::Result<Option<String>> {
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        return match crate::terminal::menu::input(initial, |value, cursor| {
            let renderer = Renderer::current(renderer.locale, renderer.style, false);
            let mut output = String::new();
            renderer.heading(&mut output, label);
            renderer.line(
                &mut output,
                &text::input_window("> ", value, cursor, renderer.width),
            );
            renderer.line(
                &mut output,
                renderer.locale.text(
                    "Enter continue · Ctrl+U clear · Esc back",
                    "Enter 继续 · Ctrl+U 清空 · Esc 返回",
                    "Enter 繼續 · Ctrl+U 清空 · Esc 返回",
                ),
            );
            output
        })? {
            crate::terminal::menu::Input::Submit(value)
                if value.trim().eq_ignore_ascii_case(":back") =>
            {
                Ok(None)
            }
            crate::terminal::menu::Input::Submit(value) => Ok(Some(value.trim().to_owned())),
            crate::terminal::menu::Input::Back | crate::terminal::menu::Input::Quit => Ok(None),
        };
    }
    if !initial.is_empty() {
        eprintln!(
            "{}: {}",
            renderer.locale.text("Current", "当前值", "目前值"),
            text::plain_text(initial)
        );
    }
    let Some(value) = prompt(label)? else {
        return Ok(None);
    };
    if value.trim().eq_ignore_ascii_case(":back")
        || (!allow_b && value.trim().eq_ignore_ascii_case("b"))
    {
        return Ok(None);
    }
    if value.trim().eq_ignore_ascii_case(":default") {
        return Ok(Some(String::new()));
    }
    if value.trim().is_empty() && !initial.is_empty() {
        return Ok(Some(initial.to_owned()));
    }
    Ok(Some(value.trim().to_owned()))
}

fn uses_cached_variant_evidence(action: Action, workbench: &Workbench) -> bool {
    if workbench.variant.is_empty() {
        return false;
    }
    let is_variant_command = |command: &Command| {
        matches!(command.action, Action::Inspect | Action::Probe)
            && command.variant == workbench.variant
    };
    match action {
        Action::LastReport => workbench
            .last_command
            .as_ref()
            .is_some_and(is_variant_command),
        Action::Details => workbench
            .displayed_command()
            .is_some_and(is_variant_command),
        Action::LastDiagnosis | Action::Calls | Action::Preview | Action::Locate => {
            workbench.diagnosis.is_some()
        }
        Action::LastPreview => workbench.saved_preview.is_some(),
        Action::Structure => workbench.inspected_structure.is_some(),
        _ => false,
    }
}

fn check_variant_revision(
    workbench: &mut Workbench,
    cli: &Cli,
    renderer: &Renderer,
) -> io::Result<bool> {
    let result = cli.execute(&["variant", "show", &workbench.variant]);
    let current = result
        .as_ref()
        .ok()
        .filter(|report| {
            report.operation == "variant_show"
                && report.status == "success"
                && report.data["name"].as_str() == Some(workbench.variant.as_str())
        })
        .and_then(|report| report.data["variant"]["revision"].as_str())
        .filter(|revision| !revision.trim().is_empty())
        .map(str::to_owned);
    if current.is_some() && current == workbench.variant_revision {
        return Ok(true);
    }
    let receipt = workbench.saved_receipt.take();
    workbench.clear_project_context();
    workbench.saved_receipt = receipt;
    workbench.variant_revision = current;
    let reason = if workbench.variant_revision.is_some() {
        renderer.locale.text("Variant revision changed or was not previously available. Session evidence was cleared; run inspection and probe again.", "变体版本已变化或此前无法获取。已清除会话证据，请重新检查和诊断。", "變體版本已變更或先前無法取得。已清除工作階段證據，請重新檢查和診斷。")
    } else {
        renderer.locale.text("Variant revision could not be read. Session evidence was cleared; check the CLI and variant before retrying.", "无法读取变体版本。已清除会话证据，请检查 CLI 和变体后重试。", "無法讀取變體版本。已清除工作階段證據，請檢查 CLI 和變體後重試。")
    };
    notice(
        renderer,
        renderer.locale.text(
            "Variant evidence is stale",
            "变体证据已过期",
            "變體證據已過期",
        ),
        reason,
    )?;
    Ok(false)
}

#[derive(Clone)]
struct VariantChoice {
    name: String,
    revision: Option<String>,
    selectable: bool,
}

fn variant_screen(
    renderer: &Renderer,
    variants: &[VariantChoice],
    selected: usize,
    rows: usize,
) -> String {
    let mut output = String::new();
    renderer.heading(
        &mut output,
        renderer
            .locale
            .text("Select variant", "选择变体", "選擇變體"),
    );
    renderer.line(
        &mut output,
        renderer.locale.text(
            "Listing is not validation or a pin",
            "列表不验证或固定版本",
            "清單不驗證或固定版本",
        ),
    );
    renderer.line(
        &mut output,
        renderer.locale.text(
            "Selecting clears session evidence",
            "选择后清除会话证据",
            "選擇後清除工作階段證據",
        ),
    );
    let count = variants.len() + 1;
    let visible = rows.saturating_sub(6).max(1);
    let start = selected
        .saturating_sub(visible - 1)
        .min(count.saturating_sub(visible));
    for index in start..(start + visible).min(count) {
        let marker = if selected == index { ">" } else { " " };
        let label = if index == 0 {
            renderer
                .locale
                .text("CLI default", "CLI 默认", "CLI 預設")
                .to_owned()
        } else {
            let item = &variants[index - 1];
            let revision = item
                .revision
                .as_deref()
                .map(|value| text::truncate(&text::plain_text(value).replace('\n', " "), 12))
                .unwrap_or_else(|| "?".to_owned());
            let status = if item.selectable {
                ""
            } else {
                renderer
                    .locale
                    .text(" · unavailable", " · 不可用", " · 無法使用")
            };
            format!(
                "{} · {}{status}",
                text::plain_text(&item.name).replace('\n', " "),
                revision
            )
        };
        renderer.line(
            &mut output,
            &text::truncate(&format!("{marker} {index}. {label}"), renderer.width),
        );
    }
    renderer.line(
        &mut output,
        &format!(
            "{} {}",
            if selected == count { ">" } else { " " },
            renderer.locale.text("b. Back", "b. 返回", "b. 返回")
        ),
    );
    renderer.line(
        &mut output,
        if renderer.line_mode {
            renderer.locale.text(
                "Enter a number · b back",
                "输入序号 · b 返回",
                "輸入序號 · b 返回",
            )
        } else {
            renderer.locale.text(
                "↑↓/Tab select · Enter · b back",
                "↑↓/Tab 选择 · Enter · b 返回",
                "↑↓/Tab 選擇 · Enter · b 返回",
            )
        },
    );
    output
}

struct VariantSelection {
    name: String,
    revision: Option<String>,
}

fn choose_variant(
    cli: &Cli,
    current: &str,
    renderer: &Renderer,
) -> io::Result<Option<VariantSelection>> {
    eprintln!(
        "{}",
        renderer.locale.text(
            "Reading CLI variants…",
            "正在读取 CLI 变体…",
            "正在讀取 CLI 變體…"
        )
    );
    let result = cli.execute(&["variant", "list"]);
    let records = match &result {
        Ok(report) if report.operation == "variant_list" && report.status == "success" => {
            report.data["variants"].as_array()
        }
        _ => None,
    };
    if records.is_none() {
        let mut message = renderer.result(&result);
        renderer.line(
            &mut message,
            renderer.locale.text(
                "Variant list unavailable; CLI default remains selectable.",
                "变体列表不可用；仍可选择 CLI 默认。",
                "變體清單無法取得；仍可選擇 CLI 預設。",
            ),
        );
        #[cfg(unix)]
        if !renderer.line_mode && crate::terminal::menu::supported() {
            if !view_report(&message, renderer.locale, renderer.style)? {
                return Ok(None);
            }
        } else {
            print!("{message}");
        }
        #[cfg(not(unix))]
        print!("{message}");
    }
    let records = records.cloned().unwrap_or_default();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for record in &records {
        if let Some(name) = record["name"].as_str().filter(|name| !name.is_empty()) {
            *counts.entry(name).or_default() += 1;
        }
    }
    let variants: Vec<_> = records
        .iter()
        .filter_map(|record| {
            let name = record["name"].as_str().filter(|name| !name.is_empty())?;
            Some(VariantChoice {
                name: name.to_owned(),
                revision: record["revision"].as_str().map(str::to_owned),
                selectable: record["readable"] == true
                    && counts.get(name) == Some(&1)
                    && !name.chars().any(char::is_control),
            })
        })
        .collect();
    let initial = if current.is_empty() {
        0
    } else {
        variants
            .iter()
            .position(|item| item.name == current)
            .map_or(variants.len() + 1, |index| index + 1)
    };
    let mut focus = initial;
    loop {
        #[cfg(unix)]
        let selection = if !renderer.line_mode && crate::terminal::menu::supported() {
            let mut names: Vec<String> = (0..=variants.len())
                .map(|index| index.to_string())
                .collect();
            names.push("b".into());
            let keys: Vec<&str> = names.iter().map(String::as_str).collect();
            crate::terminal::menu::select_from(&keys, focus, |selected| {
                variant_screen(renderer, &variants, selected, crate::terminal::menu::rows())
            })?
        } else {
            print!(
                "{}",
                variant_screen(renderer, &variants, focus, variants.len() + 7)
            );
            let Some(input) = prompt(renderer.locale.text(
                "Variant number (b back)",
                "变体序号（b 返回）",
                "變體序號（b 返回）",
            ))?
            else {
                return Ok(None);
            };
            if input.eq_ignore_ascii_case("b") || input.eq_ignore_ascii_case("q") {
                return Ok(None);
            }
            input.parse::<usize>().ok()
        };
        #[cfg(not(unix))]
        let selection = {
            print!(
                "{}",
                variant_screen(renderer, &variants, focus, variants.len() + 7)
            );
            let Some(input) = prompt(renderer.locale.text(
                "Variant number (b back)",
                "变体序号（b 返回）",
                "變體序號（b 返回）",
            ))?
            else {
                return Ok(None);
            };
            if input.eq_ignore_ascii_case("b") || input.eq_ignore_ascii_case("q") {
                return Ok(None);
            }
            input.parse::<usize>().ok()
        };
        let Some(index) = selection else {
            continue;
        };
        if index == variants.len() + 1 {
            return Ok(None);
        }
        if index == 0 {
            return Ok(Some(VariantSelection {
                name: String::new(),
                revision: None,
            }));
        }
        if let Some(item) = variants.get(index - 1) {
            focus = index;
            if item.selectable {
                return Ok(Some(VariantSelection {
                    name: item.name.clone(),
                    revision: item
                        .revision
                        .clone()
                        .filter(|revision| !revision.trim().is_empty()),
                }));
            }
            notice(
                renderer,
                renderer
                    .locale
                    .text("Variant not selected", "未选择变体", "未選擇變體"),
                renderer.locale.text(
                    "This variant is unreadable or has an ambiguous name.",
                    "此变体不可读取或名称存在歧义。",
                    "此變體無法讀取或名稱有歧義。",
                ),
            )?;
        }
    }
}

fn keymap_screen(
    renderer: &Renderer,
    bindings: &[Binding],
    selected: usize,
    rows: usize,
) -> String {
    let mut output = String::new();
    renderer.heading(
        &mut output,
        renderer
            .locale
            .text("Workbench menu keys", "工作台菜单键位", "工作台選單鍵位"),
    );
    renderer.line(
        &mut output,
        renderer.locale.text(
            "Graph keys use the selected preset",
            "图内键位使用操作预设",
            "圖內鍵位使用操作預設",
        ),
    );
    renderer.line(
        &mut output,
        renderer.locale.text(
            "s Settings · q Quit · fixed",
            "s 设置 · q 退出 · 固定",
            "s 設定 · q 離開 · 固定",
        ),
    );
    let visible = rows.saturating_sub(6).max(1);
    let start = selected
        .saturating_sub(visible - 1)
        .min(bindings.len().saturating_sub(visible));
    for (index, binding) in bindings.iter().enumerate().skip(start).take(visible) {
        let marker = if index == selected { ">" } else { " " };
        renderer.line(
            &mut output,
            &text::truncate(
                &format!(
                    "{marker} {}. {}",
                    binding.key,
                    binding.action.label(renderer.locale)
                ),
                renderer.width,
            ),
        );
    }
    renderer.line(
        &mut output,
        &format!(
            "{} {}",
            if selected == bindings.len() { ">" } else { " " },
            renderer.locale.text("b. Back", "b. 返回", "b. 返回")
        ),
    );
    renderer.line(
        &mut output,
        if renderer.line_mode {
            renderer.locale.text(
                "Enter action key · b back",
                "输入动作键位 · b 返回",
                "輸入動作鍵位 · b 返回",
            )
        } else {
            renderer.locale.text(
                "↑↓ select · Enter edit · b back",
                "↑↓ 选择 · Enter 修改 · b 返回",
                "↑↓ 選擇 · Enter 修改 · b 返回",
            )
        },
    );
    output
}

fn notice(renderer: &Renderer, heading: &str, message: &str) -> io::Result<()> {
    let mut output = String::new();
    renderer.heading(&mut output, heading);
    renderer.line(&mut output, message);
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        let _ = view_report(&output, renderer.locale, renderer.style)?;
        return Ok(());
    }
    print!("{output}");
    Ok(())
}

fn matching_workbench_commands<'a>(
    page: &'a MenuPage<'_>,
    locale: Locale,
    query: &str,
) -> Vec<&'a Binding> {
    let query = query.trim().to_lowercase();
    page.bindings
        .iter()
        .filter(|binding| {
            [
                binding.action.label(locale),
                binding.action.description(locale),
                binding.action.id(),
                binding.key,
            ]
            .iter()
            .any(|value| value.to_lowercase().contains(&query))
        })
        .collect()
}

fn search_workbench_commands(
    page: &MenuPage<'_>,
    renderer: &Renderer,
) -> io::Result<Option<Action>> {
    let locale = renderer.locale;
    #[cfg(unix)]
    let query = if !renderer.line_mode && crate::terminal::menu::supported() {
        match crate::terminal::menu::input("", |value, cursor| {
            let renderer = Renderer::current(renderer.locale, renderer.style, false);
            let mut output = String::new();
            renderer.heading(
                &mut output,
                locale.text("Search commands", "搜索操作", "搜尋操作"),
            );
            renderer.line(
                &mut output,
                &text::input_window("> ", value, cursor, renderer.width),
            );
            let matches = matching_workbench_commands(page, locale, value);
            renderer.line(
                &mut output,
                &format!(
                    "{} {}",
                    locale.text("Matches", "匹配", "符合"),
                    matches.len()
                ),
            );
            let visible = crate::terminal::menu::rows().saturating_sub(6);
            for binding in matches.iter().take(visible) {
                let marker = if page.available(binding.action) {
                    " "
                } else {
                    "!"
                };
                renderer.line(
                    &mut output,
                    &text::truncate(
                        &format!("{marker} {}. {}", binding.key, binding.action.label(locale)),
                        renderer.width,
                    ),
                );
            }
            renderer.line(
                &mut output,
                locale.text(
                    "! unavailable · Enter results · Esc back",
                    "! 不可用 · Enter 查看 · Esc 返回",
                    "! 不可用 · Enter 檢視 · Esc 返回",
                ),
            );
            output
        })? {
            crate::terminal::menu::Input::Submit(value) => Some(value),
            crate::terminal::menu::Input::Back | crate::terminal::menu::Input::Quit => None,
        }
    } else {
        init_input(
            renderer,
            locale.text("Search commands", "搜索操作", "搜尋操作"),
            "",
        )?
    };
    #[cfg(not(unix))]
    let query = init_input(
        renderer,
        locale.text("Search commands", "搜索操作", "搜尋操作"),
        "",
    )?;
    let Some(query) = query else {
        return Ok(None);
    };
    let matches = matching_workbench_commands(page, locale, &query);
    if matches.is_empty() {
        notice(
            renderer,
            locale.text("No commands found", "没有匹配的操作", "沒有符合的操作"),
            locale.text(
                "Try another search.",
                "请尝试其他搜索词。",
                "請嘗試其他搜尋詞。",
            ),
        )?;
        return Ok(None);
    }
    let mut focus = 0;
    loop {
        let render = |selected: usize, rows: usize| {
            let mut output = String::new();
            renderer.heading(
                &mut output,
                locale.text("Command results", "操作搜索结果", "操作搜尋結果"),
            );
            let total = matches.len() + 1;
            let visible = rows.saturating_sub(4).max(1);
            let start = selected
                .saturating_add(1)
                .saturating_sub(visible)
                .min(total.saturating_sub(visible));
            for index in start..(start + visible).min(total) {
                let marker = if index == selected { ">" } else { " " };
                let line = if let Some(binding) = matches.get(index) {
                    let disabled = if page.available(binding.action) {
                        " "
                    } else {
                        "!"
                    };
                    format!(
                        "{marker}{disabled} {}. {}",
                        binding.key,
                        binding.action.label(locale)
                    )
                } else {
                    format!("{marker} {}", locale.text("b. Back", "b. 返回", "b. 返回"))
                };
                renderer.line(&mut output, &text::truncate(&line, renderer.width));
            }
            renderer.line(
                &mut output,
                locale.text(
                    "! unavailable · ↑↓ Enter · b back",
                    "! 不可用 · ↑↓ Enter · b 返回",
                    "! 不可用 · ↑↓ Enter · b 返回",
                ),
            );
            output
        };
        #[cfg(unix)]
        let selected = if !renderer.line_mode && crate::terminal::menu::supported() {
            let mut keys: Vec<&str> = matches.iter().map(|binding| binding.key).collect();
            keys.push("b");
            crate::terminal::menu::select_from(&keys, focus, |selected| {
                render(selected, crate::terminal::menu::rows())
            })?
        } else {
            print!("{}", render(focus, matches.len() + 5));
            let Some(choice) = prompt(locale.text(
                "Command key (b back)",
                "操作键位（b 返回）",
                "操作按鍵（b 返回）",
            ))?
            else {
                return Ok(None);
            };
            if choice.eq_ignore_ascii_case("b") {
                None
            } else {
                matches
                    .iter()
                    .position(|binding| binding.key.eq_ignore_ascii_case(&choice))
            }
        };
        #[cfg(not(unix))]
        let selected = {
            print!("{}", render(focus, matches.len() + 5));
            let Some(choice) = prompt(locale.text(
                "Command key (b back)",
                "操作键位（b 返回）",
                "操作按鍵（b 返回）",
            ))?
            else {
                return Ok(None);
            };
            if choice.eq_ignore_ascii_case("b") {
                None
            } else {
                matches
                    .iter()
                    .position(|binding| binding.key.eq_ignore_ascii_case(&choice))
            }
        };
        let Some(index) = selected.filter(|index| *index < matches.len()) else {
            return Ok(None);
        };
        let action = matches[index].action;
        if page.available(action) {
            return Ok(Some(action));
        }
        focus = index;
        let reason = match action {
            Action::Init | Action::CreateClassification | Action::CreateCustom => locale.text(
                "Project setup is available for an unconfigured project.",
                "项目未配置时可设置入口。",
                "專案尚未設定時可設定入口。",
            ),
            Action::Assembly => locale.text(
                "Save a project model before editing assembly.",
                "请先保存项目模型入口，再编辑装配。",
                "請先儲存專案模型入口，再編輯裝配。",
            ),
            Action::LastReport | Action::Details => locale.text(
                "Run a check or diagnosis first.",
                "请先运行检查或诊断。",
                "請先執行檢查或診斷。",
            ),
            Action::LastDiagnosis | Action::Calls | Action::Locate | Action::Preview => locale
                .text(
                    "Inspect and diagnose a model first; the required evidence must be available.",
                    "请先检查并诊断模型，并确保所需证据可用。",
                    "請先檢查並診斷模型，並確認所需證據可用。",
                ),
            Action::LastPreview => locale.text(
                "Run a repair preview first.",
                "请先运行修复预览。",
                "請先執行修復預覽。",
            ),
            _ => locale.text(
                "This action is unavailable in the current context.",
                "当前状态下无法执行此操作。",
                "目前狀態無法執行此操作。",
            ),
        };
        notice(
            renderer,
            locale.text("Command unavailable", "操作暂不可用", "操作暫不可用"),
            reason,
        )?;
    }
}
fn keymap_notice(renderer: &Renderer, message: &str) -> io::Result<()> {
    notice(
        renderer,
        renderer
            .locale
            .text("Key not changed", "键位未修改", "鍵位未修改"),
        message,
    )
}

fn configure_workbench_keys(
    keymaps: &mut WorkbenchKeymaps,
    preferences: Preferences,
    path: Option<&Path>,
    renderer: &Renderer,
) -> io::Result<()> {
    let preset = preferences.preset;
    loop {
        let all = keymaps.bindings(preset).expect("validated keymap");
        let editable: Vec<Binding> = all
            .iter()
            .copied()
            .filter(|binding| !matches!(binding.action, Action::Settings | Action::Quit))
            .collect();
        #[cfg(unix)]
        let selected = if !renderer.line_mode && crate::terminal::menu::supported() {
            // The back key takes precedence; an action currently bound to b is
            // still reachable using the arrow keys and Enter.
            let mut keys: Vec<&str> = editable
                .iter()
                .map(|binding| if binding.key == "b" { "" } else { binding.key })
                .collect();
            keys.push("b");
            crate::terminal::menu::select(&keys, |selected| {
                keymap_screen(renderer, &editable, selected, crate::terminal::menu::rows())
            })?
        } else {
            print!(
                "{}",
                keymap_screen(renderer, &editable, 0, editable.len() + 6)
            );
            let Some(input) = prompt(renderer.locale.text(
                "Action key (b back)",
                "动作键位（b 返回）",
                "動作鍵位（b 返回）",
            ))?
            else {
                return Ok(());
            };
            if input.eq_ignore_ascii_case("b") || input.eq_ignore_ascii_case("q") {
                return Ok(());
            }
            editable
                .iter()
                .position(|binding| binding.key.eq_ignore_ascii_case(input.trim()))
        };
        #[cfg(not(unix))]
        let selected = {
            print!(
                "{}",
                keymap_screen(renderer, &editable, 0, editable.len() + 6)
            );
            let Some(input) = prompt(renderer.locale.text(
                "Action key (b back)",
                "动作键位（b 返回）",
                "動作鍵位（b 返回）",
            ))?
            else {
                return Ok(());
            };
            if input.eq_ignore_ascii_case("b") || input.eq_ignore_ascii_case("q") {
                return Ok(());
            }
            editable
                .iter()
                .position(|binding| binding.key.eq_ignore_ascii_case(input.trim()))
        };
        let Some(index) = selected else {
            continue;
        };
        let Some(binding) = editable.get(index) else {
            return Ok(());
        };
        let baseline = preset
            .bindings()
            .iter()
            .find(|item| item.action == binding.action)
            .expect("known action")
            .key;
        let label = format!(
            "{} · {}",
            binding.action.label(renderer.locale),
            if renderer.line_mode {
                renderer.locale.text(
                    "New key (a-z/0-9; :default resets; b cancels)",
                    "新键位（字母或数字；:default 恢复；b 取消）",
                    "新鍵位（字母或數字；:default 恢復；b 取消）",
                )
            } else {
                renderer.locale.text(
                    "New key (a-z/0-9; clear resets; Esc cancels)",
                    "新键位（字母或数字；清空恢复；Esc 取消）",
                    "新鍵位（字母或數字；清空恢復；Esc 取消）",
                )
            }
        );
        let Some(value) = init_input(renderer, &label, binding.key)? else {
            continue;
        };
        let key = if value.is_empty() {
            baseline.to_owned()
        } else {
            value.to_ascii_lowercase()
        };
        if key == "b" || key == "q" || !MENU_KEYS.contains(&key.as_str()) {
            keymap_notice(
                renderer,
                renderer.locale.text(
                    "Use one letter or digit; b and q are reserved.",
                    "请输入单个字母或数字；b 和 q 保留。",
                    "請輸入單個字母或數字；b 和 q 保留。",
                ),
            )?;
            continue;
        }
        if let Some(conflict) = all
            .iter()
            .find(|other| other.action != binding.action && other.key == key)
        {
            keymap_notice(
                renderer,
                &format!(
                    "{} {key}: {}",
                    renderer
                        .locale
                        .text("Already assigned", "键位已分配", "鍵位已指派"),
                    conflict.action.label(renderer.locale)
                ),
            )?;
            continue;
        }
        let mut candidate = keymaps.clone();
        if key == baseline {
            candidate.map_mut(preset).remove(binding.action.id());
        } else {
            candidate
                .map_mut(preset)
                .insert(binding.action.id().to_owned(), key);
        }
        let saved = path
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "preferences directory unavailable")
            })
            .and_then(|path| candidate.save(path, preferences));
        match saved {
            Ok(()) => *keymaps = candidate,
            Err(error) => keymap_notice(
                renderer,
                &format!(
                    "{}: {error}",
                    renderer.locale.text(
                        "Could not save preferences",
                        "无法保存偏好设置",
                        "無法儲存偏好設定"
                    )
                ),
            )?,
        }
    }
}

fn configure_probe_options(workbench: &mut Workbench, renderer: &Renderer) -> io::Result<()> {
    let Page::Batch { model, draft } = &workbench.page else {
        return Ok(());
    };
    let initial = workbench
        .probe_options_draft
        .clone()
        .or_else(|| {
            workbench
                .diagnosis
                .as_ref()
                .filter(|d| {
                    d.command.model == *model
                        && d.command.batch == *draft
                        && d.command.variant == workbench.variant
                        && workbench
                            .model_draft
                            .as_ref()
                            .is_some_and(|params| *params == d.command.probe_options.model_params)
                })
                .map(|d| d.command.probe_options.clone())
        })
        .unwrap_or_default();
    let binding = loop {
        let Some(value) = init_input(
            renderer,
            renderer.locale.text(
                "Batch binding: auto, envelope, single, args or kwargs (b cancels)",
                "批次绑定：auto、envelope、single、args 或 kwargs（b 取消）",
                "批次繫結：auto、envelope、single、args 或 kwargs（b 取消）",
            ),
            &initial.binding,
        )?
        else {
            return Ok(());
        };
        let value = value.to_ascii_lowercase();
        if matches!(
            value.as_str(),
            "" | "auto" | "envelope" | "single" | "args" | "kwargs"
        ) {
            break if value == "auto" {
                String::new()
            } else {
                value
            };
        }
        eprintln!(
            "{}",
            renderer.locale.text(
                "Unknown binding; choose a listed value.",
                "绑定方式无效，请选择列出的值。",
                "繫結方式無效，請選擇列出的值。"
            )
        );
    };
    let Some(fingerprint) = init_input(
        renderer,
        renderer.locale.text(
            "Dataset fingerprint (optional; update when data changes; b cancels)",
            "数据集指纹（可选；数据变化时需更新；b 取消）",
            "資料集指紋（選填；資料變化時需更新；b 取消）",
        ),
        &initial.fingerprint,
    )?
    else {
        return Ok(());
    };
    workbench.probe_options_draft = Some(ProbeOptions {
        binding,
        fingerprint,
        ..initial
    });
    Ok(())
}

fn parse_factory_parameters(input: &str) -> Result<BTreeMap<String, Value>, String> {
    struct Parameters;
    impl<'de> serde::de::Visitor<'de> for Parameters {
        type Value = BTreeMap<String, Value>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object with unique parameter names")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut values = BTreeMap::new();
            while let Some((name, value)) = map.next_entry::<String, Value>()? {
                if name.trim().is_empty()
                    || name.contains('=')
                    || name.chars().any(char::is_control)
                {
                    return Err(serde::de::Error::custom("invalid parameter name"));
                }
                if values.insert(name, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate parameter name"));
                }
            }
            Ok(values)
        }
    }
    use serde::de::Deserializer;
    let mut parser = serde_json::Deserializer::from_str(input);
    let values = parser
        .deserialize_map(Parameters)
        .map_err(|error| error.to_string())?;
    parser.end().map_err(|error| error.to_string())?;
    Ok(values)
}

fn configure_batch_parameters(workbench: &mut Workbench, renderer: &Renderer) -> io::Result<()> {
    let Page::Batch { model, draft } = &workbench.page else {
        return Ok(());
    };
    let mut options = workbench
        .probe_options_draft
        .clone()
        .or_else(|| {
            workbench
                .diagnosis
                .as_ref()
                .filter(|diagnosis| {
                    diagnosis.command.model == *model
                        && diagnosis.command.batch == *draft
                        && diagnosis.command.variant == workbench.variant
                        && workbench.model_draft.as_ref().is_some_and(|params| {
                            *params == diagnosis.command.probe_options.model_params
                        })
                })
                .map(|diagnosis| diagnosis.command.probe_options.clone())
        })
        .unwrap_or_default();
    let mut initial = serde_json::to_string(&options.batch_params).map_err(io::Error::other)?;
    loop {
        let Some(value) = init_input(
            renderer,
            renderer.locale.text(
                "Batch factory parameters as a JSON object (b cancels)",
                "批次工厂参数：JSON 对象（b 取消）",
                "批次工廠參數：JSON 物件（b 取消）",
            ),
            &initial,
        )?
        else {
            return Ok(());
        };
        let input = if value.trim().is_empty() {
            "{}"
        } else {
            value.trim()
        };
        match parse_factory_parameters(input) {
            Ok(parameters) => {
                options.batch_params = parameters;
                workbench.probe_options_draft = Some(options);
                return Ok(());
            }
            Err(_) => {
                notice(renderer, renderer.locale.text("Invalid batch parameters", "批次参数无效", "批次參數無效"), renderer.locale.text(
                    "Enter a JSON object with unique, nonempty names; names cannot contain '=' or control characters.",
                    "请输入 JSON 对象；参数名必须非空且不重复，不能包含 = 或控制字符。",
                    "請輸入 JSON 物件；參數名稱必須非空且不重複，不能包含 = 或控制字元。",
                ))?;
                initial = value;
            }
        }
    }
}

fn configure_execution_mode(workbench: &mut Workbench, renderer: &Renderer) -> io::Result<()> {
    let Page::Batch { model, draft } = &workbench.page else {
        return Ok(());
    };
    let mut options = workbench
        .probe_options_draft
        .clone()
        .or_else(|| {
            workbench
                .diagnosis
                .as_ref()
                .filter(|diagnosis| {
                    diagnosis.command.model == *model
                        && diagnosis.command.batch == *draft
                        && diagnosis.command.variant == workbench.variant
                        && workbench.model_draft.as_ref().is_some_and(|params| {
                            *params == diagnosis.command.probe_options.model_params
                        })
                })
                .map(|diagnosis| diagnosis.command.probe_options.clone())
        })
        .unwrap_or_default();
    let initial = if options.training { "train" } else { "eval" };
    loop {
        let Some(value) = init_input(
            renderer,
            renderer.locale.text(
                "Forward mode: eval or train (b cancels)",
                "前向模式：eval 或 train（b 取消）",
                "前向模式：eval 或 train（b 取消）",
            ),
            initial,
        )?
        else {
            return Ok(());
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "eval" => options.training = false,
            "train" => options.training = true,
            _ => {
                notice(
                    renderer,
                    renderer
                        .locale
                        .text("Invalid forward mode", "前向模式无效", "前向模式無效"),
                    renderer.locale.text(
                        "Choose eval or train.",
                        "请选择 eval 或 train。",
                        "請選擇 eval 或 train。",
                    ),
                )?;
                continue;
            }
        }
        workbench.probe_options_draft = Some(options);
        return Ok(());
    }
}

fn configure_input_snapshot(workbench: &mut Workbench, renderer: &Renderer) -> io::Result<()> {
    let Page::Batch { model, draft } = &workbench.page else {
        return Ok(());
    };
    let mut options = workbench
        .probe_options_draft
        .clone()
        .or_else(|| {
            workbench
                .diagnosis
                .as_ref()
                .filter(|diagnosis| {
                    diagnosis.command.model == *model
                        && diagnosis.command.batch == *draft
                        && diagnosis.command.variant == workbench.variant
                        && workbench.model_draft.as_ref().is_some_and(|params| {
                            *params == diagnosis.command.probe_options.model_params
                        })
                })
                .map(|diagnosis| diagnosis.command.probe_options.clone())
        })
        .unwrap_or_default();
    loop {
        let Some(value) = init_input(renderer, renderer.locale.text(
            "Input snapshot: on/off. Contains input values in a temporary file; abnormal exit may leave it (b cancels)",
            "输入快照：on/off。临时文件含输入值；异常退出可能残留（b 取消）",
            "輸入快照：on/off。暫存檔含輸入值；異常結束可能殘留（b 取消）",
        ), if options.capture_snapshot { "on" } else { "off" })? else { return Ok(()); };
        match value.trim().to_ascii_lowercase().as_str() {
            "on" => options.capture_snapshot = true,
            "off" | "" => options.capture_snapshot = false,
            _ => {
                notice(
                    renderer,
                    renderer
                        .locale
                        .text("Invalid snapshot option", "快照选项无效", "快照選項無效"),
                    renderer.locale.text(
                        "Choose on or off.",
                        "请选择 on 或 off。",
                        "請選擇 on 或 off。",
                    ),
                )?;
                continue;
            }
        }
        workbench.probe_options_draft = Some(options);
        return Ok(());
    }
}

fn configure_model_parameters(workbench: &mut Workbench, renderer: &Renderer) -> io::Result<()> {
    let Page::Model { draft, .. } = &workbench.page else {
        return Ok(());
    };
    let params = workbench
        .model_draft
        .clone()
        .or_else(|| {
            workbench
                .model_context
                .as_ref()
                .filter(|(model, _)| model == draft)
                .map(|(_, params)| params.clone())
        })
        .unwrap_or_default();
    let mut initial = serde_json::to_string(&params).map_err(io::Error::other)?;
    loop {
        let Some(value) = init_input(
            renderer,
            renderer.locale.text(
                "Model factory parameters as a JSON object (b cancels)",
                "模型工厂参数：JSON 对象（b 取消）",
                "模型工廠參數：JSON 物件（b 取消）",
            ),
            &initial,
        )?
        else {
            return Ok(());
        };
        let input = if value.trim().is_empty() {
            "{}"
        } else {
            value.trim()
        };
        match parse_factory_parameters(input) {
            Ok(parameters) => {
                workbench.model_draft = Some(parameters);
                workbench.model_draft_owner = None;
                return Ok(());
            }
            Err(_) => {
                notice(renderer, renderer.locale.text("Invalid model parameters", "模型参数无效", "模型參數無效"), renderer.locale.text(
                    "Enter a JSON object with unique, nonempty names; names cannot contain '=' or control characters.",
                    "请输入 JSON 对象；参数名必须非空且不重复，不能包含 = 或控制字符。",
                    "請輸入 JSON 物件；參數名稱必須非空且不重複，不能包含 = 或控制字元。",
                ))?;
                initial = value;
            }
        }
    }
}

#[cfg(unix)]
fn historical_recheck_status(
    result: &io::Result<Report>,
    diagnosis_content: &Value,
    requested_training: bool,
    replay: bool,
    locale: Locale,
) -> String {
    let status = match result {
        Ok(report) if report.operation == "probe" && report.status == "success" => {
            let mode_ok =
                reported_training_mode(&report.data["observed_mode"]) == Some(requested_training);
            let replay_ok = !replay || report.data["batch_snapshot"]["status"] == "replayed";
            let input_ok = compare_input_content(diagnosis_content, &report.data["batch_content"])
                == InputContentComparison::Same;
            let fresh_ok = matches!(
                report.data["evidence_cache"]["status"].as_str(),
                Some("stored" | "not_stored")
            );
            if mode_ok && replay_ok && input_ok && fresh_ok {
                match report.data["trace_truncated"].as_bool() {
                    Some(false) => locale.text(
                        "Recheck · CLI reports success",
                        "重验 · CLI 报告成功",
                        "重驗 · CLI 報告成功",
                    ),
                    Some(true) => locale.text(
                        "Recheck · success, trace truncated",
                        "重验 · 成功，调用记录截断",
                        "重驗 · 成功，呼叫記錄截斷",
                    ),
                    None => locale.text(
                        "Recheck · success, coverage unknown",
                        "重验 · 成功，覆盖未知",
                        "重驗 · 成功，覆蓋未知",
                    ),
                }
            } else {
                locale.text(
                    "Recheck · success, checks unverified",
                    "重验 · 成功，条件未核对",
                    "重驗 · 成功，條件未核對",
                )
            }
        }
        Ok(report)
            if report.operation == "probe"
                && report.status == "error"
                && report
                    .diagnostics
                    .iter()
                    .any(|item| item.code == "FORWARD_FAILED") =>
        {
            locale.text(
                "Recheck · CLI reports forward failure",
                "重验 · CLI 报告前向失败",
                "重驗 · CLI 報告前向失敗",
            )
        }
        Ok(_) => locale.text(
            "Recheck · unverified or rejected",
            "重验 · 未确认或已拒绝",
            "重驗 · 未確認或已拒絕",
        ),
        Err(_) => locale.text("Recheck · unavailable", "重验 · 不可用", "重驗 · 不可用"),
    };
    status.to_owned()
}

fn preview_linear(
    cli: &Cli,
    locale: Locale,
    style: Style,
    target: &str,
    command: &Command,
    evidence: &str,
    diagnosis_report: &Report,
    snapshot: Option<&tempfile::TempDir>,
    inspected: Option<&[Value]>,
    saved_preview: &mut Option<PreviewRecord>,
    saved_receipt: &mut Option<PreviewRecord>,
    line_mode: bool,
    attempted_commit: &mut bool,
) -> io::Result<bool> {
    let renderer = Renderer::current(locale, style, line_mode);
    let diagnosis_content = &diagnosis_report.data["batch_content"];
    let mut instructions = String::new();
    renderer.heading(
        &mut instructions,
        locale.text(
            "Preview Linear input size",
            "预览 Linear 输入维度",
            "預覽 Linear 輸入維度",
        ),
    );
    if command.variant.is_empty() {
        renderer.line(
            &mut instructions,
            locale.text(
                "CLI default: preview may use the existing working variant.",
                "CLI 默认：预览可能使用已有 working 变体。",
                "CLI 預設：預覽可能使用已有 working 變體。",
            ),
        );
    } else {
        renderer.line(
            &mut instructions,
            &format!(
                "{}  {}",
                locale.text("Selected variant", "所选变体", "所選變體"),
                text::plain_text(&command.variant)
            ),
        );
    }
    renderer.line(
        &mut instructions,
        &format!(
            "{}  {}",
            locale.text("Last observed call", "最后观测到的调用", "最後觀測到的呼叫"),
            target
        ),
    );
    if line_mode {
        renderer.line(
            &mut instructions,
            locale.text("Enter b to cancel.", "输入 b 取消。", "輸入 b 取消。"),
        );
    }
    instructions.push_str(evidence);
    renderer.line(
        &mut instructions,
        locale.text(
            "Enter a positive in_features value. The CLI will validate the candidate.",
            "输入正整数 in_features，CLI 将验证候选修改。",
            "輸入正整數 in_features，CLI 將驗證候選修改。",
        ),
    );
    renderer.line(
        &mut instructions,
        if snapshot.is_some() {
            locale.text(
                "Validation replays the temporary captured input.",
                "验证会重放临时保存的输入。",
                "驗證會重放暫存的輸入。",
            )
        } else {
            locale.text(
                "Validation runs the batch factory again; the sample may differ.",
                "验证会再次调用批次工厂，样本可能不同。",
                "驗證會再次呼叫批次工廠，樣本可能不同。",
            )
        },
    );
    let snapshot_path = snapshot.map(|directory| directory.path().join("input.snapshot"));
    let snapshot_path = snapshot_path.as_deref().and_then(Path::to_str);
    #[cfg(unix)]
    let value = if !line_mode && crate::terminal::menu::supported() {
        match crate::terminal::menu::input("", |value, cursor| {
            let rows = crate::terminal::menu::rows();
            let lines: Vec<&str> = instructions.lines().collect();
            let available = rows.saturating_sub(5);
            let start = lines.len().saturating_sub(available);
            let visible = lines[start..].join("\n");
            let marker = if start > 0 { "…\n" } else { "" };
            let width = layout::content_width(layout::terminal_width());
            let entry = text::input_window("in_features  ", value, cursor, width);
            format!("{marker}{visible}\n{entry}\n\nEnter · OK   Esc · Back\n")
        })? {
            crate::terminal::menu::Input::Submit(value) => Some(value),
            crate::terminal::menu::Input::Back => return Ok(true),
            crate::terminal::menu::Input::Quit => return Ok(false),
        }
    } else {
        print!("{instructions}");
        prompt("in_features")?
    };
    #[cfg(not(unix))]
    let value = {
        print!("{instructions}");
        prompt("in_features")?
    };
    let Some(value) = value else {
        return Ok(false);
    };
    if line_mode && value.trim().eq_ignore_ascii_case("b") {
        return Ok(true);
    }
    let mut content = String::new();
    #[cfg(unix)]
    let mut diff = None;
    let mut executed = false;
    let mut preview_json: Option<String> = None;
    let mut commit_guard: Option<(String, String, String, String, Option<String>)> = None;
    if let Ok(size) = value.trim().parse::<u64>() {
        if size > 0 {
            let parameter = format!("in_features={size}");
            let mut args = vec!["layer", "set", target, "--param", &parameter, "--dry-run"];
            if !command.model.is_empty() {
                args.extend(["--model", &command.model]);
            }
            let model_params: Vec<String> = command
                .probe_options
                .model_params
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            for parameter in &model_params {
                args.extend(["--model-param", parameter]);
            }
            if !command.batch.is_empty() {
                args.extend(["--batch", &command.batch]);
            }
            if !command.variant.is_empty() {
                args.extend(["--variant", &command.variant]);
            }
            if !command.probe_options.binding.is_empty() {
                args.extend(["--batch-binding", &command.probe_options.binding]);
            }
            if command.probe_options.training {
                args.push("--training");
            }
            let batch_params: Vec<String> = command
                .probe_options
                .batch_params
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            for parameter in &batch_params {
                args.extend(["--batch-param", parameter]);
            }
            args.push("--require-forward");
            if let Some(path) = snapshot_path {
                let digest = diagnosis_content["digest"].as_str().unwrap_or("");
                args.extend(["--replay-input-from", path, "--if-input-content", digest]);
            }
            let mut preview_command = command.clone();
            preview_command.action = Action::Preview;
            preview_command.announce(&renderer, Some(target), true)?;
            let result = cli.execute(&args);
            preview_json = result.as_ref().ok().map(|report| report.raw_json.clone());
            executed = true;
            renderer.line(
                &mut content,
                &format!(
                    "{target} · {parameter} · {}",
                    locale.text("preview", "预览", "預覽")
                ),
            );
            renderer.line(
                &mut content,
                if inspected.is_some() {
                    locale.text(
                        "Inspection snapshot is not verified against the preview base revision.",
                        "检查快照尚未与预览基准版本核对。",
                        "檢查快照尚未與預覽基準版本核對。",
                    )
                } else {
                    locale.text(
                        "No matching inspected snapshot; inspect the model to compare structure.",
                        "没有匹配的检查快照；请检查模型以对照结构。",
                        "沒有相符的檢查快照；請檢查模型以對照結構。",
                    )
                },
            );
            if result.as_ref().is_ok_and(|report| {
                report.operation == "edit"
                    && mode_mismatch(
                        &report.data["validation"]["mode"],
                        command.probe_options.training,
                    )
            }) {
                renderer.heading(
                    &mut content,
                    locale.text(
                        "Candidate mode differs from request",
                        "候选模式与请求不一致",
                        "候選模式與請求不一致",
                    ),
                );
                renderer.line(
                    &mut content,
                    locale.text(
                        "Forward result does not validate the requested mode. Run preview again.",
                        "前向结果不能验证请求的模式，请重新预览。",
                        "前向結果無法驗證請求的模式，請重新預覽。",
                    ),
                );
            } else if result.as_ref().is_ok_and(|report| {
                report.operation == "edit"
                    && reported_training_mode(&report.data["validation"]["mode"]).is_none()
            }) {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Candidate mode unverified; CLI did not report train or eval.",
                        "候选模式无法核对；CLI 未报告 train 或 eval。",
                        "候選模式無法核對；CLI 未報告 train 或 eval。",
                    ),
                );
            }
            let content_match = result
                .as_ref()
                .ok()
                .filter(|report| {
                    report.operation == "edit"
                        && report.data["validation"]["forward"] == "passed"
                        && report.data["validation"]["forward_evidence"] == "executed"
                })
                .map_or(InputContentComparison::Unknown, |report| {
                    compare_input_content(
                        diagnosis_content,
                        &report.data["validation"]["batch_content"],
                    )
                });
            renderer.line(
                &mut content,
                match content_match {
                    InputContentComparison::Same => locale.text(
                        "Bound input content matches diagnosis · CLI fingerprints",
                        "前向输入内容与诊断匹配 · CLI 指纹",
                        "前向輸入內容與診斷相符 · CLI 指紋",
                    ),
                    InputContentComparison::Different => locale.text(
                        "Bound input content differs from diagnosis · CLI fingerprints",
                        "前向输入内容与诊断不同 · CLI 指纹",
                        "前向輸入內容與診斷不同 · CLI 指紋",
                    ),
                    InputContentComparison::Unknown => locale.text(
                        "Bound input content cannot be compared with diagnosis.",
                        "无法核对前向输入内容与诊断是否相同。",
                        "無法核對前向輸入內容與診斷是否相同。",
                    ),
                },
            );
            if snapshot_path.is_some() {
                let replayed = result.as_ref().is_ok_and(|report| {
                    report.operation == "edit"
                        && report.data["validation"]["batch_snapshot"]["status"] == "replayed"
                });
                renderer.line(
                    &mut content,
                    if replayed {
                        locale.text(
                            "CLI reports candidate used captured input.",
                            "CLI 报告候选使用了保存的输入。",
                            "CLI 報告候選使用了保存的輸入。",
                        )
                    } else {
                        locale.text(
                            "Captured input replay unconfirmed; commit unavailable.",
                            "无法确认输入快照已重放；不能提交。",
                            "無法確認輸入快照已重放；不能提交。",
                        )
                    },
                );
            }
            if content_match == InputContentComparison::Same {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Content match does not verify model source or runtime.",
                        "内容匹配不验证模型源码或运行环境。",
                        "內容相符不驗證模型原始碼或執行環境。",
                    ),
                );
            }
            content.push_str(&renderer.result(&result));
            if let Ok(report) = result {
                let base = &report.data["variant_base"];
                let name = base["name"].as_str().unwrap_or("");
                let token = base["token"].as_str().unwrap_or("");
                let is_digest = |value: &str| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                };
                let digest = diagnosis_content["digest"].as_str().unwrap_or("");
                let source = &report.data["source_manifest_evidence"];
                let source_valid = source.is_null()
                    || (source["algorithm"] == "canonical-json-sha256-v1"
                        && source["scope"]
                            == "worker-recorded-project-sources-and-component-manifests"
                        && source["digest"].as_str().is_some_and(is_digest));
                let source_digest = if source_valid {
                    source["digest"].as_str().map(str::to_owned)
                } else {
                    None
                };
                renderer.line(
                    &mut content,
                    if source.is_null() {
                        locale.text(
                            "Project source evidence unavailable.",
                            "项目源码证据不可用。",
                            "專案原始碼證據不可用。",
                        )
                    } else if source_valid {
                        locale.text(
                            "Commit will compare the worker-recorded project source manifest.",
                            "提交将核对 worker 记录的项目源码清单。",
                            "提交將核對 worker 記錄的專案原始碼清單。",
                        )
                    } else {
                        locale.text(
                            "Malformed source evidence; commit unavailable.",
                            "源码证据格式异常，无法提交。",
                            "原始碼證據格式異常，無法提交。",
                        )
                    },
                );
                let selected_name = if command.variant.is_empty() {
                    "working"
                } else {
                    &command.variant
                };
                if report.operation == "edit"
                    && report.status == "success"
                    && report.data["committed"] == false
                    && report.data["variant_name"] == name
                    && name == selected_name
                    && base["algorithm"] == "canonical-json-sha256-v1"
                    && (token == "absent" || is_digest(token))
                    && content_match == InputContentComparison::Same
                    && is_digest(digest)
                    && (snapshot_path.is_none()
                        || report.data["validation"]["batch_snapshot"]["status"] == "replayed")
                    && reported_training_mode(&report.data["validation"]["mode"])
                        == Some(command.probe_options.training)
                    && !command.model.is_empty()
                    && !command.batch.is_empty()
                    && source_valid
                {
                    commit_guard = Some((
                        name.to_owned(),
                        token.to_owned(),
                        digest.to_owned(),
                        parameter.clone(),
                        source_digest,
                    ));
                }
                #[cfg(unix)]
                if report.operation == "edit" {
                    diff = Some(crate::graph::diff::Diff::live_preview(
                        &report,
                        inspected,
                        target,
                        locale,
                        command.probe_options.training,
                    ));
                }
                if report.operation == "edit" {
                    content.push_str(&renderer.preview_effect_summary(&report));
                    for key in ["edit_effects", "validation", "variant_name", "committed"] {
                        if !report.data[key].is_null() {
                            renderer.line(&mut content, &format!("{key}  {}", report.data[key]));
                        }
                    }
                } else {
                    renderer.line(
                        &mut content,
                        locale.text(
                            "Unexpected CLI operation · preview unverified; commit unavailable.",
                            "CLI 操作类型不匹配 · 预览未确认；不能提交。",
                            "CLI 操作類型不相符 · 預覽未確認；無法提交。",
                        ),
                    );
                }
            }
        }
    }
    if content.is_empty() {
        renderer.line(
            &mut content,
            locale.text(
                "Enter a positive integer.",
                "请输入正整数。",
                "請輸入正整數。",
            ),
        );
    }
    if executed {
        *saved_preview = Some(PreviewRecord {
            content: content.clone(),
            #[cfg(unix)]
            diff: diff.clone(),
        });
    }
    #[cfg(unix)]
    if !line_mode && crate::terminal::menu::supported() {
        if !view_report_with_diff_state(&content, locale, style, &mut diff)? {
            return Ok(false);
        }
        if executed && let Some(preview) = saved_preview.as_mut() {
            preview.diff = diff.clone();
        }
    } else {
        print!("{content}");
    }
    #[cfg(not(unix))]
    {
        let _ = inspected;
        print!("{content}");
    }
    if let Some((name, base, digest, parameter, source_digest)) = commit_guard {
        let Some(confirmed) = confirm_variant_commit(
            &renderer,
            &name,
            snapshot_path.is_some(),
            source_digest.is_some(),
        )?
        else {
            return Ok(false);
        };
        if confirmed {
            *attempted_commit = true;
            let model_params: Vec<_> = command
                .probe_options
                .model_params
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect();
            let batch_params: Vec<_> = command
                .probe_options
                .batch_params
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect();
            let mut args = vec![
                "layer",
                "set",
                target,
                "--param",
                &parameter,
                "--model",
                &command.model,
                "--batch",
                &command.batch,
                "--variant",
                &name,
                "--require-forward",
                "--if-variant-base",
                &base,
                "--if-input-content",
                &digest,
            ];
            if let Some(source_digest) = &source_digest {
                args.extend(["--if-source-manifest", source_digest]);
            }
            for value in &model_params {
                args.extend(["--model-param", value]);
            }
            for value in &batch_params {
                args.extend(["--batch-param", value]);
            }
            if !command.probe_options.binding.is_empty() {
                args.extend(["--batch-binding", &command.probe_options.binding]);
            }
            if command.probe_options.training {
                args.push("--training");
            }
            if let Some(path) = snapshot_path {
                args.extend(["--replay-input-from", path]);
            }
            let mut commit_command = command.clone();
            commit_command.action = Action::Preview;
            commit_command.announce(&renderer, Some(target), false)?;
            let result = cli.execute(&args);
            let committed = result.as_ref().is_ok_and(|report| {
                report.operation == "edit"
                    && report.status == "success"
                    && report.data["committed"] == true
                    && report.data["variant_name"] == name
            });
            #[cfg(unix)]
            let historical_commit = if committed {
                locale.text(
                    "Historical · commit confirmed",
                    "历史 · 已确认提交",
                    "歷史 · 已確認提交",
                )
            } else {
                locale.text(
                    "Historical · commit unconfirmed",
                    "历史 · 提交未确认",
                    "歷史 · 提交未確認",
                )
            }
            .to_owned();
            #[cfg(unix)]
            let mut historical_status = locale
                .text("Recheck · not run", "重验 · 未执行", "重驗 · 未執行")
                .to_owned();
            let committed_base = result.as_ref().ok().and_then(|report| {
                let base = &report.data["variant_committed"];
                let token = base["token"].as_str()?;
                (base["name"] == name
                    && base["algorithm"] == "canonical-json-sha256-v1"
                    && token.len() == 64
                    && token
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
                .then(|| token.to_owned())
            });
            let mut outcome = renderer.result(&result);
            let mut recheck_json: Option<String> = None;
            if committed && let Some(committed_base) = committed_base {
                let mut recheck = vec![
                    "model",
                    "probe",
                    "--model",
                    &command.model,
                    "--batch",
                    &command.batch,
                    "--variant",
                    &name,
                    "--if-input-content",
                    &digest,
                    "--if-variant-base",
                    &committed_base,
                ];
                for value in &model_params {
                    recheck.extend(["--model-param", value]);
                }
                for value in &batch_params {
                    recheck.extend(["--batch-param", value]);
                }
                if !command.probe_options.binding.is_empty() {
                    recheck.extend(["--batch-binding", &command.probe_options.binding]);
                }
                if command.probe_options.training {
                    recheck.push("--training");
                }
                if let Some(path) = snapshot_path {
                    recheck.extend(["--replay-input-from", path]);
                }
                if !command.probe_options.fingerprint.is_empty() {
                    recheck.extend(["--batch-fingerprint", &command.probe_options.fingerprint]);
                }
                renderer.heading(
                    &mut outcome,
                    if snapshot_path.is_some() {
                        locale.text(
                            "Guarded recheck · captured input replay",
                            "条件重验 · 重放已保存输入",
                            "條件重驗 · 重放已保存輸入",
                        )
                    } else {
                        locale.text(
                            "Guarded recheck · regenerated input",
                            "条件重验 · 重新生成输入",
                            "條件重驗 · 重新產生輸入",
                        )
                    },
                );
                renderer.line(&mut outcome, locale.text(
                    "Recheck requires the committed variant content; source, weights and runtime are not pinned.",
                    "重验要求变体内容与提交一致；源码、权重和运行环境未固定。",
                    "重驗要求變體內容與提交一致；原始碼、權重和執行環境未固定。",
                ));
                let mut probe_command = command.clone();
                probe_command.action = Action::Probe;
                probe_command.announce(&renderer, None, true)?;
                let recheck_result = cli.execute(&recheck);
                recheck_json = recheck_result
                    .as_ref()
                    .ok()
                    .map(|report| report.raw_json.clone());
                #[cfg(unix)]
                {
                    historical_status = historical_recheck_status(
                        &recheck_result,
                        diagnosis_content,
                        command.probe_options.training,
                        snapshot_path.is_some(),
                        locale,
                    );
                }
                if snapshot_path.is_some()
                    && !recheck_result
                        .as_ref()
                        .is_ok_and(|report| report.data["batch_snapshot"]["status"] == "replayed")
                {
                    renderer.line(
                        &mut outcome,
                        locale.text(
                            "CLI did not confirm captured input replay for this recheck.",
                            "CLI 未确认本次重验重放了保存的输入。",
                            "CLI 未確認本次重驗重放了保存的輸入。",
                        ),
                    );
                }
                if recheck_result.as_ref().is_ok_and(|report| {
                    report.operation == "probe"
                        && report.status == "success"
                        && reported_training_mode(&report.data["observed_mode"])
                            != Some(command.probe_options.training)
                }) {
                    renderer.line(&mut outcome, locale.text(
                        "Recheck mode differs from the request or was not reported; the requested mode remains unverified.",
                        "重验模式与请求不符或未报告；请求的模式仍未验证。",
                        "重驗模式與請求不符或未報告；請求的模式仍未驗證。",
                    ));
                }
                if let Ok(report) = &recheck_result
                    && report.operation == "probe"
                    && report.status == "success"
                {
                    match report.data["evidence_cache"]["status"].as_str() {
                        Some("stored" | "not_stored") => {}
                        Some("reused") => renderer.line(
                            &mut outcome,
                            locale.text(
                                "Recheck reused prior evidence; a new forward was not confirmed.",
                                "重验复用了旧证据；未确认执行新的前向。",
                                "重驗重用了舊證據；未確認執行新的前向。",
                            ),
                        ),
                        _ => renderer.line(
                            &mut outcome,
                            locale.text(
                                "Recheck freshness was not reported; a new forward is unverified.",
                                "未报告重验新鲜度；新前向未确认。",
                                "未報告重驗新鮮度；新前向未確認。",
                            ),
                        ),
                    }
                    match report.data["trace_truncated"].as_bool() {
                        Some(false) => {}
                        Some(true) => renderer.line(&mut outcome, locale.text(
                            "Recheck call trace was truncated; recorded coverage is incomplete.",
                            "重验调用记录已截断；记录覆盖不完整。",
                            "重驗呼叫記錄已截斷；記錄覆蓋不完整。",
                        )),
                        None => renderer.line(&mut outcome, locale.text(
                            "Recheck call trace coverage was not reported.",
                            "未报告重验调用记录的覆盖情况。",
                            "未報告重驗呼叫記錄的覆蓋情況。",
                        )),
                    }
                    match compare_input_content(diagnosis_content, &report.data["batch_content"]) {
                        InputContentComparison::Same => {}
                        InputContentComparison::Different => renderer.line(&mut outcome, locale.text(
                            "Recheck input content differs from diagnosis; the sample is not verified.",
                            "重验输入内容与诊断不同；样本未核对。",
                            "重驗輸入內容與診斷不同；樣本未核對。",
                        )),
                        InputContentComparison::Unknown => renderer.line(&mut outcome, locale.text(
                            "Recheck input content cannot be compared with diagnosis.",
                            "无法核对重验输入内容与诊断是否相同。",
                            "無法核對重驗輸入內容與診斷是否相同。",
                        )),
                    }
                }
                outcome.push_str(&renderer.result(&recheck_result));
            } else if committed {
                renderer.line(
                    &mut outcome,
                    locale.text(
                        "Committed variant token unavailable; guarded recheck was not run.",
                        "无法获取已提交变体的标识；未执行条件重验。",
                        "無法取得已提交變體的識別；未執行條件重驗。",
                    ),
                );
            } else {
                renderer.line(
                    &mut outcome,
                    locale.text(
                        "Commit not confirmed; refresh project and variant state before retrying.",
                        "提交未确认；重试前请刷新项目和变体状态。",
                        "提交未確認；重試前請重新整理專案和變體狀態。",
                    ),
                );
            }
            let mut receipt = String::new();
            renderer.heading(
                &mut receipt,
                locale.text(
                    "Commit and recheck receipt",
                    "提交与重验回执",
                    "提交與重驗回執",
                ),
            );
            let version = cli.reported_version();
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text(
                        "CLI version queried after operations · executable not pinned",
                        "操作后查询的 CLI 版本 · 未固定可执行文件",
                        "操作後查詢的 CLI 版本 · 未固定可執行檔",
                    ),
                    version
                        .as_deref()
                        .unwrap_or(locale.text("unavailable", "不可用", "無法取得")),
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text("Model", "模型", "模型"),
                    text::plain_text(&command.model)
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text("Batch", "批次", "批次"),
                    text::plain_text(&command.batch)
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text("Variant", "变体", "變體"),
                    text::plain_text(&name)
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {} · {}",
                    locale.text("Target", "目标", "目標"),
                    text::plain_text(target),
                    text::plain_text(&parameter)
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text("Preview base", "预览基准", "預覽基準"),
                    base
                ),
            );
            renderer.line(
                &mut receipt,
                &format!(
                    "{}  {}",
                    locale.text("Input fingerprint", "输入指纹", "輸入指紋"),
                    digest
                ),
            );
            if let Some(token) = result
                .as_ref()
                .ok()
                .filter(|report| report.operation == "edit" && committed)
                .and_then(|report| report.data["variant_committed"]["token"].as_str())
            {
                renderer.line(
                    &mut receipt,
                    &format!(
                        "{}  {}",
                        locale.text(
                            "Committed variant token · CLI",
                            "已提交变体标识 · CLI",
                            "已提交變體識別 · CLI"
                        ),
                        text::plain_text(token)
                    ),
                );
            }
            receipt.push_str(&outcome);
            renderer.heading(
                &mut receipt,
                locale.text(
                    "Preview before commit · historical",
                    "提交前预览 · 历史",
                    "提交前預覽 · 歷史",
                ),
            );
            receipt.push_str(&content);
            append_receipt_report(
                &renderer,
                &mut receipt,
                locale.text("Diagnosis CLI JSON", "诊断 CLI JSON", "診斷 CLI JSON"),
                Some(&diagnosis_report.raw_json),
            );
            append_receipt_report(
                &renderer,
                &mut receipt,
                locale.text("Preview CLI JSON", "预览 CLI JSON", "預覽 CLI JSON"),
                preview_json.as_deref(),
            );
            append_receipt_report(
                &renderer,
                &mut receipt,
                locale.text("Commit CLI JSON", "提交 CLI JSON", "提交 CLI JSON"),
                result.as_ref().ok().map(|report| report.raw_json.as_str()),
            );
            append_receipt_report(
                &renderer,
                &mut receipt,
                locale.text("Recheck CLI JSON", "重验 CLI JSON", "重驗 CLI JSON"),
                recheck_json.as_deref(),
            );
            #[cfg(unix)]
            let mut saved_diff = saved_preview
                .as_ref()
                .and_then(|record| record.diff.clone());
            #[cfg(unix)]
            if let Some(diff) = &mut saved_diff {
                diff.attach_receipt(historical_commit, historical_status, &receipt, locale);
            }
            *saved_receipt = Some(PreviewRecord {
                content: receipt,
                #[cfg(unix)]
                diff: saved_diff.clone(),
            });
            #[cfg(unix)]
            if !line_mode && crate::terminal::menu::supported() {
                let shown = view_report_with_diff_state(&outcome, locale, style, &mut saved_diff)?;
                if let Some(receipt) = saved_receipt.as_mut() {
                    receipt.diff = saved_diff;
                }
                return Ok(shown);
            }
            print!("{outcome}");
        }
    }
    Ok(true)
}

fn append_receipt_report(renderer: &Renderer, output: &mut String, label: &str, raw: Option<&str>) {
    const LIMIT: usize = 65_536;
    renderer.heading(output, label);
    let Some(raw) = raw.filter(|raw| !raw.is_empty()) else {
        renderer.line(
            output,
            renderer
                .locale
                .text("Report unavailable.", "报告不可用。", "報告無法取得。"),
        );
        return;
    };
    let mut end = raw.len().min(LIMIT);
    while !raw.is_char_boundary(end) {
        end -= 1;
    }
    // Prefix each pre-wrapped line so the normal terminal wrapper cannot drop
    // leading spaces that are part of a JSON string.
    let width = renderer.width.saturating_sub(2).max(1);
    let mut line = String::new();
    let mut columns = 0usize;
    for ch in text::plain_text(&raw[..end]).chars() {
        if ch == '\n' {
            renderer.line(output, &format!("│ {line}"));
            line.clear();
            columns = 0;
            continue;
        }
        let char_width = text::display_width(&ch.to_string());
        if columns + char_width > width && !line.is_empty() {
            renderer.line(output, &format!("│ {line}"));
            line.clear();
            columns = 0;
        }
        line.push(ch);
        columns += char_width;
    }
    if !line.is_empty() {
        renderer.line(output, &format!("│ {line}"));
    }
    if end < raw.len() {
        renderer.line(
            output,
            &format!(
                "{} {end}/{} {}",
                renderer.locale.text(
                    "Report excerpt truncated:",
                    "报告摘录已截断：",
                    "報告摘錄已截斷："
                ),
                raw.len(),
                renderer.locale.text("bytes", "字节", "位元組")
            ),
        );
    } else {
        renderer.line(
            output,
            renderer.locale.text(
                "Complete report shown, wrapped for display.",
                "已显示完整报告，按屏宽换行。",
                "已顯示完整報告，依螢幕寬度換行。",
            ),
        );
    }
}

fn confirm_variant_commit(
    renderer: &Renderer,
    name: &str,
    replay: bool,
    source_guard: bool,
) -> io::Result<Option<bool>> {
    let locale = renderer.locale;
    let title = locale.text(
        "Commit guarded variant?",
        "提交条件变体？",
        "提交條件變體？",
    );
    let warning = if source_guard && replay {
        locale.text("Uses captured input and checks recorded project sources; external code, weights and runtime remain unpinned.", "使用保存的输入并核对记录的项目源码；外部代码、权重和运行环境未固定。", "使用儲存的輸入並核對記錄的專案原始碼；外部程式碼、權重和執行環境未固定。")
    } else if source_guard {
        locale.text("Regenerates input and checks recorded project sources; external code, weights and runtime remain unpinned.", "重新生成输入并核对记录的项目源码；外部代码、权重和运行环境未固定。", "重新產生輸入並核對記錄的專案原始碼；外部程式碼、權重和執行環境未固定。")
    } else if replay {
        locale.text(
            "Uses captured input; source, weights and runtime are not pinned.",
            "将使用保存的输入；源码、权重和运行环境未固定。",
            "將使用保存的輸入；原始碼、權重和執行環境未固定。",
        )
    } else {
        locale.text(
            "Inputs are regenerated; source, weights and runtime are not pinned.",
            "输入将重新生成；源码、权重和运行环境未固定。",
            "輸入將重新產生；原始碼、權重和執行環境未固定。",
        )
    };
    #[cfg(unix)]
    if !renderer.line_mode && crate::terminal::menu::supported() {
        let selection = crate::terminal::menu::select(&["b", "y"], |selected| {
            let mut output = String::new();
            renderer.heading(&mut output, title);
            renderer.line(
                &mut output,
                &format!(
                    "{}  {}",
                    locale.text("Variant", "变体", "變體"),
                    text::plain_text(name)
                ),
            );
            renderer.line(&mut output, warning);
            renderer.line(
                &mut output,
                &format!(
                    "{} {}",
                    if selected == 0 { ">" } else { " " },
                    locale.text("b. Cancel", "b. 取消", "b. 取消")
                ),
            );
            renderer.line(
                &mut output,
                &format!(
                    "{} {}",
                    if selected == 1 { ">" } else { " " },
                    locale.text("y. Commit", "y. 提交", "y. 提交")
                ),
            );
            output
        })?;
        return Ok(selection.map(|selected| selected == 1));
    }
    println!("{title}  {}\n{warning}", text::plain_text(name));
    Ok(prompt(locale.text(
        "y to commit, Enter to cancel",
        "输入 y 提交，回车取消",
        "輸入 y 提交，Enter 取消",
    ))?
    .map(|answer| answer.trim().eq_ignore_ascii_case("y")))
}

#[cfg(unix)]
fn view_report(content: &str, locale: Locale, style: Style) -> io::Result<bool> {
    view_report_with_diff(content, locale, style, None)
}

#[cfg(unix)]
fn view_report_with_diff(
    content: &str,
    locale: Locale,
    style: Style,
    mut diff: Option<crate::graph::diff::Diff>,
) -> io::Result<bool> {
    view_report_with_diff_state(content, locale, style, &mut diff)
}

#[cfg(unix)]
fn view_report_with_diff_state(
    content: &str,
    locale: Locale,
    style: Style,
    diff: &mut Option<crate::graph::diff::Diff>,
) -> io::Result<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let mut offset = 0;
    let mut show_diff = false;
    loop {
        if show_diff {
            let Some(action) = crate::terminal::menu::select(
                &["b", "d", "v", "\t", "n", "p", "[", "]", "g"],
                |_| {
                    diff.as_ref().unwrap().render(
                        crate::terminal::layout::viewport_width(),
                        crate::terminal::menu::rows(),
                        locale,
                    )
                },
            )?
            else {
                return Ok(false);
            };
            let diff = diff.as_mut().unwrap();
            match action {
                0 | 1 if diff.detail_open() => diff.toggle_details(),
                0 | 1 => show_diff = false,
                2 => diff.toggle_details(),
                3 if !diff.detail_open() => diff.change_side(),
                4 | 5 => diff.scroll(
                    crate::terminal::layout::viewport_width(),
                    crate::terminal::menu::rows(),
                    action == 4,
                    true,
                ),
                6 | 7 if !diff.detail_open() => {
                    diff.scroll_horizontal(crate::terminal::layout::viewport_width(), action == 7)
                }
                8 if !diff.detail_open() => diff.toggle_diagram(),
                _ => {}
            }
            continue;
        }
        let keys: &[&str] = if diff.is_some() {
            &["n", "p", "b", "d"]
        } else {
            &["n", "p", "b"]
        };
        let Some(action) = crate::terminal::menu::select(keys, |_| {
            let renderer = Renderer::current(locale, style, false);
            let visible = crate::terminal::menu::rows().saturating_sub(3).max(1);
            let start = offset.min(lines.len().saturating_sub(1));
            let mut screen = lines
                .iter()
                .skip(start)
                .take(visible)
                .copied()
                .collect::<Vec<_>>()
                .join("\n");
            screen.push('\n');
            renderer.line(
                &mut screen,
                &format!(
                    "{} {}/{}",
                    locale.text("Lines", "行", "行"),
                    start + 1,
                    lines.len()
                ),
            );
            renderer.line(
                &mut screen,
                if diff.is_some() {
                    locale.text(
                        "n Next   p Previous   d Difference   b Back",
                        "n 下一页   p 上一页   d 差异图   b 返回",
                        "n 下一頁   p 上一頁   d 差異圖   b 返回",
                    )
                } else {
                    locale.text(
                        "n Next   p Previous   b Back",
                        "n 下一页   p 上一页   b 返回",
                        "n 下一頁   p 上一頁   b 返回",
                    )
                },
            );
            screen
        })?
        else {
            return Ok(false);
        };
        let step = crate::terminal::menu::rows().saturating_sub(3).max(1);
        match action {
            0 => offset = (offset + step).min(lines.len().saturating_sub(1)),
            1 => offset = offset.saturating_sub(step),
            3 => show_diff = true,
            _ => return Ok(true),
        }
    }
}

#[cfg(unix)]
fn browse_details(items: &[Value], locale: Locale, style: Style) -> io::Result<bool> {
    let mut names: Vec<String> = (1..=items.len()).map(|index| index.to_string()).collect();
    names.push("b".to_owned());
    let keys: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut focus = 0;
    loop {
        let Some(index) = crate::terminal::menu::select_from(&keys, focus, |selected| {
            Renderer::current(locale, style, false).detail_list(
                items,
                selected,
                crate::terminal::menu::rows(),
            )
        })?
        else {
            return Ok(false);
        };
        let Some(item) = items.get(index) else {
            return Ok(true);
        };
        focus = index;
        let content = Renderer::current(locale, style, false).detail(item);
        if !view_report(&content, locale, style)? {
            return Ok(false);
        }
    }
}

pub(crate) fn reported_alias_effects(effect: Option<&Value>, locale: Locale) -> Vec<String> {
    let aliases = effect.and_then(|item| item["affected_aliases"].as_array());
    let mut lines = Vec::new();
    let Some(aliases) = aliases.filter(|aliases| !aliases.is_empty()) else {
        lines.push(
            locale
                .text(
                    "Affected aliases not reported",
                    "未报告受影响的别名",
                    "未報告受影響的別名",
                )
                .to_owned(),
        );
        return lines;
    };
    lines.push(
        locale
            .text(
                "Affected aliases · CLI report",
                "受影响别名 · CLI 报告",
                "受影響別名 · CLI 報告",
            )
            .to_owned(),
    );
    let mut incomplete = false;
    for alias in aliases {
        match alias.as_str().filter(|path| !path.trim().is_empty()) {
            Some(path) => lines.push(format!("  {}", text::plain_text(path))),
            None => incomplete = true,
        }
    }
    if incomplete {
        lines.push(
            locale
                .text(
                    "Some alias paths are unavailable",
                    "部分别名路径不可用",
                    "部分別名路徑無法取得",
                )
                .to_owned(),
        );
    }
    lines
}

pub(crate) fn path_tail(path: &str, width: usize) -> String {
    let path = text::plain_text(path).replace('\n', " ");
    if text::display_width(&path) <= width {
        return path;
    }
    if width <= 1 {
        return text::truncate(&path, width);
    }
    let mut start = path.len();
    for (index, _) in path.char_indices().rev() {
        if text::display_width(&path[index..]) + 1 > width {
            break;
        }
        start = index;
    }
    format!("…{}", &path[start..])
}

fn call_search_index(call: &Value) -> (String, bool) {
    fn visit(
        node: &Value,
        path: &str,
        depth: usize,
        count: &mut usize,
        limited: &mut bool,
        out: &mut String,
    ) {
        if depth > 16 || *count >= 4096 {
            *limited = true;
            return;
        }
        *count += 1;
        match node["kind"].as_str() {
            Some("dict") => {
                if let Some(items) = node["items"].as_object() {
                    for (key, value) in items {
                        if *count >= 4096 {
                            *limited = true;
                            break;
                        }
                        visit(
                            value,
                            &format!("{path}.{key}"),
                            depth + 1,
                            count,
                            limited,
                            out,
                        );
                    }
                }
            }
            Some("tuple" | "list") => {
                if let Some(items) = node["items"].as_array() {
                    for (index, value) in items.iter().enumerate() {
                        if *count >= 4096 {
                            *limited = true;
                            break;
                        }
                        visit(
                            value,
                            &format!("{path}[{index}]"),
                            depth + 1,
                            count,
                            limited,
                            out,
                        );
                    }
                }
            }
            Some(kind) => {
                out.push_str(&format!(" {path} {kind}"));
                if let Some(dtype) = node["dtype"].as_str() {
                    out.push_str(&format!(" {dtype}"));
                }
                if let Some(shape) = node["shape"].as_array() {
                    let dims: Vec<String> = shape
                        .iter()
                        .map(|v| v.as_i64().map_or("?".to_owned(), |n| n.to_string()))
                        .collect();
                    out.push_str(&format!(
                        " {} {} [{}]",
                        dims.join("×"),
                        dims.join("x"),
                        dims.join(",")
                    ));
                }
            }
            None => {}
        }
    }
    let mut out = String::new();
    for field in [
        "alias_paths",
        "paths",
        "module_type",
        "module_instance_id",
        "invocation_id",
        "invocation_index",
        "parent_invocation_id",
        "forward_id",
        "probe_id",
        "batch_id",
    ] {
        if let Some(value) = call.get(field) {
            out.push_str(&format!(" {value}"));
        }
    }
    let mut count = 0;
    let mut limited = false;
    visit(
        &call["inputs"],
        "inputs",
        0,
        &mut count,
        &mut limited,
        &mut out,
    );
    visit(
        &call["outputs"],
        "outputs",
        0,
        &mut count,
        &mut limited,
        &mut out,
    );
    (out.to_lowercase(), limited)
}

fn browse_recorded_calls(
    workbench: &mut Workbench,
    locale: Locale,
    style: Style,
    line_mode: bool,
) -> io::Result<bool> {
    let Some(calls) = workbench.recorded_calls().cloned() else {
        return Ok(true);
    };
    let Some(diagnosis) = &workbench.diagnosis else {
        return Ok(true);
    };
    let mut diagnostics = Vec::new();
    let mut collect_diagnostics = |report: &Report| {
        let raw: std::sync::Arc<str> = std::sync::Arc::from(report.raw_json.as_str());
        diagnostics.extend(
            report
                .diagnostics
                .iter()
                .cloned()
                .map(|item| ReportDiagnostic {
                    operation: report.operation.clone(),
                    item,
                    raw: raw.clone(),
                }),
        );
    };
    collect_diagnostics(&diagnosis.report);
    if workbench
        .last_command
        .as_ref()
        .is_some_and(|command| command.action != Action::Probe)
    {
        if let Some(Ok(report)) = &workbench.last_result {
            collect_diagnostics(report);
        }
    }
    let reused = crate::cli::reused_evidence(&diagnosis.report.data);
    let report_raw = diagnosis.report.raw_json.clone();
    let identity_report: Value = serde_json::from_str(&report_raw).unwrap_or(Value::Null);
    let last = diagnosis.report.data["failure_location"]["last_observed_module_call"].clone();
    let is_failure = diagnosis.report.status == "error";
    let trace_truncated = diagnosis.report.data["trace_truncated"] == true
        || diagnosis.report.data["calls_observed"]
            .as_u64()
            .is_some_and(|n| n > calls.len() as u64);
    let Some((_, modules)) = &workbench.inspected_structure else {
        return Ok(true);
    };
    let modules = modules.clone();
    let nesting = call_nesting(&calls);
    let index: Vec<(String, bool)> = calls.iter().map(call_search_index).collect();
    let limited = index.iter().any(|(_, limited)| *limited);
    let mut last_focus = CallSearchResult::Call(workbench.call_focus);
    loop {
        let renderer = Renderer::current(locale, style, line_mode);
        let query = workbench.call_query.trim().to_lowercase();
        let mut matches: Vec<CallSearchResult> = index
            .iter()
            .enumerate()
            .filter_map(|(i, (text, _))| text.contains(&query).then_some(CallSearchResult::Call(i)))
            .collect();
        if !query.is_empty() {
            matches.extend(
                diagnostics
                    .iter()
                    .enumerate()
                    .filter_map(|(i, diagnostic)| {
                        let text = format!(
                            "{} {} {} {}",
                            diagnostic.operation,
                            diagnostic.item.code,
                            diagnostic.item.message,
                            diagnostic.item.hint.as_deref().unwrap_or("")
                        )
                        .to_lowercase();
                        text.contains(&query)
                            .then_some(CallSearchResult::Diagnostic(i))
                    }),
            );
        }
        let focus = matches
            .iter()
            .position(|&result| result == last_focus)
            .unwrap_or(0);
        #[cfg(unix)]
        let selected = if !line_mode && crate::terminal::menu::supported() {
            let mut names: Vec<String> = matches.iter().map(|result| result.key()).collect();
            names.push("b".into());
            names.push("/".into());
            let keys: Vec<&str> = names.iter().map(String::as_str).collect();
            let Some(position) = crate::terminal::menu::select_from(&keys, focus, |selected| {
                Renderer::current(locale, style, false).call_list(
                    &calls,
                    &nesting,
                    &matches,
                    &diagnostics,
                    selected,
                    crate::terminal::menu::rows(),
                    &workbench.call_query,
                    limited,
                    trace_truncated,
                    reused,
                )
            })?
            else {
                return Ok(false);
            };
            Some(position)
        } else {
            print!(
                "{}",
                renderer.call_list(
                    &calls,
                    &nesting,
                    &matches,
                    &diagnostics,
                    focus,
                    matches.len().saturating_add(8),
                    &workbench.call_query,
                    limited,
                    trace_truncated,
                    reused,
                )
            );
            let Some(input) = prompt(locale.text(
                "Call number (@number opens even when filtered, / filter, b back, q quit)",
                "调用序号（@序号跨筛选打开，/ 筛选，b 返回，q 退出）",
                "呼叫序號（@序號跨篩選開啟，/ 篩選，b 返回，q 離開）",
            ))?
            else {
                return Ok(false);
            };
            if input.eq_ignore_ascii_case("q") {
                return Ok(false);
            }
            if input.eq_ignore_ascii_case("b") {
                return Ok(true);
            }
            if input == "/" {
                Some(matches.len() + 1)
            } else {
                select_line_call(&input, &mut matches, calls.len())
            }
        };
        #[cfg(not(unix))]
        let selected = {
            print!(
                "{}",
                renderer.call_list(
                    &calls,
                    &nesting,
                    &matches,
                    &diagnostics,
                    focus,
                    matches.len().saturating_add(8),
                    &workbench.call_query,
                    limited,
                    trace_truncated,
                    reused,
                )
            );
            let Some(input) = prompt(locale.text(
                "Call number (@number opens even when filtered, / filter, b back, q quit)",
                "调用序号（@序号跨筛选打开，/ 筛选，b 返回，q 退出）",
                "呼叫序號（@序號跨篩選開啟，/ 篩選，b 返回，q 離開）",
            ))?
            else {
                return Ok(false);
            };
            if input.eq_ignore_ascii_case("q") {
                return Ok(false);
            }
            if input.eq_ignore_ascii_case("b") {
                return Ok(true);
            }
            if input == "/" {
                Some(matches.len() + 1)
            } else {
                select_line_call(&input, &mut matches, calls.len())
            }
        };
        let Some(position) = selected else {
            continue;
        };
        if position == matches.len() {
            return Ok(true);
        }
        if position == matches.len() + 1 {
            #[cfg(unix)]
            let entered = if !line_mode && crate::terminal::menu::supported() {
                match crate::terminal::menu::input(&workbench.call_query, |value, cursor| {
                    let renderer = Renderer::current(locale, style, false);
                    let mut output = String::new();
                    renderer.heading(
                        &mut output,
                        locale.text("Filter recorded calls", "筛选已记录调用", "篩選已記錄呼叫"),
                    );
                    renderer.line(
                        &mut output,
                        &text::input_window("> ", value, cursor, renderer.width),
                    );
                    renderer.line(
                        &mut output,
                        locale.text(
                            "Enter filter · Ctrl+U clear · Esc cancel",
                            "Enter 筛选 · Ctrl+U 清空 · Esc 取消",
                            "Enter 篩選 · Ctrl+U 清空 · Esc 取消",
                        ),
                    );
                    output
                })? {
                    crate::terminal::menu::Input::Submit(value) => Some(value),
                    crate::terminal::menu::Input::Back => None,
                    crate::terminal::menu::Input::Quit => return Ok(false),
                }
            } else {
                prompt(locale.text(
                    "Filter (empty clears)",
                    "筛选（留空清除）",
                    "篩選（留空清除）",
                ))?
            };
            #[cfg(not(unix))]
            let entered = prompt(locale.text(
                "Filter (empty clears)",
                "筛选（留空清除）",
                "篩選（留空清除）",
            ))?;
            if let Some(value) = entered {
                workbench.call_query = value.trim().to_owned();
            }
            continue;
        }
        let Some(&result) = matches.get(position) else {
            continue;
        };
        last_focus = result;
        let index = match result {
            CallSearchResult::Call(index) => index,
            CallSearchResult::Diagnostic(index) => {
                let diagnostic = &diagnostics[index];
                let mut content = String::new();
                renderer.heading(
                    &mut content,
                    locale.text("Report diagnostic", "报告级诊断", "報告級診斷"),
                );
                renderer.line(
                    &mut content,
                    &format!(
                        "{}  {}",
                        locale.text("Source report", "来源报告", "來源報告"),
                        text::plain_text(&diagnostic.operation)
                    ),
                );
                renderer.line(
                    &mut content,
                    locale.text(
                        "Report-level diagnostic; no call association is asserted.",
                        "这是报告级诊断，未关联到具体调用。",
                        "這是報告級診斷，未關聯至具體呼叫。",
                    ),
                );
                renderer.line(
                    &mut content,
                    &format!(
                        "[{}] {}",
                        text::plain_text(&diagnostic.item.code),
                        text::plain_text(&diagnostic.item.message)
                    ),
                );
                if let Some(hint) = &diagnostic.item.hint {
                    renderer.line(&mut content, &text::plain_text(hint));
                }
                renderer.line(
                    &mut content,
                    locale.text("Original CLI report", "CLI 原始报告", "CLI 原始報告"),
                );
                renderer.line(&mut content, &diagnostic.raw);
                #[cfg(unix)]
                if !line_mode && crate::terminal::menu::supported() {
                    if !view_report(&content, locale, style)? {
                        return Ok(false);
                    }
                } else {
                    print!("{content}");
                }
                #[cfg(not(unix))]
                print!("{content}");
                continue;
            }
        };
        let Some(call) = calls.get(index) else {
            continue;
        };
        workbench.call_focus = index;
        let Some(path) = workbench.call_location(call).map(str::to_owned) else {
            let mut content = String::new();
            if reused {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Reused evidence · no new forward",
                        "复用证据 · 本次未执行前向",
                        "重用證據 · 本次未執行前向",
                    ),
                );
            }
            renderer.line(
                &mut content,
                locale.text(
                    "No unique module matches this call in the inspected snapshot.",
                    "已检查快照中没有唯一匹配此调用的模块。",
                    "已檢查快照中沒有唯一符合此呼叫的模組。",
                ),
            );
            content.push_str(&renderer.detail(call));
            if line_mode {
                renderer.identity_matches(&mut content, call, &identity_report);
            }
            #[cfg(unix)]
            if !line_mode && crate::terminal::menu::supported() {
                if !view_report(&content, locale, style)? {
                    return Ok(false);
                }
            } else {
                print!("{content}");
            }
            #[cfg(not(unix))]
            print!("{content}");
            continue;
        };
        let is_last = is_failure
            && !last.is_null()
            && &last == call
            && calls.iter().filter(|candidate| *candidate == &last).count() == 1;
        #[cfg(unix)]
        let selection = if !query.is_empty() {
            let (candidates, incomplete) = crate::graph::matching_call_inputs(call, &query);
            if !candidates.is_empty() && (candidates.len() > 1 || incomplete) {
                match choose_matching_input(&candidates, incomplete, locale, style, line_mode)? {
                    InputChoice::Selected(index) => Some(index),
                    InputChoice::Back => continue,
                    InputChoice::Quit => return Ok(false),
                }
            } else {
                candidates.first().map(|candidate| candidate.index)
            }
        } else {
            None
        };
        #[cfg(not(unix))]
        let selection = None;
        if line_mode {
            let mut content = String::new();
            if reused {
                renderer.line(
                    &mut content,
                    locale.text(
                        "Reused evidence · no new forward",
                        "复用证据 · 本次未执行前向",
                        "重用證據 · 本次未執行前向",
                    ),
                );
            }
            renderer.line(
                &mut content,
                &format!(
                    "{}  {path}",
                    locale.text("Matched module", "匹配模块", "符合模組")
                ),
            );
            #[cfg(unix)]
            if let Some(index) = selection {
                let (candidates, _) = crate::graph::matching_call_inputs(call, &query);
                if let Some(candidate) =
                    candidates.iter().find(|candidate| candidate.index == index)
                {
                    renderer.line(
                        &mut content,
                        &format!(
                            "{}  {} · {}",
                            locale.text("Selected input", "已选输入", "已選輸入"),
                            text::plain_text(&candidate.path),
                            text::plain_text(&candidate.shape)
                        ),
                    );
                }
            }
            renderer.line(
                &mut content,
                locale.text(
                    "Inspection snapshot identity is not verified against this call.",
                    "检查快照身份尚未与此调用核对。",
                    "檢查快照身分尚未與此呼叫核對。",
                ),
            );
            content.push_str(&renderer.detail(call));
            renderer.identity_matches(&mut content, call, &identity_report);
            print!("{content}");
        } else {
            let state = crate::graph::show_focused(
                &modules,
                &path,
                call,
                &report_raw,
                is_last,
                true,
                (!query.is_empty()).then_some(query.as_str()),
                selection,
                workbench.call_views.get(&index),
                false,
                locale,
            )?;
            workbench.call_views.insert(index, state);
        }
    }
}

#[cfg(unix)]
enum InputChoice {
    Selected(usize),
    Back,
    Quit,
}

#[cfg(unix)]
fn choose_matching_input(
    candidates: &[crate::graph::InputCandidate],
    incomplete: bool,
    locale: Locale,
    style: Style,
    line_mode: bool,
) -> io::Result<InputChoice> {
    fn render(
        renderer: &Renderer,
        candidates: &[crate::graph::InputCandidate],
        incomplete: bool,
        selected: usize,
        rows: usize,
    ) -> String {
        let mut output = String::new();
        renderer.heading(
            &mut output,
            renderer
                .locale
                .text("Choose matching input", "选择匹配的输入", "選擇相符的輸入"),
        );
        if incomplete {
            renderer.line(
                &mut output,
                &text::truncate(
                    renderer.locale.text(
                        "Input trace incomplete · recorded matches only",
                        "输入记录不完整 · 仅显示已记录项",
                        "輸入記錄不完整 · 僅顯示已記錄項",
                    ),
                    renderer.width,
                ),
            );
        }
        let visible = rows.saturating_sub(5 + usize::from(incomplete)).max(1);
        let start = selected
            .saturating_sub(visible - 1)
            .min(candidates.len().saturating_sub(visible));
        for (index, item) in candidates.iter().enumerate().skip(start).take(visible) {
            let prefix = format!(
                "{} {}. ",
                if index == selected { ">" } else { " " },
                index + 1
            );
            let metadata = item
                .metadata
                .strip_prefix("tensor · ")
                .unwrap_or(&item.metadata);
            let suffix = format!(" · {} {metadata}", item.shape);
            let budget = renderer
                .width
                .saturating_sub(text::display_width(&prefix) + text::display_width(&suffix));
            renderer.line(
                &mut output,
                &text::truncate(
                    &format!("{prefix}{}{suffix}", path_tail(&item.path, budget)),
                    renderer.width,
                ),
            );
        }
        renderer.line(
            &mut output,
            &format!(
                "{} {}",
                if selected == candidates.len() {
                    ">"
                } else {
                    " "
                },
                renderer.locale.text("b. Back", "b. 返回", "b. 返回")
            ),
        );
        renderer.line(
            &mut output,
            renderer.locale.text(
                "↑/↓ move · Enter choose · b back",
                "↑/↓ 移动 · Enter 选择 · b 返回",
                "↑/↓ 移動 · Enter 選擇 · b 返回",
            ),
        );
        output
    }
    if !line_mode && crate::terminal::menu::supported() {
        let mut names: Vec<String> = (1..=candidates.len())
            .map(|index| index.to_string())
            .collect();
        names.push("b".into());
        let keys: Vec<&str> = names.iter().map(String::as_str).collect();
        let Some(index) = crate::terminal::menu::select(&keys, |selected| {
            render(
                &Renderer::current(locale, style, false),
                candidates,
                incomplete,
                selected,
                crate::terminal::menu::rows(),
            )
        })?
        else {
            return Ok(InputChoice::Quit);
        };
        return Ok(candidates
            .get(index)
            .map_or(InputChoice::Back, |item| InputChoice::Selected(item.index)));
    }
    loop {
        print!(
            "{}",
            render(
                &Renderer::current(locale, style, true),
                candidates,
                incomplete,
                0,
                candidates.len().saturating_add(6)
            )
        );
        let Some(input) = prompt(locale.text(
            "Input number (b back, q quit)",
            "输入序号（b 返回，q 退出）",
            "輸入序號（b 返回，q 離開）",
        ))?
        else {
            return Ok(InputChoice::Quit);
        };
        if input.eq_ignore_ascii_case("b") {
            return Ok(InputChoice::Back);
        }
        if input.eq_ignore_ascii_case("q") {
            return Ok(InputChoice::Quit);
        }
        if let Some(candidate) = input
            .parse::<usize>()
            .ok()
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| candidates.get(index))
        {
            return Ok(InputChoice::Selected(candidate.index));
        }
    }
}

fn prompt(label: &str) -> io::Result<Option<String>> {
    eprint!("{label}: ");
    io::stderr().flush()?;
    let mut line = String::new();
    Ok((io::stdin().read_line(&mut line)? != 0).then(|| line.trim().to_owned()))
}

impl Renderer {
    fn preview_effect_summary(&self, report: &Report) -> String {
        let mut output = String::new();
        if let Some(name) = report.data["variant_name"].as_str() {
            self.line(
                &mut output,
                &format!(
                    "{}  {name}",
                    self.locale
                        .text("Candidate variant", "候选变体", "候選變體")
                ),
            );
        }
        if report.data["committed"] == false {
            self.line(
                &mut output,
                self.locale.text(
                    "Preview · not committed",
                    "预览 · 尚未提交",
                    "預覽 · 尚未提交",
                ),
            );
        }
        let base = &report.data["variant_base"];
        let token = base["token"].as_str().filter(|token| {
            *token == "absent"
                || (token.len() == 64
                    && token
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
        });
        let name = base["name"].as_str().filter(|name| !name.trim().is_empty());
        if base["algorithm"] == "canonical-json-sha256-v1"
            && name.is_some()
            && name == report.data["variant_name"].as_str()
            && token.is_some()
        {
            self.line(
                &mut output,
                &format!(
                    "{}  {} · {}",
                    self.locale
                        .text("Variant base · CLI", "变体基准 · CLI", "變體基準 · CLI"),
                    text::plain_text(name.unwrap()),
                    token.unwrap()
                ),
            );
            self.line(
                &mut output,
                self.locale.text(
                    "Variant base does not fix model source or sample.",
                    "变体基准不固定模型源码或样本。",
                    "變體基準不固定模型原始碼或樣本。",
                ),
            );
        } else {
            self.line(
                &mut output,
                self.locale.text(
                    "Variant base unavailable or unrecognized.",
                    "变体基准不可用或无法识别。",
                    "變體基準無法取得或無法辨識。",
                ),
            );
        }
        if let Some(effects) = report.data["edit_effects"].as_array() {
            for effect in effects {
                let path = effect["path"].as_str().unwrap_or("?");
                for (key, label) in [
                    (
                        "initialized",
                        self.locale.text("Initialized", "重新初始化", "重新初始化"),
                    ),
                    ("preserved", self.locale.text("Preserved", "保留", "保留")),
                ] {
                    if let Some(names) = effect[key].as_array() {
                        for name in names.iter().filter_map(Value::as_str) {
                            self.line(&mut output, &format!("{label}  {path}.{name}"));
                        }
                    }
                }
                if let Some(shared) = effect["preserved_external_state_sharing"].as_array() {
                    for entry in shared {
                        self.line(
                            &mut output,
                            &format!(
                                "{} {} · {}",
                                self.locale.text(
                                    "Shared state retained",
                                    "保留共享状态",
                                    "保留共享狀態"
                                ),
                                text::plain_text(entry["name"].as_str().unwrap_or("?")),
                                text::plain_text(&entry["external_paths"].to_string())
                            ),
                        );
                    }
                }
                for line in reported_alias_effects(Some(effect), self.locale) {
                    self.line(&mut output, &line);
                }
            }
            if effects.is_empty() {
                for line in reported_alias_effects(None, self.locale) {
                    self.line(&mut output, &line);
                }
            }
        } else {
            for line in reported_alias_effects(None, self.locale) {
                self.line(&mut output, &line);
            }
        }
        for (key, label) in [
            (
                "construction",
                self.locale.text("Construction", "构造检查", "建構檢查"),
            ),
            (
                "forward",
                self.locale
                    .text("Candidate forward", "候选前向检查", "候選前向檢查"),
            ),
        ] {
            let result = report.data["validation"][key]
                .as_str()
                .unwrap_or_else(|| self.locale.text("unknown", "未知", "未知"));
            self.line(&mut output, &format!("{label}  {result}"));
        }
        let mode = report.data["validation"]["mode"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(text::plain_text)
            .unwrap_or_else(|| self.locale.text("unknown", "未知", "未知").to_owned());
        self.line(
            &mut output,
            &format!(
                "{}  {mode}",
                self.locale.text(
                    "Candidate mode · CLI report",
                    "候选模式 · CLI 报告",
                    "候選模式 · CLI 報告"
                )
            ),
        );
        output
    }

    fn diagnosis_summary(
        &self,
        report: &Report,
        modules: Option<&Vec<Value>>,
        show_action: bool,
        requested_training: Option<bool>,
    ) -> String {
        let call = &report.data["failure_location"]["last_observed_module_call"];
        if call.is_null() {
            return String::new();
        }
        let mut output = String::new();
        self.heading(
            &mut output,
            self.locale
                .text("Last observed call", "最后观测到的调用", "最後觀測到的呼叫"),
        );
        self.call(&mut output, call);
        let mode = report.data["observed_mode"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(text::plain_text)
            .unwrap_or_else(|| self.locale.text("unknown", "未知", "未知").to_owned());
        self.line(
            &mut output,
            &format!(
                "{}  {mode}",
                self.locale.text("Observed mode", "观测模式", "觀測模式")
            ),
        );
        let inputs = &call["inputs"]["items"];
        let candidates = [
            (
                &inputs["args"]["items"][0],
                self.locale.text(
                    "Input 0 · observed",
                    "输入 0 · 实际观测",
                    "輸入 0 · 實際觀測",
                ),
            ),
            (
                &inputs["kwargs"]["items"]["input"],
                self.locale.text(
                    "Keyword input · observed",
                    "关键字 input · 实际观测",
                    "關鍵字 input · 實際觀測",
                ),
            ),
        ];
        let mut observed = false;
        for (input, label) in candidates {
            if input["kind"] != "tensor" {
                continue;
            }
            observed = true;
            let dimensions = input["shape"].as_array().map_or_else(
                || "?".to_owned(),
                |shape| {
                    if shape.is_empty() {
                        "[]".to_owned()
                    } else {
                        shape
                            .iter()
                            .map(Value::to_string)
                            .collect::<Vec<_>>()
                            .join(" × ")
                    }
                },
            );
            self.line(&mut output, &format!("{label}  {dimensions}"));
        }
        if !observed {
            self.line(
                &mut output,
                self.locale.text(
                    "Direct input tensor not recorded.",
                    "未记录直接输入张量。",
                    "未記錄直接輸入張量。",
                ),
            );
        }
        if call["module_type"] == "Linear" {
            let paths = call["alias_paths"].as_array();
            let mut matches = modules.into_iter().flatten().filter(|module| {
                paths.is_some_and(|paths| {
                    paths.iter().any(|path| {
                        path.as_str()
                            .is_some_and(|path| module["path"].as_str() == Some(path))
                    })
                }) && module["type"] == "Linear"
            });
            let first = matches.next();
            let expected = if matches.next().is_none() {
                first.and_then(|module| module["editable_params"]["in_features"].as_u64())
            } else {
                None
            };
            self.line(
                &mut output,
                &format!(
                    "{}  {}",
                    if self.width < 50 {
                        self.locale
                            .text("Snapshot input features", "快照输入维度", "快照輸入維度")
                    } else {
                        self.locale.text(
                            "Input features · inspected snapshot",
                            "输入维度 · 检查时的结构快照",
                            "輸入維度 · 檢查時的結構快照",
                        )
                    },
                    expected.map_or_else(|| "?".to_owned(), |value| value.to_string())
                ),
            );
            if show_action
                && !requested_training.is_some_and(|requested| {
                    mode_mismatch(&report.data["observed_mode"], requested)
                })
            {
                let hint = if preview_alias(call, modules.map(Vec::as_slice)).is_some() {
                    self.locale.text(
                        "Back, then choose Preview Linear input size.",
                        "返回后可选择预览 Linear 输入维度。",
                        "返回後可選擇預覽 Linear 輸入維度。",
                    )
                } else if modules.is_some() {
                    self.locale.text(
                        "Preview unavailable: call and inspected Linear are not uniquely matched. Inspect again.",
                        "无法预览：调用与已检查 Linear 未唯一匹配，请重新检查。",
                        "無法預覽：呼叫與已檢查 Linear 未唯一符合，請重新檢查。",
                    )
                } else {
                    self.locale.text(
                        "Preview unavailable: the call has no usable target path.",
                        "无法预览：调用没有可用的目标路径。",
                        "無法預覽：呼叫沒有可用的目標路徑。",
                    )
                };
                self.line(&mut output, hint);
            }
        }
        if show_action {
            self.line(
                &mut output,
                self.locale.text(
                    "Input tensor producer unavailable in this view.",
                    "此视图没有输入张量的生产者信息。",
                    "此畫面沒有輸入張量的產生者資訊。",
                ),
            );
            if let Some(parent_id) = call["parent_invocation_id"].as_str() {
                let mut parents = report.data["calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|candidate| candidate["invocation_id"].as_str() == Some(parent_id));
                if let Some(parent) = parents.next().filter(|_| parents.next().is_none()) {
                    self.heading(
                        &mut output,
                        self.locale.text(
                            "Parent call · execution nesting",
                            "上级调用 · 执行嵌套",
                            "上層呼叫 · 執行巢狀",
                        ),
                    );
                    self.call(&mut output, parent);
                } else {
                    self.line(
                        &mut output,
                        self.locale.text(
                            "Parent call details unavailable.",
                            "上级调用详情不可用。",
                            "上層呼叫詳情無法取得。",
                        ),
                    );
                }
            }
            if let Some(batch) = report.data["batch_id"].as_str() {
                self.line(
                    &mut output,
                    &format!(
                        "{}  {batch}",
                        self.locale.text("Observed batch", "观测批次", "觀測批次")
                    ),
                );
                if report.data["batch_identity_stable"] == false {
                    self.line(
                        &mut output,
                        self.locale.text(
                            "Batch identity is not stable across runs.",
                            "批次身份在不同运行之间不稳定。",
                            "批次身分在不同執行之間不穩定。",
                        ),
                    );
                }
            }
        }
        output
    }

    fn report(&self, report: &Report) -> String {
        let locale = self.locale;
        let mut output = String::new();
        let status = match report.status.as_str() {
            "success" => locale.text("Completed", "操作完成", "操作完成"),
            "error" => locale.text("Failed", "操作失败", "操作失敗"),
            _ => locale.text("Inconclusive", "尚无结论", "尚無結論"),
        };
        output.push('\n');
        self.heading(&mut output, status);
        for diagnostic in &report.diagnostics {
            self.line(
                &mut output,
                &format!("[{}] {}", diagnostic.code, diagnostic.message),
            );
            if let Some(hint) = &diagnostic.hint {
                self.line(&mut output, hint);
            }
        }

        match report.operation.as_str() {
            "assembly_generate" if report.status == "success" => {
                self.line(
                    &mut output,
                    &format!(
                        "{}: {}",
                        locale.text("Generated file", "生成文件", "產生檔案"),
                        text::plain_text(report.data["output"].as_str().unwrap_or("?"))
                    ),
                );
                self.line(
                    &mut output,
                    &format!(
                        "{}: {}",
                        locale.text(
                            "Generated code re-import and forward",
                            "生成代码重导入与前向",
                            "產生程式碼重匯入與前向"
                        ),
                        text::plain_text(
                            &serde_json::json!({
                                "construction":report.data["generated_validation"]["construction"],
                                "forward":report.data["generated_validation"]["forward"],
                                "weights":report.data["generated_validation"]["weights"],
                                "scope":report.data["generated_validation"]["scope"]
                            })
                            .to_string()
                        )
                    ),
                );
                self.line(
                    &mut output,
                    &format!(
                        "{}: {}",
                        locale.text(
                            "Observed project source check",
                            "已观察项目源码检查",
                            "已觀察專案原始碼檢查"
                        ),
                        text::plain_text(
                            report.data["source_evidence"]["status"]
                                .as_str()
                                .unwrap_or("unknown")
                        )
                    ),
                );
                self.line(
                    &mut output,
                    &format!(
                        "{}: {}",
                        locale.text("Validation inputs", "验证输入", "驗證輸入"),
                        text::plain_text(
                            report.data["input_comparison"]["status"]
                                .as_str()
                                .unwrap_or("unknown")
                        )
                    ),
                );
                self.line(&mut output, locale.text("Trained weights are not included; project code, packages and data remain required.", "不包含训练权重；仍需要项目代码、依赖包和数据。", "不含訓練權重；仍需要專案程式碼、相依套件和資料。"));
            }
            "assembly_add" | "assembly_set" | "assembly_remove" | "assembly_patch"
                if report.status == "success" && report.data["committed"] == true =>
            {
                self.line(
                    &mut output,
                    locale.text(
                        "Assembly saved; previous diagnosis has been cleared.",
                        "装配已保存；之前的诊断已清除。",
                        "裝配已儲存；先前的診斷已清除。",
                    ),
                );
                self.line(
                    &mut output,
                    &format!(
                        "Validation: {}",
                        text::plain_text(&report.data["validation"].to_string())
                    ),
                );
                self.line(
                    &mut output,
                    locale.text(
                        "Choose Probe a batch to inspect the new assembly.",
                        "选择前向诊断以检查新的装配。",
                        "選擇前向診斷以檢查新的裝配。",
                    ),
                );
            }
            "create" | "scaffold"
                if report.status == "success" && report.data["committed"] == true =>
            {
                self.line(
                    &mut output,
                    if report.operation == "scaffold" {
                        locale.text(
                            "Custom model project created.",
                            "自定义模型项目已创建。",
                            "自訂模型專案已建立。",
                        )
                    } else {
                        locale.text(
                            "Classification project created.",
                            "分类项目已创建。",
                            "分類專案已建立。",
                        )
                    },
                );
                for file in report.data["files"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    self.line(&mut output, &text::plain_text(file));
                }
                self.line(
                    &mut output,
                    locale.text(
                        "Choose Probe a batch to run the first forward.",
                        "选择前向诊断，运行首次前向。",
                        "選擇前向診斷，執行首次前向。",
                    ),
                );
            }
            "init" => {
                let state = match report.data["committed"].as_bool() {
                    Some(true) => {
                        locale.text("Project defaults saved", "项目入口已保存", "專案入口已儲存")
                    }
                    Some(false) => locale.text(
                        "Preview only · not saved",
                        "仅预览 · 尚未保存",
                        "僅預覽 · 尚未儲存",
                    ),
                    None => locale.text("Save state unavailable", "保存状态未知", "儲存狀態未知"),
                };
                self.line(&mut output, state);
                if let Some(path) = report.data["path"].as_str() {
                    self.line(&mut output, &text::plain_text(path));
                }
            }
            "inspect" => {
                if let Some(modules) = report.data["modules"].as_array() {
                    for module in modules {
                        self.line(
                            &mut output,
                            &format!(
                                "{}  {}",
                                module["path"].as_str().unwrap_or("—"),
                                module["type"].as_str().unwrap_or(""),
                            ),
                        );
                    }
                }
            }
            "probe" => self.probe(&mut output, &report.data),
            "doctor" if report.status == "success" => {
                for key in ["python_version", "torch_version"] {
                    self.line(
                        &mut output,
                        &format!("{key}  {}", report.data[key].as_str().unwrap_or("—")),
                    );
                }
                if let Some(capabilities) = report.data["capabilities"].as_array() {
                    for capability in capabilities {
                        if let (Some(id), Some(state)) =
                            (capability["id"].as_str(), capability["state"].as_str())
                        {
                            self.line(&mut output, &format!("{id}  {state}"));
                        }
                    }
                }
            }
            _ => {}
        }
        output
    }

    fn probe(&self, output: &mut String, data: &Value) {
        let locale = self.locale;
        match data["batch_snapshot"]["status"].as_str() {
            Some("captured") => self.line(
                output,
                locale.text(
                    "CLI reports input snapshot captured · temporary session file.",
                    "CLI 报告已保存输入快照 · 临时会话文件。",
                    "CLI 報告已儲存輸入快照 · 暫存工作階段檔案。",
                ),
            ),
            Some("unsupported") => self.line(
                output,
                locale.text(
                    "Input snapshot unsupported · replay unavailable; diagnosis still ran.",
                    "输入不支持快照 · 无法重放；诊断仍已执行。",
                    "輸入不支援快照 · 無法重放；診斷仍已執行。",
                ),
            ),
            Some("replayed") => self.line(
                output,
                locale.text(
                    "CLI reports captured input replayed.",
                    "CLI 报告已重放保存的输入。",
                    "CLI 報告已重放保存的輸入。",
                ),
            ),
            _ => {}
        }
        let mode = data["observed_mode"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(text::plain_text)
            .unwrap_or_else(|| locale.text("unknown", "未知", "未知").to_owned());
        self.line(
            output,
            &format!(
                "{}  {mode}",
                locale.text("Observed mode", "观测模式", "觀測模式")
            ),
        );
        if crate::cli::reused_evidence(data) {
            self.line(
                output,
                locale.text(
                    "Reused evidence · no new forward was run.",
                    "复用历史证据 · 本次未执行前向。",
                    "重用歷史證據 · 本次未執行前向。",
                ),
            );
        }
        if let Some(id) = data["batch_id"].as_str().filter(|id| !id.trim().is_empty()) {
            self.line(
                output,
                &format!(
                    "{}  {}",
                    locale.text("Reported batch ID", "报告批次 ID", "報告批次 ID"),
                    text::plain_text(id)
                ),
            );
            match data["batch_identity_stable"].as_bool() {
                Some(true) => self.line(output, locale.text("Stable batch identity reported; depends on the supplied fingerprint and sample IDs.", "报告为稳定批次身份；依赖提供的指纹和样本 ID。", "報告為穩定批次身分；依賴提供的指紋和樣本 ID。")),
                Some(false) => self.line(output, locale.text("Batch identity is not stable across runs.", "批次身份在不同运行之间不稳定。", "批次身分在不同執行之間不穩定。")),
                None => self.line(output, locale.text("Batch identity stability not reported.", "未报告批次身份是否稳定。", "未報告批次身分是否穩定。")),
            }
        }
        let location = &data["failure_location"];
        let last = &location["last_observed_module_call"];
        if !last.is_null() {
            self.heading(
                output,
                locale.text("Last observed call", "最后观测到的调用", "最後觀測到的呼叫"),
            );
            self.call(output, last);
        }
        if let Some(frame) = location["project_stack"]
            .as_array()
            .and_then(|frames| frames.last())
        {
            if let (Some(file), Some(line)) = (
                frame["file"].as_str().or_else(|| frame["path"].as_str()),
                frame["line"].as_u64(),
            ) {
                self.line(output, &format!("{file}:{line}"));
            }
        }
        if let Some(calls) = data["calls"].as_array() {
            self.heading(
                output,
                locale.text(
                    "Observed calls for this batch",
                    "本批次观测到的调用",
                    "此批次觀測到的呼叫",
                ),
            );
            for call in calls {
                self.call(output, call);
            }
        }
        if data["trace_truncated"] == true {
            self.line(
                output,
                locale.text(
                    "The trace was truncated.",
                    "调用记录已截断。",
                    "呼叫記錄已截斷。",
                ),
            );
        }
    }

    fn identity_matches(&self, output: &mut String, call: &Value, report: &Value) {
        let locale = self.locale;
        self.heading(
            output,
            locale.text("Earlier tensor matches", "较早的张量匹配", "較早的張量符合"),
        );
        if report["data"]["tensor_identity_trace"]["kind"] != "same_tensor_object_and_version" {
            self.line(
                output,
                locale.text(
                    "Identity evidence unavailable.",
                    "张量身份证据不可用。",
                    "張量身分證據不可用。",
                ),
            );
            return;
        }
        self.line(output, locale.text(
            "Partial observation of the same object and version; not proof of a producer or dataflow.",
            "相同对象及版本的部分观测；不能证明生产者或数据流。",
            "相同物件及版本的部分觀測；不能證明生產者或資料流。",
        ));
        let (matches, limited) = crate::graph::line_identity_matches(call, report);
        if matches.is_empty() {
            self.line(
                output,
                locale.text(
                    "No validated matches recorded; a source may still exist.",
                    "没有已核验的匹配记录；仍可能存在来源。",
                    "沒有已核驗的符合記錄；仍可能存在來源。",
                ),
            );
        }
        for observed in matches {
            let invocation = observed
                .invocation
                .map(|n| format!("#{n}"))
                .unwrap_or_else(|| "#?".into());
            self.line(
                output,
                &format!(
                    "  {} ← {} {} · {}{}{}",
                    text::plain_text(&observed.input),
                    text::plain_text(&observed.alias),
                    invocation,
                    text::plain_text(&observed.output),
                    if observed.ambiguous_alias {
                        locale.text(
                            " · shared instance; registration unknown",
                            " · 共享实例，注册路径未知",
                            " · 共享實例，註冊路徑未知",
                        )
                    } else {
                        ""
                    },
                    observed
                        .call_number
                        .map(|n| format!(" · @{n}"))
                        .unwrap_or_default()
                ),
            );
        }
        self.heading(
            output,
            locale.text("Later input matches", "后续输入匹配", "後續輸入符合"),
        );
        let (consumers, later_limited) = crate::graph::line_identity_consumers(call, report);
        if consumers.is_empty() {
            self.line(
                output,
                locale.text(
                    "No validated later matches recorded; others may exist.",
                    "没有已核验的后续匹配记录；仍可能存在其他匹配。",
                    "沒有已核驗的後續符合記錄；仍可能存在其他符合。",
                ),
            );
        }
        for observed in consumers {
            let invocation = observed
                .invocation
                .map(|n| format!("#{n}"))
                .unwrap_or_else(|| "#?".into());
            self.line(
                output,
                &format!(
                    "  {} → {} {} · {}{}{}",
                    text::plain_text(&observed.output),
                    text::plain_text(&observed.alias),
                    invocation,
                    text::plain_text(&observed.input),
                    if observed.ambiguous_alias {
                        locale.text(
                            " · shared instance; registration unknown",
                            " · 共享实例，注册路径未知",
                            " · 共享實例，註冊路徑未知",
                        )
                    } else {
                        ""
                    },
                    observed
                        .call_number
                        .map(|n| format!(" · @{n}"))
                        .unwrap_or_default()
                ),
            );
        }
        self.line(
            output,
            locale.text(
                "At the call list, enter @number to open a listed call, even with a filter.",
                "在调用列表输入 @序号可打开对应调用，即使启用了筛选。",
                "在呼叫清單輸入 @序號可開啟對應呼叫，即使啟用了篩選。",
            ),
        );
        if limited
            || later_limited
            || report["data"]["tensor_identity_trace"]["truncated"] == true
            || report["data"]["trace_truncated"] == true
        {
            self.line(
                output,
                locale.text(
                    "Recorded matches may be truncated.",
                    "匹配记录可能已截断。",
                    "符合記錄可能已截斷。",
                ),
            );
        }
    }

    fn call(&self, output: &mut String, call: &Value) {
        let paths = call["alias_paths"]
            .as_array()
            .map(|paths| paths.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let path = paths.first().copied().unwrap_or("—");
        let invocation = call["invocation_index"]
            .as_u64()
            .map(|index| format!("#{index}"))
            .unwrap_or_else(|| {
                self.locale
                    .text("call index unknown", "调用序号未知", "呼叫序號未知")
                    .to_owned()
            });
        self.line(
            output,
            &format!(
                "  {path}  {invocation}  {}",
                call["module_type"].as_str().unwrap_or("")
            ),
        );
        if crate::cli::ambiguous_registration(call) {
            self.line(
                output,
                self.locale.text(
                    "  Shared instance · invoked registration unknown",
                    "  共享实例 · 实际调用的注册路径未知",
                    "  共享實例 · 實際呼叫的註冊路徑未知",
                ),
            );
        }
        for alias in paths.iter().skip(1) {
            self.line(
                output,
                &format!("  {}  {alias}", self.locale.text("Alias", "别名", "別名")),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_reports_distinguish_complete_missing_and_bounded_excerpts() {
        let renderer = Renderer {
            locale: Locale::English,
            style: Style::Default,
            width: 120,
            line_mode: true,
        };
        let mut output = String::new();
        append_receipt_report(
            &renderer,
            &mut output,
            "Diagnosis CLI JSON",
            Some("{\"status\":\"error\"}"),
        );
        assert!(output.contains("{\"status\":\"error\"}"));
        assert!(output.contains("Complete report shown"));
        output.clear();
        append_receipt_report(&renderer, &mut output, "Recheck CLI JSON", None);
        assert!(output.contains("Report unavailable"));
        output.clear();
        let long = format!("{}界", "a".repeat(65_535));
        append_receipt_report(&renderer, &mut output, "Preview CLI JSON", Some(&long));
        assert!(output.contains("Report excerpt truncated: 65535/65538 bytes"));
        assert!(!output.contains("Complete report shown"));
    }

    #[test]
    fn line_identity_evidence_uses_validated_same_forward_matches() {
        let tensor = serde_json::json!({"kind":"tensor", "dtype":"float32", "shape":[2,4]});
        let call = serde_json::json!({
            "call_id":2,"forward_id":"f", "invocation_id":"f:2",
            "alias_paths":["model.head"], "invocation_index":1,
            "inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[tensor.clone()]}}},
            "input_tensor_matches":[{"input_path":"inputs.args[0]","prior_outputs":[{"invocation_id":"f:1","output_path":"outputs"}]}]
        });
        let mut report = serde_json::json!({"data":{
            "forward_id":"f","tensor_identity_trace":{"kind":"same_tensor_object_and_version","scope":"partial","truncated":true},
            "calls":[{"call_id":1,"forward_id":"f","invocation_id":"f:1","invocation_index":3,"alias_paths":["model.src","model.alias"],"outputs":tensor},call.clone()]
        }});
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            let renderer = Renderer {
                locale,
                style: Style::Default,
                width: 160,
                line_mode: true,
            };
            let mut output = String::new();
            renderer.identity_matches(&mut output, &call, &report);
            assert!(
                output.contains("inputs.args[0] ← model.src #3 · outputs"),
                "{output}"
            );
            assert!(output.contains(match locale {
                Locale::English => "shared instance",
                Locale::Simplified => "共享实例",
                Locale::Traditional => "共享實例",
            }));
            assert!(output.contains(match locale {
                Locale::English => "truncated",
                Locale::Simplified => "截断",
                Locale::Traditional => "截斷",
            }));
            assert!(output.contains("· @1"), "{output}");
        }
        let renderer = Renderer {
            locale: Locale::English,
            style: Style::Default,
            width: 160,
            line_mode: true,
        };
        let mut later = String::new();
        renderer.identity_matches(&mut later, &report["data"]["calls"][0], &report);
        assert!(
            later.contains("outputs → model.head #1 · inputs.args[0] · @2"),
            "{later}"
        );
        let mut filtered = vec![CallSearchResult::Call(1)];
        assert_eq!(select_line_call("@1", &mut filtered, 2), Some(1));
        assert!(matches!(filtered[1], CallSearchResult::Call(0)));
        assert_eq!(select_line_call("@3", &mut filtered, 2), None);
        assert_eq!(select_line_call("@0", &mut filtered, 2), None);
        let mut duplicate = report.clone();
        duplicate["data"]["calls"]
            .as_array_mut()
            .unwrap()
            .push(call.clone());
        let mut output = String::new();
        renderer.identity_matches(&mut output, &duplicate["data"]["calls"][0], &duplicate);
        assert!(!output.contains("outputs → model.head"));
        assert!(output.contains("No validated later matches"));
        report["data"]["calls"][0]["forward_id"] = "other".into();
        let renderer = Renderer {
            locale: Locale::English,
            style: Style::Default,
            width: 160,
            line_mode: true,
        };
        let mut output = String::new();
        renderer.identity_matches(&mut output, &call, &report);
        assert!(!output.contains("model.src"));
        assert!(output.contains("No validated matches recorded"));
        report["data"]["tensor_identity_trace"]["kind"] = "unknown".into();
        output.clear();
        renderer.identity_matches(&mut output, &call, &report);
        assert!(output.contains("Identity evidence unavailable"));
    }

    #[test]
    fn batch_parameter_json_rejects_ambiguous_or_invalid_names() {
        let parsed =
            parse_factory_parameters(r#"{"size":48,"config":{"name":"a b"},"enabled":true}"#)
                .unwrap();
        assert_eq!(parsed["size"], 48);
        assert_eq!(parsed["config"]["name"], "a b");
        for input in [
            "[]",
            "null",
            r#"{"x":1,"x":2}"#,
            r#"{"":1}"#,
            r#"{"a=b":1}"#,
            r#"{"x\n":1}"#,
            "{} {}",
        ] {
            assert!(parse_factory_parameters(input).is_err(), "{input}");
        }
    }

    #[test]
    fn mode_comparison_only_rejects_a_recognized_opposite_mode() {
        assert!(mode_mismatch(&serde_json::json!("eval"), true));
        assert!(mode_mismatch(&serde_json::json!("train"), false));
        assert!(!mode_mismatch(&serde_json::json!("train"), true));
        assert!(!mode_mismatch(&serde_json::json!("eval"), false));
        for mode in [
            Value::Null,
            serde_json::json!(""),
            serde_json::json!("other"),
        ] {
            assert!(!mode_mismatch(&mode, true));
            assert!(!mode_mismatch(&mode, false));
        }
    }

    #[cfg(unix)]
    #[test]
    fn historical_recheck_labels_only_the_reported_outcome() {
        let report = |operation: &str,
                      status: &str,
                      data: Value,
                      diagnostics: Value|
         -> io::Result<Report> {
            Ok(serde_json::from_value(serde_json::json!({"protocol_version":1,"operation":operation,"status":status,"data":data,"diagnostics":diagnostics})).unwrap())
        };
        let content = serde_json::json!({"status":"available", "algorithm":"sha256-forward-input-v1", "digest":"a".repeat(64)});
        let success = || {
            report(
                "probe",
                "success",
                serde_json::json!({"observed_mode":"eval","batch_content":content,"batch_snapshot":{"status":"replayed"},"evidence_cache":{"status":"not_stored"},"trace_truncated":false}),
                serde_json::json!([]),
            )
        };
        assert!(
            historical_recheck_status(&success(), &content, false, true, Locale::English)
                .contains("CLI reports success")
        );
        assert!(
            historical_recheck_status(&success(), &content, true, true, Locale::English)
                .contains("checks unverified")
        );
        let truncated = report(
            "probe",
            "success",
            serde_json::json!({"observed_mode":"eval","batch_content":content,"evidence_cache":{"status":"not_stored"},"trace_truncated":true}),
            serde_json::json!([]),
        );
        assert!(
            historical_recheck_status(&truncated, &content, false, false, Locale::English)
                .contains("trace truncated")
        );
        let missing_coverage = report(
            "probe",
            "success",
            serde_json::json!({"observed_mode":"eval","batch_content":content,"evidence_cache":{"status":"not_stored"}}),
            serde_json::json!([]),
        );
        assert!(
            historical_recheck_status(&missing_coverage, &content, false, false, Locale::English)
                .contains("coverage unknown")
        );
        let no_replay = report(
            "probe",
            "success",
            serde_json::json!({"observed_mode":"eval"}),
            serde_json::json!([]),
        );
        assert!(
            historical_recheck_status(&no_replay, &content, false, true, Locale::English)
                .contains("checks unverified")
        );
        assert!(
            historical_recheck_status(&no_replay, &content, false, false, Locale::English)
                .contains("checks unverified")
        );
        let reused = report(
            "probe",
            "success",
            serde_json::json!({"observed_mode":"eval","batch_content":content,"evidence_cache":{"status":"reused"}}),
            serde_json::json!([]),
        );
        assert!(
            historical_recheck_status(&reused, &content, false, false, Locale::English)
                .contains("checks unverified")
        );
        let failure = report(
            "probe",
            "error",
            serde_json::json!({}),
            serde_json::json!([{"code":"FORWARD_FAILED","message":"failed"}]),
        );
        assert!(
            historical_recheck_status(&failure, &content, false, false, Locale::English)
                .contains("forward failure")
        );
        let conflict = report(
            "probe",
            "error",
            serde_json::json!({}),
            serde_json::json!([{"code":"VARIANT_BASE_CONFLICT","message":"changed"}]),
        );
        assert!(
            historical_recheck_status(&conflict, &content, false, false, Locale::English)
                .contains("unverified")
        );
        let wrong = report(
            "edit",
            "success",
            serde_json::json!({"observed_mode":"eval"}),
            serde_json::json!([]),
        );
        assert!(
            historical_recheck_status(&wrong, &content, false, false, Locale::English)
                .contains("unverified")
        );
        let unavailable = Err(io::Error::new(io::ErrorKind::TimedOut, "timeout"));
        assert!(
            historical_recheck_status(&unavailable, &content, false, false, Locale::English)
                .contains("unavailable")
        );
    }

    #[test]
    fn input_content_comparison_requires_complete_matching_algorithm_and_digest() {
        let identity = |digest: &str| serde_json::json!({"status":"available", "algorithm":"sha256-forward-input-v1", "digest":digest});
        let a = identity(&"a".repeat(64));
        let b = identity(&"b".repeat(64));
        assert_eq!(compare_input_content(&a, &a), InputContentComparison::Same);
        assert_eq!(
            compare_input_content(&a, &b),
            InputContentComparison::Different
        );
        for invalid in [
            Value::Null,
            identity("not-a-digest"),
            serde_json::json!({"status":"unsupported","algorithm":"sha256-forward-input-v1"}),
            serde_json::json!({"status":"available","algorithm":"other","digest":"a".repeat(64)}),
        ] {
            assert_eq!(
                compare_input_content(&a, &invalid),
                InputContentComparison::Unknown
            );
        }
    }

    #[test]
    fn preview_summary_uses_reported_weight_effects() {
        let renderer = Renderer {
            locale: Locale::English,
            width: 100,
            style: Style::Default,
            line_mode: false,
        };
        let report: Report = serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "edit", "status": "success",
            "data": {"committed": false, "variant_name": "working", "edit_effects": [{
                "path": "model.head", "initialized": ["weight"], "preserved": ["bias"],
                "affected_aliases":["model.head", "model.alias"]
            }], "validation": {"construction": "passed", "forward": "passed", "mode":"train"}}
        }))
        .unwrap();
        let summary = renderer.preview_effect_summary(&report);
        assert!(summary.contains("Initialized  model.head.weight"));
        assert!(summary.contains("Preserved  model.head.bias"));
        assert!(summary.contains("Affected aliases · CLI report"));
        assert!(summary.contains("model.alias"));
        assert!(summary.contains("Preview · not committed"));
        assert!(summary.contains("Candidate forward  passed"));
        assert!(summary.contains("Candidate mode · CLI report  train"));
        assert!(summary.contains("Variant base unavailable or unrecognized"));
        let with_base: Report = serde_json::from_value(serde_json::json!({
            "protocol_version":1,"operation":"edit","status":"success",
            "data":{"committed":false,"variant_name":"working", "variant_base":{"name":"working", "token":"absent", "algorithm":"canonical-json-sha256-v1"}}
        })).unwrap();
        let base_summary = renderer.preview_effect_summary(&with_base);
        assert!(base_summary.contains("Variant base · CLI  working · absent"));
        assert!(base_summary.contains("does not fix model source or sample"));
        let wrong_name: Report = serde_json::from_value(serde_json::json!({
            "protocol_version":1,"operation":"edit","status":"success",
            "data":{"variant_name":"working", "variant_base":{"name":"other", "token":"absent", "algorithm":"canonical-json-sha256-v1"}}
        })).unwrap();
        assert!(
            renderer
                .preview_effect_summary(&wrong_name)
                .contains("Variant base unavailable or unrecognized")
        );
        let incomplete: Report = serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "edit", "status": "success", "data": {"committed": false}
        })).unwrap();
        let summary = renderer.preview_effect_summary(&incomplete);
        assert!(summary.contains("Construction  unknown"));
        assert!(summary.contains("Candidate forward  unknown"));
        assert!(summary.contains("Candidate mode · CLI report  unknown"));
        assert!(summary.contains("Affected aliases not reported"));
        let partial = serde_json::json!({"affected_aliases":["model.head",null,""]});
        let lines = reported_alias_effects(Some(&partial), Locale::English);
        assert!(lines.iter().any(|line| line.contains("model.head")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Some alias paths are unavailable"))
        );
    }

    #[test]
    fn candidate_state_effects_distinguish_missing_evidence_and_new_layers() {
        let renderer = Renderer {
            locale: Locale::English,
            width: 100,
            style: Style::Default,
            line_mode: true,
        };
        let mut output = String::new();
        render_candidate_state_effects(
            &renderer,
            &mut output,
            &serde_json::json!([
                {"action":"set", "path":"model.old", "preserved":["weight"], "old_state_not_copied":null},
                {"action":"add", "path":"model.seq", "name":"extra", "initialized":["weight"]}
            ]),
        );
        assert!(output.contains("Previous module state not copied · model.old: unknown"));
        assert!(output.contains("Candidate initialized state · model.seq.extra"));
        assert!(output.contains("not a saved training checkpoint"));
    }

    #[test]
    fn failure_summary_keeps_observation_and_snapshot_distinct() {
        let renderer = Renderer {
            locale: Locale::English,
            width: 100,
            style: Style::Default,
            line_mode: false,
        };
        let report: Report = serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "probe", "status": "error",
            "data": {"failure_location": {"last_observed_module_call": {
                "alias_paths": ["model.head"], "module_type": "Linear",
                "inputs": {"items": {"args": {"items": [{"kind": "tensor", "shape": [2, 48]}]}}}
            }}}
        }))
        .unwrap();
        let modules = vec![
            serde_json::json!({"path": "model.head", "type": "Linear", "editable_params": {"in_features": 32}}),
        ];
        let summary = renderer.diagnosis_summary(&report, Some(&modules), true, None);
        assert!(summary.contains("Input 0 · observed  2 × 48"));
        assert!(summary.contains("Input features · inspected snapshot  32"));
        assert!(
            renderer
                .diagnosis_summary(&report, None, true, None)
                .contains("Input features · inspected snapshot  ?")
        );
        let ambiguous = [modules.clone(), modules].concat();
        assert!(
            renderer
                .diagnosis_summary(&report, Some(&ambiguous), true, None)
                .contains("Input features · inspected snapshot  ?")
        );
    }

    #[test]
    fn preview_target_prefers_unique_inspected_linear_and_rejects_conflicts() {
        let call = serde_json::json!({"module_type":"Linear", "alias_paths":["missing.path", "model.head"]});
        let linear = serde_json::json!({"path":"model.head", "type":"Linear"});
        assert_eq!(preview_alias(&call, None), Some("missing.path"));
        assert_eq!(
            preview_alias(&call, Some(std::slice::from_ref(&linear))),
            Some("model.head")
        );
        let other = serde_json::json!({"path":"missing.path", "type":"Linear"});
        assert!(preview_alias(&call, Some(&[linear.clone(), other])).is_none());
        assert!(preview_alias(&call, Some(&[linear.clone(), linear])).is_none());
        assert!(
            preview_alias(
                &call,
                Some(&[serde_json::json!({"path":"model.head", "type":"ReLU"})])
            )
            .is_none()
        );
        assert!(preview_alias(&call, Some(&[])).is_none());
        let unusable = serde_json::json!({"alias_paths":["", null, "  "]});
        assert!(preview_alias(&unusable, None).is_none());
        let usable = serde_json::json!({"alias_paths":["", null, "model.head"]});
        assert_eq!(preview_alias(&usable, None), Some("model.head"));
    }

    #[test]
    fn failure_summary_reports_keyword_input_and_missing_shapes_honestly() {
        let mut report: Report = serde_json::from_value(serde_json::json!({
            "protocol_version":1, "operation":"probe", "status":"error",
            "data":{"failure_location":{"last_observed_module_call":{
                "alias_paths":["model.head"], "module_type":"Linear",
                "inputs":{"kind":"dict","items":{
                    "args":{"kind":"tuple","items":[]},
                    "kwargs":{"kind":"dict","items":{"input":{"kind":"tensor","shape":[2,48]}}}
                }}
            }}}
        }))
        .unwrap();
        let modules = vec![
            serde_json::json!({"path":"model.head", "type":"Linear", "editable_params":{"in_features":32}}),
        ];
        for (locale, observed, missing) in [
            (
                Locale::English,
                "Keyword input · observed",
                "Direct input tensor not recorded.",
            ),
            (
                Locale::Simplified,
                "关键字 input · 实际观测",
                "未记录直接输入张量。",
            ),
            (
                Locale::Traditional,
                "關鍵字 input · 實際觀測",
                "未記錄直接輸入張量。",
            ),
        ] {
            let renderer = Renderer {
                locale,
                width: 100,
                style: Style::Default,
                line_mode: false,
            };
            let summary = renderer.diagnosis_summary(&report, Some(&modules), true, None);
            assert!(
                summary.contains(&format!("{observed}  2 × 48")),
                "{summary}"
            );
            assert!(!summary.contains(missing), "{summary}");
            let input = &mut report.data["failure_location"]["last_observed_module_call"]["inputs"]
                ["items"]["kwargs"]["items"]["input"];
            input["shape"] = Value::Null;
            let summary = renderer.diagnosis_summary(&report, Some(&modules), true, None);
            assert!(summary.contains(&format!("{observed}  ?")), "{summary}");
            input_restore(&mut report);
            report.data["failure_location"]["last_observed_module_call"]["inputs"]["items"]["kwargs"]
                ["items"]["input"] = Value::Null;
            let summary = renderer.diagnosis_summary(&report, Some(&modules), true, None);
            assert!(summary.contains(missing), "{summary}");
            input_restore(&mut report);
        }

        fn input_restore(report: &mut Report) {
            report.data["failure_location"]["last_observed_module_call"]["inputs"]["items"]["kwargs"]
                ["items"]["input"] = serde_json::json!({"kind":"tensor","shape":[2,48]});
        }
    }

    #[test]
    fn failure_summary_labels_parent_as_nesting_and_handles_missing_or_ambiguous_parent() {
        let renderer = Renderer {
            locale: Locale::English,
            width: 100,
            style: Style::Default,
            line_mode: false,
        };
        let mut report: Report = serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "probe", "status": "error",
            "data": {"failure_location": {"last_observed_module_call": {
                "alias_paths": ["model.head"], "module_type": "Linear", "parent_invocation_id": "forward:call-1"
            }}, "calls": [{"invocation_id":"forward:call-1", "alias_paths":["model"], "module_type":"Net", "invocation_index":1}]}
        })).unwrap();
        let summary = renderer.diagnosis_summary(&report, None, true, None);
        assert!(summary.contains("Parent call · execution nesting"));
        assert!(summary.contains("model  #1  Net"));
        assert!(summary.contains("Input tensor producer unavailable in this view."));
        let duplicate = report.data["calls"][0].clone();
        report.data["calls"].as_array_mut().unwrap().push(duplicate);
        let summary = renderer.diagnosis_summary(&report, None, true, None);
        assert!(summary.contains("Parent call details unavailable."));
        assert!(!summary.contains("Parent call · execution nesting"));
    }

    #[test]
    fn inspected_structure_survives_probe_for_the_same_model() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
            inspected_structure: None,
        };
        let command = |action, model: &str| Command {
            action,
            model: model.to_owned(),
            batch: String::new(),
            variant: String::new(),
            probe_options: ProbeOptions::default(),
        };
        let report = |operation: &str, data: Value| {
            serde_json::from_value(serde_json::json!({
                "protocol_version": 1,
                "operation": operation,
                "status": "success",
                "data": data,
            }))
            .unwrap()
        };
        workbench.record_result(
            command(Action::Inspect, "a:build"),
            Ok(report(
                "inspect",
                serde_json::json!({"modules": [{"path": "model.head"}]}),
            )),
        );
        workbench.record_result(
            command(Action::Probe, "a:build"),
            Ok(report("probe", serde_json::json!({}))),
        );
        assert_eq!(
            workbench.inspected_structure.as_ref().unwrap().1[0]["path"],
            "model.head"
        );
        workbench.record_result(
            command(Action::Probe, "b:build"),
            Ok(report("probe", serde_json::json!({}))),
        );
        assert!(workbench.inspected_structure.is_none());
        workbench.record_result(
            command(Action::Inspect, ""),
            Ok(report(
                "inspect",
                serde_json::json!({"modules": [{"path": "model"}]}),
            )),
        );
        workbench.record_result(
            command(Action::Probe, ""),
            Ok(report("probe", serde_json::json!({}))),
        );
        assert!(workbench.inspected_structure.is_none());
    }

    #[test]
    fn batch_comparison_requires_two_nonempty_stable_ids() {
        let report = |id: Value, stable: Value| -> Report {
            serde_json::from_value(serde_json::json!({"protocol_version":1,"operation":"probe","status":"success","data":{"batch_id":id,"batch_identity_stable":stable}})).unwrap()
        };
        let a = report(serde_json::json!("batch-a"), serde_json::json!(true));
        let b = report(serde_json::json!("batch-b"), serde_json::json!(true));
        assert_eq!(compare_batch_ids(&a, &a), BatchComparison::Same);
        assert_eq!(compare_batch_ids(&a, &b), BatchComparison::Different);
        for uncertain in [
            report(serde_json::json!("batch-a"), serde_json::json!(false)),
            report(serde_json::json!(""), serde_json::json!(true)),
            report(Value::Null, Value::Null),
        ] {
            assert_eq!(compare_batch_ids(&a, &uncertain), BatchComparison::Unknown);
            assert_eq!(compare_batch_ids(&uncertain, &a), BatchComparison::Unknown);
        }
    }

    #[test]
    fn timed_out_retry_keeps_only_matching_explicit_diagnosis_and_disables_actions() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            inspected_structure: Some((
                "demo:build".into(),
                vec![serde_json::json!({"path":"model.head", "type":"Linear"})],
            )),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
        };
        let command = |model: &str, batch: &str| Command {
            action: Action::Probe,
            model: model.into(),
            batch: batch.into(),
            variant: String::new(),
            probe_options: ProbeOptions::default(),
        };
        let report = || {
            serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "probe", "status": "error",
            "data": {"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear"}}, "calls":[{"alias_paths":["model.head"]}]}
        })).unwrap()
        };
        workbench.record_result(command("demo:build", "demo:batch"), Ok(report()));
        let snapshot = tempfile::tempdir().unwrap();
        let snapshot_path = snapshot.path().to_owned();
        workbench.diagnosis.as_mut().unwrap().snapshot = Some(snapshot);
        assert!(workbench.preview_target().is_some());
        workbench.record_result(
            command("demo:build", "demo:batch"),
            Err(io::Error::new(io::ErrorKind::TimedOut, "timeout")),
        );
        assert!(workbench.diagnosis.is_some() && workbench.diagnosis_retry_issue.is_some());
        assert!(snapshot_path.exists());
        assert!(
            workbench.preview_target().is_none()
                && workbench.observed_location().is_none()
                && workbench.recorded_calls().is_none()
        );
        workbench.record_result(command("demo:build", "demo:batch"), Ok(report()));
        assert!(!snapshot_path.exists());
        workbench.record_result(
            command("demo:build", "demo:batch"),
            Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled")),
        );
        assert!(matches!(
            workbench.diagnosis_retry_issue,
            Some(RetryIssue::Cancelled)
        ));
        assert!(workbench.diagnosis.is_some() && workbench.preview_target().is_none());
        workbench.record_result(
            command("demo:build", "another:batch"),
            Err(io::Error::new(io::ErrorKind::TimedOut, "timeout")),
        );
        assert!(workbench.diagnosis.is_none() && workbench.diagnosis_retry_issue.is_none());
        workbench.record_result(command("demo:build", "demo:batch"), Ok(report()));
        let mut changed = command("demo:build", "demo:batch");
        changed.probe_options.fingerprint = "different-dataset".into();
        workbench.record_result(
            changed,
            Err(io::Error::new(io::ErrorKind::TimedOut, "timeout")),
        );
        assert!(workbench.diagnosis.is_none() && workbench.diagnosis_retry_issue.is_none());
        workbench.record_result(command("", ""), Ok(report()));
        workbench.record_result(
            command("", ""),
            Err(io::Error::new(io::ErrorKind::TimedOut, "timeout")),
        );
        assert!(workbench.diagnosis.is_none());
    }

    #[test]
    fn diagnosis_survives_other_commands_until_model_context_changes() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
            inspected_structure: None,
        };
        let command = |action, model: &str| Command {
            action,
            model: model.into(),
            batch: "fixture:batch".into(),
            variant: String::new(),
            probe_options: ProbeOptions::default(),
        };
        let report = |operation: &str, status: &str, data: Value| {
            serde_json::from_value(serde_json::json!({
                "protocol_version": 1, "operation": operation, "status": status, "data": data,
            }))
            .unwrap()
        };
        let modules = serde_json::json!({"modules":[{"path":"model.head", "type":"Linear"}]});
        workbench.record_result(
            command(Action::Inspect, "demo:build"),
            Ok(report("inspect", "success", modules.clone())),
        );
        workbench.record_result(command(Action::Probe, "demo:build"), Ok(report("probe", "error", serde_json::json!({
            "failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear"}},
            "calls":[{"invocation_id":"forward:call-1"}]
        }))));
        workbench.record_result(
            command(Action::Doctor, ""),
            Ok(report("doctor", "success", serde_json::json!({}))),
        );
        assert_eq!(workbench.preview_target(), Some("model.head"));
        assert_eq!(workbench.observed_location(), Some("model.head"));
        workbench.select(Action::LastDiagnosis);
        assert_eq!(workbench.displayed_report().unwrap().operation, "probe");
        assert_eq!(
            workbench.displayed_command().unwrap().batch,
            "fixture:batch"
        );
        assert_eq!(workbench.detail_items().unwrap().len(), 1);
        workbench.select(Action::LastReport);
        assert_eq!(workbench.displayed_report().unwrap().operation, "doctor");
        workbench.record_result(
            command(Action::Inspect, "demo:build"),
            Ok(report("inspect", "success", modules.clone())),
        );
        assert_eq!(workbench.preview_target(), Some("model.head"));
        workbench.record_result(
            command(Action::Inspect, "other:build"),
            Ok(report("inspect", "success", modules)),
        );
        assert!(workbench.diagnosis.is_none());
        assert!(workbench.observed_location().is_none());
    }

    #[test]
    fn observed_location_requires_a_unique_matching_snapshot() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
            inspected_structure: Some((
                "demo:build".into(),
                vec![serde_json::json!({"path":"model.head"})],
            )),
        };
        let probe: Report = serde_json::from_value(serde_json::json!({
            "protocol_version": 1, "operation": "probe", "status": "error",
            "data": {"failure_location": {"last_observed_module_call": {"alias_paths": ["model.head"]}},
                "calls":[{"alias_paths":["model.head"],"invocation_index":1},{"alias_paths":["model.head"],"invocation_index":2}]}
        })).unwrap();
        workbench.record_result(
            Command {
                action: Action::Probe,
                model: "demo:build".into(),
                batch: "fixture:batch".into(),
                variant: String::new(),
                probe_options: ProbeOptions::default(),
            },
            Ok(probe),
        );
        assert_eq!(workbench.observed_location(), Some("model.head"));
        assert_eq!(workbench.recorded_calls().unwrap().len(), 2);
        let call = workbench.recorded_calls().unwrap()[0].clone();
        assert_eq!(workbench.call_location(&call), Some("model.head"));
        assert!(
            workbench
                .call_location(&serde_json::json!({"alias_paths":["unknown"]}))
                .is_none()
        );
        workbench
            .inspected_structure
            .as_mut()
            .unwrap()
            .1
            .push(serde_json::json!({"path":"model.head"}));
        assert!(workbench.observed_location().is_none());
        assert!(workbench.call_location(&call).is_none());
        workbench.inspected_structure = None;
        assert!(workbench.observed_location().is_none());
    }

    #[test]
    fn preferences_round_trip_and_reject_unknown_values() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings/preferences.json");
        let initial = Preferences::load(&path).unwrap();
        assert!(initial.style == Style::Default && initial.preset == Preset::Default);
        Preferences {
            style: Style::Compact,
            preset: Preset::Shortcuts,
        }
        .save(&path)
        .unwrap();
        let restored = Preferences::load(&path).unwrap();
        assert!(restored.style == Style::Compact && restored.preset == Preset::Shortcuts);
        Preferences::default().save(&path).unwrap();
        assert!(Preferences::load(&path).unwrap().style == Style::Default);
        fs::write(&path, r#"{"preset":"shortcuts"}"#).unwrap();
        let (old, invalid) = Preferences::load_for_diagnosis(&path).unwrap();
        assert!(old.style == Style::Default && old.preset == Preset::Shortcuts);
        assert!(!invalid);
        fs::write(&path, r#"{"style":"unknown"}"#).unwrap();
        assert_eq!(
            Preferences::load(&path).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
        restored.save(&path).unwrap();
        assert!(Preferences::load(&path).unwrap() == restored);
        fs::write(&path, "{").unwrap();
        assert_eq!(
            Preferences::load(&path).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn shared_call_warns_that_registered_path_is_not_observed() {
        let call = serde_json::json!({"alias_paths":["model.first", "model.again"], "invocation_index":2, "module_type":"Linear"});
        let duplicate = serde_json::json!({"alias_paths":["model.first", "model.first"]});
        assert!(crate::cli::ambiguous_registration(&call));
        assert!(!crate::cli::ambiguous_registration(&duplicate));
        assert!(crate::cli::unique_call_registration(
            &duplicate,
            "model.first"
        ));
        for uncertain in [
            serde_json::json!({}),
            serde_json::json!({"alias_paths": []}),
            serde_json::json!({"alias_paths": ["model.first", null]}),
            call.clone(),
        ] {
            assert!(!crate::cli::unique_call_registration(
                &uncertain,
                "model.first"
            ));
        }
        assert!(!crate::cli::unique_call_registration(
            &duplicate,
            "model.other"
        ));
        let calls = vec![call.clone()];
        for (locale, list_hint, detail_hint) in [
            (
                Locale::English,
                "* Shared call · registration unknown",
                "Shared instance · invoked registration unknown",
            ),
            (
                Locale::Simplified,
                "* 共享调用 · 注册路径未知",
                "共享实例 · 实际调用的注册路径未知",
            ),
            (
                Locale::Traditional,
                "* 共享呼叫 · 註冊路徑未知",
                "共享實例 · 實際呼叫的註冊路徑未知",
            ),
        ] {
            let renderer = Renderer {
                locale,
                width: 80,
                style: Style::Default,
                line_mode: false,
            };
            let mut detail = String::new();
            renderer.call(&mut detail, &call);
            assert!(detail.contains(detail_hint), "{detail}");
            let frame = renderer.call_list(
                &calls,
                &call_nesting(&calls),
                &[CallSearchResult::Call(0)],
                &[],
                0,
                24,
                "",
                false,
                false,
                false,
            );
            assert!(
                frame.contains(list_hint) && frame.contains("1.* model.first"),
                "{frame}"
            );
        }
    }

    #[test]
    fn call_nesting_requires_unique_ordered_and_consistent_parent_identities() {
        let calls = vec![
            serde_json::json!({"invocation_id":"root", "parent_invocation_id":null,"forward_id":"f"}),
            serde_json::json!({"invocation_id":"child", "parent_invocation_id":"root","forward_id":"f"}),
            serde_json::json!({"invocation_id":"grand", "parent_invocation_id":"child","forward_id":"f"}),
            serde_json::json!({"invocation_id":"orphan", "parent_invocation_id":"missing"}),
            serde_json::json!({"invocation_id":"orphan-child", "parent_invocation_id":"orphan"}),
            serde_json::json!({"invocation_id":"dup"}),
            serde_json::json!({"invocation_id":"dup"}),
            serde_json::json!({"invocation_id":"ambiguous", "parent_invocation_id":"dup"}),
            serde_json::json!({"invocation_id":"conflict", "parent_invocation_id":"root","forward_id":"other"}),
            serde_json::json!({"invocation_id":"early", "parent_invocation_id":"later"}),
            serde_json::json!({"invocation_id":"later", "parent_invocation_id":"early"}),
        ];
        let nodes = call_nesting(&calls);
        assert_eq!((nodes[0].depth, nodes[0].incomplete), (0, false));
        assert_eq!((nodes[1].depth, nodes[1].incomplete), (1, false));
        assert_eq!((nodes[2].depth, nodes[2].incomplete), (2, false));
        assert_eq!((nodes[3].depth, nodes[3].incomplete), (0, true));
        assert_eq!((nodes[4].depth, nodes[4].incomplete), (1, true));
        assert!(
            nodes[7].incomplete
                && nodes[8].incomplete
                && nodes[9].incomplete
                && nodes[10].incomplete
        );
        assert!(call_search_index(&calls[2]).0.contains("child"));
    }

    #[test]
    fn call_search_indexes_observed_nested_paths_shapes_and_call_identity() {
        let call = serde_json::json!({
            "alias_paths":["model.编码器.head"], "module_type":"Linear", "module_instance_id":"module-7",
            "invocation_id":"forward-a:call-2", "invocation_index":2,
            "inputs":{"kind":"dict","items":{"kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","dtype":"torch.float32","shape":[2,64]}}}}}
        });
        let (text, limited) = call_search_index(&call);
        assert!(!limited);
        for term in [
            "编码器",
            "linear",
            "module-7",
            "forward-a:call-2",
            "inputs.kwargs.mask",
            "torch.float32",
            "2×64",
            "2x64",
            "[2,64]",
        ] {
            assert!(text.contains(term), "missing {term}: {text}");
        }
        assert!(!text.contains("producer"));
        let many = serde_json::json!({"inputs":{"kind":"list","items":vec![serde_json::json!({"kind":"tensor","shape":[1]}); 4100]}});
        assert!(call_search_index(&many).1);
    }

    #[test]
    fn missing_parent_is_visible_in_the_call_list_without_a_claim_of_tensor_flow() {
        let calls = vec![
            serde_json::json!({"invocation_id":"child","parent_invocation_id":"missing", "alias_paths":["model.child"], "invocation_index":1}),
        ];
        let nesting = call_nesting(&calls);
        let renderer = Renderer {
            locale: Locale::English,
            width: 40,
            style: Style::Default,
            line_mode: false,
        };
        let frame = renderer.call_list(
            &calls,
            &nesting,
            &[CallSearchResult::Call(0)],
            &[],
            0,
            24,
            "",
            false,
            true,
            false,
        );
        assert!(frame.contains("> 1. ? model.child"), "{frame}");
        assert!(frame.contains("CLI call trace is truncated"), "{frame}");
        assert!(
            frame.contains("Nesting does not establish tensor flow"),
            "{frame}"
        );
        assert!(frame.lines().all(|line| text::display_width(line) <= 40));
    }

    #[test]
    fn call_list_keeps_original_numbers_and_distinguishes_long_paths() {
        let calls = vec![
            serde_json::json!({"alias_paths":["model.extremely_long_common_prefix.first_head"], "invocation_index":1}),
            serde_json::json!({"alias_paths":["model.extremely_long_common_prefix.second_head"], "invocation_index":2}),
        ];
        let renderer = Renderer {
            locale: Locale::English,
            width: 38,
            style: Style::Default,
            line_mode: false,
        };
        let output = renderer.call_list(
            &calls,
            &[],
            &[CallSearchResult::Call(1)],
            &[],
            0,
            24,
            "second",
            false,
            false,
            false,
        );
        assert!(output.contains("> 2."), "{output}");
        assert!(output.contains("second_head"), "{output}");
        assert!(output.contains("#2"), "{output}");
        assert!(!output.contains("first_head"), "{output}");
        assert!(output.lines().all(|line| text::display_width(line) <= 38));
        let truncated =
            renderer.call_list(&calls, &[], &[], &[], 0, 24, "missing", true, true, false);
        assert!(truncated.contains("CLI call trace is truncated"));
        assert!(truncated.contains("4096"));
        assert!(truncated.contains("No matching results"));
        assert!(truncated.lines().count() <= 24);
    }

    #[test]
    fn configuration_retry_only_keeps_evidence_for_a_verified_matching_revision() {
        let report = |revision: Option<&str>| -> io::Result<Report> {
            Ok(serde_json::from_value(serde_json::json!({
            "protocol_version":1, "operation":"config_history", "status":"success", "data":{"current_revision":revision}
        })).unwrap())
        };
        assert!(same_project_configuration(
            &report(Some("v1")),
            &report(Some("v1"))
        ));
        assert!(!same_project_configuration(
            &report(Some("v1")),
            &report(Some("v2"))
        ));
        assert!(!same_project_configuration(&report(None), &report(None)));
        let unconfigured = || -> io::Result<Report> {
            Ok(serde_json::from_value(serde_json::json!({
                "protocol_version":1, "operation":"config_history", "status":"error",
                "diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"missing"}]
            }))
            .unwrap())
        };
        assert!(same_project_configuration(&unconfigured(), &unconfigured()));
        assert!(!same_project_configuration(
            &unconfigured(),
            &report(Some("v1"))
        ));
    }

    #[test]
    fn command_search_matches_localized_text_and_current_keys() {
        let bindings = [
            Binding {
                key: "x",
                action: Action::Inspect,
            },
            Binding {
                key: "7",
                action: Action::Preview,
            },
        ];
        let page = MenuPage {
            project: Path::new("/project"),
            project_state: ProjectState::Configured,
            variant: "",
            bindings: &bindings,
            context: None,
            has_report: false,
            has_diagnosis: false,
            diagnosis_retry_issue: None,
            has_saved_preview: false,
            has_details: false,
            has_structure: false,
            has_preview: false,
            has_location: false,
            has_calls: false,
        };
        for (locale, query) in [
            (Locale::Simplified, "预览"),
            (Locale::Traditional, "預覽"),
            (Locale::English, "PREVIEW"),
        ] {
            let matches = matching_workbench_commands(&page, locale, query);
            assert_eq!(matches.len(), 1);
            assert!(matches[0].action == Action::Preview);
            assert!(!page.available(matches[0].action));
        }
        let matches = matching_workbench_commands(&page, Locale::English, " X ");
        assert_eq!(matches.len(), 1);
        assert!(matches[0].action == Action::Inspect);
        assert!(matching_workbench_commands(&page, Locale::English, "no-such-command").is_empty());
    }

    #[test]
    fn menu_renders_active_bindings_within_terminal_width() {
        let bindings = [Binding {
            key: "p",
            action: Action::Probe,
        }];
        let page = MenuPage {
            project: std::path::Path::new("/a/very/long/project/directory"),
            project_state: ProjectState::Unconfigured,
            variant: "",
            bindings: &bindings,
            context: None,
            has_report: false,
            has_diagnosis: false,
            diagnosis_retry_issue: None,
            has_saved_preview: false,
            has_details: false,
            has_structure: false,
            has_preview: false,
            has_location: false,
            has_calls: false,
        };
        let renderer = Renderer {
            locale: Locale::English,
            width: 20,
            style: Style::Default,
            line_mode: false,
        };
        let menu = renderer.menu(&page, 0);
        assert!(menu.contains("p. Probe a batch"));
        assert!(page.action("P") == Some(Action::Probe));
        assert!(page.action("") == Some(Action::Probe));
        assert!(page.action("2").is_none());
        assert!(menu.lines().all(|line| text::display_width(line) <= 20));
    }

    #[test]
    fn menu_enter_uses_the_recommended_available_action() {
        let mut page = MenuPage {
            project: Path::new("/project"),
            project_state: ProjectState::Unconfigured,
            variant: "",
            bindings: &DEFAULT_BINDINGS,
            context: None,
            has_report: false,
            has_diagnosis: false,
            diagnosis_retry_issue: None,
            has_saved_preview: false,
            has_details: false,
            has_structure: false,
            has_preview: false,
            has_location: false,
            has_calls: false,
        };
        assert!(page.action("") == Some(Action::Init));
        page.project_state = ProjectState::Configured;
        assert!(page.action("") == Some(Action::Probe));
        page.has_diagnosis = true;
        assert!(page.action("") == Some(Action::LastDiagnosis));
        page.diagnosis_retry_issue = Some(RetryIssue::TimedOut);
        assert!(page.action("") == Some(Action::Probe));
        page.diagnosis_retry_issue = None;
        page.project_state = ProjectState::NeedsReview;
        assert!(page.action("") == Some(Action::ProjectStatus));
        page.project_state = ProjectState::Unavailable;
        assert!(page.action("") == Some(Action::ProjectStatus));
    }

    #[test]
    fn menu_scrolls_with_focus_and_shows_session_context() {
        let context = ("fixture:build".to_owned(), Some("fixture:batch".to_owned()));
        for bindings in [&DEFAULT_BINDINGS[..], &SHORTCUT_BINDINGS[..]] {
            let page = MenuPage {
                project: Path::new("/project"),
                project_state: ProjectState::Unconfigured,
                variant: "",
                bindings,
                context: Some(&context),
                has_report: true,
                has_diagnosis: true,
                diagnosis_retry_issue: None,
                has_saved_preview: true,
                has_details: true,
                has_structure: true,
                has_preview: true,
                has_location: true,
                has_calls: true,
            };
            for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
                for (width, height) in [(38, 16), (38, 24), (98, 30)] {
                    for style in [Style::Default, Style::Compact, Style::Story] {
                        let renderer = Renderer {
                            locale,
                            width,
                            style,
                            line_mode: false,
                        };
                        for (selected, binding) in bindings
                            .iter()
                            .filter(|binding| page.available(binding.action))
                            .enumerate()
                        {
                            let output = renderer.menu_with_rows(&page, selected, height);
                            assert!(
                                output.contains("fixture:build")
                                    && output.contains("fixture:batch"),
                                "{output}"
                            );
                            assert!(output.contains(&format!("> {}.", binding.key)), "{output}");
                            assert!(output.lines().count() <= height, "{output}");
                            assert!(
                                output
                                    .lines()
                                    .all(|line| text::display_width(line) <= width),
                                "{output}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn variant_list_keeps_selection_visible_in_a_small_terminal() {
        let variants: Vec<_> = (0..100)
            .map(|index| VariantChoice {
                name: format!("branch-{index}"),
                revision: Some("revision-123456789".into()),
                selectable: true,
            })
            .collect();
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            let renderer = Renderer {
                locale,
                width: 38,
                style: Style::Default,
                line_mode: false,
            };
            let output = variant_screen(&renderer, &variants, 91, 10);
            assert!(output.contains("> 91. branch-90"), "{output}");
            assert!(output.lines().count() <= 10, "{output}");
            assert!(
                output.lines().all(|line| text::display_width(line) <= 38),
                "{output}"
            );
        }
    }

    #[test]
    fn workbench_keymaps_validate_conflicts_and_preserve_other_preferences() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        fs::write(
            &path,
            r#"{"other_feature":{"kept":true},"preset":"default"}"#,
        )
        .unwrap();
        let preferences = Preferences::default();
        let mut maps = WorkbenchKeymaps::default();
        maps.default.insert("inspect".into(), "v".into());
        let binding = maps
            .bindings(Preset::Default)
            .unwrap()
            .into_iter()
            .find(|item| item.action == Action::Inspect)
            .unwrap();
        assert_eq!(binding.key, "v");
        assert_eq!(
            maps.bindings(Preset::Shortcuts)
                .unwrap()
                .into_iter()
                .find(|item| item.action == Action::Inspect)
                .unwrap()
                .key,
            "i"
        );
        maps.save(&path, preferences).unwrap();
        let stored: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(stored["other_feature"]["kept"], true);
        assert_eq!(
            WorkbenchKeymaps::load(&path)
                .unwrap()
                .bindings(Preset::Default)
                .unwrap()[0]
                .key,
            "v"
        );
        let mut changed_style = preferences;
        changed_style.cycle_style();
        changed_style.save(&path).unwrap();
        assert_eq!(
            WorkbenchKeymaps::load(&path)
                .unwrap()
                .bindings(Preset::Default)
                .unwrap()[0]
                .key,
            "v"
        );
        maps.default.insert("inspect".into(), "2".into());
        assert!(maps.bindings(Preset::Default).is_err());
        maps.default.insert("inspect".into(), "b".into());
        assert!(maps.bindings(Preset::Default).is_err());
        maps.default.clear();
        maps.default.insert("quit".into(), "x".into());
        assert!(maps.bindings(Preset::Default).is_err());
        maps.default.clear();
        maps.default.insert("unknown".into(), "x".into());
        assert!(maps.bindings(Preset::Default).is_err());
        fs::write(
            &path,
            r#"{"other_feature":{"kept":true},"workbench_keys":{"default":{"inspect":"2"}}}"#,
        )
        .unwrap();
        assert_eq!(
            WorkbenchKeymaps::load(&path).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
        WorkbenchKeymaps::default()
            .save(&path, preferences)
            .unwrap();
        let restored: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(restored["other_feature"]["kept"], true);
        assert!(WorkbenchKeymaps::load(&path).is_ok());
    }

    #[test]
    fn menu_and_settings_fit_common_terminal_sizes() {
        let page = MenuPage {
            project: Path::new("/very/long/模型/project/name/that/needs/truncation"),
            project_state: ProjectState::NeedsReview,
            variant: "",
            bindings: &DEFAULT_BINDINGS,
            context: None,
            has_report: true,
            has_diagnosis: true,
            diagnosis_retry_issue: None,
            has_saved_preview: true,
            has_details: true,
            has_structure: true,
            has_preview: false,
            has_location: false,
            has_calls: false,
        };
        for width in [38, 78, 98] {
            for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
                for style in [Style::Default, Style::Compact, Style::Story] {
                    let renderer = Renderer {
                        locale,
                        width,
                        style,
                        line_mode: false,
                    };
                    for screen in [
                        renderer.menu(&page, 1),
                        renderer.settings(Preset::Default, "", LanguageChoice::System, 1),
                    ] {
                        assert!(screen.lines().count() <= 24, "{screen}");
                        assert!(
                            screen
                                .lines()
                                .all(|line| text::display_width(line) <= width),
                            "{screen}"
                        );
                        assert!(screen.contains("> 2."), "{screen}");
                    }
                    let style_cue = match style {
                        Style::Default => "› ─",
                        Style::Compact => "▸ ┄",
                        Style::Story => "◆ ━",
                    };
                    let settings =
                        renderer.settings(Preset::Default, "", LanguageChoice::System, 0);
                    assert!(settings.contains(style_cue), "{settings}");
                    for option in [
                        "› ─",
                        locale.text("Default", "默认", "預設"),
                        "▸ ┄",
                        locale.text("Compact", "紧凑", "緊湊"),
                        "◆ ━",
                        locale.text("Story", "叙事", "敘事"),
                    ] {
                        assert!(settings.contains(option), "{settings}");
                    }
                    let back = locale.text("Back", "返回", "返回");
                    assert!(
                        settings.contains("Enter") && settings.contains(back),
                        "{settings}"
                    );
                }
            }
        }
    }

    #[test]
    fn detail_list_keeps_focused_entry_visible_in_narrow_view() {
        let items: Vec<Value> = (0..100)
            .map(|index| serde_json::json!({"path": format!("model.branch.layer{index}"), "type": "Linear"}))
            .collect();
        let renderer = Renderer {
            locale: Locale::English,
            width: 38,
            style: Style::Default,
            line_mode: false,
        };
        let output = renderer.detail_list(&items, 92, 24);
        assert!(output.contains("> 93. model.branch.layer92"));
        assert!(output.contains("Linear"));
        assert!(output.lines().count() <= 24);
        assert!(output.lines().all(|line| text::display_width(line) <= 38));
    }

    #[test]
    fn form_context_reuses_batch_only_for_its_model() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
            inspected_structure: None,
        };
        let command = |action, model: &str, batch: &str| Command {
            action,
            model: model.into(),
            batch: batch.into(),
            variant: String::new(),
            probe_options: ProbeOptions::default(),
        };
        workbench.record_result(
            command(Action::Probe, "demo:build", "demo:batch"),
            Err(io::Error::other("probe failed")),
        );
        workbench.record_result(
            command(Action::Doctor, "", ""),
            Err(io::Error::other("doctor failed")),
        );
        workbench.select(Action::Probe);
        assert!(matches!(&workbench.page, Page::Model { draft, .. } if draft == "demo:build"));
        workbench.submit("demo:build".into());
        assert!(matches!(&workbench.page, Page::Batch { draft, .. } if draft == "demo:batch"));
        workbench.select(Action::Probe);
        workbench.submit("other:build".into());
        assert!(matches!(&workbench.page, Page::Batch { draft, .. } if draft.is_empty()));
        workbench.record_result(
            command(Action::Inspect, "other:build", ""),
            Err(io::Error::other("inspect failed")),
        );
        workbench.select(Action::Probe);
        assert!(matches!(&workbench.page, Page::Model { draft, .. } if draft == "other:build"));
        workbench.submit("demo:build".into());
        assert!(matches!(&workbench.page, Page::Batch { draft, .. } if draft.is_empty()));
    }

    #[test]
    fn probe_form_preserves_input_across_pages() {
        let mut workbench = Workbench {
            page: Page::Menu,
            style: Style::Default,
            preset: Preset::Default,
            variant: String::new(),
            variant_revision: None,
            last_result: None,
            last_command: None,
            diagnosis: None,
            diagnosis_retry_issue: None,
            saved_preview: None,
            saved_receipt: None,
            report_focus: ReportFocus::Latest,
            form_context: None,
            probe_options_draft: None,
            model_context: None,
            model_draft: None,
            model_draft_owner: None,
            inspected_model_params: BTreeMap::new(),
            call_focus: 0,
            call_query: String::new(),
            call_views: HashMap::new(),
            inspected_structure: None,
        };
        assert!(workbench.select(Action::Probe).is_none());
        assert!(workbench.submit("q".to_owned()).is_none());
        let command = workbench.submit("example:batch".to_owned()).unwrap();
        assert!(command.action == Action::Probe);
        assert_eq!(command.model, "q");
        assert_eq!(command.batch, "example:batch");
    }

    #[test]
    fn report_rendering_preserves_evidence_and_removes_terminal_controls() {
        let report: Report = serde_json::from_str(r#"{
            "protocol_version": 1,
            "operation": "probe",
            "status": "error",
            "diagnostics": [{"code":"FORWARD_FAILED","message":"actual 48, expected 32\u001b[2J"}],
            "data": {
                "failure_location": {
                    "last_observed_module_call": {"alias_paths":["head"],"invocation_index":2,"module_type":"Linear"},
                    "project_stack":[{"file":"model.py","line":12}]
                },
                "calls": [{"alias_paths":["head"],"invocation_index":1,"module_type":"Linear"}],
                "trace_truncated": true
            }
        }"#).unwrap();
        let renderer = Renderer {
            locale: Locale::English,
            width: 80,
            style: Style::Default,
            line_mode: false,
        };
        let output = renderer.report(&report);
        assert!(output.contains("actual 48, expected 32"));
        assert!(output.contains("head  #2  Linear"));
        assert!(output.contains("head  #1  Linear"));
        assert!(output.contains("model.py:12"));
        assert!(output.contains("The trace was truncated."));
        assert!(!output.contains('\u{1b}'));
    }

    #[test]
    fn incomplete_trace_preserves_aliases_and_reports_unknown_index() {
        let report: Report = serde_json::from_str(
            r#"{
            "protocol_version": 1,
            "operation": "probe",
            "status": "error",
            "data": {
                "failure_location": {
                    "last_observed_module_call": {
                        "alias_paths": ["left.shared", "right.shared"],
                        "module_type": "Linear"
                    }
                },
                "trace_truncated": true
            }
        }"#,
        )
        .unwrap();
        let renderer = Renderer {
            locale: Locale::English,
            width: 80,
            style: Style::Default,
            line_mode: false,
        };
        let output = renderer.report(&report);
        assert!(output.contains("left.shared  call index unknown"));
        assert!(output.contains("Alias  right.shared"));
        assert!(output.contains("The trace was truncated."));
        assert!(!output.contains("#0"));
    }
}

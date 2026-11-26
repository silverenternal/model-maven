mod cli;
mod graph;
mod nvim_workspace;
mod pending_start;
mod penrose;
mod scaffold_draft;
mod terminal;
mod welcome;
mod workbench;

use clap::Parser;
use std::{
    io::{self, IsTerminal},
    path::PathBuf,
    process::ExitCode,
};
use terminal::locale::Locale;

#[derive(Parser)]
#[command(
    name = "m-model-tui",
    version,
    about = "Interactive terminal frontend for m-model"
)]
struct Options {
    /// Render a compact model graph from a JSON request on stdin.
    #[arg(long, hide = true)]
    render_model_preview: bool,
    /// Internal interactive model-preview pane; reads the supplied snapshot.
    #[arg(long, hide = true, value_name = "FILE")]
    live_model_preview: Option<PathBuf>,
    /// Print the source revision supplied when the binary was built.
    #[arg(long, hide = true)]
    build_source_revision: bool,
    /// Print the embedded workspace sources for distribution identity checks.
    #[arg(long, hide = true)]
    workspace_sources: bool,

    /// Open the bundled Drift-L 3D graph without launching m-model.
    #[arg(long, visible_alias = "drifting-demo")]
    demo_graph: bool,

    /// Open the small graph with the segmented editor.
    #[arg(long, conflicts_with = "demo_graph")]
    editor_demo: bool,

    /// Open the offline 48/32 diagnosis without launching m-model.
    #[arg(long, conflicts_with_all = ["demo_graph", "editor_demo"])]
    diagnosis_demo: bool,

    /// Select alternate offline diagnostic evidence.
    #[arg(long, requires = "diagnosis_demo", value_enum)]
    diagnosis_case: Option<DiagnosisCase>,

    /// Use line-by-line controls, including with redirected input.
    #[arg(long, conflicts_with_all = ["demo_graph", "editor_demo", "drifting_source"])]
    line_mode: bool,

    /// Open the Neovim workspace immediately, bypassing the welcome screen.
    #[arg(long, conflicts_with_all = ["line_mode", "demo_graph", "editor_demo", "diagnosis_demo", "drifting_source"])]
    skip_welcome: bool,

    /// Read the official Drifting generator and preset in a local source checkout.
    ///
    /// From the TUI repository, use `--drifting-source ./fixtures/drifting`
    /// to try a source directory, or `--drifting-demo` to open the bundled preview.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["demo_graph", "editor_demo", "diagnosis_demo"])]
    drifting_source: Option<PathBuf>,

    /// Override the official latent Drift-L preset used for source inspection.
    /// Relative YAML paths are resolved from the current directory.
    #[arg(long, value_name = "YAML", requires = "drifting_source")]
    drifting_config: Option<PathBuf>,

    /// Path to m-model. Defaults to MODEL_MAVEN_CLI, then m-model on PATH.
    #[arg(long, value_name = "PATH")]
    cli: Option<PathBuf>,
    /// Project directory. Defaults to the current directory.
    #[arg(long, value_name = "DIR")]
    project: Option<PathBuf>,
    /// Override the terminal locale for this session.
    #[arg(long, value_enum, value_name = "LANG")]
    lang: Option<Language>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Language {
    En,
    ZhCn,
    ZhTw,
}

impl From<Language> for Locale {
    fn from(language: Language) -> Self {
        match language {
            Language::En => Self::English,
            Language::ZhCn => Self::Simplified,
            Language::ZhTw => Self::Traditional,
        }
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum DiagnosisCase {
    Mismatch,
    Large,
    TwinMerge,
    CrossedPorts,
    LastObserved,
    SnapshotMissing,
}

fn main() -> ExitCode {
    let options = Options::parse();
    if let Some(path) = options.live_model_preview.as_deref() {
        #[cfg(unix)]
        return match graph::preview::interactive(path, options.lang.map_or_else(workbench::preferred_locale, Locale::from)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("Model preview: {error}"); ExitCode::FAILURE }
        };
        #[cfg(not(unix))]
        { let _ = path; eprintln!("Interactive model preview is unavailable on this platform."); return ExitCode::FAILURE; }
    }
    if options.render_model_preview {
        return match graph::preview::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("Model preview: {error}"); ExitCode::FAILURE }
        };
    }
    if options.build_source_revision {
        println!("{}", option_env!("MODEL_MAVEN_BUILD_REVISION").unwrap_or("unknown"));
        return ExitCode::SUCCESS;
    }
    if options.workspace_sources {
        println!("{}", nvim_workspace::workspace_sources());
        return ExitCode::SUCCESS;
    }
    let locale = options
        .lang
        .map_or_else(workbench::preferred_locale, Locale::from);
    if !options.line_mode && (!io::stdin().is_terminal() || !io::stderr().is_terminal()) {
        eprintln!(
            "{}",
            locale.text(
                "m-model-tui requires an interactive terminal.",
                "m-model-tui 需要交互式终端。",
                "m-model-tui 需要互動式終端。",
            )
        );
        return ExitCode::from(2);
    }
    // Explicit legacy demos and line mode remain available for existing users;
    // normal interactive use opens the welcome screen and shared workspace.
    if !options.line_mode && !options.demo_graph && !options.editor_demo
        && !options.diagnosis_demo && options.drifting_source.is_none()
    {
        let language = options.lang.map(|lang| match lang {
            Language::En => "en", Language::ZhCn => "zh-cn", Language::ZhTw => "zh-tw",
        });
        return match nvim_workspace::launch(
            options.project,
            options.cli,
            language,
            locale,
            options.skip_welcome,
        ) {
            Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
            Err(error) => { eprintln!("Model-Maven: {error}"); ExitCode::FAILURE }
        };
    }
    match run(options, locale) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}", terminal::text::plain_text(&error.to_string()));
            ExitCode::FAILURE
        }
    }
}

fn run(options: Options, locale: Locale) -> io::Result<()> {
    if options.demo_graph {
        return graph::show_drifting_demo(locale);
    }
    if let Some(root) = options.drifting_source.as_deref() {
        return graph::show_source(root, options.drifting_config.as_deref(), locale);
    }
    if options.diagnosis_demo {
        use model_maven_tui::diagnosis::Case;
        let case = match options.diagnosis_case.unwrap_or(DiagnosisCase::Mismatch) {
            DiagnosisCase::Mismatch => Case::Mismatch,
            DiagnosisCase::Large => Case::Large,
            DiagnosisCase::TwinMerge => Case::TwinMerge,
            DiagnosisCase::CrossedPorts => Case::CrossedPorts,
            DiagnosisCase::LastObserved => Case::LastObserved,
            DiagnosisCase::SnapshotMissing => Case::SnapshotMissing,
        };
        return graph::show_diagnosis(locale, case, options.line_mode);
    }
    if options.editor_demo {
        return graph::show_editor_demo(locale);
    }
    let current = std::env::current_dir()?;
    let project = options
        .project
        .unwrap_or_else(|| current.clone())
        .canonicalize()?;
    if !project.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            project.display().to_string(),
        ));
    }
    let backend = options
        .cli
        .or_else(|| std::env::var_os("MODEL_MAVEN_CLI").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("m-model"));
    let backend = if backend.is_relative() && backend.components().count() > 1 {
        current.join(backend)
    } else {
        backend
    };
    let cli = cli::Cli::new(backend, project);
    workbench::run(&cli, locale, options.line_mode)
}

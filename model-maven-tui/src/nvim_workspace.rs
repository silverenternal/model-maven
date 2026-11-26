//! Launch the same Lua workspace used inside an existing Neovim session.
use crate::{terminal::locale::Locale, welcome};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

include!(concat!(env!("OUT_DIR"), "/workspace_sources.rs"));

/// The distributor compares these exact embedded sources with the plugin it
/// ships for existing Neovim sessions. This command does not start Neovim.
pub(crate) fn workspace_sources() -> serde_json::Value {
    WORKSPACE_SOURCES.iter().map(|(name, source)| ((*name).to_owned(), serde_json::Value::String((*source).to_owned())))
        .collect::<serde_json::Map<String, serde_json::Value>>().into()
}

// The child Neovim changes directory to the selected project. Resolve PATH
// entries first so a relative entry keeps referring to the launch directory,
// rather than silently selecting a different CLI from inside the project.
fn absolute_search_path(path: &std::ffi::OsStr, base: &Path) -> io::Result<std::ffi::OsString> {
    let entries = env::split_paths(path).map(|entry| {
        if entry.as_os_str().is_empty() {
            base.to_path_buf()
        } else if entry.is_absolute() {
            entry
        } else {
            base.join(entry)
        }
    });
    env::join_paths(entries).map_err(|error| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("invalid PATH for Neovim launch: {error}"))
    })
}

fn resolve_explicit_path(path: PathBuf, base: &Path) -> PathBuf {
    if path.is_relative() && path.components().count() > 1 {
        base.join(path)
    } else {
        path
    }
}

pub(crate) fn launch(
    project: Option<PathBuf>,
    cli: Option<PathBuf>,
    language: Option<&str>,
    locale: Locale,
    skip_welcome: bool,
) -> io::Result<i32> {
    let launch_directory = env::current_dir()?;
    let project = project.unwrap_or_else(|| launch_directory.clone());
    let project = resolve_explicit_path(project, &launch_directory).canonicalize()?;
    if !project.is_dir() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "project path is not a directory"));
    }
    if !skip_welcome && !welcome::enter_workspace(&project, locale)? {
        return Ok(0);
    }
    // A private temporary runtime makes the packaged binary independent of the
    // source checkout. It lives until the child Neovim process exits.
    let runtime = tempfile::Builder::new().prefix("model-maven-workspace-").tempdir()?;
    for (name, source) in WORKSPACE_SOURCES {
        let destination = runtime.path().join(name);
        if let Some(parent) = destination.parent() { fs::create_dir_all(parent)?; }
        fs::write(destination, source)?;
    }

    let executable = env::var_os("MODEL_MAVEN_NVIM").unwrap_or_else(|| "nvim".into());
    let executable = resolve_explicit_path(PathBuf::from(executable), &launch_directory);
    let search_path = env::var_os("PATH")
        .map(|path| absolute_search_path(&path, &launch_directory))
        .transpose()?;
    let mut command = Command::new(executable);
    command.current_dir(&project)
        .env("MODEL_MAVEN_RUNTIME_DIR", runtime.path())
        .env("MODEL_MAVEN_PROJECT", &project)
        .env("MODEL_MAVEN_TUI_RENDERER", env::current_exe()?)
        .arg("--cmd")
        .arg("lua vim.opt.runtimepath:prepend(vim.env.MODEL_MAVEN_RUNTIME_DIR)")
        .arg("-c")
        .arg(format!("lua assert(loadstring({:?}))()", include_str!("nvim_start.lua")))
        .stdin(Stdio::inherit()).stdout(Stdio::inherit()).stderr(Stdio::inherit());
    if let Some(path) = search_path {
        command.env("PATH", path);
    }
    let cli = cli.or_else(|| env::var_os("MODEL_MAVEN_CLI").map(PathBuf::from));
    if let Some(path) = cli {
        command.env("MODEL_MAVEN_CLI", resolve_explicit_path(path, &launch_directory));
    }
    if let Some(lang) = language { command.env("MODEL_MAVEN_LANGUAGE", lang); }
    let status = command.status().map_err(|err| match err.kind() {
        io::ErrorKind::NotFound => io::Error::new(
            err.kind(),
            "Neovim could not be started. Install Neovim or set MODEL_MAVEN_NVIM to its executable path.",
        ),
        io::ErrorKind::PermissionDenied => io::Error::new(
            err.kind(),
            format!("Neovim is not executable or launch permission was denied: {err}"),
        ),
        _ => io::Error::new(err.kind(), format!("could not start Neovim: {err}")),
    })?;
    Ok(status.code().unwrap_or(1))
}

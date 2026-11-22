use super::*;

pub(super) fn handle<B: Backend>(
    application: &Application<B>,
    cli: &Cli,
    command: &Command,
    context: ExecutionContext<'_>,
) -> Option<Report> {
    match command {
        Command::Env { command } => {
            let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                Ok(root) => root,
                Err(error) => {
                    return Some(Report::error(
                        "environment",
                        "PROJECT_UNAVAILABLE",
                        error.to_string(),
                        "",
                    ));
                }
            };
            let libraries = match command {
                EnvironmentCommand::Libraries { libraries, .. }
                | EnvironmentCommand::Probe { libraries, .. } => Some(libraries),
                _ => None,
            };
            if let EnvironmentCommand::Configure {
                system_site_packages,
                isolate_system_site_packages: _,
                venv,
            } = command
            {
                return Some(application.backend.execute(&json!({
                    "protocol_version": PROTOCOL_VERSION,
                    "operation": "environment_configure",
                    "system_site_packages": system_site_packages,
                    "venv": venv.as_ref().map(|path| path.to_string_lossy().into_owned()),
                    "interactive_progress": context.show_progress,
                    "runtime": {
                        "python_executable": cli.python.as_deref().unwrap_or("python3"),
                        "timeout_seconds": cli.timeout.unwrap_or(30),
                        "project_root": root
                    }
                })));
            }
            let (operation, requirements, venv, refresh, relative, default_timeout) = match command
            {
                EnvironmentCommand::Lock {
                    requirements,
                    refresh,
                } => (
                    "environment_lock",
                    Some(requirements.to_string_lossy().into_owned()),
                    None,
                    *refresh,
                    None,
                    1800,
                ),
                EnvironmentCommand::Configure { .. } => unreachable!("handled above"),
                EnvironmentCommand::Setup {
                    requirements,
                    venv,
                    update_lock,
                } => (
                    "environment_setup",
                    Some(requirements.to_string_lossy().into_owned()),
                    venv.as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    *update_lock,
                    None,
                    1800,
                ),
                EnvironmentCommand::Check { venv } => (
                    "environment_check",
                    None,
                    venv.as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    false,
                    None,
                    30,
                ),
                EnvironmentCommand::Libraries { venv, libraries: _ } => (
                    "environment_library_inventory",
                    None,
                    venv.as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    false,
                    None,
                    60,
                ),
                EnvironmentCommand::Probe {
                    venv,
                    libraries: _,
                    check_runtime,
                } => (
                    "environment_library_probe",
                    None,
                    venv.as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    false,
                    None,
                    if *check_runtime { 900 } else { 300 },
                ),
                EnvironmentCommand::Wheelhouse { command } => match command {
                    WheelhouseCommand::Archive { path } => (
                        "environment_wheelhouse_archive",
                        None,
                        None,
                        false,
                        path.as_ref()
                            .map(|path| path.to_string_lossy().into_owned()),
                        1800,
                    ),
                    WheelhouseCommand::Check { path } => (
                        "environment_wheelhouse_check",
                        None,
                        None,
                        false,
                        path.as_ref()
                            .map(|path| path.to_string_lossy().into_owned()),
                        30,
                    ),
                    WheelhouseCommand::Replay { venv, path } => (
                        "environment_wheelhouse_replay",
                        None,
                        Some(venv.to_string_lossy().into_owned()),
                        false,
                        path.as_ref()
                            .map(|path| path.to_string_lossy().into_owned()),
                        1800,
                    ),
                },
            };
            Some(application.backend.execute(&json!({
                "protocol_version": PROTOCOL_VERSION,
                "operation": operation,
                "requirements": requirements,
                "venv": venv,
                "refresh_lock": refresh,
                "relative": relative,
                "libraries": libraries,
                "check_runtime": matches!(
                    command,
                    EnvironmentCommand::Probe {
                        check_runtime: true,
                        ..
                    }
                ),
                "python_explicit": cli.python.is_some(),
                "interactive_progress": context.show_progress,
                "runtime": {
                    "python_executable": cli.python.as_deref().unwrap_or("python3"),
                    "timeout_seconds": cli.timeout.unwrap_or(default_timeout),
                    "project_root": root
                }
            })))
        }
        Command::Doctor => Some(
            application
                .backend
                .execute(&json!({"protocol_version": PROTOCOL_VERSION, "operation": "doctor"})),
        ),
        _ => None,
    }
}

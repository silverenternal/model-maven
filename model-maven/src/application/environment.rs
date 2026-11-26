use super::*;

pub(super) fn handle<B: Backend>(
    application: &Application<B>,
    cli: &Cli,
    command: &Command,
    context: ExecutionContext<'_>,
) -> Option<Report> {
    match command {
        Command::Env { command } => {
            let python = match require_python(cli, "environment") {
                Ok(value) => value,
                Err(report) => return Some(report),
            };
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
                        "python_executable": python,
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
                EnvironmentCommand::Devices { venv } => (
                    "environment_devices",
                    None,
                    venv.as_ref().map(|path| path.to_string_lossy().into_owned()),
                    false,
                    None,
                    120,
                ),
                EnvironmentCommand::Probe {
                    venv,
                    libraries: _,
                    check_runtime,
                    ..
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
                    WheelhouseCommand::Replay { venv, path, .. } => (
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
            // Resolve the interpreter and import paths from the same config
            // snapshot. Loading twice can combine two different project
            // revisions when another process edits the config between reads.
            let project_config = match application.project_config.load(&root) {
                Ok(config) => config,
                Err(error) if !cli.python_explicit || matches!(operation, "environment_library_probe" | "environment_devices") => {
                    return Some(Report::error(operation, "PROJECT_CONFIG_INVALID", error,
                        "检查项目配置；需要项目运行时路径的操作不会忽略无效配置。"));
                }
                Err(_) => None,
            };
            let project_python = project_config.as_ref()
                .and_then(|config| config["runtime_python"].as_str().map(str::to_owned));
            if crate::project_python_selection_changed(&root, cli, project_python.as_deref()) {
                return Some(Report::error(operation, "PROJECT_CONFIG_CONFLICT", "项目 Python 选择在启动时发生变化。", "重新执行环境操作。"));
            }
            let mut request = json!({
                "protocol_version": PROTOCOL_VERSION,
                "operation": operation,
                "project_root": root,
                "requirements": requirements,
                "venv": venv,
                "refresh_lock": refresh,
                "relative": relative,
                "create_environment": matches!(command, EnvironmentCommand::Wheelhouse { command: WheelhouseCommand::Replay { create: true, .. } }),
                "libraries": libraries,
                "device_inventory": operation == "environment_devices",
                "layer_catalog": matches!(command, EnvironmentCommand::Probe { layer_catalog: true, .. }),
                "operator_catalog": matches!(command, EnvironmentCommand::Probe { operator_catalog: true, .. }),
                "check_runtime": matches!(
                    command,
                    EnvironmentCommand::Probe {
                        check_runtime: true,
                        ..
                    }
                ),
                "python_explicit": cli.python_explicit || project_python.is_some(),
                "interactive_progress": context.show_progress,
                "runtime": {
                    "python_executable": python,
                    "timeout_seconds": cli.timeout.unwrap_or(default_timeout),
                    "project_root": root
                }
            });
            if matches!(operation, "environment_library_probe" | "environment_devices") {
                if let Some(config) = project_config.as_ref() {
                    super::run::apply_project_runtime_defaults(config, &mut request["runtime"]);
                }
                if let Err(report) = super::run::resolve_worker_directory(operation, &root, &mut request["runtime"], false) {
                    return Some(report);
                }
                if let Err(report) = super::run::resolve_worker_import_roots(operation, &root, &mut request["runtime"], false) {
                    return Some(report);
                }
            }
            Some(application.backend.execute(&request))
        }
        Command::Doctor => {
            let python = match require_python(cli, "doctor") {
                Ok(value) => value,
                Err(report) => return Some(report),
            };
            Some(application.backend.execute(&json!({"protocol_version": PROTOCOL_VERSION, "operation": "doctor",
                "runtime": {"python_executable": python}})))
        },
        _ => None,
    }
}

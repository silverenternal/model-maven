mod adapters;
mod app;
mod cli;
mod domain;
mod ports;
pub mod presentation;

use adapters::exports::FileArtifactExporter;
use adapters::previews::FilePreviewStore;
use adapters::project_config::FileProjectConfigStore;
use adapters::project_source::FileProjectSourceWriter;
use adapters::python::PythonBackend;
use adapters::runs::FileRunStore;
use adapters::source_edits::FileSourceEditStore;
use adapters::supervisor::NativeRunSupervisor;
use adapters::variants::FileVariantStore;
use adapters::{component_scaffold::FileComponentScaffolder, components::FileComponentRegistry};
use app::Application;
use clap::{CommandFactory, Parser, error::ErrorKind};
use cli::*;
use domain::Report;
use ports::{Backend, RunRepository, RunSupervisor};
use presentation::{
    terminal::display,
    workbench::{interactive, run_command},
};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{
    fs::OpenOptions,
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

fn internal_run(project_root: &Path, run_id: &str, cli: &Cli) -> Report {
    let root = match project_root.canonicalize() {
        Ok(root) => root,
        Err(error) => return Report::error("train", "PROJECT_UNAVAILABLE", error.to_string(), ""),
    };
    let mut encoded = Vec::new();
    if let Err(error) = io::stdin().take(1024 * 1024 + 1).read_to_end(&mut encoded) {
        return Report::error("train", "BACKGROUND_REQUEST_INVALID", error.to_string(), "");
    }
    if encoded.len() > 1024 * 1024 {
        return Report::error(
            "train",
            "BACKGROUND_REQUEST_INVALID",
            "后台请求超过 1 MiB。",
            "",
        );
    }
    let mut request: Value = match serde_json::from_slice(&encoded) {
        Ok(request) => request,
        Err(error) => {
            return Report::error("train", "BACKGROUND_REQUEST_INVALID", error.to_string(), "");
        }
    };
    if request["operation"] != "train" || request["project_root"].as_str() != root.to_str() {
        return Report::error(
            "train",
            "BACKGROUND_REQUEST_INVALID",
            "请求操作或项目路径不匹配。",
            "",
        );
    }
    let runs = FileRunStore;
    let record = match runs.show(&root, run_id) {
        Ok(record) => record,
        Err(error) => return Report::error("train", "RUN_STORE_FAILED", error, ""),
    };
    let supervisor = NativeRunSupervisor;
    let pid = record["record"]["supervisor"]["pid"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok());
    let token = record["record"]["supervisor"]["start_token"].as_str();
    if !matches!(
        record["record"]["status"].as_str(),
        Some("running" | "stopping")
    ) || pid != Some(std::process::id())
        || token.is_none_or(|token| !supervisor.is_alive(std::process::id(), token))
    {
        return Report::error(
            "train",
            "BACKGROUND_IDENTITY_INVALID",
            "当前进程与登记的运行监督身份不一致。",
            "",
        );
    }
    let run_directory = match runs.run_directory(&root, run_id) {
        Ok(directory) => directory,
        Err(error) => return Report::error("train", "RUN_STORE_FAILED", error, ""),
    };
    let expected_checkpoint = run_directory.join("checkpoints");
    let expected_progress = run_directory.join("progress.jsonl");
    let expected_stop = run_directory.join("stop.request");
    let expected_log = run_directory.join("worker.log");
    for (key, expected) in [
        ("checkpoint_dir", &expected_checkpoint),
        ("progress_path", &expected_progress),
        ("stop_file", &expected_stop),
        ("log_path", &expected_log),
    ] {
        if request[key].as_str() != expected.to_str() {
            return Report::error(
                "train",
                "BACKGROUND_REQUEST_INVALID",
                format!("后台请求的 {key} 路径不匹配运行目录。"),
                "",
            );
        }
    }
    let token = token.expect("the supervisor token was validated above");
    let ready_path = run_directory.join("supervisor.ready");
    let mut ready_options = OpenOptions::new();
    ready_options.write(true).create_new(true);
    #[cfg(unix)]
    ready_options.mode(0o600);
    let mut ready = match ready_options.open(&ready_path) {
        Ok(file) => file,
        Err(error) => {
            return Report::error(
                "train",
                "BACKGROUND_HANDSHAKE_FAILED",
                error.to_string(),
                "检查运行目录中的后台监督启动标记。",
            );
        }
    };
    if let Err(error) =
        writeln!(ready, "{}:{token}", std::process::id()).and_then(|()| ready.sync_all())
    {
        let _ = std::fs::remove_file(&ready_path);
        return Report::error(
            "train",
            "BACKGROUND_HANDSHAKE_FAILED",
            error.to_string(),
            "检查运行目录写权限。",
        );
    }
    request["run_id"] = serde_json::json!(run_id);
    let heartbeat_active = Arc::new(AtomicBool::new(true));
    let heartbeat_stop = Arc::clone(&heartbeat_active);
    let heartbeat_root = root.clone();
    let heartbeat_run_id = run_id.to_owned();
    let heartbeat_token = token.to_owned();
    let heartbeat_thread = thread::spawn(move || {
        let heartbeat_store = FileRunStore;
        while heartbeat_active.load(Ordering::Relaxed) {
            let _ = heartbeat_store.record_heartbeat(
                &heartbeat_root,
                &heartbeat_run_id,
                std::process::id(),
                &heartbeat_token,
            );
            for _ in 0..20 {
                if !heartbeat_active.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    });
    let backend = PythonBackend {
        executable: request["runtime"]["python_executable"]
            .as_str()
            .unwrap_or("python3")
            .to_owned(),
        timeout: Duration::from_secs(
            request["runtime"]["timeout_seconds"]
                .as_u64()
                .unwrap_or(86_400),
        ),
        verbose: request["verbose"].as_bool().unwrap_or(cli.verbose),
    };
    let mut report = backend.execute(&request);
    heartbeat_stop.store(false, Ordering::Relaxed);
    let _ = heartbeat_thread.join();
    if let Some(data) = report.data.as_object_mut() {
        data.insert("run_id".into(), serde_json::json!(run_id));
        if let Some(parent) = request["parent_run_id"].as_str() {
            data.insert("parent_run_id".into(), serde_json::json!(parent));
        }
    }
    let serialized = match serde_json::to_value(&report) {
        Ok(value) => value,
        Err(error) => return Report::error("train", "RUN_REPORT_INVALID", error.to_string(), ""),
    };
    if let Err(error) = runs.finish(&root, run_id, &serialized) {
        report = Report::error(
            "train",
            "RUN_STORE_FAILED",
            format!("训练已退出，但结果未完整写入运行记录：{error}"),
            "检查 .model-maven/runs 中的磁盘和目录权限。",
        );
    }
    report
}

fn project_python_executable() -> Option<String> {
    let config_dir = Path::new(".model-maven");
    let config_path = config_dir.join("environment.json");
    let venv = match std::fs::symlink_metadata(&config_path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() > 16 * 1024
            {
                return None;
            }
            if std::fs::symlink_metadata(config_dir)
                .is_ok_and(|parent| parent.file_type().is_symlink() || !parent.is_dir())
            {
                return None;
            }
            let contents = std::fs::read(&config_path).ok()?;
            let value: Value = serde_json::from_slice(&contents).ok()?;
            if value["schema_version"] != 1 {
                return None;
            }
            let selected = Path::new(value["venv"].as_str()?);
            if selected.is_absolute()
                || selected.as_os_str().is_empty()
                || selected
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                return None;
            }
            selected.to_path_buf()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Path::new(".venv").into(),
        Err(_) => return None,
    };
    let mut cursor = Path::new("").to_path_buf();
    for component in venv.components() {
        cursor.push(component.as_os_str());
        if let Ok(metadata) = std::fs::symlink_metadata(&cursor)
            && metadata.file_type().is_symlink()
        {
            return None;
        }
    }
    let is_virtualenv = std::fs::symlink_metadata(&venv)
        .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink());
    if !is_virtualenv {
        return None;
    }
    let python_path = if cfg!(windows) {
        venv.join("Scripts/python.exe")
    } else {
        venv.join("bin/python")
    };
    python_path
        .is_file()
        .then(|| python_path.to_string_lossy().into_owned())
}

#[derive(Parser)]
#[command(name = "internal-run")]
struct InternalRunArgs {
    #[arg(long)]
    project_root: PathBuf,
    #[arg(long)]
    run_id: String,
}

fn main() -> ExitCode {
    #[cfg(unix)]
    let _early_interrupt = PythonBackend::prepare_interrupt_handling();
    let raw_args = std::env::args_os().collect::<Vec<_>>();
    if raw_args
        .get(1)
        .is_some_and(|argument| argument == "internal-run")
    {
        let internal = match InternalRunArgs::try_parse_from(raw_args.into_iter().skip(1)) {
            Ok(arguments) => arguments,
            Err(error) => {
                let _ = error.print();
                return ExitCode::from(2);
            }
        };
        let internal_cli = Cli {
            describe_interface: false,
            json: true,
            no_interactive: true,
            python: None,
            python_explicit: false,
            timeout: None,
            variant: None,
            if_variant_base: None,
            if_input_content: None,
            if_source_manifest: None,
            if_source_plan: None,
            if_source_config: None,
            capture_input_to: None,
            replay_input_from: None,
            seed: None,
            verbose: false,
            command: None,
        };
        let report = internal_run(&internal.project_root, &internal.run_id, &internal_cli);
        println!(
            "{}",
            serde_json::to_string(&report).expect("report serializes")
        );
        return ExitCode::from(report.exit_code() as u8);
    }
    clap_complete::CompleteEnv::with_factory(Cli::command)
        .bin("m-model")
        .complete();
    let mut cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let help_or_version = matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            );
            let json_requested = std::env::args_os().any(|argument| argument == "--json");
            if json_requested && !help_or_version {
                let report = Report::error(
                    "cli",
                    "CLI_ARGUMENT_INVALID",
                    error.to_string(),
                    "运行 m-model --help 查看命令参数；修正参数后重试。",
                );
                display(&report, true);
                return ExitCode::from(report.exit_code() as u8);
            }
            let _ = error.print();
            return if help_or_version {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            };
        }
    };
    if cli.describe_interface {
        let report = if cli.command.is_some() {
            Report::error(
                "interface",
                "CLI_ARGUMENT_INVALID",
                "--describe-interface 不能与子命令同时使用。",
                "",
            )
        } else {
            let capabilities = [
                "input_snapshot.capture_replay",
                "project.classification_create",
                "project.custom_scaffold",
                "project.scaffold_custom_module",
                "project.scaffold_external_batch",
                "project.scaffold_external_batch_params",
                "project.scaffold_initial_bindings",
                "project.scaffold_native_encoder_output",
                "project.scaffold_encoder_output_edit",
                "project.scaffold_encoder_output_evidence",
                "project.scaffold_named_outputs",
                "project.scaffold_objectives",
                "project.scaffold_objective_weights",
                "data.csv_batch_template",
                "data.csv_batch_typed",
                "project.scaffold_input_routes",
                "project.scaffold_route_edit",
                "project.scaffold_route_evidence",
                "project.scaffold_setting_snapshot",
                "project.conditional_batch_edit",
                "assembly.conditional_config_edit",
                "assembly.atomic_patch",
                "assembly.atomic_patch_encoder_output",
                "assembly.atomic_patch_route",
                "assembly.atomic_patch_evidence",
                "assembly.single_source_evidence",
                "assembly.single_input_evidence",
                "assembly.single_input_snapshot",
                "component.conditional_uninstall",
                "component.project_package_lifecycle",
                "assembly.atomic_patch_snapshot",
                "assembly.generated_source_verify",
                "training.conditional_config_start",
                "training.request_identity",
                "runs.lookup_request_id",
                "training.conditional_supervision",
                "training.conditional_variant_start",
                "training.conditional_resume_checkpoint",
                "evaluation.conditional_checkpoint",
                "export.conditional_checkpoint",
                "training.data_fingerprint",
                "runs.comparison_unverified_data",
                "augmentation.conditional_config_edit",
                "edit.conditional_source_manifest",
                "edit.conditional_source_plan",
                "edit.starter_migration",
                "edit.conditional_source_input",
                "edit.conditional_source_config",
                "edit.conditional_source_project_manifest",
                "layer.builtin_catalog",
            ]
            .into_iter()
            .map(|name| (name, true))
            .collect::<std::collections::BTreeMap<_, _>>();
            Report::ok(
                "interface",
                serde_json::json!({
                    "interface_version": 1,
                    "cli_version": env!("CARGO_PKG_VERSION"),
                    "report_protocol_version": domain::PROTOCOL_VERSION,
                    "capabilities": capabilities,
                    "scope": "CLI interface only; runtime availability is determined separately"
                }),
            )
        };
        display(&report, cli.json);
        return ExitCode::from(report.exit_code() as u8);
    }
    cli.python_explicit = cli.python.is_some();
    if let Some(Command::Completions { shell }) = cli.command.as_ref() {
        if let Err(error) = presentation::completion::generate_script(shell, &mut io::stdout()) {
            eprintln!("无法生成补全脚本：{error}");
            return ExitCode::from(2);
        }
        return ExitCode::SUCCESS;
    }
    let environment_command = matches!(
        cli.command.as_ref(),
        Some(Command::Env {
            command: EnvironmentCommand::Lock { .. }
                | EnvironmentCommand::Configure { .. }
                | EnvironmentCommand::Setup { .. }
                | EnvironmentCommand::Check { .. }
                | EnvironmentCommand::Libraries { .. }
                | EnvironmentCommand::Wheelhouse { .. }
        })
    );
    if cli.python.is_none() && !environment_command {
        cli.python = project_python_executable();
    }
    let mut app = Application {
        backend: PythonBackend {
            executable: cli.python.clone().unwrap_or_else(|| "python3".into()),
            timeout: Duration::from_secs(cli.timeout.unwrap_or(30)),
            verbose: cli.verbose,
        },
        runtime_capabilities: std::cell::RefCell::new(None),
        variants: Box::new(FileVariantStore),
        runs: Box::new(FileRunStore),
        exporter: Box::new(FileArtifactExporter),
        supervisor: Box::new(NativeRunSupervisor),
        components: Box::new(FileComponentRegistry),
        scaffolder: Box::new(FileComponentScaffolder),
        source_edits: Box::new(FileSourceEditStore),
        project_config: Box::new(FileProjectConfigStore),
        project_source_writer: Box::new(FileProjectSourceWriter),
        previews: Box::new(FilePreviewStore),
    };
    if let Some(command) = &cli.command {
        let report = run_command(&app, &cli, command);
        display(&report, cli.json);
        ExitCode::from(report.exit_code() as u8)
    } else if !cli.json
        && !cli.no_interactive
        && io::stdin().is_terminal()
        && io::stderr().is_terminal()
    {
        interactive(&mut app, &mut cli);
        ExitCode::SUCCESS
    } else {
        if cli.json {
            display(
                &Report::error(
                    "help",
                    "COMMAND_REQUIRED",
                    "请选择一个命令。",
                    "m-model --help",
                ),
                true,
            );
            return ExitCode::from(2);
        }
        let _ = Cli::command().print_help();
        println!();
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod interactive_schema_tests {
    use super::presentation::forms::schema::validate_value as validate_prompt_schema_value;
    use serde_json::json;

    #[test]
    fn provider_schema_form_validation_covers_nested_values_and_bounds() {
        let schema = json!({
            "type":"object",
            "properties":{
                "strategy":{"type":"string","enum":["first","second"]},
                "probability":{"type":"number","minimum":0.0,"maximum":1.0},
                "options":{
                    "type":"object",
                    "properties":{"inplace":{"type":"boolean"}},
                    "required":["inplace"],
                    "additionalProperties":false
                }
            },
            "required":["strategy","options"],
            "additionalProperties":false
        });
        let payload = json!({
            "strategy":"first",
            "probability":0.35,
            "options":{"inplace":false}
        });

        assert!(validate_prompt_schema_value(&payload, &schema, "payload", 0).is_ok());
        assert!(
            validate_prompt_schema_value(
                &json!({"strategy":"unknown","options":{"inplace":false}}),
                &schema,
                "payload",
                0
            )
            .is_err()
        );
        assert!(
            validate_prompt_schema_value(
                &json!({"strategy":"first","probability":1.5,"options":{"inplace":false}}),
                &schema,
                "payload",
                0
            )
            .is_err()
        );
        assert!(
            validate_prompt_schema_value(
                &json!({"strategy":"first","options":{}}),
                &schema,
                "payload",
                0
            )
            .is_err()
        );
    }
}

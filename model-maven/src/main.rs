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
use ports::{Backend, ProjectConfigStore, RunRepository, RunSupervisor};
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
    if !matches!(request["operation"].as_str(), Some("train" | "generic_loop_train")) || request["project_root"].as_str() != root.to_str() {
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
    if record["record"]["operation"] != request["operation"] {
        return Report::error("train", "BACKGROUND_IDENTITY_INVALID", "请求操作与登记运行不匹配。", "");
    }
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
    let expected_snapshot = run_directory.join("snapshot");
    for (key, expected) in [
        ("checkpoint_dir", &expected_checkpoint),
        ("progress_path", &expected_progress),
        ("stop_file", &expected_stop),
        ("log_path", &expected_log),
        ("snapshot_dir", &expected_snapshot),
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
            .unwrap_or_default()
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

// Keep the project choice separately from the fallback interpreter selected
// later. This lets an operation detect that a saved choice was removed, not
// just that it changed to a different nonempty value.
static STARTUP_PROJECT_PYTHON: std::sync::Mutex<Option<(std::path::PathBuf, Option<String>)>> = std::sync::Mutex::new(None);

fn record_project_python_selection(root: &Path, choice: Option<String>) {
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    *STARTUP_PROJECT_PYTHON.lock().unwrap_or_else(|error| error.into_inner()) = Some((canonical_root, choice));
}

fn project_python_selection_changed(root: &Path, cli: &Cli, current: Option<&str>) -> bool {
    if cli.python_explicit { return false; }
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let snapshot = STARTUP_PROJECT_PYTHON.lock().unwrap_or_else(|error| error.into_inner());
    match snapshot.as_ref() {
        Some((startup_root, choice)) if *startup_root == canonical_root => {
            choice.as_deref() != current || current.is_some_and(|python| cli.python.as_deref() != Some(python))
        },
        // Embedded or later-switched projects may lack a startup snapshot.
        // Retain the previous conservative check for an explicit current value.
        _ => current.is_some_and(|python| cli.python.as_deref() != Some(python)),
    }
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

// Follow PATH order (including an activated environment) without starting or
// installing Python. Keep the path as selected rather than canonicalizing a
// virtualenv interpreter symlink to its base interpreter.
fn path_python_executable() -> Option<String> {
    let path = std::env::var_os("PATH")?;
    let current = std::env::current_dir().ok()?;
    let names: &[&str] = if cfg!(windows) { &["python.exe", "python3.exe"] } else { &["python3", "python"] };
    for directory in std::env::split_paths(&path) {
        let directory = if directory.as_os_str().is_empty() { current.clone() }
            else if directory.is_absolute() { directory }
            else { current.join(directory) };
        for name in names {
            let candidate = directory.join(name);
            let Ok(metadata) = std::fs::metadata(&candidate) else { continue; };
            if !metadata.is_file() { continue; }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if metadata.permissions().mode() & 0o111 == 0 { continue; }
            }
            if let Some(value) = candidate.to_str() { return Some(value.to_owned()); }
        }
    }
    None
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
    if raw_args.len() == 2 && raw_args[1] == "--build-source-revision" {
        println!("{}", option_env!("MODEL_MAVEN_BUILD_REVISION").unwrap_or("unknown"));
        return ExitCode::SUCCESS;
    }
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
            let mut capabilities = [
                "input_snapshot.capture_replay",
                "probe.force_fresh_environment",
                "probe.model_device_v1",
                "supervision.mark_device_v1",
                "supervision.preview_device_v1",
                "data.preview_no_split_v1",
                "data.preview_augment_choice_v1",
                "runtime.workspace_paths_v1",
                "runtime.project_paths_config_v1",
                "runtime.project_python_config_v1",
                "runtime.python_selection_hint_v1",
                "runtime.data_preview_paths_v1",
                "runtime.device_inventory_v1",
                "runtime.offline_environment_create_v1",
                "project.classification_create",
                "project.classification_recipe_v1",
                "project.classification_recipe_training_supervision_v1",
                "project.classification_recipe_source_snapshot_v1",
                "project.classification_recipe_bundle_v1",
                "project.graph_recipe_bundle_v1",
                "project.scaffold_recipe_bundle_v1",
                "project.classification_recipe_runtime_paths_v1",
                "project.scaffold_recipe_sync_v1",
                "project.discovery_file_candidates_v1",
                "project.custom_scaffold",
                "project.scaffold_recipe_v1",
                "project.scaffold_recipe_assembly_v1",
                "project.scaffold_recipe_encoder_layers_v1",
                "project.scaffold_recipe_encoder_widths_v1",
                "project.scaffold_recipe_native_encoders_v1",
                "project.scaffold_recipe_component_params_v1",
                "project.scaffold_recipe_data_v1",
                "project.scaffold_recipe_training_v1",
                "project.scaffold_recipe_optimizer_v1",
                "project.scaffold_recipe_supervision_ref_v1",
                "training.recipe_entry_v1",
                "project.scaffold_recipe_factory_layers_v1",
                "project.scaffold_residual_layers_v1",
                "project.scaffold_parallel_layers_v1",
                "project.scaffold_custom_loss_v1",
                "project.scaffold_layer_ids_v1",
                "project.scaffold_one_batch_evaluator_v1",
                "project.scaffold_mse_shape_contract_v1",
                "project.recipe_layer_move_plan_v1",
                "project.recipe_layer_edit_plan_v1",
                "project.recipe_layer_cross_copy_plan_v1",
                "project.graph_topology_plan_v1",
                "project.graph_edit_batch_plan_v1",
                "project.graph_boundary_annotations_v1",
                "project.graph_data_objectives_v1",
                "project.graph_edit_group_copy_v1",
                "project.graph_edit_group_set_v1",
                "project.graph_runtime_containers_v1",
                "project.graph_container_list_children_v1",
                "project.graph_edit_settings_v1",
                "project.graph_generate_replace_v1",
                "project.graph_draft_replay_v1",
                "project.graph_import_scaffold_preview_v1",
                "project.graph_import_editable_layers_v1",
                "project.graph_import_project_data_v1",
                "project.graph_unicode_ids_v1",
                "project.graph_output_name_v1",
                "project.graph_legacy_snapshot_v1",
                "project.graph_cast_parameter_dtype_v1",
                "project.graph_dynamic_mean_v1",
                "project.graph_factory_contract_v1",
                "project.graph_dtype_sequence_guard_v1",
                "project.graph_clone_node_v1",
                "project.graph_assert_batch_v1",
                "project.graph_after_node_v1",
                "project.graph_list_call_v1",
                "project.graph_bind_project_v1",
                "project.graph_bind_create_v1",
                "project.graph_optional_training_v1",
                "project.graph_reorder_dtype_sync_v1",
                "project.graph_reorder_plan_v1",
                "project.graph_reorder_after_v1",
                "project.graph_declared_shapes_v1",
                "project.graph_port_dtype_v1",
                "project.graph_training_step_v1",
                "project.graph_training_modes_v1",
                "project.graph_supervision_reference_v1",
                "project.graph_training_recipe_v1",
                "project.graph_merge_output_shapes_v1",
                "project.graph_call_output_style_v1",
                "project.graph_named_components_v1",
                "project.graph_source_preview_v1",
                "project.graph_source_create_v1",
                "project.graph_registered_components_v1",
                "project.graph_save_v1",
                "project.graph_delivery_review_v1",
                "project.graph_delivery_recovery_v1",
                "project.graph_delivery_status_v1",
                "project.graph_delivery_archive_v1",
                "project.graph_generic_batch_transforms_v1",
                "project.graph_runtime_shape_contract_v1",
                "project.scaffold_recipe_update_v1",
                "project.scaffold_reconcile_assembly_v1",
                "project.scaffold_runtime_dtype_v1",
                "project.scaffold_default_dtype_v1",
                "project.scaffold_32_inputs_outputs_v1",
                "project.scaffold_rank8_v1",
                "project.scaffold_custom_module",
                "project.scaffold_external_batch",
                "project.scaffold_external_batch_params",
                "project.scaffold_initial_bindings",
                "project.scaffold_native_encoder_output",
                "project.scaffold_encoder_output_edit",
                "project.scaffold_encoder_output_evidence",
                "project.scaffold_named_outputs",
                "project.scaffold_unicode_outputs_v1",
                "project.scaffold_unicode_inputs_v1",
                "project.scaffold_nfkc_collision_v1",
                "project.scaffold_objectives",
                "project.scaffold_objective_weights",
                "data.csv_batch_template",
                "data.csv_batch_typed",
                "data.csv_batch_runtime_dtype_v1",
                "data.csv_batch_rank8_v1",
                "data.csv_batch_optional_targets_v1",
                "data.csv_batch_row_split_v1",
                "project.scaffold_input_routes",
                "project.scaffold_route_edit",
                "project.scaffold_route_evidence",
                "project.scaffold_setting_snapshot",
                "project.conditional_batch_edit",
                "project.conditional_batch_edit_previous",
                "project.conditional_model_edit",
                "assembly.conditional_config_edit",
                "assembly.atomic_patch",
                "assembly.direct_factory_params_v1",
                "assembly.atomic_patch_encoder_output",
                "assembly.atomic_patch_route",
                "assembly.atomic_patch_evidence",
                "assembly.single_source_evidence",
                "assembly.single_runtime_paths_v1",
                "assembly.single_input_evidence",
                "assembly.single_input_snapshot",
                "component.conditional_uninstall",
                "component.project_package_lifecycle",
                "assembly.atomic_patch_snapshot",
                "assembly.generated_source_verify",
                "assembly.generate_runtime_paths_v1",
                "assembly.show_runtime_paths_v1",
                "training.conditional_config_start",
                "training.external_loop_model_choice_v1",
                "training.generic_loop_trial_cli_v1",
                "training.generic_loop_train_cli_v1",
                "training.generic_loop_interface_v1",
                "training.batch_binding_v1",
                "training.batch_device_policy_v1",
                "training.repeat_input_guard_v1",
                "training.custom_optimizer_factory_v1",
                "training.generic_device_v1",
                "training.request_identity",
                "training.atomic_run_admission",
                "runs.lookup_request_id",
                "training.conditional_supervision",
                "training.random_run_seed_v1",
                "training.model_initialization_v1",
                "data.raw_batch_binding_v1",
                "data.preview_whole_batch_v1",
                "edit.source_factory_device_v1",
                "runtime.torch_layer_catalog_v1",
                "runtime.operator_catalog_v1",
                "edit.variant_device_policy_v1",
                "edit.variant_layer_rng_v1",
                "edit.variant_base_rng_v1",
                "edit.conditional_variant_edit_v1",
                "model.conditional_variant_probe_v1",
                "data.preview_random_seed_v1",
                "supervision.conditional_retag",
                "supervision.conditional_rule_add_v1",
                "supervision.rule_copy_v1",
                "supervision.bundle_v1",
                "supervision.extended_axis_cli_v1",
                "supervision.environment_rule_v1",
                "supervision.observed_anchor_guard_v1",
                "supervision.root_mark_v1",
                "model.contract_identity_v1",
                "model.contract_preload_v1",
                "supervision.post_forward_scope_v1",
                "supervision.external_source_anchor_v1",
                "project.static_entrypoint_discovery",
                "project.static_entrypoint_discovery_roots_v1",
                "training.conditional_variant_start",
                "training.conditional_resume_checkpoint",
                "training.resume_allow_state_reset_v1",
                "runtime.historical_python_v1",
                "evaluation.conditional_checkpoint",
                "evaluation.model_device_v1",
                "export.conditional_checkpoint",
                "export.explicit_python_fallback_v1",
                "training.data_fingerprint",
                "runs.comparison_unverified_data",
                "augmentation.conditional_config_edit",
                "augmentation.catalog_v1",
                "edit.conditional_source_manifest",
                "edit.conditional_source_plan",
                "edit.atomic_source_transaction",
                "edit.mixed_source_transaction_v1",
                "edit.starter_migration",
                "edit.conditional_source_input",
                "edit.conditional_source_config",
                "edit.conditional_source_project_manifest",
                "layer.builtin_catalog",
                "workspace.authorization_identity_v1",
            ]
            .into_iter()
            .map(|name| (name, true))
            .collect::<std::collections::BTreeMap<_, _>>();
            capabilities.insert("training.generic_loop_detached_cli_v1", cfg!(target_os = "linux"));
            capabilities.insert("training.detached_cli_v1", cfg!(target_os = "linux"));
            capabilities.insert("supervision.conditional_variant_mark_v1", true);
            capabilities.insert("supervision.conditional_variant_retag_v1", true);
            capabilities.insert("supervision.conditional_variant_preview_v1", true);
            // Metadata only: inspect the same project/PATH candidates used by
            // ordinary non-environment, non-historical commands. Do not run
            // Python; the candidate can still change before execution.
            let python_hint = if let Some(python) = cli.python.as_deref() {
                serde_json::json!({"status":"selected", "source":"explicit", "python":python})
            } else {
                match std::env::current_dir().map_err(|e| e.to_string())
                    .and_then(|root| FileProjectConfigStore.load(&root)) {
                    Err(_) => serde_json::json!({"status":"unknown", "source":"invalid_project_config"}),
                    Ok(config) => {
                        let saved = config.as_ref().and_then(|value| value["runtime_python"].as_str());
                        if let Some(python) = saved {
                            serde_json::json!({"status":"selected", "source":"project", "python":python})
                        } else if let Some(python) = project_python_executable() {
                            serde_json::json!({"status":"selected", "source":"project_environment", "python":python})
                        } else if std::fs::symlink_metadata(".model-maven/environment.json").is_ok() {
                            serde_json::json!({"status":"unknown", "source":"project_environment_unavailable"})
                        } else if let Some(python) = path_python_executable() {
                            serde_json::json!({"status":"selected", "source":"path", "python":python})
                        } else {
                            serde_json::json!({"status":"unavailable", "source":"none"})
                        }
                    }
                }
            };
            Report::ok(
                "interface",
                serde_json::json!({
                    "interface_version": 1,
                    "cli_version": env!("CARGO_PKG_VERSION"),
                    "report_protocol_version": domain::PROTOCOL_VERSION,
                    "capabilities": capabilities,
                    "classification_create_presets": cli::CLASSIFICATION_CREATE_PRESETS,
                    "scaffold_encoder_layers": app::scaffold_encoder_layer_catalog(),
                    "python_selection_hint": python_hint,
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
                | EnvironmentCommand::Devices { .. }
                | EnvironmentCommand::Probe { .. }
                | EnvironmentCommand::Wheelhouse { .. }
        })
    );
    // These handlers do not run the Python worker. Let them report their
    // own configuration errors or help recover a broken environment.
    let environment_independent = matches!(cli.command.as_ref(),
        Some(Command::Discover { .. } | Command::Init { .. } | Command::Config { .. } | Command::Runs { .. }
            | Command::Assembly { command: AssemblyCommand::Show { config_only: true, .. } }
            | Command::Layer { command: LayerCommand::Catalog }
            | Command::Data { command: DataCommand::CsvTemplate { .. } }
            | Command::Component { command: ComponentCommand::Scaffold { .. } | ComponentCommand::List { .. }
                | ComponentCommand::Show { .. } | ComponentCommand::Check { .. } }
            | Command::Scaffold { .. } | Command::RecipeLayerPlan { .. } | Command::RecipeLayerEditPlan { .. }));
    // Export is based on the saved run and may not need Python at all. Do not
    // bind it to a newly selected or broken project environment before the
    // export handler can inspect the recorded runtime.
    let historical_export = matches!(cli.command.as_ref(), Some(Command::Export { .. } | Command::LoopRun { resume: Some(_), .. }));
    if cli.python.is_none() && !historical_export {
        match std::env::current_dir().map_err(|error| error.to_string())
            .and_then(|root| FileProjectConfigStore.load(&root).map(|config| (root, config))) {
            Ok((root, config)) => {
                let choice = config.as_ref().and_then(|value| value["runtime_python"].as_str()).map(str::to_owned);
                record_project_python_selection(&root, choice.clone());
                cli.python = choice;
            },
            Err(error) if !environment_independent => {
                let report = Report::error("environment", "PROJECT_CONFIG_INVALID", error,
                    "检查项目配置或明确使用 --python；没有自动忽略配置中的运行时选择。");
                display(&report, cli.json);
                return ExitCode::from(report.exit_code() as u8);
            },
            Err(_) => {},
        }
    }
    if cli.python.is_none() && !environment_command && !historical_export {
        cli.python = project_python_executable();
        let saved_environment_exists = std::fs::symlink_metadata(".model-maven/environment.json").is_ok();
        if cli.python.is_none() && saved_environment_exists && !environment_independent {
            let report = Report::error(
                "environment",
                "PROJECT_ENVIRONMENT_UNAVAILABLE",
                "项目保存的 Python 环境配置无效或所选解释器不可用。",
                "检查 .model-maven/environment.json 和所选环境，或明确使用 --python 指定解释器；未自动改用其他环境。",
            );
            display(&report, cli.json);
            return ExitCode::from(report.exit_code() as u8);
        }
    }
    if cli.python.is_none() && !historical_export {
        cli.python = path_python_executable();
    }
    let mut app = Application {
        backend: PythonBackend {
            executable: cli.python.clone().unwrap_or_default(),
            timeout: Duration::from_secs(cli.timeout.unwrap_or(30)),
            verbose: cli.verbose,
        },
        runtime_capabilities: std::cell::RefCell::new(None),
        runtime_declarations: std::cell::RefCell::new(None),
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

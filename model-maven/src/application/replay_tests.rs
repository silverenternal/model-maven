use super::*;
use crate::presentation::workbench::run_command;
use crate::{
    adapters::{
        component_scaffold::FileComponentScaffolder, components::FileComponentRegistry,
        exports::FileArtifactExporter, project_config::FileProjectConfigStore, runs::FileRunStore,
        source_edits::FileSourceEditStore, supervisor::NativeRunSupervisor,
        variants::FileVariantStore,
    },
    domain::Report,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Exchange {
    request: Value,
    response: Report,
}

#[derive(Clone)]
struct Recorder(Arc<Mutex<Vec<Exchange>>>);

impl Backend for Recorder {
    fn execute(&self, request: &Value) -> Report {
        let operation = request["operation"].as_str().unwrap_or("unknown");
        let response = if operation == "doctor" {
            Report::ok(
                operation,
                json!({
                    "compatibility": {
                        "worker_protocol_version": 1,
                        "worker_api_version": 1,
                        "state_formats": {}
                    },
                    "capabilities": [{
                        "id":"model.inspect",
                        "state":"supported",
                        "provider":"recorded"
                    }]
                }),
            )
        } else {
            Report::ok(operation, json!({"runtime":"recorded"}))
        };
        self.0.lock().unwrap().push(Exchange {
            request: request.clone(),
            response: response.clone(),
        });
        response
    }
}

struct Replayer(RefCell<VecDeque<Exchange>>);

struct FixedConfig(Option<Value>);
impl crate::ports::ProjectConfigStore for FixedConfig {
    fn load(&self, _: &std::path::Path) -> Result<Option<Value>, String> {
        Ok(self.0.clone())
    }
    fn create(&self, _: &std::path::Path, _: &Value) -> Result<std::path::PathBuf, String> {
        unreachable!()
    }
    fn update(
        &self,
        _: &std::path::Path,
        _: &Value,
        _: &Value,
    ) -> Result<std::path::PathBuf, String> {
        unreachable!()
    }
}

#[test]
fn training_guard_rechecks_augmentation_config_and_rejects_deletion() {
    let baseline = json!({"schema_version":1,"model":"x:model","batch":"x:batch"});
    let expected = crate::domain::input::content_revision(&baseline).unwrap();
    for changed in [
        Some(
            json!({"schema_version":1,"model":"x:model","batch":"x:batch","augment":{"train":[]}}),
        ),
        None,
    ] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut app = application(Recorder(Arc::clone(&calls)));
        app.project_config = Box::new(FixedConfig(changed));
        let mut job = empty_job();
        job.operation = "train";
        let mut request = json!({});
        let error = app
            .attach_training_augmentation(
                &mut job,
                std::path::Path::new("/"),
                &mut request,
                Some(&expected),
            )
            .unwrap_err();
        assert_eq!(error.diagnostics[0].code, "PROJECT_CONFIG_CONFLICT");
        assert!(request.get("train_augment").is_none());
        assert!(calls.lock().unwrap().is_empty());
    }
}

impl Backend for Replayer {
    fn execute(&self, request: &Value) -> Report {
        let exchange = self.0.borrow_mut().pop_front().expect("recorded exchange");
        assert_eq!(&exchange.request, request);
        exchange.response
    }
}

#[derive(Default)]
struct RuntimeSwitchBackend(RefCell<Option<String>>);

impl Backend for RuntimeSwitchBackend {
    fn execute(&self, request: &Value) -> Report {
        Report::ok(
            request["operation"].as_str().unwrap_or("unknown"),
            json!({}),
        )
    }
}

impl crate::ports::PythonRuntimeControl for RuntimeSwitchBackend {
    fn select_python(&mut self, executable: String) {
        *self.0.borrow_mut() = Some(executable);
    }
}

fn application<B: Backend>(backend: B) -> Application<B> {
    Application {
        backend,
        runtime_capabilities: RefCell::new(None),
        runtime_declarations: RefCell::new(None),
        variants: Box::new(FileVariantStore),
        runs: Box::new(FileRunStore),
        exporter: Box::new(FileArtifactExporter),
        supervisor: Box::new(NativeRunSupervisor),
        components: Box::new(FileComponentRegistry),
        scaffolder: Box::new(FileComponentScaffolder),
        source_edits: Box::new(FileSourceEditStore),
        project_config: Box::new(FileProjectConfigStore),
        project_source_writer: Box::new(crate::adapters::project_source::FileProjectSourceWriter),
        previews: Box::new(crate::adapters::previews::FilePreviewStore),
    }
}

fn doctor_cli() -> Cli {
    Cli {
        describe_interface: false,
        json: true,
        no_interactive: false,
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
        command: Some(Command::Doctor),
    }
}

#[test]
fn application_requests_can_be_recorded_and_replayed_without_terminal_or_python_runtime() {
    let recording = Arc::new(Mutex::new(Vec::new()));
    let recorded = application(Recorder(recording.clone())).run(&doctor_cli(), &Command::Doctor);
    let transcript = serde_json::to_vec(&*recording.lock().unwrap()).unwrap();
    let exchanges: VecDeque<Exchange> = serde_json::from_slice(&transcript).unwrap();
    assert_eq!(exchanges.len(), 1);
    assert_eq!(exchanges[0].request["operation"], "doctor");

    let replayed =
        application(Replayer(RefCell::new(exchanges))).run(&doctor_cli(), &Command::Doctor);
    assert_eq!(
        serde_json::to_value(recorded).unwrap(),
        serde_json::to_value(replayed).unwrap()
    );
}

#[test]
fn human_and_json_cli_modes_share_the_same_application_service() {
    let recording = Arc::new(Mutex::new(Vec::new()));
    let app = application(Recorder(recording.clone()));
    let command = Command::Model {
        command: ModelCommand::Inspect {
            path: None,
            inputs: Inputs {
                model: Some("fixture:build_model".into()),
                ..Inputs::default()
            },
        },
    };
    let mut human_cli = doctor_cli();
    human_cli.json = false;
    let json_cli = doctor_cli();

    let human_report = run_command(&app, &human_cli, &command);
    let json_report = run_command(&app, &json_cli, &command);

    assert_eq!(
        serde_json::to_value(human_report).unwrap(),
        serde_json::to_value(json_report).unwrap()
    );
    let exchanges = recording.lock().unwrap();
    assert_eq!(exchanges.len(), 3);
    assert_eq!(exchanges[0].request["operation"], "doctor");
    assert_eq!(exchanges[1].request["operation"], "inspect");
    assert_eq!(exchanges[1].request, exchanges[2].request);
}

struct LiveTestSupervisor;

impl crate::ports::RunSupervisor for LiveTestSupervisor {
    fn identify(&self, _pid: u32) -> Option<String> {
        Some("test-start-token".into())
    }

    fn launch(
        &self,
        _executable: &Path,
        _project_root: &Path,
        _run_id: &str,
        _request: &Value,
        _log_path: &Path,
        _register_identity: &mut dyn FnMut(u32, &str) -> Result<(), String>,
    ) -> Result<(), String> {
        Err("launch is not used by this test".into())
    }

    fn is_alive(&self, pid: u32, token: &str) -> bool {
        pid == std::process::id() && token == "test-start-token"
    }

    fn interrupt(&self, _pid: u32, _token: &str) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn resume_rejects_a_live_run_and_a_run_with_unknown_supervisor_identity() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let backend = application(Recorder(Arc::new(Mutex::new(Vec::new()))));
    let active_id = backend
        .runs
        .create(
            &root_path,
            &json!({"operation":"train","project_root":root_path,"training_mode":"manual-step"}),
        )
        .unwrap();
    backend
        .runs
        .set_supervisor(
            &root_path,
            &active_id,
            std::process::id(),
            "test-start-token",
        )
        .unwrap();
    let mut backend = backend;
    backend.supervisor = Box::new(LiveTestSupervisor);

    let active_error = backend.resume_record(&root_path, &active_id).unwrap_err();
    assert!(active_error.contains("仍由 PID"), "{active_error}");
    backend
        .runs
        .finish(
            &root_path,
            &active_id,
            &json!({"status":"success","data":{}}),
        )
        .unwrap();

    let unknown_id = backend
        .runs
        .create(
            &root_path,
            &json!({"operation":"train","project_root":root_path,"training_mode":"manual-step"}),
        )
        .unwrap();
    let metadata_path = backend
        .runs
        .run_directory(&root_path, &unknown_id)
        .unwrap()
        .join("run.json");
    let mut metadata: Value =
        serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
    metadata["status"] = json!("unknown");
    std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    let unknown_error = backend.resume_record(&root_path, &unknown_id).unwrap_err();
    assert!(
        unknown_error.contains("当前无法确认它已停止"),
        "{unknown_error}"
    );
}

#[test]
fn a_run_with_a_dead_verified_supervisor_is_interrupted_before_new_admission() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let app = application(Recorder(Arc::new(Mutex::new(Vec::new()))));
    let lost_id = app
        .runs
        .create(
            &root_path,
            &json!({"operation":"training_probe","project_root":root_path,"run_name":"lost"}),
        )
        .unwrap();
    app.runs
        .set_supervisor(&root_path, &lost_id, u32::MAX, "dead-process-token")
        .unwrap();

    app.reconcile_lost_runs(&root_path, "training_probe")
        .unwrap();
    let lost = app.runs.show(&root_path, &lost_id).unwrap();
    assert_eq!(lost["record"]["status"], "interrupted");
    assert_eq!(lost["report"]["diagnostics"][0]["code"], "RUN_PROCESS_LOST");
    assert!(
        app.runs
            .create(
                &root_path,
                &json!({"operation":"training_probe","project_root":root_path,"run_name":"next"}),
            )
            .is_ok()
    );
}

#[test]
fn stale_detached_heartbeat_is_reported_unknown_even_when_pid_exists() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let app = application(Recorder(Arc::new(Mutex::new(Vec::new()))));
    let id = app
        .runs
        .create(
            &root_path,
            &json!({"operation":"train","project_root":root_path,"run_name":"stale heartbeat"}),
        )
        .unwrap();
    let pid = std::process::id();
    let token = app.supervisor.identify(pid).unwrap();
    app.runs
        .set_supervisor(&root_path, &id, pid, &token)
        .unwrap();
    let directory = app.runs.run_directory(&root_path, &id).unwrap();
    let metadata_path = directory.join("run.json");
    let mut metadata: Value =
        serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
    metadata["supervisor"]["kind"] = json!("detached");
    std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    std::fs::write(
        directory.join("heartbeat.json"),
        serde_json::to_vec(&json!({"pid":pid,"start_token":token,"at_ms":0})).unwrap(),
    )
    .unwrap();

    let mut shown = app.runs.show(&root_path, &id).unwrap();
    app.annotate_status(&mut shown);
    assert_eq!(shown["record"]["status"], "unknown");
    assert_eq!(shown["record"]["last_observed_status"], "running");
    assert_eq!(shown["record"]["heartbeat_status"], "stale");
}

#[test]
fn selecting_a_python_runtime_invalidates_the_cached_capability_catalog() {
    let mut app = application(RuntimeSwitchBackend::default());
    app.runtime_capabilities
        .replace(Some(crate::domain::capability::CapabilityCatalog {
            worker_protocol_version: 1,
            worker_api_version: 1,
            state_formats: Default::default(),
            capabilities: vec![],
        }));

    crate::presentation::service::ApplicationUiServices::select_python(
        &mut app,
        "project/.venv/bin/python".into(),
    );

    assert!(app.runtime_capabilities.borrow().is_none());
    assert_eq!(
        app.backend.0.borrow().as_deref(),
        Some("project/.venv/bin/python")
    );
}

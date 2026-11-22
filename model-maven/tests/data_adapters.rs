use serde_json::Value;
use std::{path::Path, process::Command};

fn run_python(program: &str) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package = project.join("python");
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(program)
        .env("PYTHONPATH", package)
        .output()
        .expect("start Python data adapter fixture");
    assert!(
        output.status.success(),
        "Python fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("Python fixture emits JSON")
}

#[test]
fn indexed_dataset_sampling_is_seeded_and_keeps_nested_multi_input_data() {
    let result = run_python(
        r#"
import hashlib, json
from model_maven_worker.data.adapters import DatasetAdapter

class Dataset:
    def __init__(self, split, seed):
        self.split = split
        self.seed = seed
    def __len__(self):
        return 7
    def __getitem__(self, index):
        return {"image": {"pixels": [index, index + 1]}, "tokens": [index], "metadata": {"split": self.split}}
    def sample_id(self, index):
        return f"sample-{index}"

adapter = DatasetAdapter("fixture:dataset", Dataset)
first = adapter.sample(count=3, seed=17, split="train", fingerprint="sha256:data-v1")
again = adapter.sample(count=3, seed=17, split="train", fingerprint="sha256:data-v1")
print(json.dumps({
    "same_ids": first.observation.sample_ids == again.observation.sample_ids,
    "ids_stable": first.observation.stable_sample_ids,
    "count": first.observation.actual_samples,
    "split": first.observation.split,
    "image_nested": "image" in first.batch[0] and "pixels" in first.batch[0]["image"],
    "has_labels_requirement": "labels" in first.observation.coverage,
    "algorithm": first.observation.algorithm,
    "coverage": first.observation.coverage,
    "matches_legacy": first.observation.coverage["selected_indices"] == sorted(sorted(range(7), key=lambda i: hashlib.sha256(f"model-maven:17:{i}".encode()).digest())[:3]),
    "can_resume": first.observation.capabilities.resume_cursor,
}, sort_keys=True))
"#,
    );
    assert_eq!(result["same_ids"], true);
    assert_eq!(result["ids_stable"], true);
    assert_eq!(result["count"], 3);
    assert_eq!(result["split"], "train");
    assert_eq!(result["image_nested"], true);
    assert_eq!(result["has_labels_requirement"], false);
    assert_eq!(result["algorithm"], "sha256-index-rank-v1");
    assert_eq!(result["matches_legacy"], true);
    assert_eq!(result["can_resume"], false);
}

#[test]
fn huge_indexed_dataset_sampling_is_bounded_and_deterministic() {
    let result = run_python(
        r#"
import hashlib, json, time
import model_maven_worker.data.adapters as adapters
original = hashlib.sha256
calls = 0
def counted(*args, **kwargs):
    global calls
    calls += 1
    if calls > 1000: raise AssertionError('sampling scanned the dataset')
    return original(*args, **kwargs)
adapters.hashlib.sha256 = counted
class Huge:
    def __len__(self): return 1_000_000_000_000
    def __getitem__(self, index): return index
    def sample_id(self, index): return f'id-{index}'
adapter = adapters.DatasetAdapter('fixture:huge', Huge)
start = time.monotonic()
a = adapter.sample(count=12, seed=17, fingerprint='data-v1')
b = adapter.sample(count=12, seed=17, fingerprint='data-v1')
c = adapter.sample(count=12, seed=18, fingerprint='data-v1')
indices = a.observation.coverage['selected_indices']
print(json.dumps({'indices':indices, 'algorithm':a.observation.algorithm,
 'same':indices == b.observation.coverage['selected_indices'],
 'different':indices != c.observation.coverage['selected_indices'],
 'stable':a.observation.stable_sample_ids, 'hash_calls':calls, 'elapsed':time.monotonic()-start}))
"#,
    );
    assert_eq!(result["algorithm"], "sha256-counter-rejection-v2");
    assert_eq!(result["same"], true);
    assert_eq!(result["different"], true);
    assert_eq!(result["stable"], true);
    let indices = result["indices"].as_array().unwrap();
    assert_eq!(indices.len(), 12);
    assert!(
        indices
            .windows(2)
            .all(|pair| pair[0].as_u64() < pair[1].as_u64())
    );
    assert!(result["hash_calls"].as_u64().unwrap() < 1000);
}

#[test]
fn iterable_dataset_reports_prefix_coverage_and_missing_resume_protocol() {
    let result = run_python(
        r#"
import json
from model_maven_worker.data.adapters import DatasetAdapter

created = []
class Stream:
    def __init__(self):
        self.yielded = 0
    def __iter__(self):
        for index in range(5):
            self.yielded += 1
            yield {"video": [[index]], "tokens": [index, index + 1]}

def build_stream(seed=0):
    result = Stream()
    created.append(result)
    return result

adapter = DatasetAdapter("fixture:stream", build_stream)
sample = adapter.sample(count=2, seed=99)
another = adapter.sample(count=2, seed=99)
result = {
    "count": sample.observation.actual_samples,
    "kind": sample.observation.coverage["kind"],
    "representative": sample.observation.coverage["representative"],
    "stable_ids": sample.observation.stable_sample_ids,
    "resume": sample.observation.capabilities.resume_cursor,
    "deterministic": sample.observation.coverage["deterministic_selection"],
    "same_prefix": sample.observation.coverage["deterministic_given_same_prefix"],
    "seed_forwarded": sample.observation.coverage["seed_forwarded_to_factory"],
    "yielded": created[0].yielded,
    "created_independent_instance": created[0] is not created[1],
    "second_yielded": created[1].yielded,
    "reason": sample.observation.capabilities.reason,
}
print(json.dumps(result, sort_keys=True))
"#,
    );
    assert_eq!(result["count"], 2);
    assert_eq!(result["kind"], "bounded-prefix");
    assert_eq!(result["representative"], false);
    assert_eq!(result["stable_ids"], false);
    assert_eq!(result["resume"], false);
    assert_eq!(result["deterministic"], false);
    assert_eq!(result["same_prefix"], true);
    assert_eq!(result["seed_forwarded"], true);
    assert_eq!(result["yielded"], 2);
    assert_eq!(result["created_independent_instance"], true);
    assert_eq!(result["second_yielded"], 2);
    assert!(result["reason"].as_str().unwrap().contains("稳定样本 ID"));
}

#[test]
fn iterable_dataset_state_requires_and_uses_explicit_restore_hooks() {
    let result = run_python(
        r#"
import json
from model_maven_worker.data.adapters import DatasetAdapter

class Cursor:
    def __init__(self): self.position = 0
    def __iter__(self):
        while self.position < 5:
            value = self.position
            self.position += 1
            yield value
    def state_dict(self): return {"position": self.position}
    def load_state_dict(self, state): self.position = state["position"]

adapter = DatasetAdapter("fixture:cursor", Cursor)
sample = adapter.sample(count=2, seed=0)
restored = Cursor()
status = adapter.restore_state(restored, sample.state)
next_value = next(iter(restored))
print(json.dumps({
    "saved_position": sample.state["position"],
    "checkpoint_save": sample.observation.capabilities.checkpoint_save,
    "checkpoint_load": sample.observation.capabilities.checkpoint_load,
    "resume_cursor": sample.observation.capabilities.resume_cursor,
    "status": status,
    "next_value": next_value,
}, sort_keys=True))
"#,
    );
    assert_eq!(result["saved_position"], 2);
    assert_eq!(result["checkpoint_save"], true);
    assert_eq!(result["checkpoint_load"], true);
    assert_eq!(result["resume_cursor"], true);
    assert_eq!(result["status"], "restored");
    assert_eq!(result["next_value"], 2);
}

#[test]
fn direct_batch_factory_still_accepts_nested_envelopes_and_dataset_factories() {
    let result = run_python(
        r#"
import json
from types import SimpleNamespace
from model_maven_worker.data.batch import make_batch

class Tensor:
    pass
torch = SimpleNamespace(Tensor=Tensor)

class Dataset:
    def __len__(self): return 4
    def __getitem__(self, index):
        return {"args": ({"image": [index], "tokens": [index, index + 1]},), "kwargs": {}}
    def sample_id(self, index): return f"id-{index}"

class TupleDataset:
    def __len__(self): return 3
    def __getitem__(self, index): return ([index], [index + 10])

request = {"batch": "fixture:make", "batch_params": {}, "seed": 3}
args, kwargs, batch = make_batch(request, torch, lambda _: (lambda **__: Dataset()))

direct_request = {"batch": "fixture:envelope", "batch_params": {}}
direct = {"args": [{"video": [[1]], "text": [2]}], "kwargs": {}}
direct_args, direct_kwargs, _ = make_batch(direct_request, torch, lambda _: (lambda **__: direct))
tuple_request = {"batch": "fixture:tuple", "batch_binding": "args", "seed": 4}
tuple_args, tuple_kwargs, _ = make_batch(tuple_request, torch, lambda _: (lambda **__: TupleDataset()))
print(json.dumps({
    "dataset_args": args,
    "dataset_kwargs": kwargs,
    "dataset_source": request["_data_observation"]["source"],
    "dataset_actual_samples": request["_data_observation"]["sampling"]["actual_samples"],
    "dataset_coverage": request["_data_observation"]["coverage"]["kind"],
    "dataset_selected_index": request["_data_observation"]["coverage"]["selected_indices"][0],
    "dataset_value_index": args[0]["image"][0],
    "direct_args": direct_args,
    "direct_kwargs": direct_kwargs,
    "tuple_args": tuple_args,
    "tuple_kwargs": tuple_kwargs,
}, sort_keys=True))
"#,
    );
    assert_eq!(
        result["dataset_args"][0]["image"][0],
        result["dataset_selected_index"]
    );
    assert_eq!(result["dataset_kwargs"], serde_json::json!({}));
    assert_eq!(result["dataset_source"], "fixture:make");
    assert_eq!(result["dataset_actual_samples"], 1);
    assert_eq!(result["dataset_coverage"], "indexed-subset");
    assert_eq!(result["direct_args"][0]["video"][0][0], 1);
    assert_eq!(result["direct_kwargs"], serde_json::json!({}));
    assert_eq!(result["tuple_args"].as_array().unwrap().len(), 2);
    assert_eq!(result["tuple_kwargs"], serde_json::json!({}));
}

#[test]
fn iterable_window_sampling_is_bounded_and_conditional_on_source_order() {
    let result = run_python(
        r#"
import json
from model_maven_worker.data.adapters import DatasetAdapter
from model_maven_worker.protocol import UserError
created = []
class Stream:
    def __init__(self, length): self.length, self.reads = length, 0
    def __iter__(self):
        for i in range(self.length):
            self.reads += 1
            yield i
def factory(seed=0, length=100):
    instance = Stream(length)
    created.append(instance)
    return instance
adapter = DatasetAdapter('fixture:stream', factory)
a = adapter.sample(count=4, seed=5, scan_limit=20)
b = adapter.sample(count=4, seed=5, scan_limit=20)
c = adapter.sample(count=4, seed=6, scan_limit=20)
short = DatasetAdapter('fixture:stream', factory, {'length': 7}).sample(count=4, seed=5, scan_limit=20)
errors = []
for limit in (3, 100001, True):
    try: adapter.sample(count=4, seed=5, scan_limit=limit)
    except UserError as e: errors.append(e.code)
map_adapter = DatasetAdapter('fixture:map', lambda: [1,2,3,4])
try: map_adapter.sample(count=2, seed=5, scan_limit=4)
except UserError as e: errors.append(e.code)
print(json.dumps({'a':a.observation.coverage, 'b':b.observation.coverage, 'c':c.observation.coverage, 'short':short.observation.coverage, 'reads':[x.reads for x in created], 'errors':errors}))
"#,
    );
    assert_eq!(result["a"]["examined_items"], 20);
    assert_eq!(
        result["a"]["selected_offsets"],
        result["b"]["selected_offsets"]
    );
    assert_ne!(
        result["a"]["selected_offsets"],
        result["c"]["selected_offsets"]
    );
    assert_eq!(result["a"]["complete"], false);
    assert_eq!(result["a"]["deterministic_selection"], false);
    assert_eq!(result["short"]["complete"], false);
    assert_eq!(result["short"]["stream_exhausted"], true);
    assert_eq!(result["short"]["examined_items"], 7);
    assert_eq!(result["reads"], serde_json::json!([20, 20, 20, 7]));
    assert_eq!(
        result["errors"],
        serde_json::json!([
            "DATA_SCAN_LIMIT_INVALID",
            "DATA_SCAN_LIMIT_INVALID",
            "DATA_SCAN_LIMIT_INVALID",
            "DATA_SCAN_LIMIT_UNSUPPORTED"
        ])
    );
}

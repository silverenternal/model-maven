use std::{fs, io, path::Path};

struct WorkerAsset {
    path: &'static str,
    contents: &'static str,
    export_compatibility: bool,
}

const WORKER_ASSETS: &[WorkerAsset] = &[
    WorkerAsset {
        path: "worker.py",
        contents: include_str!("../../python/worker.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker.py",
        contents: include_str!("../../python/worker.py"),
        export_compatibility: true,
    },
    WorkerAsset {
        path: "model_maven_worker/__init__.py",
        contents: include_str!("../../python/model_maven_worker/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/protocol.py",
        contents: include_str!("../../python/model_maven_worker/protocol.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/runtime/__init__.py",
        contents: include_str!("../../python/model_maven_worker/runtime/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/runtime/capabilities.py",
        contents: include_str!("../../python/model_maven_worker/runtime/capabilities.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/runtime/entrypoints.py",
        contents: include_str!("../../python/model_maven_worker/runtime/entrypoints.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/__init__.py",
        contents: include_str!("../../python/model_maven_worker/environment/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/common.py",
        contents: include_str!("../../python/model_maven_worker/environment/common.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/lock.py",
        contents: include_str!("../../python/model_maven_worker/environment/lock.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/setup.py",
        contents: include_str!("../../python/model_maven_worker/environment/setup.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/snapshot.py",
        contents: include_str!("../../python/model_maven_worker/environment/snapshot.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/wheelhouse.py",
        contents: include_str!("../../python/model_maven_worker/environment/wheelhouse.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/environment/adapters.py",
        contents: include_str!("../../python/model_maven_worker/environment/adapters.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/__init__.py",
        contents: include_str!("../../python/model_maven_worker/extensions/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/catalog.py",
        contents: include_str!("../../python/model_maven_worker/extensions/catalog.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/metadata.py",
        contents: include_str!("../../python/model_maven_worker/extensions/metadata.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/conformance.py",
        contents: include_str!("../../python/model_maven_worker/extensions/conformance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/layer_conformance.py",
        contents: include_str!("../../python/model_maven_worker/extensions/layer_conformance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/value_conformance.py",
        contents: include_str!("../../python/model_maven_worker/extensions/value_conformance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/trainer_conformance.py",
        contents: include_str!("../../python/model_maven_worker/extensions/trainer_conformance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/source_inspection.py",
        contents: include_str!("../../python/model_maven_worker/extensions/source_inspection.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/lifecycle.py",
        contents: include_str!("../../python/model_maven_worker/extensions/lifecycle.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/bundle.py",
        contents: include_str!("../../python/model_maven_worker/extensions/bundle.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/provenance.py",
        contents: include_str!("../../python/model_maven_worker/extensions/provenance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/references.py",
        contents: include_str!("../../python/model_maven_worker/extensions/references.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/extensions/wheels.py",
        contents: include_str!("../../python/model_maven_worker/extensions/wheels.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/__init__.py",
        contents: include_str!("../../python/model_maven_worker/data/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/batch.py",
        contents: include_str!("../../python/model_maven_worker/data/batch.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/content_identity.py",
        contents: include_str!("../../python/model_maven_worker/data/content_identity.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/input_snapshot.py",
        contents: include_str!("../../python/model_maven_worker/data/input_snapshot.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/adapters.py",
        contents: include_str!("../../python/model_maven_worker/data/adapters.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/classification.py",
        contents: include_str!("../../python/model_maven_worker/data/classification.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/index.py",
        contents: include_str!("../../python/model_maven_worker/data/index.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/sampling.py",
        contents: include_str!("../../python/model_maven_worker/data/sampling.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/transforms.py",
        contents: include_str!("../../python/model_maven_worker/data/transforms.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/augment.py",
        contents: include_str!("../../python/model_maven_worker/data/augment.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/pipeline.py",
        contents: include_str!("../../python/model_maven_worker/data/pipeline.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/conformance.py",
        contents: include_str!("../../python/model_maven_worker/data/conformance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/preview.py",
        contents: include_str!("../../python/model_maven_worker/data/preview.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/renderers.py",
        contents: include_str!("../../python/model_maven_worker/data/renderers.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/data/presets.py",
        contents: include_str!("../../python/model_maven_worker/data/presets.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/tensor_inputs/__init__.py",
        contents: include_str!("../../python/model_maven_worker/tensor_inputs/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/tensor_inputs/content_identity.py",
        contents: include_str!("../../python/model_maven_worker/tensor_inputs/content_identity.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/tensor_inputs/snapshot.py",
        contents: include_str!("../../python/model_maven_worker/tensor_inputs/snapshot.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/__init__.py",
        contents: include_str!("../../python/model_maven_worker/model/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/tensor_tree.py",
        contents: include_str!("../../python/model_maven_worker/model/tensor_tree.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/contracts.py",
        contents: include_str!("../../python/model_maven_worker/model/contracts.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/evidence_store.py",
        contents: include_str!("../../python/model_maven_worker/model/evidence_store.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/local_probe.py",
        contents: include_str!("../../python/model_maven_worker/model/local_probe.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/shape.py",
        contents: include_str!("../../python/model_maven_worker/model/shape.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/structure.py",
        contents: include_str!("../../python/model_maven_worker/model/structure.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/inspection.py",
        contents: include_str!("../../python/model_maven_worker/model/inspection.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/construction.py",
        contents: include_str!("../../python/model_maven_worker/model/construction.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/discovery.py",
        contents: include_str!("../../python/model_maven_worker/model/discovery.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/loading.py",
        contents: include_str!("../../python/model_maven_worker/model/loading.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/probe.py",
        contents: include_str!("../../python/model_maven_worker/model/probe.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/identity_links.py",
        contents: include_str!("../../python/model_maven_worker/model/identity_links.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/model/semantics.py",
        contents: include_str!("../../python/model_maven_worker/model/semantics.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/__init__.py",
        contents: include_str!("../../python/model_maven_worker/train/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/device.py",
        contents: include_str!("../../python/model_maven_worker/train/device.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/runtime.py",
        contents: include_str!("../../python/model_maven_worker/train/runtime.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/state.py",
        contents: include_str!("../../python/model_maven_worker/train/state.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/managed.py",
        contents: include_str!("../../python/model_maven_worker/train/managed.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/optimizers.py",
        contents: include_str!("../../python/model_maven_worker/train/optimizers.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/manual.py",
        contents: include_str!("../../python/model_maven_worker/train/manual.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/external.py",
        contents: include_str!("../../python/model_maven_worker/train/external.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/generic_loop.py",
        contents: include_str!("../../python/model_maven_worker/train/generic_loop.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/generic_checkpoint.py",
        contents: include_str!("../../python/model_maven_worker/train/generic_checkpoint.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/checkpoint.py",
        contents: include_str!("../../python/model_maven_worker/train/checkpoint.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/probe.py",
        contents: include_str!("../../python/model_maven_worker/train/probe.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/runner.py",
        contents: include_str!("../../python/model_maven_worker/train/runner.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/evaluation.py",
        contents: include_str!("../../python/model_maven_worker/train/evaluation.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/supervision.py",
        contents: include_str!("../../python/model_maven_worker/train/supervision.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/train/custom_rules.py",
        contents: include_str!("../../python/model_maven_worker/train/custom_rules.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/run/__init__.py",
        contents: include_str!("../../python/model_maven_worker/run/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/run/artifacts.py",
        contents: include_str!("../../python/model_maven_worker/run/artifacts.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/run/file_digest.py",
        contents: include_str!("../../python/model_maven_worker/run/file_digest.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/run/provenance.py",
        contents: include_str!("../../python/model_maven_worker/run/provenance.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/run/snapshot.py",
        contents: include_str!("../../python/model_maven_worker/run/snapshot.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/__init__.py",
        contents: include_str!("../../python/model_maven_worker/edit/__init__.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/layers.py",
        contents: include_str!("../../python/model_maven_worker/edit/layers.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/initialization_rng.py",
        contents: include_str!("../../python/model_maven_worker/edit/initialization_rng.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/providers.py",
        contents: include_str!("../../python/model_maven_worker/edit/providers.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/sharing.py",
        contents: include_str!("../../python/model_maven_worker/edit/sharing.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/migration.py",
        contents: include_str!("../../python/model_maven_worker/edit/migration.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/codegen.py",
        contents: include_str!("../../python/model_maven_worker/edit/codegen.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/variants.py",
        contents: include_str!("../../python/model_maven_worker/edit/variants.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/plan.py",
        contents: include_str!("../../python/model_maven_worker/edit/plan.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/source_ast.py",
        contents: include_str!("../../python/model_maven_worker/edit/source_ast.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/runtime/source_paths.py",
        contents: include_str!("../../python/model_maven_worker/runtime/source_paths.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/starter_migration.py",
        contents: include_str!("../../python/model_maven_worker/edit/starter_migration.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/source_candidates.py",
        contents: include_str!("../../python/model_maven_worker/edit/source_candidates.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/source_set.py",
        contents: include_str!("../../python/model_maven_worker/edit/source_set.py"),
        export_compatibility: false,
    },
    WorkerAsset {
        path: "model_maven_worker/edit/source_dispatch.py",
        contents: include_str!("../../python/model_maven_worker/edit/source_dispatch.py"),
        export_compatibility: false,
    },
];

pub fn stage_worker_package(root: &Path, include_export_compatibility: bool) -> io::Result<()> {
    for asset in WORKER_ASSETS {
        if asset.export_compatibility && !include_export_compatibility {
            continue;
        }
        let path = root.join(asset.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, asset.contents)?;
    }
    Ok(())
}

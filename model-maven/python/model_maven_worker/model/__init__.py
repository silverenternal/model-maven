"""Model construction and runtime inspection services."""

from .tensor_tree import tree

from .structure import module_entries, normalize_path, resolve_module

from .inspection import (
    editable_params,
    has_module_state,
    inspect_model,
    source_constructor_fields,
    source_editable_fields,
    source_set_block_reason,
)

from .construction import construct_model, project_construction_site, validate_model_injections

from .probe import probe, verify_gradients

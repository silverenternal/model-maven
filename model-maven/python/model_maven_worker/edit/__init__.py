"""Model editing operations."""

from .layers import (
    LayerEditRuntime,
    apply_edit,
    apply_history_item,
    layer_container_kind,
    new_layer,
    replace_module,
)
from .variants import EditValidationRuntime, valid_edit_operation, validate_variant
from .plan import EditPlanRuntime, edit_candidate
from .providers import apply_provider
from .sharing import aliases_for, clone_selected, module_paths, share_selected
from .migration import migrate_state_dict
from .source_ast import (
    ast_offset,
    call_name,
    source_factory_expression,
    source_module_expression,
    source_sequential_rebind,
    unique_source_alias,
)
from .source_candidates import SourceCandidateRuntime, source_sequential_candidate

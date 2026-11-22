"""Component discovery and extension metadata services."""

from .catalog import (
    discover_components,
    distribution_manifest_path,
    distribution_module_source,
    entrypoint_module,
    project_extension_paths,
    safe_distribution_file,
)

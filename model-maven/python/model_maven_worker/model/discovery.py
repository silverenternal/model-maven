"""Construction provenance scoped to one model factory invocation."""

import os
import sys
from pathlib import Path

from ..runtime.source_paths import is_project_python_source
_MAX_CAPTURED_MODULES = 16384


def project_construction_site(project_root, module):
    """Find the project frame that constructed ``module``."""
    project_root = Path(project_root).resolve()
    module_init = getattr(type(module), "__init__", None)
    module_init_code = getattr(module_init, "__code__", None)
    frame = sys._getframe(1).f_back
    try:
        while frame is not None:
            if frame.f_code is not module_init_code:
                filename = frame.f_code.co_filename
                try:
                    path = Path(filename).resolve()
                except (OSError, ValueError, TypeError):
                    pass
                else:
                    # The helper performs the project-boundary check itself.
                    # Repeating relative_to here adds work for every module.
                    if is_project_python_source(project_root, path, resolved=True):
                        return {
                            "file": str(path),
                            "source_path": os.path.abspath(filename),
                            "line": frame.f_lineno,
                            "function": frame.f_code.co_name,
                        }
            frame = frame.f_back
    finally:
        del frame
    return None


class ConstructionSiteCollector:
    """Keep construction identity only for the duration of one factory call.

    Strong references prevent Python from reusing ``id(module)`` while the
    factory is still running. ``finish`` intersects the captures with the
    returned model, so discarded temporary modules cannot affect provenance or
    source-edit eligibility.
    """

    def __init__(self, project_root):
        self.project_root = project_root
        self._captured = {}

    def record(self, module):
        if len(self._captured) >= _MAX_CAPTURED_MODULES:
            return
        site = project_construction_site(self.project_root, module)
        if site is not None:
            self._captured[id(module)] = (module, site)

    def finish(self, model, module_entries_fn):
        locations = {}
        site_counts = {}
        modules, _aliases = module_entries_fn(model)
        for identity, module in modules.items():
            captured = self._captured.get(identity)
            if captured is None or captured[0] is not module:
                continue
            site = captured[1]
            locations[identity] = site
            site_key = (
                type(module).__module__,
                type(module).__qualname__,
                site["source_path"],
                site["line"],
            )
            site_counts[site_key] = site_counts.get(site_key, 0) + 1
        self._captured.clear()
        return locations, site_counts

    def discard(self):
        self._captured.clear()


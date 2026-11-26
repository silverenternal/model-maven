"""Bounded, metadata-only matching of observed forward tensor objects.

Matches are evidence that an input is the same Python Tensor object and has
the same PyTorch version counter as an earlier recorded module output. They
do not prove data provenance, completeness, or absence of unobserved work.
Weak references avoid retaining tensors merely to construct this evidence.
"""

import weakref
from itertools import islice

MAX_TENSORS = 4096
MAX_WALK_ITEMS = 4096
MAX_DEPTH = 6
MAX_SEQUENCE = 32
MAX_MATCHES = 8
MAX_TENSORS_PER_CALL = 128
MAX_RELATION_UNITS = 8192


class TensorIdentityTrace:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.entries = {}
        self.truncated = False
        self.relation_units = 0

    def _walk(self, roots):
        count = 0
        active = set()

        def visit(value, path, depth):
            nonlocal count
            count += 1
            if count > MAX_WALK_ITEMS or depth >= MAX_DEPTH:
                self.truncated = True
                return
            if type(value) is self.torch.Tensor:
                try:
                    version = value._version
                    if type(version) is int:
                        yield path, value, version
                except (RuntimeError, AttributeError):
                    self.truncated = True
                return
            if type(value) not in (tuple, list, dict):
                if isinstance(value, self.torch.Tensor):
                    self.truncated = True
                return
            identifier = id(value)
            if identifier in active:
                self.truncated = True
                return
            active.add(identifier)
            try:
                if type(value) is dict:
                    if len(value) > MAX_SEQUENCE:
                        self.truncated = True
                    for key, item in islice(value.items(), MAX_SEQUENCE):
                        if (type(key) is not str or not key or len(key) > 128
                                or any(ch in key for ch in ".[]")
                                or any(ord(ch) < 32 or ord(ch) == 127 for ch in key)):
                            self.truncated = True
                            continue
                        yield from visit(item, f"{path}.{key}", depth + 1)
                else:
                    if len(value) > MAX_SEQUENCE:
                        self.truncated = True
                    for index, item in enumerate(value[:MAX_SEQUENCE]):
                        yield from visit(item, f"{path}[{index}]", depth + 1)
            finally:
                active.remove(identifier)

        for value, path in roots:
            yield from visit(value, path, 0)

    def inputs(self, args, kwargs):
        # Once the relation budget is exhausted and coverage is already
        # marked partial, walking later nested inputs cannot add evidence or
        # change the coverage report.
        if self.truncated and self.relation_units >= MAX_RELATION_UNITS:
            return []
        matches = []
        for path, value, version in self._walk(((args, "inputs.args"), (kwargs, "inputs.kwargs"))):
            if len(matches) >= MAX_TENSORS_PER_CALL:
                self.truncated = True
                break
            remaining = MAX_RELATION_UNITS - self.relation_units
            if remaining <= 0:
                self.truncated = True
                break
            prior = []
            for reference, output_version, invocation_id, output_path in self.entries.get(id(value), ()):
                if reference() is value and output_version == version:
                    prior.append({"invocation_id": invocation_id, "output_path": output_path})
            if len(prior) > remaining - 1:
                prior = prior[-(remaining - 1):] if remaining > 1 else []
                self.truncated = True
            matches.append({"input_path": path, "prior_outputs": prior})
            self.relation_units += 1 + len(prior)
        return matches

    def outputs(self, value, invocation_id):
        for count, (path, tensor, version) in enumerate(self._walk(((value, "outputs"),))):
            if count >= MAX_TENSORS_PER_CALL:
                self.truncated = True
                break
            identifier = id(tensor)
            if identifier not in self.entries and len(self.entries) >= MAX_TENSORS:
                self.truncated = True
                continue
            entries = self.entries.setdefault(identifier, [])
            # Discard stale identities if Python has reused an object id.
            entries[:] = [entry for entry in entries if entry[0]() is tensor]
            if len(entries) == MAX_MATCHES:
                entries.pop(0)
                self.truncated = True
            entries.append((weakref.ref(tensor), version, invocation_id, path))

    def coverage(self):
        return {"kind": "same_tensor_object_and_version", "scope": "partial", "truncated": self.truncated}

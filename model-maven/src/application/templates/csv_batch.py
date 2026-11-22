"""Editable, bounded numeric CSV batch source for Model-Maven starters.

Use project batch parameters such as:
  path="data.csv", inputs={"features":["x1","x2"]}, targets=["y"]
For a model returning a named dictionary, targets can be
  {"score":["score_target"], "classes":["c1","c2"]}.
Lists default to float32 vectors. Structured fields support fixed shapes and
float32/float64/float16/int64/bool, for example:
  inputs={"image":{"columns":["p0","p1","p2","p3"],"shape":[2,2]},
          "tokens":{"columns":["t0","t1"],"dtype":"int64"}}
For a single scalar target: targets={"$tensor":{"columns":["label"],
                                                "shape":[],"dtype":"int64"}}
Replace this module for variable-length records or other datasets.
"""
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

import torch

_MAX_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 100_000
_MAX_COLUMNS = 4096
_MAX_SELECTED_COLUMNS = 256
_MAX_NUMERIC_VALUES = 2_000_000


def _columns(value, label):
    if not isinstance(value, list) or not value or len(value) > _MAX_COLUMNS:
        raise ValueError(f"{label} must be a nonempty list of at most {_MAX_COLUMNS} column names")
    if any(not isinstance(x, str) or not x or len(x) > 256 for x in value):
        raise ValueError(f"{label} column names must be nonempty strings")
    return value


def _field(value, label):
    if isinstance(value, list):
        columns = _columns(value, label)
        return {"columns": columns, "shape": [len(columns)], "dtype": "float32"}
    if not isinstance(value, dict) or set(value) - {"columns", "shape", "dtype"}:
        raise ValueError(f"{label} must be a column list or a columns/shape/dtype object")
    columns = _columns(value.get("columns"), f"{label}.columns")
    shape = value.get("shape", [len(columns)])
    dtype = value.get("dtype", "float32")
    if (not isinstance(shape, list) or len(shape) > 4
            or any(type(dim) is not int or dim < 1 for dim in shape)
            or math.prod(shape) != len(columns)):
        raise ValueError(f"{label}.shape must contain at most four positive dimensions with product equal to its column count; [] is scalar")
    if not isinstance(dtype, str) or dtype not in {"float32", "float64", "float16", "int64", "bool"}:
        raise ValueError(f"{label}.dtype is unsupported")
    return {"columns": columns, "shape": shape, "dtype": dtype}


def _mapping(value, label):
    if not isinstance(value, dict) or not value or len(value) > 64:
        raise ValueError(f"{label} must be a nonempty object with at most 64 names")
    if any(not isinstance(k, str) or not k or len(k) > 128 for k in value):
        raise ValueError(f"{label} names must be nonempty strings")
    return {name: _field(field, f"{label}.{name}") for name, field in value.items()}


def _scalar(value, dtype):
    if not isinstance(value, str):
        raise ValueError("missing value")
    if dtype == "bool":
        normalized = value.strip().lower()
        if normalized not in {"0", "1", "false", "true"}:
            raise ValueError("bool must be 0, 1, false or true")
        return normalized in {"1", "true"}
    if dtype == "int64":
        if not re.fullmatch(r"[+-]?[0-9]+", value.strip()):
            raise ValueError("int64 must be a decimal integer")
        result = int(value)
        if not -(1 << 63) <= result < (1 << 63):
            raise ValueError("int64 is out of range")
        return result
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("float must be finite")
    return result


class CsvBatch:
    """Cycles rows in file order; cursor and input fingerprint are checkpointed."""
    def __init__(self):
        self._cursor = 0
        self._fingerprint = None
        self._cache = None

    def state_dict(self):
        return {"version": 1, "cursor": self._cursor, "fingerprint": self._fingerprint}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get("version") != 1
                or type(state.get("cursor")) is not int or state["cursor"] < 0
                or not isinstance(state.get("fingerprint"), str)
                or len(state["fingerprint"]) != 64):
            raise ValueError("invalid CSV batch checkpoint")
        self._cursor = state["cursor"]
        self._fingerprint = state["fingerprint"]
        self._cache = None

    def __call__(self, path, inputs, targets, batch_size=2):
        if not isinstance(path, str) or not path or len(path) > 4096:
            raise ValueError("path must be a nonempty CSV path")
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be an integer from 1 to 256")
        input_columns = _mapping(inputs, "inputs")
        single_wrapped = isinstance(targets, dict) and set(targets) == {"$tensor"}
        named_targets = isinstance(targets, dict) and not single_wrapped
        target_columns = (_mapping(targets, "targets") if named_targets
                          else {"target": _field(targets["$tensor"] if single_wrapped else targets, "targets")})
        config = {"path": str(Path(path).resolve()), "inputs": input_columns,
                  "targets": target_columns, "named_targets": named_targets, "batch_size": batch_size}
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        source = Path(config["path"])
        before = source.stat()
        if not source.is_file() or before.st_size > _MAX_BYTES:
            raise ValueError("CSV must be a regular file no larger than 16 MiB")
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if self._cache is None:
            with source.open("rb") as stream:
                raw = stream.read(_MAX_BYTES + 1)
            after = source.stat()
            if len(raw) > _MAX_BYTES or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("CSV changed while being read; retry with a stable file")
            digest = hashlib.sha256()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            digest.update(config_bytes)
            fingerprint = digest.hexdigest()
            if self._fingerprint is not None and self._fingerprint != fingerprint:
                raise ValueError("CSV contents or batch mapping changed since the checkpoint")
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
            header = reader.fieldnames
            if not header or len(header) > _MAX_COLUMNS or any(not key for key in header) or len(header) != len(set(header)):
                raise ValueError("CSV needs a header with unique nonempty columns")
            fields = (*input_columns.values(), *target_columns.values())
            required = {column for field in fields for column in field["columns"]}
            conversions = {(column, field["dtype"]) for field in fields for column in field["columns"]}
            if len(required) > _MAX_SELECTED_COLUMNS or len(conversions) > _MAX_SELECTED_COLUMNS:
                raise ValueError("At most 256 distinct CSV columns may be selected")
            missing = sorted(required - set(header))
            if missing:
                raise ValueError(f"CSV columns not found: {missing}")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if len(rows) >= _MAX_ROWS or (len(rows) + 1) * len(conversions) > _MAX_NUMERIC_VALUES:
                    raise ValueError("CSV exceeds the row or selected-value limit")
                if None in row or any(row[column] is None for column in header):
                    raise ValueError(f"CSV row {row_number} does not match the header")
                try:
                    values = {(column, dtype): _scalar(row[column], dtype) for column, dtype in conversions}
                except (ValueError, TypeError) as error:
                    raise ValueError(f"CSV row {row_number} contains an invalid typed value: {error}") from error
                rows.append(values)
            if not rows:
                raise ValueError("CSV has no data rows")
            self._fingerprint = fingerprint
            self._cache = (identity, config_bytes, rows)
        cached_identity, cached_config, rows = self._cache
        if cached_identity != identity or cached_config != config_bytes:
            raise ValueError("CSV or batch mapping changed during this process; start a new run")
        selected = [rows[(self._cursor + index) % len(rows)] for index in range(batch_size)]
        def tensor(field):
            dtype = field["dtype"]
            value = torch.tensor([[row[(column, dtype)] for column in field["columns"]] for row in selected], dtype=getattr(torch, dtype))
            value = value.reshape((batch_size, *field["shape"]))
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"CSV value overflows {dtype}")
            return value
        kwargs = {name: tensor(field) for name, field in input_columns.items()}
        target_values = {name: tensor(field) for name, field in target_columns.items()}
        loss_inputs = {"targets": target_values} if named_targets else {"target": target_values["target"]}
        self._cursor = (self._cursor + batch_size) % len(rows)
        return {"args": [], "kwargs": kwargs, "loss_inputs": loss_inputs}


batch = CsvBatch()

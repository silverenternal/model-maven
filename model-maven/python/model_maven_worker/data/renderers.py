"""Concrete, self-contained renderers for data preview artifacts."""

import base64
import html
import io
import json
import math
from pathlib import Path

from ..protocol import UserError


RENDERER_VERSION = 2
MAX_GRID_SAMPLES = 64
THUMBNAIL_SIZE = (224, 224)


def render_auto(samples, output_directory, title="Data preview"):
    """Render image samples to a real PNG grid and every sample set to standalone HTML."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptors = [_sample_descriptor(sample, index) for index, sample in enumerate(samples)]
    images = []
    image_indices = []
    for index, sample in enumerate(samples[:MAX_GRID_SAMPLES]):
        value = _sample_value(sample)
        if not _is_image_candidate(value):
            continue
        try:
            images.append(_to_pil(value))
            image_indices.append(index)
        except Exception as error:
            raise UserError(
                "PREVIEW_IMAGE_RENDER_FAILED",
                f"第 {index + 1} 个图像样本无法渲染：{type(error).__name__}: {error}",
                "确认解码和变换后的 shape、dtype 与图像通道数；不会静默替换样本。",
                {"sample_index": index},
            ) from error

    artifacts = []
    image_path = None
    if images:
        image_path = output_directory / "preview.png"
        _save_grid(images, [descriptors[index] for index in image_indices], image_path, title)
        artifacts.append({"kind": "image_grid", "path": image_path.name, "media_type": "image/png"})

    html_path = output_directory / "preview.html"
    html_path.write_text(
        _build_html(title, descriptors, image_path.read_bytes() if image_path else None),
        encoding="utf-8",
    )
    artifacts.append({"kind": "preview_document", "path": html_path.name, "media_type": "text/html; charset=utf-8"})
    return {
        "renderer": "builtin.auto",
        "renderer_version": RENDERER_VERSION,
        "sample_count": len(samples),
        "rendered_image_count": len(images),
        "truncated": len(samples) > MAX_GRID_SAMPLES,
        "artifacts": artifacts,
    }


def render_summary(samples, output_directory, title="Data preview"):
    """Render bounded structure metadata; sample labels and IDs may be included."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptors = [_sample_descriptor(sample, index, include_structure=True) for index, sample in enumerate(samples)]
    html_path = output_directory / "preview.html"
    html_path.write_text(_build_html(title, descriptors, None), encoding="utf-8")
    return {
        "renderer": "builtin.summary",
        "renderer_version": RENDERER_VERSION,
        "sample_count": len(samples),
        "rendered_image_count": 0,
        "truncated": len(samples) > MAX_GRID_SAMPLES,
        "artifacts": [{"kind": "preview_document", "path": html_path.name, "media_type": "text/html; charset=utf-8"}],
    }


def invoke_custom_renderer(renderer, samples, output_directory, request):
    """Invoke an explicitly selected project renderer; validate outputs later at the boundary."""
    result = renderer(samples=samples, output_directory=Path(output_directory), request=request)
    if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
        raise UserError(
            "PREVIEW_RENDERER_INVALID",
            "项目 renderer 必须返回包含 artifacts 数组的对象。",
            "返回 artifacts: [{kind, path, media_type}]，并将文件写入请求目录。",
        )
    declared_version = str(request.get("renderer_version"))
    returned_version = str(result.get("renderer_version", declared_version))
    if returned_version != declared_version:
        raise UserError(
            "PREVIEW_RENDERER_VERSION_MISMATCH",
            "项目 renderer 返回版本与请求声明版本不一致。",
            "更新 --renderer-version 并让 renderer 返回同一版本标记。",
            {"requested": declared_version, "returned": returned_version},
        )
    return {
        "renderer": "project.custom",
        "renderer_version": declared_version,
        "sample_count": len(samples),
        "rendered_image_count": sum(item.get("media_type") == "image/png" for item in result["artifacts"] if isinstance(item, dict)),
        "truncated": False,
        "artifacts": result["artifacts"],
    }


def _save_grid(images, descriptors, path, title):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    columns = min(4, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / columns)
    label_height = 42
    cell_width, cell_height = THUMBNAIL_SIZE[0] + 16, THUMBNAIL_SIZE[1] + label_height + 16
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, descriptor) in enumerate(zip(images, descriptors)):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        thumb = ImageOps.contain(image.convert("RGB"), THUMBNAIL_SIZE)
        card = Image.new("RGB", THUMBNAIL_SIZE, color=(255, 255, 255))
        card.paste(thumb, ((THUMBNAIL_SIZE[0] - thumb.width) // 2, (THUMBNAIL_SIZE[1] - thumb.height) // 2))
        canvas.paste(card, (x + 8, y + 8))
        label = _caption(descriptor)
        draw.text((x + 8, y + THUMBNAIL_SIZE[1] + 12), label[:52], fill=(30, 41, 59), font=font)
    canvas.save(path, format="PNG", optimize=True)


def _build_html(title, descriptors, image_bytes):
    rows = "\n".join(
        "<li><strong>{}</strong><span>{}</span></li>".format(
            html.escape(_caption(item)), html.escape(_metadata_text(item))
        )
        for item in descriptors[:MAX_GRID_SAMPLES]
    )
    image = ""
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image = f'<img class="grid" alt="Rendered sample grid" src="data:image/png;base64,{encoded}">'
    truncated = "<p>Showing the first 64 sample descriptions.</p>" if len(descriptors) > MAX_GRID_SAMPLES else ""
    return "<!doctype html>\n" + """<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033;background:#f7f8fa}}
h1{{font-size:1.5rem}} .grid{{max-width:100%;height:auto;border:1px solid #d9dee8;background:white}}
ul{{padding:0;list-style:none}} li{{background:white;border:1px solid #e1e5ec;border-radius:8px;padding:10px 12px;margin:7px 0;display:flex;gap:16px;flex-wrap:wrap}}
li span{{color:#536174;font-family:ui-monospace,monospace;font-size:.9rem;white-space:pre-wrap}}
</style></head><body><h1>{title}</h1>{image}<h2>Samples</h2>{truncated}<ul>{rows}</ul></body></html>""".format(
        title=html.escape(title), image=image, truncated=truncated, rows=rows
    )


def _sample_descriptor(sample, index, include_structure=False):
    # Summary mode describes the whole user-provided structure and should not
    # recursively guess which nested tensor might represent an image.
    value = sample if include_structure else _sample_value(sample)
    descriptor = {"index": index + 1}
    if isinstance(sample, dict):
        if sample.get("label") is not None:
            descriptor["label"] = str(sample["label"])
        if sample.get("sample_id") is not None:
            descriptor["sample_id"] = str(sample["sample_id"])
    elif isinstance(sample, (tuple, list)) and len(sample) > 1:
        label = sample[1]
        if isinstance(label, (str, int, float, bool)):
            descriptor["label"] = str(label)
    kind, shape, dtype = _describe(value)
    descriptor.update({"kind": kind, "shape": shape, "dtype": dtype})
    if include_structure:
        descriptor["structure"] = _structure_summary(sample)
    return descriptor


def _structure_summary(value):
    """Describe nested inputs without retaining values; bound work and output."""
    lines = []
    remaining = [40]
    active = set()

    def visit(item, path, depth):
        if remaining[0] <= 0:
            return
        remaining[0] -= 1
        kind, shape, dtype = _describe(item)
        details = [kind]
        if shape is not None:
            details.append("shape=" + "×".join(map(str, shape[:16])) + ("…" if len(shape) > 16 else ""))
        if dtype:
            details.append("dtype=" + dtype[:128])
        lines.append(path + ": " + " · ".join(details))
        if not isinstance(item, (dict, list, tuple)):
            return
        if id(item) in active:
            lines.append(path + ": <cycle>")
            return
        if depth >= 4:
            lines.append(path + ": <further fields omitted>")
            return
        active.add(id(item))
        entries = item.items() if isinstance(item, dict) else enumerate(item)
        for key, child in entries:
            if remaining[0] <= 0:
                lines.append(path + ": <further fields omitted>")
                break
            name = key[:96] if isinstance(key, str) else str(key) if isinstance(key, int) else "<key>"
            visit(child, path + "[" + name.replace("\n", " ").replace("\r", " ") + "]", depth + 1)
        active.remove(id(item))

    visit(value, "$", 0)
    return "\n".join(lines)


def _sample_value(sample):
    if isinstance(sample, dict) and "value" in sample:
        return sample["value"]
    if isinstance(sample, dict) and "image" in sample:
        return sample["image"]
    if isinstance(sample, dict):
        for key in ("args", "kwargs"):
            candidate = _first_image_candidate(sample.get(key))
            if candidate is not None:
                return candidate
    if isinstance(sample, (tuple, list)) and sample:
        candidate = _first_image_candidate(sample[0])
        if candidate is not None:
            return candidate
    return sample


def _first_image_candidate(value):
    if _is_image_candidate(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            candidate = _first_image_candidate(item)
            if candidate is not None:
                return candidate
    if isinstance(value, (tuple, list)):
        for item in value:
            candidate = _first_image_candidate(item)
            if candidate is not None:
                return candidate
    return None


def _describe(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            shape = [int(item) for item in shape]
        except (TypeError, ValueError):
            shape = None
    if hasattr(value, "getbands") and hasattr(value, "size"):
        width, height = value.size
        shape = [height, width, len(value.getbands())]
        return "image", shape, "uint8"
    if hasattr(value, "device") and hasattr(value, "dtype"):
        return "tensor", shape, str(value.dtype).replace("torch.", "")
    if hasattr(value, "dtype"):
        return "array", shape, str(value.dtype).replace("numpy.", "")
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return "scalar", None, type(value).__name__
    if isinstance(value, dict):
        return "mapping", None, None
    if isinstance(value, (tuple, list)):
        return type(value).__name__, [len(value)], None
    return type(value).__name__, shape, None


def _to_pil(value):
    from PIL import Image

    if hasattr(value, "getbands") and hasattr(value, "size"):
        return value.copy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise ValueError("batched tensors must be split into samples before rendering")
            value = value[0]
        if value.ndim == 3:
            value = value.permute(1, 2, 0)
        value = value.numpy()
        return _array_to_pil(value)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy as np

        return _array_to_pil(np.asarray(value))
    raise TypeError(f"not an image value: {type(value).__name__}")


def _array_to_pil(array):
    import numpy as np
    from PIL import Image

    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in {2, 3}:
        raise ValueError(f"images must have rank 2 or 3, got {array.ndim}")
    if array.dtype.kind == "f":
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        low = float(array.min(initial=0.0))
        high = float(array.max(initial=0.0))
        if low < 0 or high > 1:
            array = np.clip(array, 0, 1)
        array = (array * 255).round().astype("uint8")
    elif array.dtype != "uint8":
        array = np.clip(array, 0, 255).astype("uint8")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array.copy())


def _is_image_candidate(value):
    if hasattr(value, "getbands") and hasattr(value, "size"):
        return True
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return len(value.shape) in {2, 3, 4}
    return hasattr(value, "dtype") and hasattr(value, "shape") and len(value.shape) in {2, 3}


def _caption(descriptor):
    label = descriptor.get("label")
    return f"{descriptor['index']}. {label}" if label is not None else f"Sample {descriptor['index']}"


def _metadata_text(descriptor):
    details = [descriptor["kind"]]
    if descriptor.get("shape") is not None:
        details.append("shape=" + "×".join(map(str, descriptor["shape"])))
    if descriptor.get("dtype"):
        details.append("dtype=" + descriptor["dtype"])
    if descriptor.get("sample_id"):
        details.append("id=" + descriptor["sample_id"][:48])
    if descriptor.get("structure"):
        details.append("structure (values omitted):\n" + descriptor["structure"])
    return " · ".join(details)

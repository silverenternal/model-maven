"""Read the official Drifting generator and emit a scoped, static graph plan."""

import ast
import json
import re
import sys

if sys.version_info < (3, 9):
    sys.exit("Drifting source import requires Python 3.9 or newer")


def model_config(text):
    # The official presets have a flat model section. Read scalar values only.
    values = {}
    active = False
    for candidate in text.splitlines():
        line = candidate.partition("#")[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if active:
                break
            active = line == "model:"
            continue
        if active:
            match = re.fullmatch(r"  ([a-z_][a-z0-9_]*):\s*(\S+)", line)
            if not match:
                raise ValueError("expected scalar fields in the preset's model section")
            key, value = match.groups()
            try:
                values[key] = json.loads(value)
            except json.JSONDecodeError:
                continue
    return values


def plan(source, config):
    tree = ast.parse(source, filename="models/generator.py")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def method(owner, name="__call__"):
        return next(node for node in classes[owner].body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def find(scope, form, predicate, description, ordinal=0):
        matches = sorted((node for node in ast.walk(scope)
                          if isinstance(node, form) and predicate(node)),
                         key=lambda node: (node.lineno, node.col_offset))
        if len(matches) <= ordinal:
            raise ValueError(f"{scope.name}: source anchor missing for {description}")
        return matches[ordinal]

    def call(scope, name, ordinal=0):
        return find(scope, ast.Call, lambda node: ast.unparse(node.func) == name, name, ordinal)

    def assign(scope, name, expression):
        return find(scope, ast.Assign,
                    lambda node: any(ast.unparse(target) == name for target in node.targets)
                    and expression in ast.unparse(node.value), name)

    def node(name, kind, anchor, scope=None, **attributes):
        return dict(name=name, kind=kind, line=anchor.lineno, end=anchor.end_lineno,
                    scope=scope, attributes=attributes)

    depth = config.get("depth")
    width = config.get("hidden_size")
    heads = config.get("num_heads")
    if type(depth) is not int or not 1 <= depth <= 256:
        raise ValueError("model.depth must be an integer from 1 to 256")
    if any(type(value) is not int or value <= 0 for value in (width, heads)):
        raise ValueError("model.hidden_size and model.num_heads must be positive integers")
    if width % heads:
        raise ValueError("model.hidden_size must be divisible by model.num_heads")

    head_dim = width // heads
    model_axes = ["B", "N", width]
    combined_axes = ["B", "N", 3 * width]
    packed_axes = ["B", "N", 3, heads, head_dim]
    split_axes = ["B", "N", heads, head_dim]
    head_axes = ["B", heads, "N", head_dim]
    key_axes = ["B", heads, head_dim, "N"]
    score_axes = ["B", heads, "N", "N"]

    def scope(nodes, edges, inputs=(), outputs=()):
        return dict(nodes=nodes, edges=edges, inputs=list(inputs), outputs=list(outputs))

    generator = method("DitGen")
    setup = method("DitGen", "setup")
    condition = method("DitGen", "c_cfg_noise_to_cond")
    backbone = method("LightningDiT")
    block = method("LightningDiTBlock")
    attention = method("Attention")
    output = method("FinalLayer")
    block_loop = find(backbone, ast.For,
                      lambda node: ast.unparse(node.iter) == "range(self.depth)", "block loop")
    block_call = call(block_loop, "BlockCls")
    residual_attention = assign(block, "x", "* attn(")
    residual_ffn = assign(block, "x", "* mlp(")
    first_block = "block_00"
    last_block = f"block_{depth - 1:02}"
    normal = "RMSNorm" if config.get("use_rmsnorm") is True else "LayerNorm"
    use_swiglu = config.get("use_swiglu") is True

    root = node("DitGen", "Flax generator", generator, "generator")
    scopes = {
        "generator": scope([
            node("latent_noise", "Noise", call(generator, "random.normal")),
            node("conditioning", "Conditioning", call(generator, "self.c_cfg_noise_to_cond"), "conditioning"),
            node("backbone", "LightningDiT", call(setup, "LightningDiT"), "backbone", width=width, heads=heads),
            node("samples", "Output", find(generator, ast.Return, lambda _: True, "generator return")),
        ], [["latent_noise", "backbone"], ["conditioning", "backbone"], ["backbone", "samples"]]),
        "conditioning": scope([
            node("class_embed", "Embedding", call(condition, "self.class_embed")),
            node("cfg_embedder", "TimestepEmbedder", call(condition, "self.cfg_embedder")),
            node("cfg_norm", "RMSNorm", call(condition, "self.cfg_norm")),
            node("merge", "Add", assign(condition, "cond", "cfg_scale_t *")),
        ], [["class_embed", "merge"], ["cfg_embedder", "cfg_norm"], ["cfg_norm", "merge"]],
           outputs=[(0, "merge", 0)]),
        "backbone": scope([
            node("patchify", "Patch reshape", assign(backbone, "x", "effective_p * effective_p * C")),
            node("patch_projection", "TorchLinear", call(backbone, "TorchLinear"), width=width, output_features=width),
            node("position", "Position embedding", assign(backbone, "x", "x + pos_embed"), input_features=width, output_features=width),
            node("blocks", "DiT block stack", block_loop, "blocks", stack=depth, width=width,
                 input_features=width, output_features=width),
            node("final", "FinalLayer", call(backbone, "FinalLayer"), "final"),
            node("unpatchify", "Reshape", assign(backbone, "x", "out_size, out_size")),
        ], [["patchify", "patch_projection"], ["patch_projection", "position"], ["position", "blocks"],
            ["blocks", "final"], ["final", "unpatchify"]],
           inputs=[(0, "patchify", 0), (1, "blocks", 1), (1, "final", 1)],
           outputs=[(0, "unpatchify", 0)]),
        "blocks": scope([
            node(f"block_{index:02}", "LightningDiTBlock", block_call, "block", width=width, heads=heads,
                 input_features=width, output_features=width)
            for index in range(depth)
        ], [[f"block_{index - 1:02}", f"block_{index:02}"] for index in range(1, depth)],
           inputs=[(0, first_block, 0)] + [(1, f"block_{index:02}", 1) for index in range(depth)],
           outputs=[(0, last_block, 0)]),
        "block": scope([
            node("x", "Tensor input", block),
            node("adaLN", "Condition projection", call(block, "jnp.split")),
            node("norm1", normal, call(block, "norm1")),
            node("modulate_attn", "Modulation", call(block, "modulate"), output_axes=[model_axes]),
            node("attention", "Attention", call(block, "Attention"), "attention", width=width, heads=heads,
                 input_axes=[model_axes], output_axes=[model_axes]),
            node("attention_residual", "Residual add", residual_attention),
            node("norm2", normal, call(block, "norm2")),
            node("modulate_ffn", "Modulation", call(block, "modulate", 1)),
            node("ffn", "SwiGLUFFN" if use_swiglu else "StandardMLP",
                 call(block, "SwiGLUFFN" if use_swiglu else "StandardMLP"), "swiglu" if use_swiglu else None),
            node("ffn_residual", "Residual add", residual_ffn),
        ], [["x", "norm1"], ["norm1", "modulate_attn"], ["adaLN", "modulate_attn", 0],
            ["adaLN", "modulate_attn", 1], ["modulate_attn", "attention"],
            ["x", "attention_residual"], ["attention", "attention_residual"], ["adaLN", "attention_residual", 2],
            ["attention_residual", "norm2"], ["norm2", "modulate_ffn"], ["adaLN", "modulate_ffn", 3],
            ["adaLN", "modulate_ffn", 4], ["modulate_ffn", "ffn"],
            ["attention_residual", "ffn_residual"], ["ffn", "ffn_residual"], ["adaLN", "ffn_residual", 5]],
           inputs=[(0, "x", 0), (1, "adaLN", 0)], outputs=[(0, "ffn_residual", 0)]),
        "final": scope([
            node("norm", normal, call(output, "norm_final")),
            node("adaLN", "Condition projection", call(output, "jnp.split")),
            node("modulate", "Modulation", call(output, "modulate")),
            node("pixels", "TorchLinear", call(output, "TorchLinear", 1)),
        ], [["norm", "modulate"], ["adaLN", "modulate", 0], ["adaLN", "modulate", 1], ["modulate", "pixels"]],
           inputs=[(0, "norm", 0), (1, "adaLN", 0)], outputs=[(0, "pixels", 0)]),
    }

    if config.get("n_cls_tokens", 0) > 0:
        stages = scopes["backbone"]
        stages["nodes"].extend([
            node("class_tokens", "Condition tokens", assign(backbone, "c_tokens", "jnp.tile"),
                 stack=config["n_cls_tokens"], output_features=width),
            node("prepend", "Concatenate", call(backbone, "jnp.concatenate"), input_features=width, output_features=width),
        ])
        stages["edges"].remove(["position", "blocks"])
        stages["edges"].extend([["position", "prepend"], ["class_tokens", "prepend"], ["prepend", "blocks"]])
        stages["inputs"].append((1, "class_tokens", 0))

    if config.get("noise_classes", 0) > 0:
        coordinates = config.get("noise_coords")
        if type(coordinates) is not int or not 1 <= coordinates <= 256:
            raise ValueError("model.noise_coords must be an integer from 1 to 256")
        lookup = call(condition, "embed")
        label_slice = find(condition, ast.Subscript,
                           lambda item: ast.unparse(item) == "noise_labels[:, i]",
                           "noise coordinate slice")
        accumulate = assign(condition, "cond", "cond + embed(")
        root_scope = scopes["generator"]
        root_scope["nodes"].append(
            node("noise_labels", "Noise label sampling", call(generator, "random.randint"))
        )
        root_scope["edges"].append(["noise_labels", "conditioning"])
        conditioning = scopes["conditioning"]
        conditioning["nodes"].append(
            node("noise_embeds", "Noise conditioning", accumulate, "noise_embeds", stack=coordinates)
        )
        class_path = conditioning["edges"].index(["class_embed", "merge"])
        conditioning["edges"][class_path:class_path + 1] = [
            ["class_embed", "noise_embeds"], ["noise_embeds", "merge"]
        ]
        conditioning["inputs"].append((0, "noise_embeds", 1))
        coordinate_names = [f"coord_{index:02}" for index in range(coordinates)]
        scopes["noise_embeds"] = scope([
            node(name, "Noise coordinate", accumulate, "noise_coordinate") for name in coordinate_names
        ], [list(pair) for pair in zip(coordinate_names, coordinate_names[1:])],
           inputs=[(0, coordinate_names[0], 0)] + [(1, name, 1) for name in coordinate_names],
           outputs=[(0, coordinate_names[-1], 0)])
        scopes["noise_coordinate"] = scope([
            node("label", "Tensor slice", label_slice),
            node("embedding", "Embedding", lookup),
            node("add", "Residual add", accumulate),
        ], [["label", "embedding"], ["embedding", "add"]],
           inputs=[(0, "add", 0), (1, "label", 0)], outputs=[(0, "add", 0)])

    qkv_split = assign(attention, "(q, k, v)", "qkv[")
    attention_nodes = [
        node("qkv", "TorchLinear", call(attention, "TorchLinear"), width=3 * width,
             input_axes=[model_axes], output_axes=[combined_axes]),
        node("reshape_qkv", "Reshape QKV", assign(attention, "qkv", "qkv.reshape(B, N, 3, self.num_heads, head_dim)"),
             input_axes=[combined_axes], output_axes=[packed_axes]),
    ]
    attention_edges = [["qkv", "reshape_qkv"]]
    for arm in ("q", "k", "v"):
        attention_nodes.append(node(arm, "Tensor slice", qkv_split, input_axes=[packed_axes], output_axes=[split_axes]))
        attention_edges.append(["reshape_qkv", arm])
    q, k = "q", "k"
    if config.get("use_qknorm") is True:
        norm_call = "RMSNorm" if config.get("use_rmsnorm") else "nn.LayerNorm"
        attention_nodes.extend([
            node(f"{arm}_norm", normal, call(attention, norm_call, slot),
                 input_axes=[split_axes], output_axes=[split_axes])
            for slot, arm in enumerate(("q", "k"))
        ])
        attention_edges.extend([[q, "q_norm"], [k, "k_norm"]])
        q, k = "q_norm", "k_norm"
    if config.get("use_rope") is True:
        apply_rope = next(item for item in tree.body
                          if isinstance(item, ast.FunctionDef) and item.name == "apply_rope")
        attention_nodes.append(node("rope", "Rotary embedding", call(attention, "apply_rope"), "rope",
                                    input_axes=[split_axes, split_axes], output_axes=[split_axes, split_axes]))
        attention_edges.extend([[q, "rope"], [k, "rope"]])
        q, k = "rope", "rope"
        scopes["rope"] = scope([
            node("phase", "Rotary phase", assign(apply_rope, "emb", "jnp.concatenate")),
            node("cos", "Cosine", call(apply_rope, "jnp.cos")),
            node("sin", "Sine", call(apply_rope, "jnp.sin")),
            node("q_rotated", "Rotate Q", assign(apply_rope, "q_embed", "rotate_half(q)"),
                 input_axes=[split_axes], output_axes=[split_axes]),
            node("k_rotated", "Rotate K", assign(apply_rope, "k_embed", "rotate_half(k)"),
                 input_axes=[split_axes], output_axes=[split_axes]),
        ], [["phase", "cos"], ["phase", "sin"], ["cos", "q_rotated"], ["sin", "q_rotated"],
            ["cos", "k_rotated"], ["sin", "k_rotated"]],
           inputs=[(0, "q_rotated", 0), (1, "k_rotated", 0)],
           outputs=[(0, "q_rotated", 0), (1, "k_rotated", 0)])
    transposes = [assign(attention, arm, f"jnp.transpose({arm}, (0, 2, 1, 3))") for arm in ("q", "k", "v")]
    swap_key = find(attention, ast.Call, lambda item: ast.unparse(item) == "jnp.swapaxes(k, -1, -2)", "key swapaxes")
    scopes["head_axes"] = scope([
        node(f"{arm}_heads", f"Transpose {arm.upper()}", anchor, input_axes=[split_axes], output_axes=[head_axes])
        for arm, anchor in zip(("q", "k", "v"), transposes)
    ] + [node("k_transposed", "Swap K axes", swap_key, input_axes=[head_axes], output_axes=[key_axes])],
       [["k_heads", "k_transposed"]],
       inputs=[(0, "q_heads", 0), (1, "k_heads", 0), (2, "v_heads", 0)],
       outputs=[(0, "q_heads", 0), (1, "k_transposed", 0), (2, "v_heads", 0)])
    attention_nodes.extend([
        node("head_axes", "Attention axes", transposes[0], "head_axes", input_axes=[split_axes] * 3,
             output_axes=[head_axes, key_axes, head_axes],
             input_lines=[anchor.lineno for anchor in transposes],
             output_lines=[transposes[0].lineno, swap_key.lineno, transposes[2].lineno]),
        node("logits", "QKᵀ matmul", call(attention, "jnp.matmul"),
             input_axes=[head_axes, key_axes], output_axes=[score_axes]),
        node("softmax", "Softmax", call(attention, "jax.nn.softmax"), input_axes=[score_axes], output_axes=[score_axes]),
        node("weighted_values", "Attention × V", call(attention, "jnp.matmul", 1),
             input_axes=[score_axes, head_axes], output_axes=[head_axes]),
        node("merge_heads", "Merge heads", assign(attention, "x", "jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)"),
             input_axes=[head_axes], output_axes=[model_axes]),
        node("projection", "TorchLinear", call(attention, "TorchLinear", 1), width=width,
             input_axes=[model_axes], output_axes=[model_axes]),
    ])
    attention_edges.extend([
        [q, "head_axes"], [k, "head_axes", int(q == k)], ["v", "head_axes"],
        ["head_axes", "logits"], ["head_axes", "logits", 1], ["logits", "softmax"],
        ["softmax", "weighted_values"], ["head_axes", "weighted_values", 2],
        ["weighted_values", "merge_heads"], ["merge_heads", "projection"],
    ])
    scopes["attention"] = scope(attention_nodes, attention_edges,
                                inputs=[(0, "qkv", 0)], outputs=[(0, "projection", 0)])

    if use_swiglu:
        ffn = method("SwiGLUFFN")
        scopes["swiglu"] = scope([
            node("gate", "TorchLinear", call(ffn, "TorchLinear")),
            node("up", "TorchLinear", call(ffn, "TorchLinear", 1)),
            node("silu", "SiLU", call(ffn, "nn.silu")),
            node("multiply", "Multiply", assign(ffn, "out", "* w3")),
            node("down", "TorchLinear", call(ffn, "TorchLinear", 2), width=width),
        ], [["gate", "silu"], ["silu", "multiply"], ["up", "multiply"], ["multiply", "down"]],
           inputs=[(0, "gate", 0), (0, "up", 0)], outputs=[(0, "down", 0)])
    for projection in scopes.values():
        projection["edges"] = [dict(source=edge[0], target=edge[1],
                                    outlet=edge[2] if len(edge) > 2 else 0)
                               for edge in projection["edges"]]
        for side in ("inputs", "outputs"):
            projection[side] = [dict(slot=slot, node=name, port=port)
                                for slot, name, port in projection[side]]
    return dict(root=root, scopes=scopes)


try:
    source, config = json.load(sys.stdin)
    print(json.dumps(plan(source, model_config(config))))
except (OSError, ValueError, SyntaxError, KeyError, StopIteration) as error:
    sys.exit(f"Drifting source import: {error or 'required model method is missing'}")

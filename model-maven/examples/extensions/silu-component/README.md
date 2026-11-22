# Packaged Layer component

This example is an independently buildable Python distribution. Its entry point follows standard Python packaging syntax:

```toml
[project.entry-points."model_maven.components"]
silu = "model_maven_silu.layer:make_silu"
```

The static manifest sits beside the entry module under `__model_maven__/silu.json`. It is included in the wheel as package data, so Model-Maven can inspect it without importing `model_maven_silu.layer`.

Build and install it into the same Python environment that has PyTorch:

```sh
python -m pip install .
m-model --python "$(python -c 'import sys; print(sys.executable)')" component discover --distribution model-maven-silu-example
m-model component show example/silu
m-model component test example/silu --param inplace=false
```

Discovery records the distribution version, static manifest hash, entry-point module hash, and fixture hash when present. It does not install the package or its dependencies. The example declares `torch>=2.1`; choose an environment compatible with the project's supported PyTorch version before installation.

After discovery, a registered Layer factory can be used by the existing Layer add/replace commands:

```sh
m-model layer add \
  --model my_model:build_model \
  --batch my_data:example_batch \
  --parent model.encoder \
  --after relu \
  --name silu \
  --component example/silu@0.1.0 \
  --param inplace=false
```

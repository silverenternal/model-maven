# Explicit root-model assembly adapter example

This example keeps the model implementation in `root_model.py` and the encoder factory in `components.py`. `recipe.json` records the root factory, declared injection seam, parameters, and an optional initialization seed. The generator writes `generated/model_assembly.py` as a separate file and refuses to replace an existing target.

From the repository root, with the worker package available on `PYTHONPATH`, run:

```sh
PYTHONPATH=python python examples/assembly/generate.py
PYTHONPATH=examples/assembly:examples/assembly/generated python -c 'from model_assembly import build_model; print(build_model())'
```

The output imports the user-owned factories at runtime. It does not copy their implementation or install PyTorch, component wheels, private resources, or other dependencies. A component lock is preserved as provenance text; this example generator does not independently verify or install that lock. It supports an explicit root-factory construction seam, not arbitrary Python graph extraction or internal topology rewriting. Generated source is a readable adapter and should be reviewed and kept under version control with its source factories and dependency setup.

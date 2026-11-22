"""Generate the example adapter into a separate, non-overwriting directory."""

import json
from pathlib import Path

from model_maven_worker.edit.codegen import write_code


HERE = Path(__file__).resolve().parent
RECIPE = HERE / "recipe.json"
OUTPUT = HERE / "generated" / "model_assembly.py"


if __name__ == "__main__":
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    print(write_code(recipe, OUTPUT))

#!/usr/bin/env python3
"""Run the shared full-screen form fixture in built-in Layer mode."""
import os
from pathlib import Path
import runpy

os.environ["MODEL_MAVEN_FORM_BUILTIN"] = "1"
runpy.run_path(str(Path(__file__).with_name("verify_component_form_terminal.py")), run_name="__main__")

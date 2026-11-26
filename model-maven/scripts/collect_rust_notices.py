#!/usr/bin/env python3
"""Collect license files for the Linux dependency trees of CLI and TUI."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-source", required=True, type=Path)
    parser.add_argument("--tui-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", required=True, help="native Rust target triple")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for source in (args.cli_source, args.tui_source):
        raw = subprocess.check_output([
            "cargo", "metadata", "--locked", "--offline", "--format-version", "1",
            "--filter-platform", args.target,
        ], cwd=source)
        for package in json.loads(raw)["packages"]:
            if package.get("source") is not None:
                packages[(package["name"], package["version"])] = package
    inventory = []
    for (name, version), package in sorted(packages.items()):
        root = Path(package["manifest_path"]).parent.resolve()
        candidates = [path for path in root.iterdir() if path.is_file() and
                      path.name.lower().startswith(("license", "licence", "copying", "notice"))]
        if package.get("license_file"):
            candidates.append(root / package["license_file"])
        unique = sorted({path.resolve(strict=True) for path in candidates})
        if not unique or not package.get("license"):
            raise ValueError(f"license metadata or text missing for {name} {version}")
        target = output / "crates" / f"{name}-{version}"
        target.mkdir(parents=True)
        for path in unique:
            path.relative_to(root)
            shutil.copy2(path, target / path.name)
        inventory.append({"name": name, "version": version, "license": package["license"],
                          "files": [str((target / path.name).relative_to(output)) for path in unique]})
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip())
    rust_docs = sysroot / "share/doc/rust"
    copyright_file = rust_docs / "COPYRIGHT-library.html"
    licenses_dir = rust_docs / "licenses"
    if not copyright_file.is_file() or not licenses_dir.is_dir():
        # Distribution-packaged Rust can split compiler documentation and
        # copyright notices into separate system directories.
        copyright_file = sysroot / "share/licenses/rust/COPYRIGHT-library.html.rustc"
        licenses_dir = sysroot / "share/doc/rustc/licenses"
    if not copyright_file.is_file() or not licenses_dir.is_dir():
        raise FileNotFoundError("Rust standard-library copyright or license texts are unavailable in the active toolchain")
    rust_target = output / "rust-standard-library"
    rust_target.mkdir()
    shutil.copy2(copyright_file, rust_target / "COPYRIGHT-library.html")
    shutil.copytree(licenses_dir, rust_target / "licenses")
    (output / "inventory.json").write_text(json.dumps({
        "scope": f"{args.target} cargo metadata; includes dependency trees of CLI and TUI",
        "packages": inventory,
        "rust_standard_library": "rust-standard-library/COPYRIGHT-library.html",
    }, indent=2) + "\n")
    print(f"Collected notices for {len(inventory)} crates and the Rust standard library in {output}")


if __name__ == "__main__":
    main()

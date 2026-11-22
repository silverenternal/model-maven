#!/usr/bin/env python3
"""Stage the npm launcher or one native platform package; never publish it."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "npm" / "model-maven"
PLATFORMS = ("linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "win32-x64", "win32-arm64")
BASELINE = (2, 39)


def host_platform():
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
    return f"{sys.platform}-{architecture.get(platform.machine().lower(), platform.machine().lower())}"


def inspect_binary(path, command, target, local_test):
    glibc_max = None
    if target.startswith("linux-"):
        header = subprocess.check_output(["readelf", "-h", str(path)], text=True)
        expected = "Advanced Micro Devices X86-64" if target.endswith("-x64") else "AArch64"
        if expected not in header:
            raise ValueError(f"{path} is not a {target} ELF binary")
        versions = subprocess.check_output(["readelf", "--version-info", str(path)], text=True)
        glibc = [tuple(map(int, match)) for match in re.findall(r"GLIBC_(\d+)\.(\d+)", versions)]
        if not glibc:
            raise ValueError(f"{path} has no identifiable glibc requirements")
        maximum = max(glibc)
        if maximum > BASELINE and not local_test:
            raise ValueError(f"{path} requires glibc {maximum[0]}.{maximum[1]}, above baseline 2.39")
        glibc_max = ".".join(map(str, maximum))
    output = subprocess.check_output([str(path), "--version"], text=True).strip()
    prefix = f"{command} "
    if not output.startswith(prefix):
        raise ValueError(f"unexpected version response from {path}: {output}")
    return {"version": output[len(prefix):], "glibc_max": glibc_max,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("launcher", "platform"), required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--tui", type=Path)
    parser.add_argument("--platform", choices=PLATFORMS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--third-party-notices", type=Path)
    parser.add_argument("--name", default="model-maven-local-test")
    parser.add_argument("--license", default="UNLICENSED")
    parser.add_argument("--local-test", action="store_true", help="mark package private and permit a newer Linux glibc")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*", args.name):
        parser.error("name must be a lowercase npm package name, optionally scoped")
    if not args.local_test and (args.license == "UNLICENSED" or args.name == "model-maven-local-test"):
        parser.error("a publishable bundle needs the chosen package name and license")
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    evidence = {}
    binaries = {}
    if args.kind == "platform":
        if not args.cli or not args.tui or not args.platform or not args.third_party_notices:
            parser.error("platform package requires --cli, --tui, --platform and --third-party-notices")
        if args.platform != host_platform():
            parser.error(f"platform package must be built and run natively; host is {host_platform()}")
        binaries = {"m-model": args.cli.resolve(strict=True), "m-model-tui": args.tui.resolve(strict=True)}
        evidence = {name: inspect_binary(path, name, args.platform, args.local_test) for name, path in binaries.items()}
        if any(item["version"] != version for item in evidence.values()):
            parser.error("binary versions do not match the project manifest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-maven-npm-stage-") as directory:
        stage = Path(directory) / "package"
        if args.kind == "launcher":
            shutil.copytree(TEMPLATE, stage)
            manifest = json.loads((stage / "package.json").read_text())
            manifest.update(name=args.name, version=version, license=args.license)
            manifest["optionalDependencies"] = {f"{args.name}-{item}": version for item in PLATFORMS}
            for name in ("m-model", "m-model-tui"):
                (stage / "bin" / f"{name}.cjs").chmod(0o755)
        else:
            stage.mkdir()
            os_name, cpu = args.platform.split("-", 1)
            manifest = {"name": f"{args.name}-{args.platform}", "version": version,
                        "description": f"Model-Maven native binaries for {args.platform}",
                        "license": args.license, "os": [os_name], "cpu": [cpu],
                        "files": ["bin", "README.md", "THIRD_PARTY_NOTICES"]}
            if os_name == "linux":
                manifest["libc"] = ["glibc"]
            bin_dir = stage / "bin"
            bin_dir.mkdir()
            for name, binary in binaries.items():
                target = bin_dir / (name + (".exe" if os_name == "win32" else ""))
                shutil.copy2(binary, target)
                target.chmod(0o755)
            notices = args.third_party_notices.resolve(strict=True)
            if not (notices / "inventory.json").is_file():
                parser.error("third-party notices must contain inventory.json")
            shutil.copytree(notices, stage / "THIRD_PARTY_NOTICES")
            (stage / "README.md").write_text(f"# Model-Maven {args.platform}\n\nNative binaries used by the Model-Maven npm launcher.\n")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        if args.local_test:
            manifest["private"] = True
            with (stage / "README.md").open("a") as readme:
                readme.write("\nLocal test package; do not publish.\n")
        else:
            manifest.pop("private", None)
        (stage / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
        npm = shutil.which("npm")
        if not npm:
            parser.error("npm not found")
        packed = subprocess.run([npm, "pack", "--ignore-scripts", "--json", "--pack-destination", str(output)],
                                cwd=stage, check=True, capture_output=True, text=True, shell=sys.platform == "win32")
        tarball = output / json.loads(packed.stdout)[0]["filename"]
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        (output / (tarball.name + ".sha256")).write_text(f"{digest}  {tarball.name}\n")
        (output / (tarball.name + ".manifest.json")).write_text(json.dumps({
            "package": manifest["name"], "version": version, "license": args.license,
            "platform": args.platform, "local_test_only": args.local_test, "sha256": digest,
            "binaries": evidence, "scope": "native startup checked; user manages Python/PyTorch; not published",
        }, indent=2) + "\n")
        print(tarball)


if __name__ == "__main__":
    main()

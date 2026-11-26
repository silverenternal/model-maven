#!/usr/bin/env python3
"""Stage the npm launcher or one native platform package; never publish it."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "npm" / "model-maven"
PLATFORMS = ("linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "win32-x64", "win32-arm64")
BASELINE = (2, 39)


def checkout_identity(path):
    """Identify the checkout; this does not prove which source built a binary."""
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        revision = git("rev-parse", "--verify", "HEAD^{commit}")
        dirty = bool(git("status", "--porcelain=v1", "--untracked-files=normal"))
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"unsupported Git revision at {path}")
    return {"revision": revision, "dirty": dirty}


def workspace_sources(tui_source):
    root = tui_source.resolve(strict=True)
    # Discover modules rather than maintaining a second file list: a newly
    # added workspace module must be packaged and compared against the binary.
    directories = (root / "lua" / "model_maven", root / "plugin")
    if not all(path.is_dir() for path in directories):
        raise ValueError("unified workspace module or plugin directory is missing")
    entries = [path for directory in directories for path in directory.rglob("*")]
    if any(path.is_symlink() for path in entries):
        raise ValueError("unified workspace must not contain symbolic links")
    paths = sorted(path for path in entries if path.is_file() and path.suffix == ".lua")
    sources = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    if not {"lua/model_maven/init.lua", "lua/model_maven/workspace.lua", "plugin/model_maven.lua"}.issubset(sources):
        raise ValueError("unified workspace entrypoints are missing")
    return sources


def workspace_version(tui_source):
    manifest = tui_source / "Cargo.toml"
    try:
        value = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unified workspace Cargo.toml has no readable package version: {error}") from error
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", value):
        raise ValueError("unified workspace Cargo.toml package version is not a valid release version")
    return value


def workspace_identity(sources):
    files = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256-file-map-v1", "digest": hashlib.sha256(canonical).hexdigest(),
            "files": files}


def verify_embedded_workspace(tui, sources):
    raw = subprocess.check_output([str(tui), "--workspace-sources"], timeout=30)
    embedded = json.loads(raw)
    if not isinstance(embedded, dict) or set(embedded) != set(sources):
        raise ValueError("TUI binary does not expose the expected unified workspace files")
    if any(not isinstance(embedded[name], str) or embedded[name].encode("utf-8") != content
           for name, content in sources.items()):
        raise ValueError("TUI binary embeds a different workspace from the plugin being packaged")


def host_platform():
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
    return f"{sys.platform}-{architecture.get(platform.machine().lower(), platform.machine().lower())}"


def inspect_binary(path, command, target, local_test):
    glibc_max = None
    if target.startswith(("darwin-", "win32-")):
        with path.open("rb") as file:
            header = file.read(4096)
            if target.startswith("win32-"):
                if len(header) < 64 or header[:2] != b"MZ":
                    raise ValueError(f"{path} is not a PE executable")
                pe_offset = struct.unpack_from("<I", header, 60)[0]
                file.seek(pe_offset)
                signature = file.read(6)
                expected = 0x8664 if target.endswith("-x64") else 0xAA64
                if len(signature) != 6 or signature[:4] != b"PE\0\0" or struct.unpack_from("<H", signature, 4)[0] != expected:
                    raise ValueError(f"{path} is not a {target} PE executable")
            else:
                if len(header) < 8:
                    raise ValueError(f"{path} has a truncated Mach-O header")
                expected = 0x01000007 if target.endswith("-x64") else 0x0100000C
                magic = header[:4]
                if magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"):
                    endian = "<" if magic == b"\xcf\xfa\xed\xfe" else ">"
                    architectures = [struct.unpack_from(endian + "I", header, 4)[0]]
                elif magic in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
                    endian = ">" if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf") else "<"
                    entry_size = 32 if magic in (b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca") else 20
                    count = struct.unpack_from(endian + "I", header, 4)[0]
                    if count == 0 or count > 64 or len(header) < 8 + count * entry_size:
                        raise ValueError(f"{path} has an invalid universal Mach-O header")
                    architectures = [struct.unpack_from(endian + "I", header, 8 + index * entry_size)[0] for index in range(count)]
                else:
                    raise ValueError(f"{path} is not a 64-bit Mach-O executable")
                if expected not in architectures:
                    raise ValueError(f"{path} does not contain a {target} Mach-O architecture")
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
    parser.add_argument("--tui-source", type=Path, default=ROOT.parent / "model-maven-tui",
                        help="workspace source supplying the Neovim plugin for the launcher package")
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
    try:
        sources = workspace_sources(args.tui_source)
        tui_version = workspace_version(args.tui_source)
    except (OSError, ValueError) as error:
        parser.error(f"TUI workspace source is unavailable: {error}")
    if tui_version != version:
        parser.error(f"TUI workspace version {tui_version} does not match backend version {version}")
    identities = {"backend": checkout_identity(ROOT), "workspace": checkout_identity(args.tui_source)}
    if not args.local_test:
        for name, identity in identities.items():
            if identity["revision"] is None or identity["dirty"] is not False:
                parser.error(f"publishable packages require a clean, committed {name} checkout")
    workspace = workspace_identity(sources)
    evidence = {}
    binaries = {}
    if args.kind == "platform":
        if not args.cli or not args.tui or not args.platform or not args.third_party_notices:
            parser.error("platform package requires --cli, --tui, --platform and --third-party-notices")
        if args.platform != host_platform():
            parser.error(f"platform package must be built and run natively; host is {host_platform()}")
        binaries = {"m-model": args.cli.resolve(strict=True), "m-model-tui": args.tui.resolve(strict=True)}
        evidence = {name: inspect_binary(path, name, args.platform, args.local_test) for name, path in binaries.items()}
        if not args.local_test:
            for name, source in (("m-model", "backend"), ("m-model-tui", "workspace")):
                try:
                    reported = subprocess.check_output([str(binaries[name]), "--build-source-revision"], text=True, timeout=30).strip()
                except (OSError, subprocess.SubprocessError) as error:
                    parser.error(f"cannot read {name} build source revision: {error}")
                if reported != identities[source]["revision"]:
                    parser.error(f"{name} reports build revision {reported!r}, not the {source} checkout revision")
                evidence[name]["reported_build_revision"] = reported
        try:
            verify_embedded_workspace(binaries["m-model-tui"], sources)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.error(f"cannot verify embedded Neovim workspace: {error}")
        if any(item["version"] != version for item in evidence.values()):
            parser.error("binary versions do not match the project manifest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-maven-npm-stage-") as directory:
        stage = Path(directory) / "package"
        if args.kind == "launcher":
            shutil.copytree(TEMPLATE, stage)
            for name, contents in sources.items():
                destination = stage / "nvim" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
            examples = stage / "examples"
            examples.mkdir()
            for name in ("__init__.py", "external_loop.py", "generic_loop.py", "bandit_loop.py", "gymnasium_loop.py", "tensorboard_loop.py", "from_scratch_components.py", "from_scratch_walkthrough.md"):
                shutil.copy2(ROOT / "examples" / name, examples / name)
            complex_examples = examples / "complex_acceptance"
            complex_examples.mkdir()
            for source in sorted((ROOT / "examples" / "complex_acceptance").iterdir()):
                if source.is_file() and not source.is_symlink() and (source.suffix == ".py" or source.name == "README.md"):
                    shutil.copy2(source, complex_examples / source.name)
            docs = stage / "docs"
            docs.mkdir()
            shutil.copy2(args.tui_source / "docs" / "unified-workspace-guide.md", docs / "unified-workspace-guide.md")
            for name in ("custom-managed-optimizer.md", "workspace-journey.md", "existing-project-adaptation.md", "working-directory-contract.md"):
                shutil.copy2(ROOT / "docs" / name, docs / name)
            shutil.copy2(ROOT / "docs" / "release" / "engineering-preview.md", docs / "engineering-preview.md")
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
        # Both independently staged package kinds carry the same identity so
        # the launcher can detect a mixed publication at installation time.
        manifest["modelMaven"] = {"workspace": {
            "algorithm": workspace["algorithm"], "digest": workspace["digest"]
        }}
        if not args.local_test:
            manifest["modelMaven"]["sourceCheckouts"] = identities
        if args.kind == "platform" and args.platform.startswith("linux-"):
            # Requirements come from each binary's ELF version table, not
            # from the build machine's glibc version or the release baseline.
            manifest["modelMaven"]["glibcRequirements"] = {
                name: binary["glibc_max"] for name, binary in evidence.items()
            }
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
            "binaries": evidence, "workspace": workspace, "source_checkouts": identities,
            "scope": "checkout identities are not binary build attestations; native startup checked where applicable; user manages Python/PyTorch; not published",
        }, indent=2) + "\n")
        print(tarball)


if __name__ == "__main__":
    main()

"""The CI shape: kernel surfaces, descriptor validation and a provider-gate matrix keyed by pin.

``ci/gates.json`` names each gate, the pipelines (source kinds) it exercises
with real providers and the pins it adds beyond their descriptors. A gate's
pins are therefore derived, never restated: the descriptors define pipeline
provider pins, package runtime manifests define terminal instrument pins, and
provider descriptors define providers no pipeline step names (the Julia
model worker), and ``extra_pins`` define the rest (checkers, producers,
vendored sources). The
``pin_key`` of a gate is the digest of that pin set, so a pin change is a new
matrix row and a new cache key.

    python scripts/ci_matrix.py check            # coverage and single definition of every pin
    python scripts/ci_matrix.py matrix           # GitHub matrices (kernel surfaces, provider gates)
    python scripts/ci_matrix.py pins GATE        # the exact pins a gate binds
    python scripts/ci_matrix.py run GATE [--phase provision|run]

Gate scripts read pins from the descriptors, runtime manifests and
``ci/gates.json`` and never from a moving branch; ``check`` refuses any
revision literal in a gate script or workflow, so each pin has one definition.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ci" / "gates.json"
DESCRIPTORS = ROOT / "src" / "ciw" / "pipelines" / "descriptors"
PROVIDERS = ROOT / "src" / "ciw" / "pipelines" / "providers"
PACKAGE = ROOT / "src" / "ciw"
WORKFLOWS = ROOT / ".github" / "workflows"
GATE_FIELDS = {"gate", "summary", "kinds", "providers", "manifests", "extra_pins", "os", "python", "timeout", "run", "provision",
               "artifacts", "artifacts_always", "rust", "node", "apt", "env", "cache", "linux_only_reason"}
SURFACE_FIELDS = {"gate", "summary", "os", "python", "timeout", "run", "artifacts"}
_HEX40 = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")


def load(path: Path = REGISTRY) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != "ciw.ci-gates.v1":
        raise ValueError("Unsupported CI gate registry")
    return registry


def descriptors() -> dict:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(DESCRIPTORS.glob("*.json"))}


def providers() -> dict:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(PROVIDERS.glob("*.json"))}


def provider_pins(name: str) -> list[dict]:
    """A provider descriptor's own pin (as one content digest) and its boundary pins."""
    value = providers()[name]
    own = sha256(json.dumps(value["pin"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ([{"role": value["role"], "revision": "sha256:" + own, "source": f"provider:{name}"}] +
            [{"role": item["role"], "revision": item["pin"]["revision"], "source": f"provider:{name}"}
             for item in value["boundary"]])


def manifest_pins(name: str) -> dict:
    """Terminal instrument pins from a package runtime manifest, by role."""
    value = json.loads((PACKAGE / name).read_text(encoding="utf-8"))
    if "commit" in value:  # plsr-runtime.json names one instrument
        return {"plsr": {"repository": value["repository"], "revision": value["commit"]}}
    if "revision" in value:  # esm-runtime.json names one repository
        return {"esm": {"repository": value["repository"], "revision": value["revision"]}}
    return {role: {"repository": pin["repository"], "revision": pin["revision"]} for role, pin in sorted(value.items())}


def gate_pins(gate: dict, registry: dict, pipelines: dict | None = None) -> list[dict]:
    """Every exact pin a gate binds: descriptor steps of its kinds, manifests and extra pins."""
    pipelines = descriptors() if pipelines is None else pipelines
    pins = {}
    for kind in gate.get("kinds", []):
        for step in pipelines[kind]["steps"]:
            if "pin" in step:
                pins[(step["role"], step["pin"]["revision"])] = {"role": step["role"], "revision": step["pin"]["revision"],
                                                                 "source": f"descriptor:{kind}"}
    for name in gate.get("providers", []):
        for pin in provider_pins(name):
            pins[(pin["role"], pin["revision"])] = pin
    for name in gate.get("manifests", []):
        for role, pin in manifest_pins(name).items():
            pins[(role, pin["revision"])] = {"role": role, "revision": pin["revision"], "source": f"manifest:{name}"}
    for name in gate.get("extra_pins", []):
        pin = registry["extra_pins"][name]
        pins[(name, pin["revision"])] = {"role": name, "revision": pin["revision"], "source": "ci/gates.json"}
    return [pins[key] for key in sorted(pins)]


def pin_key(pins: list[dict]) -> str:
    text = "\n".join(f"{pin['role']}@{pin['revision']}" for pin in pins)
    return sha256(text.encode()).hexdigest()[:12]


def _script_paths(entry: dict) -> set[Path]:
    paths = set()
    for command in [*entry.get("provision", []), *entry.get("run", [])]:
        for word in command:
            if isinstance(word, str) and word.startswith("scripts/") and word.endswith(".py"):
                paths.add(ROOT / word)
    return paths


def check(registry: dict | None = None, pipelines: dict | None = None) -> dict:
    """Every pinned pipeline has a gate, every pin has one definition, every gate is well formed."""
    registry = load() if registry is None else registry
    pipelines = descriptors() if pipelines is None else pipelines
    names = [entry["gate"] for entry in registry["surfaces"] + registry["gates"]]
    if len(names) != len(set(names)):
        raise ValueError("Gate names must be unique")
    for entry in registry["surfaces"]:
        if set(entry) - SURFACE_FIELDS or not entry["run"]:
            raise ValueError(f"Surface {entry['gate']} has undeclared fields or no commands")
    covered = set()
    for gate in registry["gates"]:
        if set(gate) - GATE_FIELDS or not {"gate", "summary", "kinds", "os", "python", "timeout", "run"} <= set(gate):
            raise ValueError(f"Gate {gate['gate']} does not match the gate contract")
        unknown = set(gate["kinds"]) - set(pipelines)
        if unknown:
            raise ValueError(f"Gate {gate['gate']} names undeclared pipelines {sorted(unknown)}")
        if (set(gate.get("manifests", [])) - set(registry["manifests"]) or
                set(gate.get("extra_pins", [])) - set(registry["extra_pins"]) or set(gate.get("providers", [])) - set(providers())):
            raise ValueError(f"Gate {gate['gate']} names an undeclared provider, manifest or extra pin")
        if not gate_pins(gate, registry, pipelines) and not gate["kinds"]:
            raise ValueError(f"Gate {gate['gate']} binds no pipeline and no pin")
        covered |= set(gate["kinds"])
        for path in _script_paths(gate):
            if not path.is_file():
                raise ValueError(f"Gate {gate['gate']} runs a missing script {path.relative_to(ROOT)}")
    pinned = {kind for kind, value in pipelines.items() if any("pin" in step for step in value["steps"])}
    if pinned - covered:
        raise ValueError(f"Pinned pipelines without a provider gate: {sorted(pinned - covered)}")
    ungated = set(providers()) - {name for gate in registry["gates"] for name in gate.get("providers", [])}
    if ungated:
        raise ValueError(f"Provider descriptors without a provider gate: {sorted(ungated)}")
    used_extra = {name for gate in registry["gates"] for name in gate.get("extra_pins", [])}
    used_manifests = {name for gate in registry["gates"] for name in gate.get("manifests", [])}
    if set(registry["extra_pins"]) - used_extra or set(registry["manifests"]) - used_manifests:
        raise ValueError("Every declared extra pin and manifest is bound by a gate")
    # Pins live only in the JSON registries (descriptors, manifests, ci/gates.json):
    # a revision literal in a gate script or workflow is a second definition.
    scanned = {*WORKFLOWS.glob("*.yml"), *(path for entry in registry["surfaces"] + registry["gates"] for path in _script_paths(entry))}
    scanned |= {ROOT / "scripts" / "provider_checkouts.py"}
    for path in sorted(scanned):
        found = set(_HEX40.findall(path.read_text(encoding="utf-8")))
        if found:
            shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            raise ValueError(f"{shown} restates revision(s) {sorted(found)}; read them from the "
                             "descriptors, runtime manifests or ci/gates.json")
    return registry


def matrix(registry: dict | None = None) -> dict:
    registry = load() if registry is None else registry
    pipelines = descriptors()

    def rows(entries, kind):
        out = []
        for entry in entries:
            pins = gate_pins(entry, registry, pipelines) if kind == "gate" else []
            key = pin_key(pins) if pins else "none"
            for os_name in entry["os"]:
                for python in entry["python"]:
                    cache = entry.get("cache")
                    out.append({
                        "gate": entry["gate"], "os": os_name, "python": python, "timeout": entry["timeout"],
                        "pin_key": key, "rust": entry.get("rust", ""), "node": entry.get("node", ""),
                        "apt": " ".join(entry.get("apt", [])),
                        "artifacts": entry.get("artifacts", "").replace("{output}", f"results/{entry['gate']}"),
                        "artifacts_always": bool(entry.get("artifacts_always")),
                        "provision": bool(entry.get("provision")),
                        "cache_key": f"{entry['gate']}-{cache['toolchain']}-pins-{key}" if cache else "",
                        "cache_paths": "\n".join(cache["paths"]) if cache else "",
                    })
        return out

    return {"kernel": {"include": rows(registry["surfaces"], "surface")},
            "providers": {"include": rows(registry["gates"], "gate")}}


def render_gates(registry: dict | None = None) -> str:
    """The provider-gate section of docs/PIPELINES.md."""
    registry = load() if registry is None else registry
    pipelines = descriptors()
    lines = ["## Provider gates", "",
             "CI has three jobs ([`ci.yml`](../.github/workflows/ci.yml)): `descriptors` binds every descriptor to "
             "code and derives the matrices, `kernel` runs the kernel surfaces below and `providers` runs one row "
             "per gate, platform and Python, keyed by the digest of the exact pins the gate binds. Gates are "
             "declared in [`ci/gates.json`](../ci/gates.json) and run by `python scripts/ci_matrix.py run GATE`.", "",
             "| Gate | Pin key | Pipelines and providers | Pins | Platforms |", "| --- | --- | --- | --- | --- |"]
    for gate in registry["gates"]:
        pins = gate_pins(gate, registry, pipelines)
        bound = [f"`{pipelines[kind]['pipeline_id']}`" for kind in gate["kinds"]]
        bound += [f"`{providers()[name]['provider_id']}`" for name in gate.get("providers", [])]
        bound += [f"{name} (terminal)" for name in gate.get("manifests", [])]
        platforms = ", ".join(os_name.removesuffix("-latest") for os_name in gate["os"]) + " · py" + "/".join(gate["python"])
        roles = ", ".join(sorted({pin["role"] for pin in pins})) or "none"
        lines.append(f"| `{gate['gate']}` | `{pin_key(pins) if pins else 'none'}` | {', '.join(bound) or 'none'} | "
                     f"{roles} | {platforms} |")
    lines += ["", "Kernel surfaces: " + "; ".join(f"`{entry['gate']}` ({entry['summary']})" for entry in registry["surfaces"]) + ".",
              "", "Pins beyond the descriptors:", ""]
    for name, pin in registry["extra_pins"].items():
        lines.append(f"- `{name}` — {pin['repository']} `{pin['revision'][:12]}`: {pin['purpose']}")
    return "\n".join(lines) + "\n"


def _entry(name: str, registry: dict) -> dict:
    for entry in registry["surfaces"] + registry["gates"]:
        if entry["gate"] == name:
            return entry
    raise SystemExit(f"Unknown gate {name}")


def _expand(word: str, *, output: Path, work: Path, venv: Path) -> list[str]:
    if word.startswith("{glob:") and word.endswith("}"):
        matches = sorted(ROOT.glob(word[6:-1]))
        if len(matches) != 1:
            raise SystemExit(f"{word} must match exactly one file; found {len(matches)}")
        return [str(matches[0])]
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    value = (word.replace("{python}", sys.executable).replace("{output}", str(output))
             .replace("{work}", str(work)).replace("{venv}", str(python)))
    return [os.path.expanduser(value) if value.startswith("~") else value]


def run(name: str, phase: str = "run", registry: dict | None = None) -> None:
    registry = load() if registry is None else registry
    entry = _entry(name, registry)
    output = ROOT / "results" / name
    work = Path(os.environ.get("RUNNER_TEMP") or ROOT / ".ci-work") / f"ciw-{name}"
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **entry.get("env", {}), "CIW_GATE": name}
    if entry in registry["gates"]:
        env["CIW_GATE_PIN_KEY"] = pin_key(gate_pins(entry, registry))
    for command in entry.get(phase, []):
        argv = [piece for word in command for piece in _expand(word, output=output, work=work, venv=work / "ciw-installed")]
        print("+", " ".join(argv), flush=True)
        subprocess.run(argv, cwd=ROOT, env=env, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("matrix").add_argument("--github-output", action="store_true",
                                               help="Append kernel= and providers= lines for $GITHUB_OUTPUT")
    commands.add_parser("pins").add_argument("gate")
    runner = commands.add_parser("run")
    runner.add_argument("gate")
    runner.add_argument("--phase", choices=("provision", "run"), default="run")
    args = parser.parse_args(argv)
    if args.command == "check":
        registry = check()
        print(f"PASS: {len(registry['gates'])} provider gates and {len(registry['surfaces'])} kernel surfaces; "
              "every pinned pipeline gated and every pin declared once")
    elif args.command == "matrix":
        value = matrix(check())
        if args.github_output:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
                for key, rows in value.items():
                    handle.write(f"{key}={json.dumps(rows, separators=(',', ':'))}\n")
        print(json.dumps(value, indent=2))
    elif args.command == "pins":
        registry = load()
        pins = gate_pins(_entry(args.gate, registry), registry)
        print(json.dumps({"gate": args.gate, "pin_key": pin_key(pins), "pins": pins}, indent=2))
    else:
        run(args.gate, args.phase)


if __name__ == "__main__":
    main()

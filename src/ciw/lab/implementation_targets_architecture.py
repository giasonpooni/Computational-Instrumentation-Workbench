"""Language boundaries: industrial C/C++ interface inventory and an import-graph scan (T143, T144).

The inventory records which industrial interfaces need C or C++ libraries and
the boundary CIW would use for each: a pinned subprocess provider that
exchanges retained bytes, never an in-process binding inside the evidence
layer, and never an enabled write path. It is an analytic design record; no
listed library is installed, linked or exercised here.

The scan parses every module of the installed ``ciw`` package with ``ast``
(nothing is imported or executed) and checks architectural rules: the
evidence and identity closure is standard-library Python, in-process native
loading is confined to declared hardware probes, process spawns use argument
vectors (no shell) from modules that record a runtime identity, and no
compiled extension ships inside the package. A rule scan is structural
evidence about source text, not proof of runtime behaviour.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- T143
INTERFACES = (
    {"name": "OPC UA client (read-only subscriptions)", "category": "plant data",
     "libraries": ["open62541 (C)", "Unified Automation C++ SDK", "Siemens/vendor OPC UA stacks"],
     "why_native": "Certified and vendor-supported stacks are C/C++; security policies (X.509, SignAndEncrypt) and "
                   "subscription timing are implemented there",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "absent",
     "exchange": "framed JSON/CBOR records over stdio with node ids, source timestamps and status codes retained raw",
     "identity": ["executable sha256", "library version and build flags", "server certificate fingerprint"]},
    {"name": "EtherCAT master (process data monitoring)", "category": "fieldbus",
     "libraries": ["SOEM (C)", "IgH EtherCAT Master (C, kernel module)", "Beckhoff TwinCAT ADS (C++)"],
     "why_native": "Cyclic process-data exchange needs raw sockets or kernel drivers with microsecond-scale deadlines "
                   "that an interpreter cannot meet",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "disabled",
     "exchange": "retained cyclic PDO snapshots with working counters and distributed-clock timestamps",
     "identity": ["master executable sha256", "ESI/ENI configuration digest", "kernel module version"]},
    {"name": "Vendor camera SDKs (GenICam GenTL)", "category": "vision acquisition",
     "libraries": ["Basler pylon (C++)", "Teledyne FLIR Spinnaker (C++)", "Allied Vision Vimba X (C/C++)",
                   "GenICam reference implementation (C++)"],
     "why_native": "Proprietary drivers, zero-copy buffers and hardware triggers are exposed only through C/C++ ABIs",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "absent",
     "exchange": "raw frame bytes with sha256, chunk timestamps and camera feature snapshot (GenICam XML digest)",
     "identity": ["SDK version", "camera firmware version", "GenICam XML digest"]},
    {"name": "Point cloud processing (PCL / Open3D)", "category": "3D geometry",
     "libraries": ["PCL (C++)", "Open3D (C++ core with Python bindings)"],
     "why_native": "Registration, KD-tree and surface reconstruction kernels are C++ with nondeterministic threading "
                   "unless pinned; bindings would put unaudited native state in the evidence process",
     "boundary": "pinned_subprocess", "direction": "geometry_exchange", "write_path": "absent",
     "exchange": "PLY/PCD files with sha256 in and out; parameters in a retained request",
     "identity": ["library version", "thread count", "build flags (OpenMP, SIMD)"]},
    {"name": "CAD kernel (OpenCASCADE)", "category": "CAD geometry",
     "libraries": ["OpenCASCADE Technology (C++)", "pythonOCC bindings"],
     "why_native": "B-rep, STEP/IGES translation and tolerance handling live in the C++ kernel",
     "boundary": "pinned_subprocess", "direction": "geometry_exchange", "write_path": "absent",
     "exchange": "STEP/BREP files with sha256; exported meshes and measurements as retained records",
     "identity": ["OCCT version", "translator settings digest", "tolerance settings"]},
)
BOUNDARIES = frozenset({"pinned_subprocess"})
DIRECTIONS = frozenset({"read_only", "geometry_exchange"})
WRITE_PATHS = frozenset({"absent", "disabled"})


class ArchitectureRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_inventory(entries) -> list:
    """Refuse in-process bindings, write-capable directions and entries without an identity pin."""
    for entry in entries:
        for key in ("name", "category", "libraries", "why_native", "boundary", "direction", "write_path",
                    "exchange", "identity"):
            if not entry.get(key):
                raise ArchitectureRefusal("incomplete_entry", f"Interface entry lacks {key}")
        if entry["boundary"] not in BOUNDARIES:
            raise ArchitectureRefusal("in_process_binding", f"{entry['name']}: native code must stay behind a pinned "
                                      "subprocess boundary")
        if entry["direction"] not in DIRECTIONS:
            raise ArchitectureRefusal("write_capable_direction", f"{entry['name']}: direction must be read-only")
        if entry["write_path"] not in WRITE_PATHS:
            raise ArchitectureRefusal("write_path_enabled", f"{entry['name']}: write paths stay absent or disabled")
    return list(entries)


# ----------------------------------------------------------------- T144
EVIDENCE_MODULES = ("ciw.lab.evidence", "ciw.lab.report", "ciw.core.identities")
NATIVE_ALLOWLIST = frozenset({"ciw.energy_cuda", "ciw.energy_nvml"})  # declared hardware energy probes
NATIVE_MODULES = frozenset({"ctypes", "cffi", "cppyy", "ctypes.util", "_ctypes"})
SPAWN_FUNCTIONS = frozenset({"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"})
OS_SPAWN = re.compile(r"^(system|popen|exec\w*|spawn\w*|posix_spawn\w*)$")
IDENTITY_TOKENS = ("revision", "source_tree", "runtime_identity", "sha256", "digest")


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(current: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = current.split(".")
    base = base if is_package else base[:-1]
    base = base[:len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def scan_source(name: str, text: str, is_package: bool = False) -> dict:
    """Imports, native loading, process spawns and shell use of one module's source text."""
    tree = ast.parse(text)
    imports, subprocess_aliases, spawn_names, os_aliases = set(), set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                if alias.name == "subprocess":
                    subprocess_aliases.add(alias.asname or "subprocess")
                if alias.name == "os":
                    os_aliases.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(name, is_package, node)
            imports.add(target)
            for alias in node.names:
                if target.startswith("ciw") and alias.name != "*":
                    imports.add(f"{target}.{alias.name}")
                if node.level == 0 and target == "subprocess" and alias.name in SPAWN_FUNCTIONS:
                    spawn_names.add(alias.asname or alias.name)
    spawns, shell = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_spawn = (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and ((func.value.id in subprocess_aliases and func.attr in SPAWN_FUNCTIONS)
                         or (func.value.id in os_aliases and OS_SPAWN.match(func.attr)))) \
            or (isinstance(func, ast.Name) and func.id in spawn_names)
        if is_spawn:
            spawns.append(node.lineno)
        for keyword in node.keywords:
            if keyword.arg == "shell" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False):
                shell.append(node.lineno)
    native = sorted(imp for imp in imports if imp.split(".")[0] in {m.split(".")[0] for m in NATIVE_MODULES})
    return {"imports": sorted(imports), "native": native, "spawn_lines": sorted(spawns), "shell_lines": sorted(shell),
            "identity_tokens": sorted(token for token in IDENTITY_TOKENS if token in text),
            "mentions_subprocess": "subprocess" in text}


def scan_package(root: Path = PACKAGE_ROOT) -> dict:
    modules, unparsed = {}, []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = module_name(path, root)
        try:
            modules[name] = scan_source(name, path.read_text(encoding="utf-8"), path.name == "__init__.py")
        except (SyntaxError, UnicodeDecodeError, ValueError):
            unparsed.append(name)
    compiled = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in (".so", ".pyd", ".dll"))
    return {"modules": modules, "unparsed": unparsed, "compiled_extensions": compiled}


def closure(modules: dict, roots) -> list:
    """Transitive ciw-internal import closure of ``roots`` (packages include their __init__)."""
    seen, stack = set(), [r for r in roots if r in modules]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for target in modules[current]["imports"]:
            parts = target.split(".")
            for end in range(len(parts), 0, -1):
                candidate = ".".join(parts[:end])
                if candidate in modules:
                    if candidate not in seen:
                        stack.append(candidate)
                    break
    return sorted(seen)


def violations(scan: dict) -> list:
    """Architecture rule violations as (rule, module, detail) triples."""
    modules = scan["modules"]
    stdlib = set(sys.stdlib_module_names)
    out = []
    evidence = closure(modules, EVIDENCE_MODULES)
    for name in evidence:
        info = modules[name]
        third = sorted(i for i in info["imports"] if i and i.split(".")[0] not in stdlib and not i.startswith("ciw"))
        if third:
            out.append(("evidence_closure_not_stdlib", name, ", ".join(third)))
        if info["native"]:
            out.append(("evidence_native_loading", name, ", ".join(info["native"])))
        if info["spawn_lines"]:
            out.append(("evidence_spawns_process", name, str(info["spawn_lines"])))
    for name, info in modules.items():
        if info["native"] and name not in NATIVE_ALLOWLIST:
            out.append(("native_loading_outside_allowlist", name, ", ".join(info["native"])))
        if info["shell_lines"]:
            out.append(("shell_invocation", name, str(info["shell_lines"])))
        if info["spawn_lines"] and not info["identity_tokens"]:
            out.append(("spawn_without_runtime_identity", name, str(info["spawn_lines"])))
    for path in scan["compiled_extensions"]:
        out.append(("compiled_extension_in_package", path, "native code shipped in the evidence package"))
    return sorted(out)


def mutated_scan(scan: dict, name: str, text: str) -> dict:
    """The package scan with one module's source replaced (negative tests)."""
    modules = dict(scan["modules"])
    modules[name] = scan_source(name, text, is_package=False)
    return {"modules": modules, "unparsed": scan["unparsed"], "compiled_extensions": scan["compiled_extensions"]}

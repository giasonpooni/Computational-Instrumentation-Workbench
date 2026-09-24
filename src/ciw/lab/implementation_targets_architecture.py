"""Language boundaries: industrial C/C++ interface inventory and an import-graph scan (T143, T144, T153, T154).

The inventory records which industrial interfaces are assigned to C or C++
libraries (required, or preferred where a pure-Python stack exists) and the
boundary CIW would use for each: a pinned subprocess provider that
exchanges retained bytes, never an in-process binding inside the evidence
layer, and never an enabled write path. It is an analytic design record; no
listed library is installed, linked or exercised here.

The scan parses every module of the installed ``ciw`` package with ``ast``
(nothing is imported or executed) and checks architectural rules: the
evidence and identity closure is standard-library Python, in-process native
loading is confined to declared hardware probes, process spawns (subprocess,
os, asyncio) use argument vectors and no shell, and no compiled extension
ships inside the package. Whether a spawning module names an identity in its
code is recorded as a heuristic only; whether the spawned executable is pinned
(compared with an expected identity before it runs) is not something a source
scan can establish, and PATH-resolved spawns are listed as counterexamples.
The same scan lists machine write paths (device, serial, fieldbus and
instrument libraries, device nodes, and network libraries outside declared
servers) and modules whose code names a control-like output. A rule scan is
structural evidence about source text, not proof of runtime behaviour.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- T143
INTERFACES = (
    # OPC UA does not need C/C++: pure-Python stacks exist. The C/C++ choice is a preference, stated with its reason.
    {"name": "OPC UA client (read-only subscriptions)", "category": "plant data",
     "libraries": ["open62541 (C)", "Unified Automation C++ SDK", "Siemens/vendor OPC UA stacks"],
     "native_necessity": "preferred",
     "pure_python_alternatives": ["asyncua (asyncio OPC UA client and server)",
                                  "python-opcua (package opcua; superseded by asyncua)"],
     "why_native": "Not required: the pure-Python stacks can subscribe read-only. C/C++ stacks are preferred for OPC "
                   "Foundation certification, vendor support and a maintained security-policy implementation (X.509, "
                   "SignAndEncrypt) that plant operators accept; either stack would sit behind the same pinned "
                   "subprocess boundary, never inside the evidence process",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "absent",
     "exchange": "framed JSON/CBOR records over stdio with node ids, source timestamps and status codes retained raw",
     "identity": ["executable sha256", "library version and build flags", "server certificate fingerprint"]},
    # A master (SOEM, IgH, TwinCAT) originates every bus frame, including output process data, so it can never
    # be read-only; monitoring uses a passive TAP whose capture host cannot inject frames.
    {"name": "EtherCAT passive monitoring (network TAP capture)", "category": "fieldbus",
     "libraries": ["libpcap / Npcap (C)", "Wireshark EtherCAT dissectors (C)", "TAP or probe device driver (vendor C)"],
     "why_native": "Line-rate capture with hardware timestamps needs kernel capture drivers and C dissectors; a Python "
                   "loop cannot keep up with cyclic frames at microsecond spacing",
     "native_necessity": "required",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "absent", "fieldbus_role": "passive_tap",
     "exchange": "pcapng captures with sha256; decoded PDO snapshots with working counters and TAP timestamps retained",
     "identity": ["TAP device model and firmware version", "capture library version", "ENI/ESI decoding configuration "
                  "digest"]},
    {"name": "Vendor camera SDKs (GenICam GenTL)", "category": "vision acquisition",
     "libraries": ["Basler pylon (C++)", "Teledyne FLIR Spinnaker (C++)", "Allied Vision Vimba X (C/C++)",
                   "GenICam reference implementation (C++)"],
     "native_necessity": "required",
     "why_native": "Proprietary drivers, zero-copy buffers and hardware triggers are exposed only through C/C++ ABIs",
     "boundary": "pinned_subprocess", "direction": "read_only", "write_path": "absent",
     "exchange": "raw frame bytes with sha256, chunk timestamps and camera feature snapshot (GenICam XML digest)",
     "identity": ["SDK version", "camera firmware version", "GenICam XML digest"]},
    {"name": "Point cloud processing (PCL / Open3D)", "category": "3D geometry",
     "libraries": ["PCL (C++)", "Open3D (C++ core with Python bindings)"],
     "native_necessity": "required",
     "why_native": "Registration, KD-tree and surface reconstruction kernels are C++ with nondeterministic threading "
                   "unless pinned; bindings would put unaudited native state in the evidence process",
     "boundary": "pinned_subprocess", "direction": "geometry_exchange", "write_path": "absent",
     "exchange": "PLY/PCD files with sha256 in and out; parameters in a retained request",
     "identity": ["library version", "thread count", "build flags (OpenMP, SIMD)"]},
    {"name": "CAD kernel (OpenCASCADE)", "category": "CAD geometry",
     "libraries": ["OpenCASCADE Technology (C++)", "pythonOCC bindings"],
     "native_necessity": "required",
     "why_native": "B-rep, STEP/IGES translation and tolerance handling live in the C++ kernel",
     "boundary": "pinned_subprocess", "direction": "geometry_exchange", "write_path": "absent",
     "exchange": "STEP/BREP files with sha256; exported meshes and measurements as retained records",
     "identity": ["OCCT version", "translator settings digest", "tolerance settings"]},
)
BOUNDARIES = frozenset({"pinned_subprocess"})
NECESSITIES = frozenset({"required", "preferred"})
DIRECTIONS = frozenset({"read_only", "geometry_exchange"})
WRITE_PATHS = frozenset({"absent", "disabled"})
FIELDBUS_ROLES = frozenset({"passive_tap"})
# Identity pins must name something concrete; moving labels, ranges and wildcards pin nothing. Pins are
# descriptions ("library version and build flags"), so words that are also ordinary prose ("main board", "/dev/",
# "current sensor") are refused only as the whole pin; unambiguous moving labels are refused anywhere.
FLOATING_WHOLE_PINS = ("current", "main", "master", "trunk", "head", "dev", "develop", "x")
FLOATING_LABELS = ("latest", "nightly", "snapshot", "stable", "any", "unknown", "tbd", "n/a", "na")
_FLOATING_PIN = re.compile(
    r"^\s*\Z"                                                              # empty or blank
    r"|^\s*(" + "|".join(FLOATING_WHOLE_PINS) + r")\s*\Z"                  # whole-pin branch or ref names
    r"|(?<![a-z0-9])(" + "|".join(FLOATING_LABELS) + r")(?![a-z0-9])"      # moving labels (also 2023.2_nightly)
    r"|(?-i:(?<![A-Za-z0-9])HEAD(?![A-Za-z0-9]))"                          # git HEAD anywhere
    r"|[*?<>=~^]"                                                          # wildcards and version ranges
    r"|[0-9]\.x(?![a-z0-9])",                                              # 1.x, 2024.x, 2024.1.x
    re.IGNORECASE)


class ArchitectureRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_inventory(entries) -> list:
    """Refuse in-process bindings, write-capable directions or bus roles and missing or floating identity pins.

    ``native_necessity`` says whether C/C++ is required or only preferred; a
    preferred entry must name the pure-Python alternatives it passes over.
    """
    for entry in entries:
        for key in ("name", "category", "libraries", "native_necessity", "why_native", "boundary", "direction",
                    "write_path", "exchange", "identity"):
            if not entry.get(key):
                raise ArchitectureRefusal("incomplete_entry", f"Interface entry lacks {key}")
        if entry["native_necessity"] not in NECESSITIES:
            raise ArchitectureRefusal("incomplete_entry", f"{entry['name']}: native_necessity must be required or "
                                      "preferred")
        if entry["native_necessity"] == "preferred" and not entry.get("pure_python_alternatives"):
            raise ArchitectureRefusal("incomplete_entry", f"{entry['name']}: a preferred native library must name the "
                                      "pure-Python alternatives it passes over")
        if entry["boundary"] not in BOUNDARIES:
            raise ArchitectureRefusal("in_process_binding", f"{entry['name']}: native code must stay behind a pinned "
                                      "subprocess boundary")
        if entry["direction"] not in DIRECTIONS:
            raise ArchitectureRefusal("write_capable_direction", f"{entry['name']}: direction must be read-only")
        if entry["write_path"] not in WRITE_PATHS:
            raise ArchitectureRefusal("write_path_enabled", f"{entry['name']}: write paths stay absent or disabled")
        if "fieldbus_role" in entry and entry["fieldbus_role"] not in FIELDBUS_ROLES:
            raise ArchitectureRefusal("write_capable_direction", f"{entry['name']}: a fieldbus master originates output "
                                      "process data; only a passive tap is read-only")
        pins = entry["identity"]
        if not isinstance(pins, list) or any(not isinstance(pin, str) or _FLOATING_PIN.search(pin) for pin in pins):
            raise ArchitectureRefusal("unpinned_identity", f"{entry['name']}: identity pins must be concrete, not "
                                      "empty, wildcard or moving labels")
    return list(entries)


# ----------------------------------------------------------------- T144
EVIDENCE_MODULES = ("ciw.lab.evidence", "ciw.lab.report", "ciw.core.identities")
NATIVE_ALLOWLIST = frozenset({"ciw.energy_cuda", "ciw.energy_nvml"})  # declared hardware energy probes
NATIVE_MODULES = frozenset({"ctypes", "cffi", "cppyy", "ctypes.util", "_ctypes"})
SPAWN_FUNCTIONS = frozenset({"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"})
ASYNC_SPAWN = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
OS_SPAWN = re.compile(r"^(system|popen|exec\w*|spawn\w*|posix_spawn\w*)$")
# Calls that always go through a shell, whatever their arguments.
SHELL_CALLS = frozenset({"system", "popen", "create_subprocess_shell", "getoutput", "getstatusoutput"})
IDENTITY_TOKENS = ("revision", "source_tree", "runtime_identity", "sha256", "digest")

# Write paths toward machines (T153). Device, serial, fieldbus, industrial-protocol and instrument-I/O libraries
# are refused everywhere; network libraries only in declared modules that serve the workbench's own clients.
# A dotted entry matches that module and its submodules.
DEVICE_LIBRARIES = frozenset({
    "serial", "can", "canopen", "pymodbus", "minimalmodbus", "umodbus", "pysoem", "pyads", "snap7", "pycomm3",
    "cpppo", "pylogix", "asyncua", "opcua", "usb", "hid", "ftd2xx", "pyftdi", "smbus", "smbus2", "spidev", "RPi",
    "gpiozero", "periphery", "pyvisa", "nidaqmx", "evdev", "pyudev"})
NETWORK_LIBRARIES = frozenset({
    "socket", "socketserver", "ssl", "http", "urllib.request", "xmlrpc", "ftplib", "smtplib", "telnetlib",
    "websockets", "websocket", "aiohttp", "requests", "httpx", "urllib3", "zmq", "grpc", "paho", "mcp"})
NETWORK_CALLS = frozenset({"open_connection", "start_server", "open_unix_connection", "start_unix_server",
                           "create_connection", "create_server", "create_datagram_endpoint",
                           "create_unix_connection", "create_unix_server"})
_DEVICE_NODE = re.compile(r"^(/dev/|\\\\[.?]\\|COM[0-9]+:?\Z)")
DECLARED_NETWORK = {
    "ciw.server": "workbench websocket server: sends session snapshots and results to its browser clients",
    "ciw.cli": "websocket client for the workbench server and launcher of ciw.server",
    "ciw.lab.mcp_server": "MCP tool server for lab reports, run over stdio",
}
# Control-like outputs (T154): stop or abort requests, setpoints, gain changes and command conversions named in code.
CONTROL_TERMS = ("stop request", "abort_action", "abort_on", "setpoint", "to_command", "safe torque",
                 "release a stop", "command motion", "change gains")
# The scanner's own term lists name every library and term above without using them.
SCANNER_MODULES = frozenset({"ciw.lab.implementation_targets_architecture"})


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
    """Imports, native loading, process spawns, shell use and PATH lookups of one module's source text.

    Identity tokens count only in code (identifiers and non-docstring string
    constants); comments and docstrings do not.
    """
    tree = ast.parse(text)
    imports, aliases = set(), {"subprocess": set(), "os": set(), "asyncio": set(), "shutil": set()}
    spawn_names, shell_names, which_names = set(), set(), set()
    calls, words, docstrings, strings = [], [], set(), []
    for node in ast.walk(tree):  # breadth first: a docstring's owner is visited before the docstring
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
            if not isinstance(node, ast.Module):
                words.append(node.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                if alias.name in aliases:
                    aliases[alias.name].add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(name, is_package, node)
            imports.add(target)
            for alias in node.names:
                if target.startswith("ciw") and alias.name != "*":
                    imports.add(f"{target}.{alias.name}")
                local = alias.asname or alias.name
                if node.level == 0 and ((target == "subprocess" and alias.name in SPAWN_FUNCTIONS)
                                        or (target == "os" and OS_SPAWN.match(alias.name))
                                        or (target == "asyncio" and alias.name in ASYNC_SPAWN)):
                    spawn_names.add(local)
                    if alias.name in SHELL_CALLS:
                        shell_names.add(local)
                if node.level == 0 and target == "shutil" and alias.name == "which":
                    which_names.add(local)
        elif isinstance(node, ast.Call):
            calls.append(node)
        elif isinstance(node, ast.Name):
            words.append(node.id)
        elif isinstance(node, ast.Attribute):
            words.append(node.attr)
        elif isinstance(node, ast.arg):
            words.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            words.append(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            words.append(node.value)
            strings.append(node.value)
    spawns, shell, which, network_calls = [], set(), [], set()
    for node in calls:
        func = node.func
        called = None
        if isinstance(func, ast.Attribute) and func.attr in NETWORK_CALLS:
            network_calls.add(func.attr)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = func.value.id
            if ((owner in aliases["subprocess"] and func.attr in SPAWN_FUNCTIONS)
                    or (owner in aliases["os"] and OS_SPAWN.match(func.attr))
                    or (owner in aliases["asyncio"] and func.attr in ASYNC_SPAWN)):
                called = func.attr
            if owner in aliases["shutil"] and func.attr == "which":
                which.append(node.lineno)
        elif isinstance(func, ast.Name):
            if func.id in spawn_names:
                called = func.id
            if func.id in which_names:
                which.append(node.lineno)
        if called is not None:
            spawns.append(node.lineno)
            if called in SHELL_CALLS or (isinstance(func, ast.Name) and func.id in shell_names):
                shell.add(node.lineno)
        for keyword in node.keywords:
            if keyword.arg == "shell" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False):
                shell.add(node.lineno)
    native = sorted(imp for imp in imports if imp.split(".")[0] in {m.split(".")[0] for m in NATIVE_MODULES})
    code = "\n".join(words)
    lowered = code.lower()
    return {"imports": sorted(imports), "native": native, "spawn_lines": sorted(spawns),
            "shell_lines": sorted(shell), "which_lines": sorted(which),
            "identity_tokens": sorted(token for token in IDENTITY_TOKENS if token in code),
            "mentions_subprocess": "subprocess" in text,
            "device_libraries": sorted(imp for imp in imports if _matches(imp, DEVICE_LIBRARIES)),
            "network_libraries": sorted(imp for imp in imports if _matches(imp, NETWORK_LIBRARIES)),
            "network_calls": sorted(network_calls),
            "device_nodes": sorted({value for value in strings if _DEVICE_NODE.match(value)}),
            "control_terms": sorted(term for term in CONTROL_TERMS if term in lowered)}


def _matches(module: str, entries) -> bool:
    return any(module == entry or module.startswith(entry + ".") for entry in entries)


def scan_package(root: Path = PACKAGE_ROOT) -> dict:
    """Scan every module; ``package_sha256`` identifies the scanned source tree (line endings normalized)."""
    from .implementation_targets_serial import canonical_sha256

    modules, unparsed, digests = {}, [], {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = module_name(path, root)
        data = path.read_bytes()
        digests[name] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        try:
            modules[name] = scan_source(name, data.decode("utf-8"), path.name == "__init__.py")
        except (SyntaxError, UnicodeDecodeError, ValueError):
            unparsed.append(name)
    compiled = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in (".so", ".pyd", ".dll"))
    return {"modules": modules, "unparsed": unparsed, "compiled_extensions": compiled,
            "package_sha256": canonical_sha256(digests), "files": len(digests)}


def closure(modules: dict, roots) -> list:
    """Transitive ciw-internal import closure of ``roots``.

    Importing a module executes every ancestor package's ``__init__`` first
    (``ciw``, then ``ciw.lab`` for ``ciw.lab.evidence``), so each visited
    module brings its scanned ancestor packages and their imports along.
    """
    seen, stack = set(), [r for r in roots if r in modules]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        names = current.split(".")
        stack.extend(parent for parent in (".".join(names[:end]) for end in range(1, len(names)))
                     if parent in modules and parent not in seen)
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


def path_resolved_spawners(scan: dict) -> list:
    """Modules that look an executable up on PATH (shutil.which) and spawn processes.

    Such a spawn runs whatever executable is first on PATH; recording its
    version afterwards is provenance, not a pin, because nothing compares it
    with an expected identity before it runs.
    """
    return sorted(name for name, info in scan["modules"].items() if info["which_lines"] and info["spawn_lines"])


def mutated_scan(scan: dict, name: str, text: str) -> dict:
    """The package scan with one module's source replaced (negative tests); a package keeps its package status."""
    modules = dict(scan["modules"])
    is_package = any(other.startswith(name + ".") for other in modules)
    modules[name] = scan_source(name, text, is_package=is_package)
    return dict(scan, modules=modules)


def write_paths(scan: dict) -> dict:
    """Device and network write paths per module, split into refused and declared ones (T153).

    Any device, serial, fieldbus or instrument library, and any device-node
    string such as '/dev/ttyUSB0' or 'COM3', is refused everywhere. Network
    libraries and connection calls are refused outside DECLARED_NETWORK.
    """
    device, network, declared = {}, {}, {}
    for name, info in sorted(scan["modules"].items()):
        if name in SCANNER_MODULES:
            continue
        found = info["device_libraries"] + [f"device node {node!r}" for node in info["device_nodes"]]
        if found:
            device[name] = found
        net = info["network_libraries"] + [f"call {call}" for call in info["network_calls"]]
        if net:
            (declared if name in DECLARED_NETWORK else network)[name] = net
    return {"device": device, "undeclared_network": network, "declared_network": declared}


def control_term_modules(scan: dict) -> dict:
    """Modules whose code (identifiers and non-docstring strings) names a control-like output (heuristic, T154)."""
    return {name: info["control_terms"] for name, info in sorted(scan["modules"].items())
            if info["control_terms"] and name not in SCANNER_MODULES}

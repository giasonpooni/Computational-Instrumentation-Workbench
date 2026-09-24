"""Declared pipelines: the data layer between investigations and the kernel.

A ``ciw.pipeline-descriptor.v1`` declares one source kind's pipeline: its
operation identity, inputs, provider steps with exact pins, verification,
refusal vocabulary, the named domain rules its implementation must keep, its
investigation and its operator surface. Descriptors are trusted process data
shipped with CIW; saved workspaces never supply or extend them, and a
descriptor never loads code.

Until each kind's module reads its pins from its descriptor, ``live_pins``
normalizes the pins the module currently declares (module constants and
runtime manifests) and ``check`` requires the two to agree exactly, so the
descriptor cannot drift from what executes.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module, resources
import json
from pathlib import Path
import re

DESCRIPTOR_SCHEMA = "ciw.pipeline-descriptor.v1"
INVESTIGATION_SCHEMA = "ciw.investigation.v1"
INVOCATIONS = frozenset({"pinned_subprocess", "verified_contract_file", "host_bound_binary", "ciw_reference"})
SURFACES = frozenset({"entry", "inner", "bench", "certification", "source_only"})
IMPLEMENTATIONS = frozenset({"hand_written", "declared_workflow", "generic_runner"})
UPSTREAM_CARDINALITIES = frozenset({"none", "one", "ordered_many"})
FIELDS = frozenset({"schema", "pipeline_id", "source_kind", "session_schema", "summary", "inputs", "steps",
                    "verification", "refusals", "domain_rules", "investigations", "surface", "authority",
                    "implementation", "guide"})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_ROLE = re.compile(r"[a-z][a-z0-9]{1,15}\Z")
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,63}\*?\Z")
_SYMBOL = re.compile(r"ciw(\.[a-z_][a-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_LINE_CITATION = re.compile(r"\.(py|jl|json):\d")
# Admission codes the workbench itself raises for every pipeline.
WORKBENCH_REFUSALS = ("operation_unavailable", "workbench_capacity")

# Transitional table of where each kind's module declares its pins today.
_DESCRIPTOR_DEFINED = frozenset({"calibrated-observable", "identified-design", "calibrated-window",
                                 "acquired-calibrated-window", "telemetry", "instrument-exchange"})
_PIN_FIELDS = ("revision", "source_tree", "module", "source_root", "source_sha256", "path", "sha256", "repository")
_BINARY_ROLES = frozenset({"engine", "prover", "guest"})


def _normalize(pin):
    return {key: pin[key] for key in _PIN_FIELDS if key in pin}


def _refusal_codes(tree, roles):
    import ast
    codes = set()

    def pieces(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = pieces(node.left), pieces(node.right)
            return None if left is None or right is None else [a + b for a in left for b in right]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "upper"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "role"):
            return [role.upper() for role in roles] or ["*"]
        return ["*"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and (
                (isinstance(node.func, ast.Name) and node.func.id in {"AdapterRefusal", "_refuse"}) or
                (isinstance(node.func, ast.Attribute) and node.func.attr == "AdapterRefusal")):
            for code in pieces(node.args[0]) or []:
                if code != "*":
                    codes.add(code.split("*")[0] + "*" if "*" in code else code)
    return codes


def code_refusals(descriptor: dict, descriptors: dict | None = None) -> list[str]:
    """Refusal codes raised on the pipeline's execution path.

    The path is the implementation module, its declared delegates (modules
    whose pipelines it executes), the DeclaredWorkflow base or the shared
    runner when that is its runner, the pinned-subprocess adapter when a step
    uses it, and the
    workbench's own admission codes. Role-templated codes expand over the
    roles of the pipeline that owns the module.
    """
    import ast
    package = Path(__file__).resolve().parents[1]
    owners = {value["implementation"]["module"]: value for value in (descriptors or {}).values()}
    implementation = descriptor["implementation"]
    modules = [(implementation["module"], descriptor)]
    modules += [(module, owners.get(module, descriptor)) for module in implementation["delegates"]]
    if implementation["runner"] == "declared_workflow":
        modules.append(("ciw.declared_workload", descriptor))
    if implementation["runner"] == "generic_runner":
        modules.append(("ciw.pipelines.runner", descriptor))
    codes = set(WORKBENCH_REFUSALS)
    for module, owner in modules:
        path = package / (module.removeprefix("ciw.").replace(".", "/") + ".py")
        roles = [step["role"] for step in owner["steps"]]
        codes |= _refusal_codes(ast.parse(path.read_text(encoding="utf-8")), roles)
    if any(step["invocation"] == "pinned_subprocess" for step in descriptor["steps"]):
        codes |= _refusal_codes(ast.parse((package / "adapters" / "subprocess.py").read_text(encoding="utf-8")), [])
    return sorted(codes)


def live_pins(kind: str) -> dict:
    """The pins the kind's module declares today, normalized by provider role."""
    from ..workbench import _workflow
    workflow = _workflow(kind)
    roles = sorted(set(getattr(workflow, "ROLES", ())) - _BINARY_ROLES)
    if kind in _DESCRIPTOR_DEFINED:
        # These modules read their pins from the descriptor (pin_map), so the
        # descriptor is the definition and the binding is by construction.
        return {role: _normalize(pin) for role, pin in pin_map(kind).items()}
    if kind == "variational-free-energy":
        from ..free_energy_native import PINS
        return {role: _normalize(PINS[role]) for role in roles}
    module = import_module(type(workflow).__module__ if not hasattr(workflow, "__file__") else workflow.__name__)
    pins = getattr(module, "PINS", None)
    if isinstance(pins, dict) and kind in pins:
        pin = dict(pins[kind])
        role = pin.pop("role")
        return {role: _normalize(pin)}
    if isinstance(pins, dict) and roles and set(roles) <= set(pins):
        trees = getattr(module, "SOURCE_TREES", {})
        return {role: _normalize({**pins[role], **({"source_tree": trees[role]} if role in trees else {})})
                for role in roles}
    pin = getattr(workflow, "pin", None)
    if isinstance(pin, dict) and len(roles) == 1:
        trees = getattr(module, "SOURCE_TREES", {})
        extra = {"source_tree": trees[pin["revision"]]} if pin.get("revision") in trees else {}
        return {roles[0]: _normalize({**pin, **extra})}
    if not roles:
        return {}
    raise ValueError(f"No normalized pin source for {kind}")


def pin_map(kind: str) -> dict:
    """Every pinned provider step of a pipeline, by role: the definition its module executes."""
    path = _descriptor_dir() / f"{kind}.json"
    value = validate(json.loads(path.read_text(encoding="utf-8")))
    return {step["role"]: deepcopy(step["pin"]) for step in value["steps"] if "pin" in step}


def provider_pin(kind: str, role: str | None = None) -> dict:
    """The exact pin a module executes, read from its descriptor (the definition)."""
    path = _descriptor_dir() / f"{kind}.json"
    value = validate(json.loads(path.read_text(encoding="utf-8")))
    steps = [step for step in value["steps"] if "pin" in step and (role is None or step["role"] == role)]
    if len(steps) != 1:
        raise ValueError(f"{kind} does not declare exactly one pinned provider step" + (f" for {role}" if role else ""))
    return {"role": steps[0]["role"], **deepcopy(steps[0]["pin"])}


def _descriptor_dir() -> Path:
    return Path(resources.files("ciw.pipelines") / "descriptors")


def load() -> dict:
    descriptors = {}
    for path in sorted(_descriptor_dir().glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        validate(value)
        if value["source_kind"] in descriptors or path.stem != value["source_kind"]:
            raise ValueError(f"Descriptor file {path.name} must be named after its unique source kind")
        descriptors[value["source_kind"]] = value
    return descriptors


def _keys(value, required, optional=(), name="object"):
    if not isinstance(value, dict) or not set(required) <= set(value) <= set(required) | set(optional):
        raise ValueError(f"{name} fields do not match the descriptor contract")


def _text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be bounded nonempty text")


def validate(value) -> dict:
    """Structural validation; ``check`` additionally binds a descriptor to code."""
    _keys(value, FIELDS, name="pipeline descriptor")
    if value["schema"] != DESCRIPTOR_SCHEMA:
        raise ValueError("Unsupported pipeline descriptor schema")
    for key in ("pipeline_id", "source_kind", "session_schema", "summary", "guide"):
        _text(value[key], key)
    inputs = value["inputs"]
    _keys(inputs, {"source_schema", "upstream_kinds", "upstream_cardinality", "configuration"}, name="inputs")
    if inputs["upstream_cardinality"] not in UPSTREAM_CARDINALITIES or not isinstance(inputs["upstream_kinds"], list):
        raise ValueError("inputs.upstream must declare kinds and a cardinality")
    if (inputs["upstream_cardinality"] == "none") != (not inputs["upstream_kinds"]):
        raise ValueError("An upstream cardinality needs upstream kinds and vice versa")
    if inputs["configuration"] not in {"in_source", "separate_operator_configuration", "none"}:
        raise ValueError("inputs.configuration is in_source, separate_operator_configuration or none")
    if not isinstance(value["steps"], list):
        raise ValueError("steps must be a list")
    roles = []
    for step in value["steps"]:
        _keys(step, {"role", "invocation", "purpose"}, {"pin"}, name="step")
        if not isinstance(step["role"], str) or not _ROLE.fullmatch(step["role"]) or step["invocation"] not in INVOCATIONS:
            raise ValueError("Each step needs a provider role and a declared invocation")
        _text(step["purpose"], "step purpose")
        if step["invocation"] in {"pinned_subprocess", "verified_contract_file"}:
            pin = step.get("pin")
            if not isinstance(pin, dict) or not _HEX40.fullmatch(str(pin.get("revision"))) or not set(pin) <= set(_PIN_FIELDS):
                raise ValueError(f"Step {step['role']} needs an exact 40-hex pin revision")
        elif "pin" in step:
            raise ValueError(f"A {step['invocation']} step carries no repository pin")
        roles.append(step["role"])
    verification = value["verification"]
    _keys(verification, {"method", "role"}, name="verification")
    _text(verification["method"], "verification.method", 128)
    if verification["role"] is not None and verification["role"] not in roles:
        raise ValueError("A verification role must be one of the pipeline's steps")
    for code in value["refusals"]:
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise ValueError("Refusal codes are bounded identifiers")
    if len(set(value["refusals"])) != len(value["refusals"]):
        raise ValueError("Refusal codes must be distinct")
    for rule in value["domain_rules"]:
        _keys(rule, {"rule", "evidence", "code"}, name="domain rule")
        _text(rule["rule"], "domain rule", 1024)
        _text(rule["evidence"], "domain rule evidence", 512)
        if (not isinstance(rule["code"], list) or not 1 <= len(rule["code"]) <= 8 or
                any(not isinstance(ref, str) or not _SYMBOL.fullmatch(ref) for ref in rule["code"])):
            raise ValueError("A domain rule cites 1..8 code symbols as module:qualname")
        if _LINE_CITATION.search(rule["rule"] + " " + rule["evidence"]):
            raise ValueError("Domain rules cite code symbols, not line numbers")
    if (not isinstance(value["investigations"], list) or len(set(value["investigations"])) != len(value["investigations"])):
        raise ValueError("investigations must be a list of distinct identifiers")
    if value["surface"] not in SURFACES:
        raise ValueError("surface must be entry, inner, bench, certification or source_only")
    if value["authority"] != "read_only":
        raise ValueError("Every current pipeline is read-only with respect to equipment")
    implementation = value["implementation"]
    _keys(implementation, {"module", "runner", "delegates"}, name="implementation")
    if implementation["runner"] not in IMPLEMENTATIONS or not implementation["module"].startswith("ciw."):
        raise ValueError("implementation names a CIW module and its runner class")
    if (not isinstance(implementation["delegates"], list)
            or any(not isinstance(item, str) or not item.startswith("ciw.") for item in implementation["delegates"])):
        raise ValueError("implementation.delegates lists CIW modules whose pipelines this one executes")
    return value


PROVIDER_SCHEMA = "ciw.provider-descriptor.v1"
PROVIDER_FIELDS = frozenset({"schema", "provider_id", "role", "summary", "invocation", "pin", "boundary", "operations",
                             "surface", "authority", "implementation", "guide"})
PROVIDER_INVOCATIONS = frozenset({"persistent_worker", "pinned_subprocess", "installed_package", "pinned_artifact"})
PROVIDER_SURFACES = frozenset({"terminal", "workbench"})


def _provider_dir() -> Path:
    return Path(resources.files("ciw.pipelines") / "providers")


def load_providers() -> dict:
    """Provider descriptors for providers that no frozen pipeline step names (terminal surfaces)."""
    providers = {}
    for path in sorted(_provider_dir().glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        _keys(value, PROVIDER_FIELDS, name="provider descriptor")
        if value["schema"] != PROVIDER_SCHEMA or value["invocation"] not in PROVIDER_INVOCATIONS:
            raise ValueError(f"{path.name}: unsupported provider descriptor schema or invocation")
        for key in ("provider_id", "summary", "guide"):
            _text(value[key], key)
        if (not _ROLE.fullmatch(value["role"]) or value["surface"] not in PROVIDER_SURFACES or
                value["authority"] != "read_only"):
            raise ValueError(f"{path.name}: a provider descriptor declares a role, a provider surface and read-only authority")
        if not isinstance(value["pin"], dict) or not value["pin"]:
            raise ValueError(f"{path.name}: a provider descriptor declares its pin")
        for item in value["boundary"]:
            _keys(item, {"role", "purpose", "pin"}, name="provider boundary")
            if not re.fullmatch(r"[0-9a-f]{40}", item["pin"].get("revision", "")):
                raise ValueError(f"{path.name}: a boundary pin is an exact revision")
        if path.stem in providers:
            raise ValueError(f"Duplicate provider descriptor {path.stem}")
        providers[path.stem] = value
    return providers


def provider_descriptor(name: str) -> dict:
    """One provider descriptor; its ``pin`` is the definition the implementation executes."""
    providers = load_providers()
    if name not in providers:
        raise ValueError(f"No provider descriptor {name}")
    return deepcopy(providers[name])


def check_providers(providers: dict | None = None) -> dict:
    """Bind provider descriptors to what their implementation executes.

    The implementation module's ``provider_binding()`` maps each provider it
    runs to the live pin fields it would execute (for example digests of
    packaged files) and the operations it serves; both must equal the
    descriptor.
    """
    providers = load_providers() if providers is None else providers
    root = Path(__file__).resolve().parents[3]
    for name, value in providers.items():
        bindings = import_module(value["implementation"]["module"]).provider_binding()
        if name not in bindings:
            raise ValueError(f"{name}: {value['implementation']['module']} does not bind this provider")
        binding = bindings[name]
        for key, live in binding["pin"].items():
            if value["pin"].get(key) != live:
                raise ValueError(f"{name}: pinned {key} differs from what {value['implementation']['module']} executes")
        if binding["operations"] != value["operations"]:
            raise ValueError(f"{name}: declared operations differ from the operations its implementation serves")
        if (root / "docs").is_dir() and not (root / value["guide"]).is_file():
            raise ValueError(f"{name}: guide {value['guide']} does not exist")
    return providers


def load_investigations() -> dict:
    value = json.loads((Path(resources.files("ciw.pipelines")) / "investigations.json").read_text(encoding="utf-8"))
    if value.get("schema") != INVESTIGATION_SCHEMA or not isinstance(value.get("investigations"), list):
        raise ValueError("Unsupported investigation catalog")
    result = {}
    for item in value["investigations"]:
        _keys(item, {"investigation_id", "title", "question", "default_pipeline", "pipelines"}, name="investigation")
        _text(item["title"], "investigation title")
        _text(item["question"], "investigation question", 1024)
        if item["investigation_id"] in result or not set(item["default_pipeline"]) <= set(item["pipelines"]):
            raise ValueError("Investigation identities are unique and their default pipeline uses their pipelines")
        result[item["investigation_id"]] = item
    return result


def resolve_symbol(reference: str):
    """The object a ``module:qualname`` code reference names; ValueError if it does not exist."""
    module, _, qualname = reference.partition(":")
    try:
        target = import_module(module)
        for name in qualname.split("."):
            target = getattr(target, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Code reference {reference} does not resolve") from exc
    return target


def _check_runner(kind: str, runner: str, workflow) -> None:
    """A generic runner pipeline supplies domain hooks only; the runner owns its records."""
    from .runner import HOOKS, PipelineRunner
    is_runner = isinstance(workflow, PipelineRunner)
    if runner == "generic_runner":
        if not is_runner:
            raise ValueError(f"{kind}: a generic_runner pipeline must execute through the shared runner")
        for cls in type(workflow).__mro__:
            if cls is PipelineRunner:
                break
            extra = {name for name, value in vars(cls).items() if callable(value) or isinstance(value, (staticmethod, classmethod, property))} - HOOKS
            if extra:
                raise ValueError(f"{kind}: generic_runner {cls.__name__} overrides runner methods {sorted(extra)}")
    elif runner == "declared_workflow" and not is_runner:
        raise ValueError(f"{kind}: a declared_workflow pipeline extends the shared runner")
    elif runner == "hand_written" and is_runner:
        raise ValueError(f"{kind}: runs through the shared runner; declare it generic_runner or declared_workflow")


def check(descriptors: dict | None = None) -> dict:
    """Bind descriptors to code: kinds, operation ids, roles, pins, upstreams, guides, investigations."""
    from .. import kernel
    from ..workbench import OPERATIONS, UPSTREAM_KINDS, _workflow
    descriptors = load() if descriptors is None else descriptors
    investigations = load_investigations()
    expected = kernel.FROZEN_KINDS
    if set(descriptors) != expected:
        raise ValueError(f"Descriptors must cover exactly the frozen kinds; missing {sorted(expected - set(descriptors))}, "
                         f"unexpected {sorted(set(descriptors) - expected)}")
    root = Path(__file__).resolve().parents[3]
    for kind, value in descriptors.items():
        if kind in OPERATIONS and value["pipeline_id"] != OPERATIONS[kind]:
            raise ValueError(f"{kind}: pipeline_id differs from the registered operation")
        roles = {step["role"] for step in value["steps"]}
        workflow_roles = set(getattr(_workflow(kind), "ROLES", ()))
        if roles != workflow_roles:
            raise ValueError(f"{kind}: descriptor roles {sorted(roles)} differ from bound roles {sorted(workflow_roles)}")
        declared = {step["role"]: step["pin"] for step in value["steps"] if "pin" in step}
        if declared != live_pins(kind):
            raise ValueError(f"{kind}: descriptor pins differ from the pins its module executes")
        upstream = [UPSTREAM_KINDS[kind]] if kind in UPSTREAM_KINDS else []
        if kind != "residual-monitor" and value["inputs"]["upstream_kinds"] != upstream:
            raise ValueError(f"{kind}: descriptor upstream differs from the registered upstream")
        import_module(value["implementation"]["module"])
        _check_runner(kind, value["implementation"]["runner"], _workflow(kind))
        for rule in value["domain_rules"]:
            for reference in rule["code"]:
                resolve_symbol(reference)
        for module in value["implementation"]["delegates"]:
            import_module(module)
        if value["refusals"] != code_refusals(value, descriptors):
            raise ValueError(f"{kind}: descriptor refusals differ from the codes its implementation raises")
        if (root / "docs").is_dir() and not (root / value["guide"]).is_file():
            raise ValueError(f"{kind}: guide {value['guide']} does not exist")
        for item in value["investigations"]:
            if item not in investigations or value["pipeline_id"] not in investigations[item]["pipelines"]:
                raise ValueError(f"{kind}: investigation {item} does not list this pipeline")
    by_pipeline = {value["pipeline_id"]: value for value in descriptors.values()}
    for item in investigations.values():
        if not set(item["pipelines"]) <= set(by_pipeline):
            raise ValueError(f"Investigation {item['investigation_id']} names an undeclared pipeline")
        earlier = set()
        for pipeline in item["default_pipeline"]:
            upstream = set(by_pipeline[pipeline]["inputs"]["upstream_kinds"])
            if upstream and not upstream & earlier:
                raise ValueError(f"Investigation {item['investigation_id']}: {pipeline} needs one of "
                                 f"{sorted(upstream)} earlier in its default pipeline")
            earlier.add(by_pipeline[pipeline]["source_kind"])
        for kind, value in descriptors.items():
            member = value["pipeline_id"] in item["pipelines"]
            if member != (item["investigation_id"] in value["investigations"]):
                raise ValueError(f"{kind}: investigation membership differs between descriptor and catalog")
            if value["pipeline_id"] in item["default_pipeline"] and value["surface"] != "entry":
                raise ValueError(f"{kind}: a default-pipeline stage is an entry surface")
    defaults = {pipeline for item in investigations.values() for pipeline in item["default_pipeline"]}
    for kind, value in descriptors.items():
        if (value["surface"] == "entry") != (value["pipeline_id"] in defaults):
            raise ValueError(f"{kind}: entry surfaces are exactly the default-pipeline stages")
        if value["surface"] == "inner" and not value["investigations"]:
            raise ValueError(f"{kind}: an inner pipeline belongs to an investigation")
    check_providers()
    return descriptors


def render_catalog(descriptors: dict | None = None) -> str:
    """docs/PIPELINES.md, generated from descriptors and the investigation catalog."""
    descriptors = load() if descriptors is None else descriptors
    investigations = load_investigations()
    by_pipeline = {value["pipeline_id"]: value for value in descriptors.values()}
    lines = ["# Declared pipelines", "",
             "<!-- Generated by `python scripts/generate_pipeline_catalog.py` from",
             "src/ciw/pipelines/descriptors/*.json and investigations.json. Do not edit by hand. -->", "",
             "Investigations are the operator surface (`operation.list` with `{\"view\": \"investigations\"}`).",
             "Every pipeline stays executable and replayable through the [kernel](KERNEL.md) verbs.", ""]
    for item in investigations.values():
        lines += [f"## {item['title']}", "", item["question"], "",
                  "| Stage | Pipeline | Upstream | Providers | Guide |", "| --- | --- | --- | --- | --- |"]
        for pipeline in item["pipelines"]:
            value = by_pipeline[pipeline]
            stage = (f"default {item['default_pipeline'].index(pipeline) + 1}" if pipeline in item["default_pipeline"]
                     else "inner")
            providers = ", ".join(step["role"] for step in value["steps"]) or "CIW reference"
            upstream = ", ".join(value["inputs"]["upstream_kinds"]) or "none"
            guide = value["guide"].removeprefix("docs/")
            lines.append(f"| {stage} | `{pipeline}` — {value['summary']} | {upstream} | {providers} | [{guide}]({guide}) |")
        lines.append("")
    others = sorted((value for value in descriptors.values() if not value["investigations"]), key=lambda v: v["pipeline_id"])
    lines += ["## Benches and certification", "", "| Surface | Pipeline | Providers | Verification | Guide |",
              "| --- | --- | --- | --- | --- |"]
    for value in others:
        providers = ", ".join(step["role"] for step in value["steps"]) or "CIW reference"
        guide = value["guide"].removeprefix("docs/")
        lines.append(f"| {value['surface']} | `{value['pipeline_id']}` — {value['summary']} | {providers} | "
                     f"{value['verification']['method']} | [{guide}]({guide}) |")
    matrix = provider_matrix(descriptors)
    lines += ["", "## Provider pins", "",
              f"{len(matrix)} distinct provider pins back {len(descriptors)} pipelines. A provider gate is keyed by one "
              "row: the same pin serves every pipeline listed.", "",
              "| Role | Revision | Pipelines |", "| --- | --- | --- |"]
    for entry in matrix:
        lines.append(f"| {entry['role']} | `{entry['pin']['revision'][:12]}` | "
                     + ", ".join(f"`{p}`" for p in entry["pipelines"]) + " |")
    providers = load_providers()
    if providers:
        lines += ["", "## Providers outside pipelines", "",
                  "Providers that no frozen pipeline step names are declared by a `ciw.provider-descriptor.v1`; its "
                  "`pin` is the definition the implementation executes and `pipelines.check()` binds it to "
                  "`provider_binding()` of the implementation.", "",
                  "| Provider | Role | Invocation | Operations | Boundary | Guide |", "| --- | --- | --- | --- | --- | --- |"]
        for value in providers.values():
            operations = ", ".join(f"`{op}`" for ops in value["operations"].values() for op in ops)
            boundary = ", ".join(f"{item['role']} `{item['pin']['revision'][:12]}`" for item in value["boundary"]) or "none"
            guide = value["guide"].removeprefix("docs/")
            lines.append(f"| `{value['provider_id']}` | {value['role']} | {value['invocation'].replace('_', ' ')} | "
                         f"{operations} | {boundary} | [{guide}]({guide}) |")
    common = set(WORKBENCH_REFUSALS) | {"INPUT_LIMIT", "INVALID_INPUT", "MALFORMED_RESPONSE", "OUTPUT_LIMIT",
                                         "RUNTIME_FAILED", "RUNTIME_IO", "RUNTIME_PIN_MISMATCH", "RUNTIME_UNAVAILABLE",
                                         "SOURCE_PIN_MISMATCH", "TIMEOUT"}
    lines += ["", "## Pipeline details", "",
              "Refusal codes are derived from the code on each pipeline's execution path and checked by "
              "`pipelines.check()`; every pinned-subprocess pipeline can also raise the adapter codes "
              "(`" + "`, `".join(sorted(common - set(WORKBENCH_REFUSALS))) + "`) and every pipeline the workbench "
              "admission codes (`operation_unavailable`, `workbench_capacity`). Domain rules are the checks the "
              "implementation keeps beyond the shared runner shape, with code evidence, from the verified inventory."]
    for value in sorted(descriptors.values(), key=lambda v: v["pipeline_id"]):
        specific = [code for code in value["refusals"] if code not in common]
        lines += ["", f"### `{value['pipeline_id']}`", "",
                  f"Implementation `{value['implementation']['module']}` ({value['implementation']['runner'].replace('_', ' ')}"
                  + (", delegates to " + ", ".join(f"`{d}`" for d in value["implementation"]["delegates"])
                     if value["implementation"]["delegates"] else "") + f"); verification: {value['verification']['method']}.",
                  "", "Specific refusals: " + (", ".join(f"`{code}`" for code in specific) if specific else "none") + "."]
        if value["domain_rules"]:
            lines.append("")
            lines += [(f"- {rule['rule']} ({rule['evidence']}; code: "
                       + ", ".join(f"`{ref}`" for ref in rule["code"]) + ")").replace("|", "\\|")
                      for rule in value["domain_rules"]]
    multiple = {}
    for entry in matrix:
        multiple.setdefault(entry["role"], []).append(entry["pin"]["revision"][:12])
    split = {role: revisions for role, revisions in multiple.items() if len(revisions) > 1}
    if split:
        lines += ["", "Providers pinned at more than one revision in one session: "
                  + "; ".join(f"{role} ({', '.join(revisions)})" for role, revisions in sorted(split.items())) + "."]
    return "\n".join(lines) + "\n"


def provider_matrix(descriptors: dict | None = None) -> list[dict]:
    """Distinct (provider role, revision) pins and the pipelines that use each."""
    descriptors = load() if descriptors is None else descriptors
    matrix = {}
    for kind, value in sorted(descriptors.items()):
        for step in value["steps"]:
            if "pin" not in step:
                continue
            key = (step["role"], step["pin"]["revision"])
            entry = matrix.setdefault(key, {"role": step["role"], "pin": deepcopy(step["pin"]), "pipelines": []})
            entry["pipelines"].append(value["pipeline_id"])
    return [matrix[key] for key in sorted(matrix)]


def operator_catalog(available: dict) -> list[dict]:
    """Investigations as the operator's default surface; ``available`` maps pipeline id to availability."""
    descriptors = load()
    by_pipeline = {value["pipeline_id"]: value for value in descriptors.values()}
    catalog = []
    for item in load_investigations().values():
        stages = [{"pipeline_id": pipeline, "source_kind": by_pipeline[pipeline]["source_kind"],
                   "summary": by_pipeline[pipeline]["summary"], "available": bool(available.get(pipeline))}
                  for pipeline in item["default_pipeline"]]
        catalog.append({"investigation_id": item["investigation_id"], "title": item["title"],
                        "question": item["question"], "default_pipeline": stages,
                        "inner_pipelines": sorted(set(item["pipelines"]) - set(item["default_pipeline"])),
                        "available": all(stage["available"] for stage in stages)})
    return catalog

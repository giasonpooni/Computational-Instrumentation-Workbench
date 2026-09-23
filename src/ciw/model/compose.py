"""Typed-port composition of model specifications.

A connection joins an out-port to an in-port only when their role
(commanded, measured or physical), quantity kind, dimension and frame agree,
the components share the independent-variable kind and origin, and a measured
connection preserves the exact calibration identity. Units may differ: the
connected input becomes a derived quantity whose unit-aware expression is the
source expression, so conversion is part of the model rather than an ad hoc
rescale. A measured input fed by an observation receives the noise-free
model-predicted observation; the substitution is recorded, never implied.

Cyclic dependencies through connected ports are algebraic loops. They are
refused: an algebraic constraint needs explicit solver semantics beyond
dataflow ordering.
"""
from __future__ import annotations

from copy import deepcopy

from . import expression
from .spec import Model, SpecificationError, validate_spec


def _qualify(model: Model, name: str | None) -> tuple[dict, dict]:
    """Return a qualified copy of the spec and the symbol map used."""
    spec = deepcopy(model.spec)
    independent = model.independent
    if name is None:
        if spec["provenance"]["kind"] != "composition":
            raise SpecificationError("Only a composite may enter a composition without a component name")
        return spec, {}
    if "." in name:
        raise SpecificationError("Component names must not contain '.'")
    mapping = {symbol: f"{name}.{symbol}" for symbol in model.entries if symbol != independent}
    for key in ("states", "inputs", "parameters", "derived", "observations"):
        for entry in spec[key]:
            entry["symbol"] = mapping[entry["symbol"]]
            if "latex" in entry:
                entry["latex"] = f"{{{entry['latex']}}}^{{\\mathrm{{{name.replace('_', ' ')}}}}}"
            for field in ("expression",):
                if field in entry:
                    entry[field] = expression.rename(entry[field], mapping)
    for entry in spec["dynamics"]:
        entry["state"], entry["rhs"] = mapping[entry["state"]], expression.rename(entry["rhs"], mapping)
    for port in spec["ports"]:
        port["name"], port["symbol"] = f"{name}.{port['name']}", mapping[port["symbol"]]
    for block in spec["uncertainty"].values():
        if isinstance(block, dict):
            block["order"] = [mapping[symbol] for symbol in block["order"]]
    domain = spec["validity_domain"]
    domain["bounds"] = {mapping[symbol]: bounds for symbol, bounds in domain["bounds"].items()}
    domain["notes"] = [f"{name}: {note}" for note in domain["notes"]]
    return spec, mapping


def _block_diagonal(blocks):
    present = [block for block in blocks if block is not None]
    if not present:
        return None
    order = [symbol for block in present for symbol in block["order"]]
    size, matrix, offset = len(order), [], 0
    matrix = [[0.0] * size for _ in range(size)]
    for block in present:
        for i, row in enumerate(block["matrix"]):
            for j, value in enumerate(row):
                matrix[offset + i][offset + j] = value
        offset += len(block["order"])
    return {"order": order, "matrix": matrix}


def compose(model_id: str, title: str, components: list[dict], connections: list[dict]) -> Model:
    """Compose ``components`` ([{"name": str|None, "model": Model}]) through ``connections``.

    Each connection is ``{"from": "<component>.<out-port>", "to": "<component>.<in-port>"}``
    using qualified port names. Composite state order is component order, then
    each component's declared state order.
    """
    if not isinstance(components, list) or len(components) < 1:
        raise SpecificationError("A composition needs at least one component")
    specs, models = [], []
    for item in components:
        model = item["model"]
        if not isinstance(model, Model):
            raise SpecificationError("Compose validated models only")
        spec, _ = _qualify(model, item.get("name"))
        specs.append(spec)
        models.append(model)
    first = specs[0]["independent_variable"]
    for spec in specs[1:]:
        other = spec["independent_variable"]
        if other["kind"] != first["kind"]:
            raise SpecificationError(f"Cannot compose a {first['kind']} model with a {other['kind']} model")
        if other["origin"] != first["origin"]:
            raise SpecificationError("Components declare different independent-variable origins")
    independent = deepcopy(first)
    t = independent["symbol"]
    for spec, model in zip(specs, models):
        own = spec["independent_variable"]["symbol"]
        if own != t:
            # Unit-aware expressions make the renamed variable's unit irrelevant.
            mapping = {own: t}
            for key in ("derived", "observations"):
                for entry in spec[key]:
                    entry["expression"] = expression.rename(entry["expression"], mapping)
            for entry in spec["dynamics"]:
                entry["rhs"] = expression.rename(entry["rhs"], mapping)

    merged = {key: [entry for spec in specs for entry in spec[key]]
              for key in ("states", "inputs", "parameters", "derived", "observations", "dynamics", "ports")}
    symbols = {}
    for key in ("states", "inputs", "parameters", "derived", "observations"):
        for entry in merged[key]:
            if entry["symbol"] in symbols or entry["symbol"] == t:
                raise SpecificationError(f"Composite symbol collision: {entry['symbol']}")
            symbols[entry["symbol"]] = (key, entry)
    ports = {}
    for port in merged["ports"]:
        if port["name"] in ports:
            raise SpecificationError(f"Composite port collision: {port['name']}")
        ports[port["name"]] = port

    def describe(port_name):
        port = ports.get(port_name)
        if port is None:
            raise SpecificationError(f"Unknown port: {port_name}")
        key, entry = symbols[port["symbol"]]
        role = {"inputs": entry.get("role"), "observations": "measured", "states": "physical",
                "derived": entry.get("role")}[key]
        return port, key, entry, role

    if not isinstance(connections, list):
        raise SpecificationError("connections must be an array")
    replacements, connected_inputs, lineage = {}, set(), []
    for connection in connections:
        if not isinstance(connection, dict) or connection.keys() != {"from", "to"}:
            raise SpecificationError("A connection is {from, to}")
        source, source_kind, source_entry, source_role = describe(connection["from"])
        target, _, target_entry, target_role = describe(connection["to"])
        if source["direction"] != "out" or target["direction"] != "in":
            raise SpecificationError("A connection runs from an out-port to an in-port")
        if target["symbol"] in connected_inputs:
            raise SpecificationError(f"In-port {target['name']} is already connected")
        if source_role != target_role:
            raise SpecificationError(f"Role mismatch: {source_role} {source['name']} cannot feed "
                                     f"{target_role} {target['name']}")
        if source_entry["quantity"] != target_entry["quantity"]:
            raise SpecificationError(f"Quantity mismatch: {source_entry['quantity']} -> {target_entry['quantity']}")
        source_unit = _unit(source_entry["unit"])
        if not source_unit.same_dimension(_unit(target_entry["unit"])):
            raise SpecificationError(f"Dimension mismatch: {source_entry['unit']} -> {target_entry['unit']}")
        if source_entry.get("frame") != target_entry.get("frame"):
            raise SpecificationError(f"Frame mismatch: {source_entry.get('frame')} -> {target_entry.get('frame')}")
        if target_role == "measured" and source_entry["calibration"] != target_entry["calibration"]:
            raise SpecificationError("Calibration identity mismatch: a measured input must receive the "
                                     "observation calibration it declares")
        # Observations are not in expression scope, so their expression is
        # inlined; states and derived quantities remain named references.
        replacements[target["symbol"]] = (deepcopy(source_entry["expression"]) if source_kind == "observations"
                                          else {"sym": source["symbol"]})
        connected_inputs.add(target["symbol"])
        lineage.append({"from": source["name"], "to": target["name"], "role": source_role,
                        "source_symbol": source["symbol"], "target_symbol": target["symbol"],
                        "source_unit": source_entry["unit"], "target_unit": target_entry["unit"],
                        "frame": source_entry.get("frame"),
                        "substitution": ("noise_free_predicted_observation" if source_kind == "observations"
                                         else "direct_quantity"),
                        **({"sensor": source_entry["sensor"], "calibration": deepcopy(source_entry["calibration"])}
                           if source_kind == "observations" else {})})

    # Connected inputs become derived quantities in the composite.
    derived = [deepcopy(entry) for entry in merged["derived"]]
    for entry in merged["inputs"]:
        if entry["symbol"] in connected_inputs:
            role = {"measured": "derived", "commanded": "commanded", "physical": "physical"}[entry["role"]]
            item = {"symbol": entry["symbol"], "quantity": entry["quantity"], "unit": entry["unit"],
                    "frame": entry["frame"], "role": role, "expression": replacements[entry["symbol"]],
                    "description": f"connected input (was {entry['role']})"}
            if "latex" in entry:
                item["latex"] = entry["latex"]
            derived.append(item)
    derived = _order(derived)
    inputs = [deepcopy(entry) for entry in merged["inputs"] if entry["symbol"] not in connected_inputs]
    port_list = [deepcopy(port) for port in merged["ports"] if port["symbol"] not in connected_inputs]

    blocks = {key: _block_diagonal([spec["uncertainty"][key] for spec in specs])
              for key in ("parameters", "initial_state", "measurement_noise", "process_noise")}
    assumptions = []
    for spec in specs:
        for item in spec["uncertainty"]["assumptions"]:
            if item not in assumptions:
                assumptions.append(item)
    if blocks["process_noise"] is not None and "no_process_noise" in assumptions:
        assumptions.remove("no_process_noise")
    if blocks["measurement_noise"] is not None and "no_measurement_noise_declared" in assumptions:
        assumptions.remove("no_measurement_noise_declared")
    if blocks["process_noise"] is None and "no_process_noise" not in assumptions:
        assumptions.append("no_process_noise")
    if len(specs) > 1 and "independent_component_uncertainty" not in assumptions:
        assumptions.append("independent_component_uncertainty")

    intervals = [spec["validity_domain"]["independent_variable"] for spec in specs]
    interval = None
    for item in intervals:
        if item is None:
            continue
        if interval is None:
            interval = list(item)
        else:
            low = [value for value in (interval[0], item[0]) if value is not None]
            high = [value for value in (interval[1], item[1]) if value is not None]
            interval = [max(low) if low else None, min(high) if high else None]
    bounds = {}
    for spec in specs:
        for symbol, value in spec["validity_domain"]["bounds"].items():
            if symbol not in connected_inputs:
                bounds[symbol] = value

    composite = {
        "schema": "ciw.model-spec.v1", "model_id": model_id, "title": title,
        "independent_variable": independent,
        "states": [deepcopy(entry) for entry in merged["states"]], "inputs": inputs,
        "parameters": [deepcopy(entry) for entry in merged["parameters"]],
        "dynamics": [deepcopy(entry) for entry in merged["dynamics"]],
        "observations": [deepcopy(entry) for entry in merged["observations"]],
        "derived": derived, "ports": port_list,
        "uncertainty": {**blocks, "assumptions": assumptions},
        "validity_domain": {"independent_variable": interval, "bounds": bounds,
                            "notes": [note for spec in specs for note in spec["validity_domain"]["notes"]]},
        "provenance": {"kind": "composition", "derived_from": {
            "components": [{"name": item.get("name"), "spec_digest": model.digest,
                            "assumptions": list(model.spec["uncertainty"]["assumptions"])}
                           for item, model in zip(components, models)],
            "connections": lineage}, "notes": []},
    }
    return validate_spec(composite)


def _unit(text):
    from .units import parse_unit
    return parse_unit(text)


def _order(derived: list[dict]) -> list[dict]:
    """Stable topological order; a cycle is an algebraic loop and is refused."""
    names = {entry["symbol"] for entry in derived}
    pending = list(derived)
    ordered, done = [], set()
    while pending:
        for index, entry in enumerate(pending):
            if (expression.symbols_in(entry["expression"]) & names) <= done:
                ordered.append(pending.pop(index))
                done.add(entry["symbol"])
                break
        else:
            cycle = ", ".join(entry["symbol"] for entry in pending)
            raise SpecificationError(f"Algebraic loop through connected ports: {cycle}")
    return ordered


def semantic_view(model: Model) -> dict:
    """An order-insensitive view of derived quantities; state order stays meaningful."""
    spec = deepcopy(model.spec)
    spec.pop("provenance")
    spec["derived"] = {entry["symbol"]: entry for entry in spec["derived"]}
    spec["ports"] = {port["name"]: port for port in spec["ports"]}
    spec["uncertainty"]["assumptions"] = sorted(spec["uncertainty"]["assumptions"])
    spec["model_id"] = spec["title"] = None
    return spec

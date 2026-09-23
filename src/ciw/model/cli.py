"""Terminal commands for model specifications, LaTeX views and Julia operations.

Every command that executes Julia takes an explicit executable binding.
Generated files are written to new paths only; nothing is overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

OPERATIONS = {"simulate": "ciw.model.simulate.v1", "linearize": "ciw.model.linearize.v1",
              "select": "ciw.model.measurement-selection.v1", "symbolic": "ciw.model.symbolic.v1"}


def add_parser(commands) -> None:
    model = commands.add_parser("model", help="Language-neutral models, LaTeX views and pinned Julia operations")
    actions = model.add_subparsers(dest="model_command", required=True)
    validate = actions.add_parser("validate", help="Validate a ciw.model-spec.v1 and print its symbol table")
    validate.add_argument("spec", type=Path)
    latex = actions.add_parser("latex", help="Generate the LaTeX view of a specification or a retained run")
    latex.add_argument("spec", type=Path)
    latex.add_argument("--run", type=Path, help="Bind symbols to a completed ciw.model-run.v1")
    latex.add_argument("--exploratory", type=Path, action="append", default=[],
                       help="Include a ciw.exploratory-derivation.v1 as escaped, unbound text")
    latex.add_argument("--output", type=Path, required=True, help="New .tex path")
    latex.add_argument("--view-output", type=Path, help="Also write the ciw.model-latex-view.v1 record")
    rescale = actions.add_parser("rescale", help="Consistently rescale declared units, covariances and bounds")
    rescale.add_argument("spec", type=Path)
    rescale.add_argument("--unit", action="append", required=True, metavar="SYMBOL=UNIT")
    rescale.add_argument("--output", type=Path, required=True)
    rescale.add_argument("--request", type=Path, help="Also transform a simulation request")
    rescale.add_argument("--request-output", type=Path)
    compose = actions.add_parser("compose", help="Compose specifications through typed ports")
    compose.add_argument("--component", action="append", required=True, metavar="NAME=SPEC")
    compose.add_argument("--connect", action="append", nargs=2, default=[], metavar=("OUT_PORT", "IN_PORT"))
    compose.add_argument("--model-id", required=True)
    compose.add_argument("--title", required=True)
    compose.add_argument("--output", type=Path, required=True)
    exploratory = actions.add_parser("exploratory", help="Retain handwritten LaTeX as a non-executable derivation")
    exploratory.add_argument("--title", required=True)
    exploratory.add_argument("--latex-file", type=Path, required=True)
    exploratory.add_argument("--relates-to", type=Path, help="Specification the derivation discusses")
    exploratory.add_argument("--author")
    exploratory.add_argument("--output", type=Path, required=True)
    run = actions.add_parser("run", help="Execute an operation on the pinned Julia worker and retain the run")
    run.add_argument("operation", choices=sorted(OPERATIONS))
    run.add_argument("spec", type=Path)
    run.add_argument("request", type=Path)
    run.add_argument("--julia", type=Path, required=True, help="Julia 1.10.12 executable (explicit host binding)")
    run.add_argument("--julia-depot", type=Path, help="JULIA_DEPOT_PATH holding the instantiated environment")
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--output-dir", type=Path, default=Path("results/model-runs"))
    inspect = actions.add_parser("inspect", help="Inspect a retained run without Julia")
    inspect.add_argument("path", type=Path)
    replay = actions.add_parser("replay", help="Re-execute a retained run as a fresh occurrence")
    replay.add_argument("path", type=Path)
    replay.add_argument("--julia", type=Path, required=True)
    replay.add_argument("--julia-depot", type=Path)
    replay.add_argument("--output-dir", type=Path, default=Path("results/model-runs"))


def _new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _new_json(path: Path, value) -> None:
    _new_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args, print_json) -> int:
    from ..session import read_json
    from .spec import summary, validate_spec
    command = args.model_command
    if command == "validate":
        print_json(summary(validate_spec(read_json(args.spec))))
    elif command == "latex":
        from .latex import latex_view, validate_exploratory
        from .workflow import binding_for_run, read_run
        model = validate_spec(read_json(args.spec))
        binding = None
        if args.run is not None:
            bundle = read_run(args.run)
            if bundle.get("spec_digest") != model.digest:
                raise ValueError("The run executed a different specification")
            binding = binding_for_run(bundle)
        notes = [validate_exploratory(read_json(path)) for path in args.exploratory]
        view = latex_view(model, binding=binding, exploratory=notes)
        _new_text(args.output, view["latex"])
        if args.view_output:
            _new_json(args.view_output, view)
        print_json({"latex_file": str(args.output), "status": view["status"], "spec_digest": view["spec_digest"],
                    "latex_sha256": view["latex_sha256"]})
    elif command == "rescale":
        from .transform import rescale, rescale_request
        model = validate_spec(read_json(args.spec))
        units = {}
        for item in args.unit:
            symbol, separator, unit = item.partition("=")
            if not separator or not symbol or not unit or symbol in units:
                raise ValueError("--unit takes distinct SYMBOL=UNIT entries")
            units[symbol] = unit
        rescaled = rescale(model, units)
        _new_json(args.output, rescaled.spec)
        result = {"spec_file": str(args.output), "spec_digest": rescaled.digest,
                  "derived_from": rescaled.spec["provenance"]["derived_from"]}
        if args.request is not None:
            if args.request_output is None:
                raise ValueError("--request requires --request-output")
            request = rescale_request(model, rescaled, read_json(args.request))
            _new_json(args.request_output, request)
            result["request_file"] = str(args.request_output)
        print_json(result)
    elif command == "compose":
        from .compose import compose
        components = []
        for item in args.component:
            name, separator, path = item.partition("=")
            if not separator:
                raise ValueError("--component takes NAME=SPEC")
            components.append({"name": name, "model": validate_spec(read_json(Path(path)))})
        composite = compose(args.model_id, args.title, components,
                            [{"from": source, "to": target} for source, target in args.connect])
        _new_json(args.output, composite.spec)
        print_json({"spec_file": str(args.output), "spec_digest": composite.digest,
                    "state_order": composite.state_order,
                    "connections": composite.spec["provenance"]["derived_from"]["connections"]})
    elif command == "exploratory":
        from .latex import exploratory_derivation
        relates = validate_spec(read_json(args.relates_to)).digest if args.relates_to else None
        record = exploratory_derivation(args.title, args.latex_file.read_text(encoding="utf-8"),
                                        relates_to=relates, author=args.author)
        _new_json(args.output, record)
        print_json({"derivation_file": str(args.output), "derivation_id": record["derivation_id"],
                    "executable": False})
    elif command in {"run", "replay"}:
        from .worker import OPERATION_PROFILE, JuliaBinding, JuliaWorker
        from .workflow import execute_run, inspect_run, read_run, replay_run, save_run
        if command == "run":
            operation = OPERATIONS[args.operation]
            spec, request = read_json(args.spec), read_json(args.request)
        else:
            bundle = read_run(args.path)
            operation = bundle["operation"]
        binding = JuliaBinding(julia=args.julia, profile=OPERATION_PROFILE[operation], depot=args.julia_depot)
        with JuliaWorker(binding) as worker:
            result = (execute_run(operation, spec, request, worker, timeout=args.timeout) if command == "run"
                      else replay_run(bundle, worker))
        path = save_run(result, args.output_dir)
        summary_view = inspect_run(result)
        print_json({"run_file": str(path), **summary_view, "diagnostics": result["diagnostics"]})
        if result["execution"]["status"] != "completed":
            return 3
        if result["diagnostics"] is not None and not result["diagnostics"].get("passed", False):
            return 4
    elif command == "inspect":
        from .workflow import inspect_run, read_run
        print_json(inspect_run(read_run(args.path)))
    return 0

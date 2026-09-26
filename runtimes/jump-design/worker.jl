# One request per process. Only this fixed scalar quadratic program is supported.
# Standard output is reserved for the data-only TOML response.
using TOML

const INPUT_SCHEMA = "ciw.jump-scalar-design-input.v1"
const OUTPUT_SCHEMA = "ciw.jump-scalar-design-output.v1"
const MAX_REQUEST_BYTES = 8192
const RATIONAL_BOUND = 1_000_000

function exact_keys(value, expected, name)
    value isa AbstractDict || error("$name must be a table")
    Set(keys(value)) == Set(expected) || error("$name has missing or unsupported fields")
end

function rational_field(value, name)
    exact_keys(value, ("numerator", "denominator"), name)
    numerator = value["numerator"]
    denominator = value["denominator"]
    numerator isa Integer && !(numerator isa Bool) || error("$name numerator must be an integer")
    denominator isa Integer && !(denominator isa Bool) || error("$name denominator must be an integer")
    -RATIONAL_BOUND <= numerator <= RATIONAL_BOUND || error("$name numerator exceeds the bound")
    1 <= denominator <= RATIONAL_BOUND || error("$name denominator exceeds the bound")
    gcd(numerator, denominator) == 1 || error("$name must be a reduced rational")
    return BigInt(numerator) // BigInt(denominator)
end

function read_request()
    bytes = read(stdin, MAX_REQUEST_BYTES + 1)
    length(bytes) <= MAX_REQUEST_BYTES || error("request exceeds the byte bound")
    isvalid(String, bytes) || error("request must be UTF-8")
    request = TOML.parse(String(bytes))
    exact_keys(request, ("schema", "x", "target", "regularization", "lower", "upper"), "request")
    request["schema"] == INPUT_SCHEMA || error("unsupported input schema")
    values = Dict(name => rational_field(request[name], name)
                  for name in ("x", "target", "regularization", "lower", "upper"))
    -10 <= values["x"] <= 10 || error("x is outside [-10, 10]")
    -100 <= values["target"] <= 100 || error("target is outside [-100, 100]")
    1 // 1_000_000 <= values["regularization"] <= 100 || error("regularization is outside [1e-6, 100]")
    -1 <= values["lower"] <= values["upper"] <= 1 || error("bounds require -1 <= lower <= upper <= 1")
    return values
end

# Validate the data before loading the numerical provider. No request can supply
# code, package names, executable paths, environment changes or solver options.
const REQUEST = try
    read_request()
catch exception
    println(stderr, "jump-design: rejected request: ", sprint(showerror, exception))
    exit(2)
end

redirect_stdout(stderr) do
    @eval using JuMP
    @eval using HiGHS
end

function solve_request(request)
    x = request["x"]
    # Calculate the declared model and derivative in exact rational arithmetic
    # before crossing into the solver's Float64 arithmetic.
    baseline = Float64(x^2)
    jacobian = Float64(2x)
    target = Float64(request["target"])
    regularization = Float64(request["regularization"])
    lower = Float64(request["lower"])
    upper = Float64(request["upper"])

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "threads", 1)
    set_optimizer_attribute(model, "time_limit", 10.0)
    set_optimizer_attribute(model, "qp_iteration_limit", 10_000)
    set_optimizer_attribute(model, "primal_feasibility_tolerance", 1.0e-9)
    set_optimizer_attribute(model, "dual_feasibility_tolerance", 1.0e-9)
    set_optimizer_attribute(model, "random_seed", 0)
    @variable(model, lower <= delta <= upper)
    @objective(model, Min,
               (baseline + jacobian * delta - target)^2 + regularization * delta^2)
    optimize!(model)
    termination = termination_status(model)
    primal = primal_status(model)
    termination == JuMP.MOI.OPTIMAL || error("solver termination status: $termination")
    primal == JuMP.MOI.FEASIBLE_POINT && has_values(model) || error("solver primal status: $primal")
    selected = value(delta)
    objective = objective_value(model)
    seconds = solve_time(model)
    all(isfinite, (selected, objective, seconds)) || error("solver returned a nonfinite value")
    seconds >= 0 || error("solver returned a negative elapsed time")
    return Dict(
        "schema" => OUTPUT_SCHEMA,
        "termination_status" => string(termination),
        "primal_status" => string(primal),
        "delta" => selected,
        "objective_value" => objective,
        "solve_seconds" => seconds,
        "julia_version" => string(VERSION),
        "jump_version" => string(pkgversion(JuMP)),
        "highs_version" => string(pkgversion(HiGHS)),
    )
end

try
    response = redirect_stdout(stderr) do
        solve_request(REQUEST)
    end
    TOML.print(stdout, response; sorted=true)
    println(stdout)
catch exception
    println(stderr, "jump-design: execution failed: ", sprint(showerror, exception))
    exit(2)
end

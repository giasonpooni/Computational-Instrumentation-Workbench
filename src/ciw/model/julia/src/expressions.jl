# Interpretation of the lowered, data-only model (ciw.model-lowered.v1).
# Requests carry expression trees, never Julia source. Every request compiles
# fresh closures; nothing numerical survives between requests.

const UNARY = Dict{String,Function}(
    "sin" => sin, "cos" => cos, "tan" => tan, "exp" => exp, "log" => log,
    "tanh" => tanh, "sqrt" => sqrt, "abs" => abs,
)

struct MalformedInput <: Exception
    message::String
end

function compile_expr(node, slots::Dict{String,Int})
    node isa AbstractDict || throw(MalformedInput("expression nodes must be maps"))
    if haskey(node, "sym")
        length(node) == 1 || throw(MalformedInput("symbol node has extra fields"))
        name = node["sym"]
        haskey(slots, name) || throw(MalformedInput("undeclared symbol $(name)"))
        index = slots[name]
        return vals -> vals[index]
    elseif haskey(node, "num")
        length(node) == 1 || throw(MalformedInput("literal node has extra fields"))
        value = node["num"]
        (value isa Real && !(value isa Bool) && isfinite(value)) || throw(MalformedInput("invalid literal"))
        constant = Float64(value)
        return vals -> constant
    end
    (haskey(node, "op") && haskey(node, "args") && length(node) == 2) || throw(MalformedInput("invalid expression node"))
    op, raw_args = node["op"], node["args"]
    raw_args isa AbstractVector || throw(MalformedInput("operator arguments must be a list"))
    args = [compile_expr(a, slots) for a in raw_args]
    n = length(args)
    if op == "add" && n >= 2
        return vals -> begin
            acc = args[1](vals)
            for k in 2:n
                acc = acc + args[k](vals)
            end
            acc
        end
    elseif op == "mul" && n >= 2
        return vals -> begin
            acc = args[1](vals)
            for k in 2:n
                acc = acc * args[k](vals)
            end
            acc
        end
    elseif op == "sub" && n == 2
        a, b = args
        return vals -> a(vals) - b(vals)
    elseif op == "div" && n == 2
        a, b = args
        return vals -> a(vals) / b(vals)
    elseif op == "neg" && n == 1
        a = args[1]
        return vals -> -a(vals)
    elseif op == "pow" && n == 2
        a, b = args
        return vals -> a(vals)^b(vals)
    elseif haskey(UNARY, op) && n == 1
        f, a = UNARY[op], args[1]
        return vals -> f(a(vals))
    end
    throw(MalformedInput("unsupported operator $(op)/$(n)"))
end

struct LoweredModel
    t_symbol::String
    t_scale::Float64
    states::Vector{String}
    state_scales::Vector{Float64}
    inputs::Vector{String}
    input_scales::Vector{Float64}
    parameters::Vector{String}
    parameter_scales::Vector{Float64}
    parameter_values::Vector{Float64}
    derived::Vector{String}
    derived_scales::Vector{Float64}
    derived_exprs::Vector{Any}
    dynamics::Vector{Any}
    observations::Vector{String}
    observation_scales::Vector{Float64}
    observation_exprs::Vector{Any}
    raw::Dict{String,Any}
end

positive_scale(x) = (x isa Float64 && isfinite(x) && x > 0) ? x : throw(MalformedInput("unit scales must be positive binary64"))

function entries(raw, key)
    items = raw[key]
    items isa AbstractVector || throw(MalformedInput("$(key) must be a list"))
    return items
end

function lowered_model(raw)
    raw isa AbstractDict || throw(MalformedInput("lowered model must be a map"))
    get(raw, "schema", nothing) == "ciw.model-lowered.v1" || throw(MalformedInput("unsupported lowered model schema"))
    independent = raw["independent"]
    t_symbol = independent["symbol"]
    names = String[t_symbol]
    states = [e["symbol"] for e in entries(raw, "states")]
    inputs = [e["symbol"] for e in entries(raw, "inputs")]
    parameters = [e["symbol"] for e in entries(raw, "parameters")]
    derived = [e["symbol"] for e in entries(raw, "derived")]
    append!(names, states, inputs, parameters, derived)
    length(unique(names)) == length(names) || throw(MalformedInput("duplicate symbol"))
    isempty(states) && throw(MalformedInput("a model needs at least one state"))
    slots = Dict{String,Int}(name => i for (i, name) in enumerate(names))
    # Derived quantities may only reference earlier slots; compile with a growing scope.
    base = Dict{String,Int}(name => slots[name] for name in names[1:1+length(states)+length(inputs)+length(parameters)])
    derived_exprs = Any[]
    for e in entries(raw, "derived")
        push!(derived_exprs, compile_expr(e["expression"], base))
        base[e["symbol"]] = slots[e["symbol"]]
    end
    dynamics = entries(raw, "dynamics")
    length(dynamics) == length(states) || throw(MalformedInput("one dynamics expression per state is required"))
    observation_exprs = Any[compile_expr(e["expression"], slots) for e in entries(raw, "observations")]
    return LoweredModel(
        t_symbol, positive_scale(independent["scale"]),
        states, Float64[positive_scale(e["scale"]) for e in entries(raw, "states")],
        inputs, Float64[positive_scale(e["scale"]) for e in entries(raw, "inputs")],
        parameters, Float64[positive_scale(e["scale"]) for e in entries(raw, "parameters")],
        Float64[finite_number(e["value"]) for e in entries(raw, "parameters")],
        derived, Float64[positive_scale(e["scale"]) for e in entries(raw, "derived")], derived_exprs,
        Any[compile_expr(e, slots) for e in dynamics],
        [e["symbol"] for e in entries(raw, "observations")],
        Float64[positive_scale(e["scale"]) for e in entries(raw, "observations")], observation_exprs, raw)
end

function finite_number(x)
    (x isa Real && !(x isa Bool) && isfinite(x)) || throw(MalformedInput("expected a finite number"))
    return Float64(x)
end

nslots(m::LoweredModel) = 1 + length(m.states) + length(m.inputs) + length(m.parameters) + length(m.derived)

"Fill SI values for one evaluation point; inputs and parameters are in declared units."
function environment(m::LoweredModel, t, x, u, p)
    T = promote_type(typeof(t), eltype(x), eltype(u), eltype(p))
    vals = Vector{T}(undef, nslots(m))
    vals[1] = t * m.t_scale
    k = 1
    for i in eachindex(m.states)
        vals[k+i] = x[i] * m.state_scales[i]
    end
    k += length(m.states)
    for i in eachindex(m.inputs)
        vals[k+i] = u[i] * m.input_scales[i]
    end
    k += length(m.inputs)
    for i in eachindex(m.parameters)
        vals[k+i] = p[i] * m.parameter_scales[i]
    end
    k += length(m.parameters)
    for i in eachindex(m.derived)
        vals[k+i] = m.derived_exprs[i](vals)
    end
    return vals
end

"Derivative of the declared-unit state with respect to the declared-unit independent variable."
function rhs(m::LoweredModel, t, x, u, p)
    vals = environment(m, t, x, u, p)
    return [m.dynamics[i](vals) * m.t_scale / m.state_scales[i] for i in eachindex(m.states)]
end

function observe(m::LoweredModel, t, x, u, p)
    vals = environment(m, t, x, u, p)
    return [m.observation_exprs[i](vals) / m.observation_scales[i] for i in eachindex(m.observations)]
end

function derived_values(m::LoweredModel, t, x, u, p)
    vals = environment(m, t, x, u, p)
    offset = 1 + length(m.states) + length(m.inputs) + length(m.parameters)
    return [vals[offset+i] / m.derived_scales[i] for i in eachindex(m.derived)]
end

function input_vector(m::LoweredModel, inputs)
    inputs isa AbstractDict || throw(MalformedInput("inputs must be a map"))
    Set(keys(inputs)) == Set(m.inputs) || throw(MalformedInput("inputs must name exactly the declared inputs"))
    return Float64[finite_number(inputs[name]) for name in m.inputs]
end

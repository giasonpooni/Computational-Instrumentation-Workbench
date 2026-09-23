# ModelingToolkit / Symbolics route (symbolic profile only). The structured
# lowered model is rebuilt symbolically; ModelingToolkit may reorder unknowns,
# so results are always mapped back to the declared state order.

function symbolic_expr(node, table::Dict{String,Any})
    if haskey(node, "sym")
        return table[node["sym"]]
    elseif haskey(node, "num")
        return node["num"]
    end
    op = node["op"]
    args = [symbolic_expr(a, table) for a in node["args"]]
    op == "add" && return foldl(+, args)
    op == "mul" && return foldl(*, args)
    op == "sub" && return args[1] - args[2]
    op == "div" && return args[1] / args[2]
    op == "neg" && return -args[1]
    op == "pow" && return args[1]^args[2]
    haskey(UNARY, op) && return UNARY[op](args[1])
    throw(MalformedInput("unsupported operator $(op)"))
end

function mtk_name(symbol::String)
    name = replace(symbol, "." => "__")
    Base.isidentifier(name) || throw(MalformedInput("symbol $(symbol) has no ModelingToolkit name"))
    return Symbol(name)
end

function op_symbolic(input, config)
    checked_config(config, Dict("compile" => "ModelingToolkit.mtkcompile", "jacobian" => "Symbolics.jacobian",
                                "latex" => "Latexify.latexify"))
    m = lowered_model(input["lowered"])
    raw = m.raw
    x0 = f64_vector(input["state"], "state")
    require(length(x0) == length(m.states), "state length must match the state order")
    u0 = input_vector(m, input["inputs"])
    t0 = finite_number(input["time"])
    names = [m.t_symbol; m.states; m.inputs; m.parameters]
    length(unique(mtk_name.(names))) == length(names) || throw(MalformedInput("symbol names collide after sanitizing"))
    tau = only(ModelingToolkit.@independent_variables $(mtk_name(m.t_symbol)))
    D = ModelingToolkit.Differential(tau)
    xs = [only(ModelingToolkit.@variables $(mtk_name(s))(tau)) for s in m.states]
    us = [only(ModelingToolkit.@parameters $(mtk_name(s))) for s in m.inputs]
    ps = [only(ModelingToolkit.@parameters $(mtk_name(s))) for s in m.parameters]
    # SI-valued symbols: declared variable times its unit scale.
    table = Dict{String,Any}(m.t_symbol => tau * m.t_scale)
    for (i, s) in enumerate(m.states); table[s] = xs[i] * m.state_scales[i]; end
    for (i, s) in enumerate(m.inputs); table[s] = us[i] * m.input_scales[i]; end
    for (i, s) in enumerate(m.parameters); table[s] = ps[i] * m.parameter_scales[i]; end
    for (i, e) in enumerate(raw["derived"])
        table[e["symbol"]] = symbolic_expr(e["expression"], table)
    end
    rhs_declared = [symbolic_expr(raw["dynamics"][i], table) * m.t_scale / m.state_scales[i] for i in eachindex(m.states)]
    equations = [D(xs[i]) ~ rhs_declared[i] for i in eachindex(xs)]
    system = ModelingToolkit.System(equations, tau, xs, [us; ps]; name=:ciw_model)
    compiled = try
        ModelingToolkit.mtkcompile(system)
    catch err
        throw(Halt(3, "mtkcompile failed: $(sprint(showerror, err))"))
    end
    declared_names = [string(Symbolics.tosymbol(x; escape=false)) for x in xs]
    compiled_unknowns = ModelingToolkit.unknowns(compiled)
    compiled_names = [string(Symbolics.tosymbol(x; escape=false)) for x in compiled_unknowns]
    Set(compiled_names) == Set(declared_names) || throw(Halt(3, "compiled system changed the unknown set"))
    permutation = Int64[findfirst(==(name), compiled_names) - 1 for name in declared_names]
    point = Dict{Any,Any}()
    for (i, x) in enumerate(xs); point[x] = x0[i]; end
    for (i, v) in enumerate(us); point[v] = u0[i]; end
    for (i, v) in enumerate(ps); point[v] = m.parameter_values[i]; end
    point[tau] = t0
    numeric(M) = Float64[Float64(Symbolics.value(Symbolics.substitute(entry, point))) for entry in M]
    J_declared = numeric(Symbolics.jacobian(rhs_declared, xs))
    J_compiled = numeric(ModelingToolkit.calculate_jacobian(compiled))
    order = permutation .+ 1
    J_mapped = J_compiled[order, order]
    all(isfinite, J_declared) && all(isfinite, J_mapped) || throw(Halt(4, "nonfinite symbolic Jacobian"))
    latex = string(Latexify.latexify(equations))
    return Dict{String,Any}(
        "declared_unknowns" => declared_names, "compiled_unknowns" => compiled_names,
        "permutation" => permutation, "jacobian_declared" => F64Array(J_declared),
        "jacobian_compiled_to_declared" => F64Array(J_mapped), "latex_equations" => latex)
end

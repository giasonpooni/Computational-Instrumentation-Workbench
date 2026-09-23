# Numerical operations of the core profile. Each call builds fresh problem and
# solver state from the committed input and configuration bytes.

struct Halt <: Exception
    code::Int
    detail::String
end

require(cond, message) = cond || throw(MalformedInput(message))

function f64_vector(x, name)
    x isa F64Array || throw(MalformedInput("$(name) must be a binary64 array"))
    return as_vector(x)
end

function checked_config(config, expected::Dict)
    config isa AbstractDict || throw(MalformedInput("configuration must be a map"))
    for (key, value) in expected
        get(config, key, nothing) == value || throw(MalformedInput("configuration $(key) must be $(value)"))
    end
    return config
end

function ode_settings(config, n)
    reltol = finite_number(config["reltol"])
    require(1e-14 <= reltol <= 1e-2, "reltol outside [1e-14, 1e-2]")
    abstol = f64_vector(config["abstol"], "abstol")
    require(length(abstol) == n && all(>(0), abstol), "abstol needs one positive entry per state")
    maxiters = config["maxiters"]
    require(maxiters isa Int64 && 1 <= maxiters <= 10_000_000, "maxiters outside 1..10000000")
    return reltol, abstol, maxiters
end

function op_simulate(input, config)
    checked_config(config, Dict("algorithm" => "Tsit5", "controller" => "algorithm_default",
        "initial_dt" => "solver_automatic", "internalnorm" => "ODE_DEFAULT_NORM",
        "save_policy" => "interpolated_saveat", "span" => "first_to_last_sample"))
    m = lowered_model(input["lowered"])
    n = length(m.states)
    reltol, abstol, maxiters = ode_settings(config, n)
    dtmax = config["dtmax"]
    require(dtmax === nothing || (dtmax isa Float64 && isfinite(dtmax) && dtmax > 0), "dtmax must be positive or null")
    x0 = f64_vector(input["initial_state"], "initial_state")
    require(length(x0) == n, "initial_state length must match the state order")
    u = input_vector(m, input["inputs"])
    times = f64_vector(input["sample_times"], "sample_times")
    require(2 <= length(times) <= 4096 && all(diff(times) .> 0), "sample_times must be 2..4096 increasing values")
    p = m.parameter_values
    f(x, _, t) = rhs(m, t, x, u, p)
    problem = ODEProblem{false}(f, x0, (times[1], times[end]))
    options = Dict{Symbol,Any}(:reltol => reltol, :abstol => abstol, :maxiters => maxiters,
        :saveat => times, :save_everystep => false, :dense => false)
    dtmax === nothing || (options[:dtmax] = dtmax)
    solution = try
        solve(problem, Tsit5(); options...)
    catch err
        err isa DomainError && throw(Halt(4, "domain error during integration: $(err)"))
        rethrow()
    end
    SciMLBase.successful_retcode(solution) || throw(Halt(3, "solver retcode $(solution.retcode)"))
    length(solution.t) == length(times) && solution.t == times ||
        throw(Halt(5, "saved $(length(solution.t)) of $(length(times)) requested samples"))
    X = reduce(vcat, (permutedims(state) for state in solution.u))
    all(isfinite, X) || throw(Halt(4, "nonfinite state"))
    derived = Dict{String,Any}(name => zeros(length(times)) for name in m.derived)
    observations = Dict{String,Any}(name => zeros(length(times)) for name in m.observations)
    for (k, t) in enumerate(times)
        d = derived_values(m, t, solution.u[k], u, p)
        o = observe(m, t, solution.u[k], u, p)
        for (i, name) in enumerate(m.derived)
            derived[name][k] = d[i]
        end
        for (i, name) in enumerate(m.observations)
            observations[name][k] = o[i]
        end
    end
    for series in (values(derived)..., values(observations)...)
        all(isfinite, series) || throw(Halt(4, "nonfinite derived or observed value"))
    end
    stats = solution.stats
    return Dict{String,Any}(
        "t" => F64Array(collect(solution.t)), "x" => F64Array(X),
        "derived" => Dict{String,Any}(k => F64Array(v) for (k, v) in derived),
        "observations" => Dict{String,Any}(k => F64Array(v) for (k, v) in observations),
        "retcode" => string(solution.retcode),
        "stats" => Dict{String,Any}("naccept" => Int64(stats.naccept), "nreject" => Int64(stats.nreject),
                                    "nf" => Int64(stats.nf)))
end

function matrix_rank(M, rtol)
    isempty(M) && return 0
    s = svdvals(M)
    return count(>(rtol * maximum(s)), s)
end

function op_linearize(input, config)
    checked_config(config, Dict("analysis" => "ControlSystemsBase ss/poles/ctrb/obsv",
                                "differentiation" => "ForwardDiff.jacobian"))
    rtol = finite_number(config["rank_rtol"])
    require(0 < rtol < 1, "rank_rtol must lie in (0, 1)")
    m = lowered_model(input["lowered"])
    x = f64_vector(input["state"], "state")
    require(length(x) == length(m.states), "state length must match the state order")
    u = input_vector(m, input["inputs"])
    t = finite_number(input["time"])
    p = m.parameter_values
    n, nu, ny = length(m.states), length(m.inputs), length(m.observations)
    f0 = try
        rhs(m, t, x, u, p)
    catch err
        err isa DomainError && throw(Halt(4, "domain error at operating point"))
        rethrow()
    end
    A = ForwardDiff.jacobian(z -> rhs(m, t, z, u, p), x)
    B = nu == 0 ? zeros(n, 0) : ForwardDiff.jacobian(w -> rhs(m, t, x, w, p), u)
    C = ny == 0 ? zeros(0, n) : ForwardDiff.jacobian(z -> observe(m, t, z, u, p), x)
    D = (ny == 0 || nu == 0) ? zeros(ny, nu) : ForwardDiff.jacobian(w -> observe(m, t, x, w, p), u)
    all(isfinite, A) && all(isfinite, B) && all(isfinite, C) && all(isfinite, D) && all(isfinite, f0) ||
        throw(Halt(4, "nonfinite linearization"))
    λ = if nu > 0 && ny > 0
        poles(ss(A, B, C, D))
    else
        eigvals(A)
    end
    order = sortperm(λ; by=z -> (real(z), imag(z)))
    λ = λ[order]
    controllability = nu == 0 ? 0 : matrix_rank(ctrb(A, B), rtol)
    observability = ny == 0 ? 0 : matrix_rank(obsv(A, C), rtol)
    return Dict{String,Any}(
        "A" => F64Array(A), "B" => F64Array(B), "C" => F64Array(C), "D" => F64Array(D),
        "rhs" => F64Array(f0),
        "poles_real" => F64Array(real.(λ)), "poles_imag" => F64Array(imag.(λ)),
        "controllability_rank" => Int64(controllability), "observability_rank" => Int64(observability))
end

function op_measurement_selection(input, config)
    checked_config(config, Dict("algorithm" => "Tsit5", "mip_rel_gap" => 0.0,
        "objective" => "maximin_prior_normalized_fisher_diagonal", "random_seed" => 0,
        "sensitivity" => "ForwardDiff_through_Tsit5", "solver" => "HiGHS", "threads" => 1))
    m = lowered_model(input["lowered"])
    n = length(m.states)
    reltol, abstol, maxiters = ode_settings(config, n)
    time_limit = finite_number(config["time_limit_s"])
    require(0 < time_limit <= 600, "time_limit_s must lie in (0, 600]")
    x0 = f64_vector(input["initial_state"], "initial_state")
    require(length(x0) == n, "initial_state length must match the state order")
    u = input_vector(m, input["inputs"])
    t0 = finite_number(input["start_time"])
    design = input["design_parameters"]
    require(design isa AbstractVector && !isempty(design) && allunique(design) &&
            all(d -> d in m.parameters, design), "design_parameters must name distinct declared parameters")
    positions = [findfirst(==(d), m.parameters) for d in design]
    candidates = input["candidates"]
    require(candidates isa AbstractVector && 1 <= length(candidates) <= 64, "1..64 candidates are supported")
    prior = input["prior_variance"]
    noise = input["noise_variance"]
    require(prior isa AbstractDict && Set(keys(prior)) == Set(design), "prior_variance must cover design_parameters")
    times = Float64[]
    obs_index = Int[]
    costs = Float64[]
    for c in candidates
        require(c isa AbstractDict && Set(keys(c)) == Set(["cost", "observation", "time"]), "invalid candidate")
        k = findfirst(==(c["observation"]), m.observations)
        require(k !== nothing, "candidate names an undeclared observation")
        t = finite_number(c["time"])
        require(t > t0, "candidate times must follow start_time")
        cost = finite_number(c["cost"])
        require(cost >= 0, "candidate cost must be nonnegative")
        push!(times, t); push!(obs_index, k); push!(costs, cost)
    end
    require(noise isa AbstractDict && all(name -> haskey(noise, name) && finite_number(noise[name]) > 0,
            unique(m.observations[obs_index])), "noise_variance must be positive for every candidate observation")
    budget = finite_number(input["budget"])
    max_count = input["max_count"]
    require(budget >= 0 && max_count isa Int64 && max_count >= 1, "budget and max_count must be valid")
    save_times = sort(unique(times))
    function outputs(theta)
        p = convert(Vector{eltype(theta)}, copy(m.parameter_values))
        p[positions] .= theta
        x_start = convert(Vector{eltype(theta)}, x0)
        problem = ODEProblem{false}((x, _, t) -> rhs(m, t, x, u, p), x_start, (t0, save_times[end]))
        solution = solve(problem, Tsit5(); reltol=reltol, abstol=abstol, maxiters=maxiters,
                         saveat=save_times, save_everystep=false, save_start=false, dense=false)
        SciMLBase.successful_retcode(solution) || throw(Halt(3, "sensitivity solve retcode $(solution.retcode)"))
        solution.t == save_times || throw(Halt(3, "sensitivity solve did not reach every candidate time"))
        return [observe(m, times[i], solution.u[searchsortedfirst(save_times, times[i])], u, p)[obs_index[i]]
                for i in eachindex(times)]
    end
    theta = m.parameter_values[positions]
    S = ForwardDiff.jacobian(outputs, theta)
    all(isfinite, S) || throw(Halt(4, "nonfinite sensitivities"))
    G = [S[i, j]^2 * finite_number(prior[design[j]]) / finite_number(noise[m.observations[obs_index[i]]])
         for i in eachindex(times), j in eachindex(design)]
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "mip_rel_gap", 0.0)
    set_attribute(model, "random_seed", 0)
    set_attribute(model, "threads", 1)
    set_attribute(model, "time_limit", time_limit)
    c = length(times)
    @variable(model, w[1:c], Bin)
    @variable(model, s >= 0)
    @constraint(model, sum(costs[i] * w[i] for i in 1:c) <= budget)
    @constraint(model, sum(w) <= max_count)
    for j in eachindex(design)
        @constraint(model, sum(G[i, j] * w[i] for i in 1:c) >= s)
    end
    @objective(model, Max, s)
    optimize!(model)
    termination = termination_status(model)
    primal = primal_status(model)
    termination == MOI.OPTIMAL && primal == MOI.FEASIBLE_POINT ||
        throw(Halt(3, "HiGHS terminated with $(termination), primal $(primal)"))
    selected = Int64[i - 1 for i in 1:c if value(w[i]) > 0.5]
    return Dict{String,Any}(
        "sensitivities" => F64Array(S), "information" => F64Array(G), "selected" => selected,
        "objective" => Float64(objective_value(model)),
        "termination_status" => string(termination), "primal_status" => string(primal))
end

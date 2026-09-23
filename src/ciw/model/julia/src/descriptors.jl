# Registered operation descriptors. The program bytes of a request are the
# descriptor followed by "runtime: sha256:<runtime digest>\n". Keep these
# byte-identical with ciw.model.worker.DESCRIPTORS; the handshake reports
# each descriptor's SHA-256 so the host refuses drift before any work.

const DESCRIPTORS = Dict{String,String}(
    "ciw.model.simulate.v1" => """
ciw.julia.model-simulate.v1
input: CIWB v1 {initial_state: f64[n] declared units, inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, sample_times: f64[m]}
configuration: CIWB v1 {algorithm: Tsit5, abstol: f64[n] state units, controller: algorithm_default, dtmax: f64|null, initial_dt: solver_automatic, internalnorm: ODE_DEFAULT_NORM, maxiters: i64, reltol: f64, save_policy: interpolated_saveat, span: first_to_last_sample}
semantics: x_si = scale .* x; derived in declared order; dx/dtau = rhs_si * scale(tau) ./ scale(x); binary64
output: CIWB v1 {derived: {symbol: f64[m]}, observations: {symbol: f64[m]}, retcode: str, stats: {naccept, nf, nreject}, t: f64[m], x: f64[m,n]}
faults: 2=malformed, 3=solver unsuccessful, 4=nonfinite or domain error, 5=incomplete sample coverage
""",
    "ciw.model.linearize.v1" => """
ciw.julia.model-linearize.v1
input: CIWB v1 {inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, state: f64[n], time: f64}
configuration: CIWB v1 {analysis: ControlSystemsBase ss/poles/ctrb/obsv, differentiation: ForwardDiff.jacobian, rank_rtol: f64}
semantics: A=d(dx/dtau)/dx, B=d(dx/dtau)/du, C=dy/dx, D=dy/du in declared units at the operating point
output: CIWB v1 {A, B, C, D: f64[.,.], controllability_rank: i64, observability_rank: i64, poles_imag: f64[n], poles_real: f64[n], rhs: f64[n]}
faults: 2=malformed, 4=nonfinite or domain error
""",
    "ciw.model.measurement-selection.v1" => """
ciw.julia.model-measurement-selection.v1
input: CIWB v1 {budget: f64, candidates: [{cost: f64, observation: str, time: f64}], design_parameters: [str], initial_state: f64[n], inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, max_count: i64, noise_variance: {observation: f64}, prior_variance: {parameter: f64}, start_time: f64}
configuration: CIWB v1 {abstol: f64[n], algorithm: Tsit5, maxiters: i64, mip_rel_gap: 0.0, objective: maximin_prior_normalized_fisher_diagonal, random_seed: 0, reltol: f64, sensitivity: ForwardDiff_through_Tsit5, solver: HiGHS, threads: 1, time_limit_s: f64}
semantics: G[i,j] = (dy_i/dtheta_j)^2 * prior_variance[j] / noise_variance[obs_i]; maximize s subject to sum_i w_i G[i,j] >= s, sum w_i cost_i <= budget, sum w_i <= max_count, w binary
output: CIWB v1 {information: f64[c,p], objective: f64, primal_status: str, selected: [i64], sensitivities: f64[c,p], termination_status: str}
faults: 2=malformed, 3=solver unsuccessful, 4=nonfinite or domain error
""",
    "ciw.model.symbolic.v1" => """
ciw.julia.model-symbolic.v1
input: CIWB v1 {inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, state: f64[n], time: f64}
configuration: CIWB v1 {compile: ModelingToolkit.mtkcompile, jacobian: Symbolics.jacobian, latex: Latexify.latexify}
semantics: build ModelingToolkit System from the lowered model in declared units; evaluate symbolic Jacobians at the operating point in declared state order
output: CIWB v1 {compiled_unknowns: [str], declared_unknowns: [str], jacobian_compiled_to_declared: f64[n,n], jacobian_declared: f64[n,n], latex_equations: str, permutation: [i64]}
faults: 2=malformed, 3=compilation failed, 4=nonfinite or domain error
""",
)

const PROFILE_OPERATIONS = Dict(
    "core" => ["ciw.model.simulate.v1", "ciw.model.linearize.v1", "ciw.model.measurement-selection.v1"],
    "symbolic" => ["ciw.model.symbolic.v1"],
)

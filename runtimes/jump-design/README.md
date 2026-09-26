# Scalar design provider

This Julia 1.10 environment contains a one-shot JuMP/HiGHS provider for one
bounded quadratic program. The environment was resolved with Julia 1.10.12,
JuMP 1.31.2, HiGHS.jl 1.25.4, and HiGHS_jll 1.15.1+3. The checked-in
`Manifest.toml` pins the complete resolved package graph. Instantiate it
explicitly before using the worker:

```powershell
$env:JULIA_DEPOT_PATH = Join-Path $PWD '.ciw-jump-depot'
julia --startup-file=no --history-file=no --project=runtimes/jump-design -e 'using Pkg; Pkg.instantiate()'
```

The worker does not install packages, load caller-supplied code, choose a
different model, or change solver settings in response to input. It reads one
UTF-8 TOML request from standard input, bounded to 8,192 bytes, and writes one
TOML result to standard output. Initialization and solver diagnostics go to
standard error. Rejected data and failed solves return exit code 2 with no
result. A host should also impose a process deadline, because the solver's
time limit excludes Julia startup and compilation.

The input schema is `ciw.jump-scalar-design-input.v1`. Its only other fields
are `x`, `target`, `regularization`, `lower`, and `upper`; each is a reduced
rational table with integer `numerator` and positive integer `denominator`.
Booleans are not integers for this protocol. Each integer's absolute value
is at most 1,000,000. The allowed domains are:

- `-10 <= x <= 10` and `-100 <= target <= 100`.
- `1/1000000 <= regularization <= 100`.
- `-1 <= lower <= upper <= 1`.

The worker forms `y0 = x^2` and `j = 2*x` using exact rational arithmetic,
converts those coefficients to Float64, and calls `optimize!` on:

```text
minimize   (y0 + j*delta - target)^2 + regularization*delta^2
subject to lower <= delta <= upper
```

This is adjustment against the declared local linear model; it is not
optimization of the nonlinear value `(x + delta)^2`. The positive
regularization makes this scalar quadratic strictly convex. The host's
independent exact reference remains a separate check of the numerical
candidate.

HiGHS uses one thread, a 10-second solver limit, a 10,000-iteration QP limit,
primal and dual feasibility tolerances of `1e-9`, and random seed zero.
Other options use the pinned HiGHS defaults. Solver tolerances do not prove
that every returned value meets an application-specific accuracy criterion.
For example, `x=1/1000000`, `target=1/1000`, `regularization=1/1000000`, and
bounds `[-1/10, 1/10]` produced `delta=0.0019047546466489655`, while the exact
minimizer is approximately `0.0019999919980320077`. The exact objective gap
is approximately `9.07e-15`. With small curvature, a small objective gap
does not imply a similarly small relative error in the selected adjustment.
The host must check the declared objective tolerance and describe that claim
precisely; an `OPTIMAL` status alone is insufficient.

A result is emitted only for `OPTIMAL` termination with a `FEASIBLE_POINT`
and finite numerical fields. The output schema is
`ciw.jump-scalar-design-output.v1`; fields retain `termination_status`,
`primal_status`, `delta`, `objective_value`, `solve_seconds`, `julia_version`,
`jump_version`, and `highs_version`. `solve_seconds` is the solver-reported
time, excluding process startup. Status strings and versions are provider
reports, not independent verification or permission to alter equipment.

The host should retain the exact request and response bytes alongside the
worker, project, manifest, and executable digests. An executable digest
does not inventory all Julia runtime libraries or host conditions.

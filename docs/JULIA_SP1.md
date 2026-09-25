# Julia and SP1 development contract

Julia would extend the workbench's numerical and exact mathematical operations.
SP1 would prove execution of selected registered programs or result checkers.
The integration uses the existing CIW to Scientific Computation Runtime (SCR)
boundary, retaining the distinction between a computation, a verified claim and
admission of evidence for a downstream purpose.

## Delivery order and status

| Increment | Scope | Readiness |
| --- | --- | --- |
| 1. Existing SCR proof workload | Expose bounded integer heat diffusion, retained input/output and independently checked SCR proof records in one shared investigation | First implementation increment; a successful native SP1 build/prove/verify gate is required before claiming a working cryptographic path |
| 2. Julia oscillator | Run numerical integration through the CIW/SCR-shaped provider seam and compare with CIW's analytical oscillator reference | Contract, adapter, offline replay and oracle tests are present; Julia 1.10.12 manifest and genuine provider execution remain release gates; see [JULIA_OSCILLATOR.md](JULIA_OSCILLATOR.md) |
| 3. Exact topology | Compute candidate ordinary Betti numbers over F2 in Julia, then independently recompute them in a registered Rust/SP1 checker | Planned; no topology operation, registered guest or proof is supplied by this document |
| 4. Measured scaling | Record execution, proving and verification time, peak memory and guest cycles before widening bounds | Required before expanding workloads; no performance capacity is asserted here |

An unavailable runtime, a protocol fixture and a mock verifier must remain
distinguishable from a successfully verified cryptographic computation. The Julia
oscillator is registered, but its generated manifest and genuine provider tests
remain required before it is called available. Installing Julia or SP1 does not
by itself pass an integration gate. This document specifies the broader
contracts; the executable oscillator details live in [JULIA_OSCILLATOR.md](JULIA_OSCILLATOR.md).

| Component | Responsibility |
| --- | --- |
| CIW | Operator session, operation selection, source catalog, retained results, inspection and visualization |
| SCR | Execution specifications, dispatch, runtime bindings, execution identities and proof/verification records |
| Julia provider | A declared numerical or exact mathematical operation with a pinned environment |
| SP1 backend | Prove execution of an explicitly registered computation or checker |
| Verification and admission layers | Check the precise claim and apply the policy governing downstream use |

## Reuse the execution and admission seam

SCR's `execution.specification.ExecutionSpecification` carries raw `program`,
`configuration` and `input_payload` bytes. Its identity commits to all three.
`execution.dispatcher.SpecificationDispatcher.runner` already accepts a callable
returning `ExecutionResult`, allowing a Julia worker to sit below the existing
dispatch and admission path. The GROMACS runner is a precedent for an external
scientific process; a second route into evidence admission is unnecessary.

The Julia runner must recompute program, input, output and specification
commitments from retained bytes. A completed response must answer the exact
request and return the declared output encoding. A refused request has no run;
a halted run has no output or computation identity. Neither becomes an empty or
zero-filled successful result. The result's origin remains simulated.

One identity detail matters for both future increments: SCR's current
`ComputationIdentity` covers program, input, output and exit code, while
`SpecificationIdentity` additionally binds configuration. Do not treat the former
alone as a commitment to solver settings. CIW must retain the specification
link, and a proof claim must bind every mathematically relevant setting inside
the guest's committed input or explicitly checked public values.

Repeated requests may share a specification identity. They still produce distinct
execution occurrences and retained result records. Numerical agreement on replay
is a separate comparison; matching evidence bytes do not erase execution history.

## Julia oscillator operation profile

The registered first Julia operation numerically integrates

```text
q' = v
v' = -2*gamma*v - omega_0^2*q
E = 0.5*mass*(v^2 + omega_0^2*q^2)
```

CIW's existing [analytical oscillator](../src/ciw/adapters/oscillator.py) supplies
an independent closed-form numerical reference. Its default fixture uses
`omega_0 = 2*pi*0.8 rad/s`, `gamma = 0.15 s^-1`, `mass = 1 kg`, `q0 = 1 m`,
`v0 = 0 m/s`, and 768 samples at 64 Hz over `[0, 12)` seconds.
Julia must solve the ODE; reproducing that analytical formula in Julia would
exercise a language bridge but would not test the proposed integration capability.

Use one explicit non-stiff solver initially. `OrdinaryDiffEqTsit5` provides
`ODEProblem`, `solve` and `Tsit5`; its documented installation can use the focused
solver package instead of the complete DifferentialEquations suite. Declare
`SciMLBase` as a direct dependency if the worker uses its return-code API.
ModelingToolkit, OSCAR and JuliaCall are not prerequisites for this oscillator.
See the [SciML solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/Tsit5/).

The first numerical profile should accept bounded, non-stiff underdamped cases
and the undamped limit. Proposed starting bounds are `0 < omega_0 <= 20 rad/s`,
`0 <= gamma <= 0.5*omega_0`, `0 < mass <= 100 kg`, `abs(q0) <= 10 m`,
`abs(v0) <= 100 m/s`, duration at most 12 seconds and 2 to 4096 retained samples.
These are proposed software limits, to be finalized with the implementation and
its fixtures. They are not physical validity limits or a promise of accuracy
for arbitrary oscillators. Reject Boolean, nonfinite and out-of-bound numerical
fields before dispatch.

The source and configuration contract must specify:

- Exact initial conditions, model parameters, units and state order `[q, v]`.
- Exact requested sample times, their origin and the half-open endpoint rule.
  The saved time grid is distinct from the adaptive solver's internal steps.
- Binary64 arithmetic; exact algorithm and version; absolute and relative
  tolerances; initial/maximum step policy; iteration limit; interpolation and
  output-saving options. Defaults that affect results must become explicit.
- Successful solver termination, retained diagnostics and the rule for handling
  nonfinite output or incomplete time coverage. Partial trajectories may be
  retained as diagnostics, but must not masquerade as a completed result.
- No measurement uncertainty or physical calibration inferred from solver
  tolerance. Numerical error checks, assumed parameter uncertainty and physical
  calibration are distinct claims. This initial profile has no inferred covariance.

Retain exact worker input and output bytes before decoding them for display.
Use a versioned, bounded encoding with explicit endianness and field order;
binary64 values and array dimensions must round-trip without locale-dependent
text formatting. The minimum output contains the requested time vector, `q`,
`v`, derived energy and solver outcome. A JSON view is a derived representation,
not a replacement for the committed raw output. Display geometry is likewise
derived from source values and never fed back into computation.

### Worker lifecycle and environment

A host-owned persistent worker amortizes Julia startup and compilation. The
worker supports a fixed operation allowlist; requests contain data, never Julia
source, package names to install or executable paths to resolve. Runtime paths
are explicit host bindings and must not be reconstructed from saved workspaces.

Start with one request in flight per worker and a bounded queue. Use framed
messages, request identifiers, size limits and a strict response parser. Keep
diagnostics on stderr so compilation messages cannot become protocol data.
The worker handshake must report the loaded operation and environment identities;
the host compares them with its expected artifacts before accepting work.

Each request constructs fresh problem and solver state. Reuse compiled code and
loaded packages, not previous trajectory state or mutable numerical buffers.
Record a worker session identity and monotonically increasing occurrence number
separately from content identities. A timeout, cancellation, malformed response,
unexpected EOF or process failure ends that worker session. Terminate and reap
the worker, fail the active occurrence, and start a new session before later work.
Do not silently retry a failed occurrence or reuse a late response as the next
request's result. Replay is an explicit fresh occurrence.

Run with an explicit project, disabled startup file and a declared thread count.
Provision and precompile the environment separately from execution; do not
resolve or update packages while serving requests. The runtime identity must
retain the exact Julia version, platform/architecture, executable digest, worker
source digest, project and manifest digests, relevant package/artifact identities,
thread settings, numerical preferences and any nonstandard system image.
An environment mismatch requires a new declared runtime, not silent substitution.

A candidate initial Julia version is **1.10.12 LTS**, pending actual installation,
package resolution and Windows/Linux execution tests. This is a candidate, not
an installed or accepted pin. At the 2026-09-23 review, Julia's official
support table listed 1.13.0 as stable and 1.10.12 as LTS; native Windows x86-64 and
Linux glibc x86-64 are Tier 1 platforms. A newer candidate may be adopted through
the same gate. See [Julia platform support](https://julialang.org/downloads/support/).

Commit the real `Project.toml` and machine-generated `Manifest.toml` only after
instantiating and testing the selected environment. Retain the exact runtime and
package artifact checksums. A hand-written lockfile or a floating compatibility
range is not a verified execution environment. Julia's
[environment documentation](https://pkgdocs.julialang.org/v1/environments/)
describes project activation and reconstruction from a manifest.

Floating-point replay should initially claim agreement under declared numeric
tolerances. Same-host byte stability and cross-platform numerical agreement need
separate fixtures; pinning packages does not prove bitwise equality across CPUs.

### Smallest useful oscillator acceptance set

| Fixture | Required evidence |
| --- | --- |
| Existing CIW default | All 768 `q`, `v` and energy values compared against the retained analytical reference; exact requested sample coverage |
| Mixed initial state | Nonzero `q0` and `v0`, including a negative component, to catch state-order and sign errors |
| Undamped limit | Analytical phase agreement and bounded energy drift over the declared interval |
| Tightened tolerances | Recorded changes in analytical error, accepted/rejected step counts and cost; no assumption that requested tolerance is itself a global error bound |
| Repeated and interleaved runs | A, B, A on the same worker and A on a restarted worker; no state leakage, fresh occurrence IDs and declared numerical agreement |
| Refusal and failure | Invalid numbers/grid, wrong environment, oversized frame, timeout, worker crash and truncated response; no completed result fabricated |
| Offline restore | Inspect retained raw result and derived views without Julia installed; execution/replay still requires an explicit matching runtime |

Define componentwise absolute/relative error thresholds against the analytical
oracle before accepting the implementation. Record the actual maximum errors
and energy diagnostics. Run the genuine worker tests on both Windows and Linux;
protocol mocks are useful for failure tests but cannot satisfy numerical gates.

## Planned F2 topology and SP1 checker

For a first exact topology profile, use ordinary simplicial homology over the
field F2. A proposed bounded input is a nonempty complex with at most 64 vertices,
256 explicitly listed nonempty simplices and dimension at most three. These
bounds require execution and proving measurements before adoption or expansion.
The empty face is implicit; the empty complex and reduced homology are outside
this initial profile.

Use canonical vertex identifiers and order simplices first by dimension, then
lexicographically. Every simplex lists distinct increasing vertices. Require
unique simplices and every codimension-one face; reject malformed input instead
of silently adding faces, dropping duplicates or relabeling evidence. Listing
the complete bounded face set avoids an unbounded expansion from supplied facets.
Any upstream import or OSCAR relabeling must retain the original bytes and the
explicit map into this canonical complex.

Julia constructs the boundary matrices over F2 and computes candidate ranks and
Betti numbers. The later Rust checker independently reconstructs those matrices
from the retained complex, verifies `D_k * D_(k+1) = 0`, computes ranks by exact
finite-field elimination and checks

```text
beta_k = number_of_k_simplices - rank(D_k) - rank(D_(k+1))
```

For ordinary homology, `D_0` is zero and `beta_0` counts connected components.
The map above the declared maximum dimension is zero. There is no floating-point
tolerance, and no integral torsion or physical topology claim follows.

OSCAR's `betti_numbers(K)` documents **reduced rational** Betti numbers;
`homology(K, i)` documents **integral** homology. Neither output can be relabeled
as ordinary F2 Betti numbers. The Julia operation must explicitly compute ranks
over F2, using a declared exact implementation or appropriate finite-field matrix
operations. See [OSCAR's simplicial-complex API](https://docs.oscar-system.org/v1/Combinatorics/simplicialcomplexes/).

OSCAR is a later provider choice, not a dependency of the oscillator or a
requirement for a small F2 implementation. Its Windows guidance requires Linux
inside WSL because some underlying systems lack native Windows support; see the
[OSCAR FAQ](https://docs.oscar-system.org/v1/General/faq/). Provisioning and testing
that environment is a separate gate from native Julia/SciML on Windows.

### Exact proof claim and bindings

Let Julia produce candidate bytes `y` for retained complex bytes `x`. The SP1
guest executes a registered checker `G(x, y, configuration)` that accepts exactly
when the decoded candidate equals its independently recomputed ordinary F2
Betti vector under the declared bounded profile. Direct recomputation needs no
auxiliary certificate initially; a later certificate checker would be a new
profile with its own claim and tests.

The guest must compute commitments from the bytes it actually consumes and
commit to the acceptance outcome. Bind the exact complex, candidate result,
coefficient field, homology convention, bounds and checker version. Keeping
these fields in one canonical guest input allows SCR's existing input commitment
to bind them without pretending that its separate configuration field is already
part of the current SP1 public-values convention. The registered guest build and
verification key identify which checker was proved.

The verifier must validate the proof independently and compare its public
commitments, guest identity and result with the retained workbench records. A
matching output hash alone proves identity, not correctness. A successful checker
proof establishes the declared F2 result for the supplied complex; it does not
prove that Julia executed, that the complex describes measured geometry, that
integral homology agrees, or that downstream admission should automatically pass.
Julia's execution record and this checker verification remain linked but distinct.

SP1 currently documents native Linux and macOS execution. Use the reviewed SCR
toolchain and SDK pin for its existing guests; current upstream documentation is
not permission to upgrade the registered build or rewrite its public-values
layout. Windows operators need an explicitly provisioned compatible Linux worker
or backend for this proof path. See [SP1 installation](https://docs.succinct.xyz/docs/sp1/getting-started/install)
and [offchain verification](https://docs.succinct.xyz/docs/sp1/generating-proofs/off-chain-verification).

### Topology acceptance set

Start with a point `(1)`, two isolated points `(2)`, an edge `(1, 0)`, a triangle
boundary `(1, 1)`, a filled triangle `(1, 0, 0)`, a tetrahedron boundary
`(1, 0, 1)` and a filled tetrahedron `(1, 0, 0, 0)`. Retain each full canonical
complex and expected vector through its highest simplex dimension.
Include an explicitly retained triangulation of the real projective plane with
F2 vector `(1, 1, 1)` to catch accidental rational-coefficient substitution.

Refuse a missing face, repeated vertex, duplicate simplex, out-of-range vertex,
wrong ordering, excessive size/dimension and inconsistent vector length. Negative
proof tests must change the complex, candidate, coefficient convention, guest or
configuration one at a time, as well as truncate/corrupt proof bytes. A valid
proof for one record must never attach to another through an unchecked identifier.

## Asynchronous proof production and admission

For the planned worker/checker integration, proof production should run
asynchronously so visualization can display a retained numerical candidate while
verification is pending. This scheduling contract is not a claim that the first
bounded heat operation already supplies an asynchronous proof service.
Store proof execution and independent verification as separate
occurrences with explicit unavailable, pending, failed and verified outcomes.
Only a successful verifier result may support the displayed verified claim.
Cancellation or proof failure must not erase the original numerical record.

A policy requiring verification before downstream admission must check the
specific expected claim and record links. A policy permitting inspection of an
unverified candidate must preserve that status. Neither policy is inferred from
the presence of a proof-shaped file. Continue using the existing admission layer.

Record cold startup/precompilation, warm numerical execution, guest execution,
proof generation and verification separately, together with peak memory, guest
cycles, proof size, exact runtime/guest identities and workload dimensions.
Only measured results should drive larger complexes, broader Julia operations
or concurrency. Symbolic modeling, algebra, integral homology and torsion can
then enter through further bounded profiles with their own acceptance fixtures.

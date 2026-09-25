# Formal specifications for the lab queue

These specifications state what each experiment family computes and what its
checks mean. Task identities point to the executable experiments; their
retained reports hold the numbers. Textbook results are cited as such; the
workbench contribution is the executable, evidence-labelled test of them.

The sections below state the evidence rules and seven experiment families in
formal notation. The other families are specified by their pages, listed in
[Specifications of record by family](#specifications-of-record-by-family):
those pages give each family's model, references and decision rules as
method descriptions, not in one uniform notation.

## Evidence labels

Let a finding be a tuple (claim, domain, value, basis). The label function
`L(basis, domain)` (`ciw.lab.evidence.supported_label`) is total on
well-formed bases and refuses malformed ones. With `C` the set of passing
reference checks, `F` the failing ones, `I` an independent check between
implementations of different origin, `P` a pinned provider execution, `G` a
declared generator, `D` a derivation and `A` a hardware acquisition record:

1. Authority domains (`machine_safety`, `industrial_readiness`,
   `customer_demand`, `actuator_authority`, `production_acceptance`):
   `L = not_established` for every basis. The domain is an input: the
   finding's author chooses it and review checks it, and `L` never infers it
   from the claim. T141 showed what that leaves open: the same acceptance
   statement filed under `computational_pipeline` with a passing check was
   labelled `numerically_verified` (and filed under `physical` with `A` it
   would be `hardware_measured`). Rule 10 is a phrase screen against it, not
   a proof that every authority statement is caught.
2. Any failed check (`F ≠ ∅` or `I` failing): `L = not_established`.
3. Physical domains (`physical`, `calibration`, `sensor_performance`):
   `L = not_established` unless `A` is present; then
   `independently_verified` if `I` passes, else `hardware_measured`.
4. Computational domains: `A` is refused (by `L` when no check fails; by
   `finding` and `validate_finding` whatever the checks say); otherwise the
   first applicable of
   `I → independently_verified`, `C ≠ ∅ → numerically_verified`,
   `P → provider_backed`, `G → synthetic`, `D → analytic`, else
   `not_established`.
5. Origin rule: `I` requires `origin(producer) ≠ origin(checker)`, where
   `origin` is the leading ASCII name token of the NFKC-normalized,
   casefolded implementation identifier, and both origins belong to a closed
   allowlist (`ciw` plus the recognised external families). The rule is
   symmetric: a pinned provider's output checked by a `ciw` reference is as
   independent as `ciw` output checked by the provider. Code in one family
   never verifies itself independently, and a `cross_implementation` check is
   never `I`. `independently_verified` therefore means independent
   implementation agreement; independent verification by another party is
   outside what the queue can establish.
6. Non-upgrade: a stated label must equal `L(basis, domain)`
   (`validate_finding`); a derived physical status is `hardware_measured` only
   if every input is (`physical_status`).
7. Comparisons: a check passes by `holds(observed, tolerance, comparison)`
   with `abs_le: |x| ≤ t`, `le: x ≤ t` (x ≥ 0 required),
   `ge: x ≥ t`, `signed_le: x ≤ t`, `signed_ge: x ≥ t`, finite `x`,
   `|t| ≤ 10¹⁰⁰`. The stated `passed` must equal the computed one.
8. Primary label of a report: with `E` the established computational labels
   and `R` the refuted computational findings (not `not_established` by
   declaration), `primary = not_established` if `R ≠ ∅` or `E = ∅`, else
   `min(E)` in the order `synthetic < analytic < provider_backed <
   numerically_verified < independently_verified`. The function is
   order-independent and monotone: adding a weaker finding never raises it.
9. Physical gate: an acquisition record enters a report only when a hardware
   probe succeeded in the same task and its `raw_sha256` equals the digest of
   a retained artifact of that task. When the raw bytes are an operator
   capture (`ctx.capture`), the probe must be of that capture's instrument
   (`runner.CAPTURE_INSTRUMENTS`); a capture whose instrument has no probe on
   the host keeps its physical findings `not_established`. The runner resets
   probes and captures when each task begins, so a probe that succeeded for
   another task, or one replayed from another task's memo, does not count.
10. Authority wording: `finding` and `validate_finding` refuse
    (`evidence.screen_authority_claim`) a claim in a computational or physical
    domain whose wording asserts an authority outcome
    (production acceptance or disposition, certification for use, machine
    safety, actuator authorization, industrial readiness, customer demand;
    `evidence.AUTHORITY_OUTCOME`). An outcome phrase is exempt only when the
    clause that holds it says, before the phrase, that the software does not
    make, mark or record it: a negated decision verb whose object reaches the
    phrase (`evidence.DECLINED_DECISION`: "cannot mark a lot accepted for
    production", "never claims that the press is safe to operate", "no claim
    that ..."), a negation directly before a phrase that starts with an active
    verb ("does not authorize actuation"), or "records <phrase> as not
    performed, refused, external or pending"
    (`evidence.RECORDED_AS_UNDECIDED`). Clauses end at `; : ! ?`, a comma or
    full stop followed by a space, dashes, parentheses and the words *and*,
    *but*, *while*, *whereas*, *although*, *though*, *yet*, *so* and
    *because* (`evidence.CLAUSE_BREAK`); a negation after a relative pronoun
    ("the lot that never failed ...") is not counted, and making the software
    the subject exempts nothing ("the workbench is ready for industrial
    deployment" is refused). So "accepted
    for production; it does not need rework" and "accepted for production,
    recorded as lot 7" are refused. Filed in an authority domain the same claim is recorded as
    `not_established` (rule 1). The screen matches phrases, not every
    paraphrase, so assigning a free-text claim to a domain stays a review
    question; the manufacturing section additionally refuses its own decision
    words outside authority domains
    (`manufacturing_records.screen_acceptance_language`, applied to that
    section only). Every retained claim passes the screen.
11. Basis components: every finding carries the sorted list of the basis
    components it declares (`evidence.basis_origin`, stored under the key
    `origin`), drawn from `acquisition` (`A`), `derivation` (`D`),
    `independent_check` (`I`), `provider` (`P`, executed only),
    `reference_checks` (`C ∪ F ≠ ∅`) and `synthetic_inputs` (`G`). These are
    parts of the basis, not the implementation origin of rule 5. They never
    enter `L`, so no label changes; they are derived like the label, a stated
    list must equal them (`validate_finding`), and every component named must
    be well formed even where `L` never inspects it (an `A` beside a failing
    check or in an authority domain). Reports and the dashboard show them
    beside every label as the finding's basis, with the identity each
    declares (`evidence.describe_basis`: generator name and seed,
    provider repository@revision, acquisition device): rule 4 ranks a passing
    check above provenance, so `numerically_verified` from synthetic inputs and
    `numerically_verified` from a provider's output differ only in their
    basis. `A` is shown as a hardware acquisition only on a physical-domain
    finding labelled `hardware_measured` or `independently_verified` (the
    findings rule 9 gates), and otherwise as a declared acquisition record
    that was not accepted. A finding retained before basis components were
    recorded has no `origin` key; readers derive it from the basis, and
    `build_report` refuses a new finding without one.
12. Workspace classification (`ciw lab classify`): `P` is formed only from a
    retained runtime identity that is a pin CIW itself declares for the
    workflow kind (the revision of a pin for the role it is recorded under, or
    of any pin of the kind for an identity nested inside a role's runtime or
    recorded in a step; the pin's module and source root where declared; and
    the source tree CIW records for that revision wherever any CIW pin table
    records one; `ciw.lab.bridge.declared_pins`). Any other identity leaves the
    result `not_established` with the reason, so a content-consistent bundle
    whose tree is not the pin (T100's fabricated bundle) is not
    `provider_backed`. Where no CIW table records a tree for the revision, any
    tree is accepted and the row shows `tree_pinned: false`
    (`ciw.lab.bridge.pins_without_tree`: today every pin of
    `calibrated-window`, `acquired-calibrated-window`, `telemetry`,
    `residual-monitor`, `schematic-assessment`, `schematic-companions` and
    `bim-quantity`, every `calibrated-observable` and `identified-design` pin
    except `gsie`, and the historical `fsrt` and `rci` adapter pins), so for
    those kinds an invented tree is not detected. The pins are public
    constants and workspace seals are unkeyed: a record that copies the pinned
    revision and tree still classifies `provider_backed`. The comparison checks
    CIW's declarations, not who produced the record.

T155 compares `L` with a reference oracle that restates rules 1–5 here,
independently of `ciw.lab.evidence`, on every basis of a finite grammar in
every domain: derivation, generator, passing and failing checks, an executed
and an unexecuted provider, acquisition, and eight independent-check variants
(none, `ciw` against `scipy`, a pinned provider against `ciw`, a failing one,
same origin, `cross_implementation` kind, an unknown family and a name that
embeds `ciw`). Refusals are part of the comparison, and every rule branch of
the oracle must be exercised. Rules 6–12 are enforced by the validator, the
report builder, the runner and the workspace classifier and tested in
`tests/test_lab_core.py` and `tests/test_lab_bridge.py`, not by T155. T100
checks that synthetic, provider-backed and physical results remain visibly
distinct.

## Geodesic equation and references

In a chart `u = (u¹, u²)` with metric `g_ij(u)`,

```
Γᵏᵢⱼ = ½ gᵏˡ (∂ᵢ gⱼₗ + ∂ⱼ gᵢₗ − ∂ₗ gᵢⱼ),      ü ᵏ + Γᵏᵢⱼ u̇ⁱ u̇ʲ = 0.
```

For an embedding `X(u)` the metric is `gᵢⱼ = Xᵢ·Xⱼ` and
`∂ₖ gᵢⱼ = Xₖᵢ·Xⱼ + Xᵢ·Xₖⱼ`; Gaussian curvature is
`K = (LN − M²)/(EG − F²)` from the second fundamental form. The speed
`g(u̇, u̇)` is a first integral; the integrators never renormalize it, so its
drift is a measured quantity (T004). References: exact geodesics on the plane,
sphere (great circles), cylinder (helices) and hyperbolic plane (semicircles);
elsewhere a 34-digit integration of independently (sympy) derived equations,
whose extrapolation integrator must agree with mpmath's Taylor-series solver
`mpmath.odefun` within its own macro-step error estimate and at its nominal
order 16, observed as log₂(estimate / gap + 1) (T002; the reference's rounding
is measured against a 44-digit rerun, not assumed). Integrator orders 1, 2, 4
and the adaptive Dormand–Prince 5(4) behavior are tested in T003.
The symmetric Gauss collocation methods (implicit midpoint, order 2; two-stage
Gauss–Legendre, order 4) solve their stage equations `K = f(y + hAK)` by
fixed-point iteration from `K = f(y)` until the stage values change by at most
`10⁻¹³` times the magnitude of the terms they are summed from, componentwise;
a step not converged within 60 iterations, or with a nonfinite iterate or one
outside the domain of `f`, is refused (`implicit_solve_not_converged`), never
returned. Their forward-then-reversed return is exact up to that residual and
rounding (T014); their accuracy on the full nonlinear systems and their
long-horizon speed error are compared with RK4 in T016.

The hyperbolic closed form for `g = I/(k² y²)` with chart heading `α` is
`x = x₀ + y₀ cos α sinh(ks)/D`, `y = y₀/D`, `D = cosh(ks) − sin α sinh(ks)`,
evaluated without cancellation for every heading, near-vertical ones included.
A chart point is regular when `g` is positive definite with
`det g / (tr g)² > 10⁻¹²` (about a reciprocal condition number, so the test
does not depend on the surface scale); otherwise it is refused as
`degenerate_metric`, and points outside a chart's domain as `outside_chart`.
A reparametrized chart `a ↦ φ(a)` keeps the base chart's domain (`φ(a)`
outside it is refused as `outside_chart`, a nonfinite `φ(a)` as
`degenerate_metric`); regularity is judged on its pullback metric `Jᵀ g J` by
the same test, so a chart map that removes a base coordinate singularity (normal
coordinates about a pole) is regular there. A frame change
`X ↦ R X` requires `max |RᵀR − I| ≤ 10⁻¹²` and `det R = +1`. A negative length
integrates backward from `s = 0` in both the fixed-step and adaptive
integrators. Observed orders are least-squares log–log slopes; zero, negative
or nonfinite errors or step sizes are refused rather than dropped, and so are
tables with fewer than two distinct step sizes (the slope is then undetermined).

## Jacobi equation, transfer matrix and Wronskian

Along a unit-speed geodesic with parallel unit normal `N`, a normal Jacobi
field `J = j N` satisfies `j'' + K(γ(s)) j = 0`. The transfer matrix

```
Φ(s) = [[j_lat,  j_head ],
        [j_lat', j_head']],   Φ(0) = I,
```

has lateral-displacement and heading columns. `det Φ ≡ 1` (Wronskian; T007).
For constant `K`: `j_head = sin(√K s)/√K`, `s`, or `sinh(√−K s)/√−K` (T005).
Conjugate points are the zeros `s ≠ 0` of `j_head` (`s > 0` on a forward
geodesic); focal points of the initial normal geodesic are the zeros of
`j_lat` (T008); both are listed in the order the geodesic meets them. On the
unit sphere they occur at `π` and `π/2`; on the outer torus equator at
`π√(r(R+r))`.

## First-order validity and focal counterexamples

A heading perturbation `ε` moves the geodesic by the separation `d(s)`
between the base and perturbed geodesics at matched arclength: the intrinsic
geodesic distance on the sphere and the hyperbolic plane, the embedded chord
on the torus (the two agree to `O(d³)`). It is unsigned and not purely normal
(it includes any along-track offset), and it linearizes to `ε |j_head(s)|`:
`d(s) = ε |j_head(s)| + r(ε, s)` with `r = C₂(s) ε² + C₃(s) ε³ + O(ε⁴)`. The
first-order prediction is valid to relative tolerance `τ` while
`|r| ≤ τ ε |j_head(s)|`, that is for `ε ≤ ε_max(s)` with
`ε_max ≈ τ |j_head(s)| / |C₂(s)|` where `C₂ ≠ 0` and
`ε_max ≈ √(τ |j_head(s)| / |C₃(s)|)` where `C₂ = 0` (T017).

`C₂`, `C₃` and `ε_max` depend on the observable. For a pure heading
perturbation on the unit sphere the relative first-order error is
`−ε²/24` for the embedded chord at every `s` (T010), `−ε² cos² s / 24` for the
geodesic distance (T017) and `−ε² cos² s / 6` for the signed distance to the
base geodesic (Fermi normal offset) `asin(sin ε sin s)` (T064).

At a conjugate point `s*` (`j_head(s*) = 0`) the domain shrinks to zero on a
generic path, linearly in `|s − s*|` because `C₂(s*) ≠ 0` (generic torus
geodesic), and on a reflection-symmetric path with `C₂ ≡ 0` like
`√|s − s*|` (torus outer equator). It does not shrink for a pure heading
perturbation on the unit sphere, which refocuses exactly: `C₂ = 0` and
`C₃ = −|sin s| cos² s / 24` vanishes with `j_head`, so
`ε_max = √(24τ)/|cos s|`, equal to `√(24τ)` at `s = π` (T017). At the
first-order zero `s₀ = 3π/4` of a combined lateral and heading perturbation on
the sphere the domain again shrinks linearly: `C₂(s₀) = ½`, an along-track
offset that a purely normal observable would not see (T017).
Approaching a generic or symmetric conjugate point the relative first-order
error diverges like `1/|s − s*|` and falls again like `1/|s − s*|` past it;
for the sphere's pure heading perturbation it stays bounded (uniform for the
chord). In every case measured the image inverts after the conjugate point
(T010).

## Chord versus geodesic distance

A unit-speed curve whose curvature is `κ₀` at the start point and changes at
rate `κ₀′` has chord `c = s − κ₀² s³/24 − κ₀ κ₀′ s⁴/24 + O(s⁵)`; with `κ`
taken at the arc midpoint instead, the `s⁴` term cancels, so `s − κ² s³/24`
with an end-point `κ` is only the leading order. `c = 2 sin(κ s/2)/κ` holds
for a plane circle (constant `κ`, torsion `τ = 0`); a helix, whose curvature is
also constant, has a chord longer by `κ² τ² s⁵ / 720` (T046). A geodesic's
space curvature equals its normal curvature. On a cylinder of radius `R`, a
helix making angle `α` with the circumferential direction has constant
`κ = cos²α / R`, so to leading order `s − c ≈ cos⁴α s³ / (24 R²)` (T046,
T047). A camera measures `c`, not `s`.

## Filter consistency (NEES/NIS)

For a linear-Gaussian filter with innovation `ν = z − H x̂⁻` and innovation
covariance `S = H P⁻ Hᵀ + R`, the normalized innovation squared `νᵀ S⁻¹ ν`
is χ² with `dim z` degrees of freedom when the model is correct. Using the raw
sensor covariance `R` in place of `S` breaks this (T066). Ignoring declared
cross-correlation breaks NEES consistency (T062). For the linear-Gaussian
bench, the filter estimate equals the batch information-form posterior (T074).

## Flat torus lattices and winding

A flat torus is `ℂ/Λ`, `Λ = ℤω₁ + ℤω₂`. Bases related by `SL(2, ℤ)` generate
the same lattice (T019); reduction preserves area, systole and length
spectrum (T026). Closed geodesics correspond to primitive vectors
`mω₁ + nω₂`, `gcd(m, n) = 1`, of length `|mω₁ + nω₂|` (T020). Because
`K = 0`, heading sensitivity at the target equals path length (T021).

## Deterministic serialization and reduction

Canonical JSON (`ciw.canonical-json.v1`): sorted keys, no insignificant
whitespace, UTF-8, finite binary64 values in shortest round-trip form, NaN and
infinities refused (T146).

Report and artifact identities do not use `ciw.canonical-json.v1`. A report's
`report_id` (and every `content_identity` in CIW) is `sha256:` over the UTF-8
bytes of `ciw.core.identities.canonical_json(value)`, which is
`json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)`
over the report without its `report_id`. Its default `ensure_ascii=True`
writes every non-ASCII character as a `\uXXXX` escape, so its bytes differ
from the specification wherever a value holds non-ASCII text (T146 lists the
differing vectors; the retained T032, T095, T096 and T146 reports hash
differently under the specification). It also accepts what the specification
refuses (integers beyond ±2⁵³, non-string keys, lone surrogates, nesting
deeper than 64), and it turns non-string keys into strings, so `{1: "x"}` and
`{"1": "x"}` share an identity (T146); a report read back from JSON has string
keys only. A second implementation reproduces report identities with this
encoding, not with `ciw.canonical-json.v1`. Floating-point summation is not associative; a declared reduction
order (fixed pairwise tree or compensated summation) makes results
reproducible across implementations (T148), and reduction order can change a
threshold decision near its boundary (T121).

## Specifications of record by family

Each page below is the specification of record for the tasks its title names
(and any others it names). T155 checks that every computational task
(T001–T154) is named by this page or one of these, and that every
specification unit is exercised by retained established findings.

| Family | Tasks | Specification |
| --- | --- | --- |
| Geodesic and Jacobi fields | T001–T009 | [GEODESIC_JACOBI.md](GEODESIC_JACOBI.md) |
| Geodesic/Jacobi limits and counterexamples | T010–T018 | [GEODESIC_JACOBI_LIMITS.md](GEODESIC_JACOBI_LIMITS.md) |
| Flat torus and topology | T019–T032 | [FLAT_TORUS_TOPOLOGY.md](FLAT_TORUS_TOPOLOGY.md) |
| Surface interface, charts and singularities | T033–T037 | [SURFACE_INTERFACE.md](SURFACE_INTERFACE.md) |
| Triangle-mesh geodesics | T038–T044 | [MESH_GEODESICS.md](MESH_GEODESICS.md) |
| Instrument observation | T045–T059 | [OBSERVATION.md](OBSERVATION.md) |
| Sensor fusion | T060–T076 | [SENSOR_FUSION.md](SENSOR_FUSION.md) |
| Exchange and provenance: identities and mutations | T077–T090 | [EXCHANGE_PROVENANCE.md](EXCHANGE_PROVENANCE.md) |
| Exchange and provenance: bundles and providers | T091–T100 | [EXCHANGE_BUNDLES.md](EXCHANGE_BUNDLES.md) |
| Lyapunov runtime | T101–T114 | [LYAPUNOV.md](LYAPUNOV.md) |
| Energy and GPU | T115–T125 | [ENERGY_GPU.md](ENERGY_GPU.md) |
| Manufacturing and robotic use cases | T126–T141 | [MANUFACTURING.md](MANUFACTURING.md) |
| Implementation targets | T142–T154 | [IMPLEMENTATION_TARGETS.md](IMPLEMENTATION_TARGETS.md) |

## Research and portfolio aggregates

T155–T168 read the reports retained for the tasks before them in queue order
(this section's earlier tasks included) and never a report of a later task.
Each aggregate is backed by a check through a second path that fails for a
wrong aggregate, never by re-reading what was just written:

- raw-text recounts of the report files (states, finding labels, domains,
  counterexample keys, numerical values without an uncertainty, assumption
  items), located by the retained layout (`runner.dumps`: one-space indent,
  sorted keys) without parsing JSON;
- parsing a written Markdown table back and comparing each row with its source
  finding: task, claim, unit, label, report identity, and a value headline that
  states only numbers of the value (`…(+N)` marks entries left out; numbers are
  never cut);
- for T158, re-executing a declared set of inexpensive figure tasks plus every
  task that declares a wall-clock timing figure (`wall_clock_timing: true` on
  the figure's generated-artifact entry, set by the task when it writes the
  figure) and comparing figure bytes; a declared timing figure is compared
  for presence and structure (series and points) only, and a declared
  rounding-level figure (`rounding_level: true`: values at binary64 rounding
  level, whose last bits follow the BLAS kernel and platform, move it) by the
  plotted values it records: the same series and point counts, x values to
  1e-12 relative and each y value within its recorded rounding bound plus
  1e-12 relative. A declared timing figure that differs in bytes is recorded
  as a counterexample to byte reproducibility, and one that reproduced byte
  for byte is reported; whether a rounding-level figure reproduced byte for
  byte (it does on the kernel of the retained run only) is recorded in
  `figure-index.json` alone, so that T158's findings, labels and prose are the
  same on every kernel; any other figure that differs, a figure no longer
  written, a declared figure whose structure differs and a rounding-level
  figure whose values move beyond their bounds refute it
  (`scripts/check_figures.py` re-executes all of them outside the queue, on
  any platform and forced OpenBLAS kernel, and counts the two declarations
  apart). With a second-platform record bound (`figure-platform-record`: a CI
  run of `scripts/check_figures.py` on another operating system, retained
  under `lab/figure-platforms/`), T158 verifies it (manifest, schemas, summary
  recounted from its figure list, text files giving back the artifact's bytes)
  and refuses one made on the run's own operating system; it counts the
  record's outcomes only over entries whose retained figure is the run's (an
  undeclared figure by its digest, a declared one by its series and points and
  recorded values within their rounding bounds, so that the count is the same
  on every kernel) and whose task source digests, recorded by
  `scripts/check_figures.py`, are those the run's report of the task records.
  A rounding-level figure agrees there within its rounding bounds with the
  record's retained copy, itself within those bounds of the run's figure. The
  record checks are `numerically_verified`, the regeneration on the second
  platform is `provider_backed` (the CI run's outcome as recorded) and a
  mismatch the record reports on a current figure refutes it. A figure
  compared neither by re-execution here nor by a current entry of a valid
  record leaves the task `partial`, and the next step names what keeps it
  partial before a third platform;
- for T168, a static tie analysis (a registered test, or its parametrized
  case, declares its task with a `lab_task` marker and mentions an evidence
  label; markers naming a task that does not register what they mark are
  counted, and, as advisory, tied tasks whose tied tests never name the task),
  itself checked on probe cases, and the JUnit outcomes the reports recorded.

Every count finding declares an exact uncertainty and a zero regression
tolerance. The release digest (T165) covers task states, headline labels,
claims and their labels, never values or artifact bytes, so two clean-room
runs give the same digest although timing figures differ.


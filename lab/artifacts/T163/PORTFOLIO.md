# Computational experimentalist portfolio

Each panel: the hypothesis, the headline findings with their evidence labels, the figure, and what is not established.

## T003 — Verify Euler, midpoint, RK4, and adaptive integrator orders.

On charts with nonzero Christoffel symbols the global endpoint error of Euler, midpoint and RK4 scales like h^1, h^2 and h^4, and the Dormand-Prince error scales like (evaluations)^-5 and like the requested tolerance, which distinguishes it from a pair advancing with y4 ((evaluations)^-4, tol^0.8).

- Explicit Euler global endpoint error converges at order 1 on every curved chart: `(object of 7 entries; see the source report)` → `numerically_verified`
- Explicit midpoint global endpoint error converges at order 2 on every curved chart: `(object of 7 entries; see the source report)` → `numerically_verified`
- Classical RK4 global endpoint error converges at order 4 on every curved chart: `(object of 7 entries; see the source report)` → `numerically_verified`
- Adaptive Dormand-Prince error falls with function evaluations at a median effective order near 5: `5.394386914005626` → `numerically_verified`

![T003](../../artifacts/T003/orders-sphere.svg)

*Physical validation:* `not_established`

## T005 — Verify the Jacobi separation law across positive, zero, and negative curvature.

Along a unit-speed geodesic of constant curvature K the heading Jacobi column is sn_K(s) and the lateral column cn_K(s), so separation oscillates (K > 0), grows linearly (K = 0) or exponentially (K < 0); the torus equators realize K = 1/(r(R+r)) and K = -1/(r(R-r)).

- The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on positive, zero and negative curvature: `(object of 6 entries; see the source report)` → `numerically_verified`
- The full integrated transfer matrix (lateral column and both rates) follows the model-space law cn_K, sn_K: `(object of 6 entries; see the source report)` → `numerically_verified`
- Neighbouring closed-form geodesics on the sphere and hyperbolic plane separate as |sn_K| (heading) and |cn_K| (lateral) per unit perturbation: `(object of 2 entries; see the source report)` → `numerically_verified`
- The torus equators are geodesics of constant curvature 1/(r(R+r)) and -1/(r(R-r)) whose Jacobi columns obey the model-space laws: `(object of 2 entries; see the source report)` → `numerically_verified`

![T005](../../artifacts/T005/heading-column.svg)

*Physical validation:* `not_established`

## T010 — Generate near-focus and post-focus counterexamples.

Where the first-order separation eps j(s) vanishes (conjugate or focal points) the true separation is of higher order in eps, so the relative first-order error diverges there; past the zero the separation changes sign (image inversion); symmetric configurations raise the order of the true separation.

- On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2: `(object of 4 entries; see the source report)` → `numerically_verified`
- On a generic torus geodesic the separation at the conjugate point is O(eps^2) while eps j vanishes: `(object of 3 entries; see the source report)` → `numerically_verified`
- The relative first-order error diverges like 1/|s - s*| approaching the conjugate point: `(object of 3 entries; see the source report)` → `numerically_verified`
- After the conjugate point the separation inverts sign and still follows eps j: `(object of 2 entries; see the source report)` → `numerically_verified`

![T010](../../artifacts/T010/relative-first-order-error.svg)

*Physical validation:* `not_established`

## T032 — Create a counterexample library for “shortest means safest.”

'Shortest means safest' fails on curved surfaces: a shortest geodesic can have larger heading amplification, a smaller focus margin, a tie with another route, or a near-conjugate endpoint, while the flat torus (T021) and simply connected negatively curved surfaces admit no such witness.

- Torus inner equator: the shortest route has larger heading amplification than a longer route: `(object of 2 entries; see the source report)` → `independently_verified`
- Torus outer equator: the shortest route has a smaller focus margin than a longer route: `(object of 2 entries; see the source report)` → `independently_verified`
- Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique: `(object of 2 entries; see the source report)` → `numerically_verified`
- Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point: `(object of 2 entries; see the source report)` → `independently_verified`

*Physical validation:* `not_established`

## T047 — Validate the cylinder chord coefficient analytically and numerically.

On a cylinder of radius R, the geodesic at angle alpha from the circumferential direction has chord deficit s - c = cos^4(alpha) s^3/(24 R^2) + O(s^5): largest circumferentially, zero on rulings, although the Gaussian curvature is zero everywhere.

- The exact helix chord expands as s - cos^4(alpha) s^3/(24 R^2) + (kappa^4/1920 + kappa^2 tau^2/720) s^5 with kappa = cos^2(alpha)/R, tau = sin(alpha)cos(alpha)/R: `(object of 2 entries; see the source report)` → `independently_verified`
- Small-s fits of exact helix chords recover cos^4(alpha)/(24 R^2) at every angle and radius: `(object of 2 entries; see the source report)` → `numerically_verified`
- Geodesics integrated on ciw.lab.surfaces.Cylinder reproduce the exact helix chord and coefficient: `(object of 2 entries; see the source report)` → `numerically_verified`
- Axial rulings (alpha = 90 deg) have zero chord correction; circumferential paths have the largest coefficient 1/(24 R^2): `(object of 2 entries; see the source report)` → `numerically_verified`

![T047](../../artifacts/T047/coefficient-vs-angle.svg)

*Physical validation:* `not_established`

## T062 — Test independent-noise and correlated-noise cases.

NEES/NIS consistency holds for a filter with the correct cross-correlated R and fails for one that drops the cross-covariance; the failure size is predictable in closed form.

- With the correct cross-correlated measurement covariance, run-averaged NEES and NIS lie inside their 99% chi-square intervals and the whitened innovations have identity covariance: `(object of 4 entries; see the source report)` → `numerically_verified`
- Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES exceeds the 99% upper bound at nearly every tick: `(object of 2 entries; see the source report)` → `numerically_verified`
- The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); only the full whitened-innovation covariance test exposes the missing cross-correlation: `(object of 3 entries; see the source report)` → `numerically_verified`
- Monte Carlo grand-mean NEES and NIS of both filters agree with the exact second-moment prediction tr(P_f^-1 E[e e^T]) and tr(S_f^-1 E[nu nu^T]) within 4 run-level standard errors: `(object of 3 entries; see the source report)` → `numerically_verified`

![T062](../../artifacts/T062/anees.svg)

*Physical validation:* `not_established`

## T066 — Verify that filtered residuals use filter covariance, not raw sensor covariance.

Filter residuals must be normalized by the filter's own covariance: S for innovations and R - H P+ H^T for post-fit residuals (the two normalized statistics coincide). The raw sensor covariance R over-states the first and under-states the second by predictable amounts.

- Innovations normalized by S = H P- H^T + R are chi-square(2) consistent: run-averaged NIS lies inside its per-tick 99% interval at 90% or more of ticks and the grand mean matches 2: `(object of 6 entries; see the source report)` → `numerically_verified`
- Post-fit residuals z - H x+ have covariance R - H P+ H^T = R S^-1 R, and normalizing them by it reproduces the innovation NIS exactly, so the post-fit test carries no information beyond the innovation test: `(object of 2 entries; see the source report)` → `numerically_verified`
- Normalizing innovations by the raw sensor covariance R inflates NIS to tr(R^-1 S) and fails the chi-square test at nearly every tick: `(object of 6 entries; see the source report)` → `numerically_verified`
- Normalizing post-fit residuals by the raw R deflates them to tr(S^-1 R) < 2, which would hide an inconsistent filter: `(object of 6 entries; see the source report)` → `numerically_verified`

![T066](../../artifacts/T066/anis_normalizers.svg)

*Physical validation:* `not_established`

## T084 — Mutate replay receipt source digest.

Predicted kills: a source digest edited without resealing (stale replay_id); resealed to a forged digest (the energy workflow rebuilds the receipt verification with subject = source and refuses the stale subject); re-pointed together with its subject at a digest that is not retained, at the replay itself, or at a bundle of another source log (workbench._validate_links requires a retained source bundle of the same kind, source_id and upstream). Predicted survivor: receipt-source.sibling-execution, because a sibling execution of the same source bytes satisfies every one of those checks and every digest is unkeyed.

- Replay receipt source digest forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message: `(list of 5 entries; see the source report)` → `numerically_verified`
- Surviving mutant receipt-source.sibling-execution: a receipt re-pointed, with its verification subject, at a sibling execution of the same bytes reopens: `{"mutant": "receipt-source.sibling-execution", "observed": "accepted"}` → `numerically_verified`
- Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens: `{"surviving_mutants_in_this_task": ["receipt-source.sibling-execution"]}` → `not_established`

*Physical validation:* `not_established`

## T121 — Detect whether GPU reductions alter numerical conclusions.

GPU-style reduction orders keep float32 sums within their order-specific a priori bounds, yet the order differences suffice to flip sign and pass/fail decisions near their boundary; float64 suffers the same with data scaled to its precision.

- Every emulated float32 summation order of the positive dataset stays within its order-specific a priori error bound; relative spread across orders: `(object of 3 entries; see the source report)` → `numerically_verified`
- Every emulated float64 summation order of the same data stays within its order-specific bound; relative spread: `(object of 3 entries; see the source report)` → `numerically_verified`
- Reduction order alone flips the sign of a float32 sum whose exact value is +0.25 (the one-thread sequential CPU fold against the GPU-style orders): `(object of 14 entries; see the source report)` → `numerically_verified`
- float64 accumulation of these float32-valued cancellation inputs is exact in every emulated order (all errors 0), hence sign-correct: `{"exact_zero_errors": 14, "max_abs_error": 0.0}` → `numerically_verified`

![T121](../../artifacts/T121/reduction-errors.svg)

*Physical validation:* `not_established`

## T137 — Rank paths by focus margin.

The focus margin (distance to the nearest focal or conjugate point relative to route length) ranks routes differently from length: the straight route over the dome is the shortest candidate route to the far edge yet has a focal point inside it.

- Candidate coupon routes ranked by focus margin (nearest focal or conjugate point / route length): `(object of 3 entries; see the source report)` → `numerically_verified`
- The shortest candidate route has the worst focus margin: `(object of 3 entries; see the source report)` → `numerically_verified`
- The second variation of route length equals the Jacobi index form j_head(L) j_head'(L): `{"finite_difference_mm": 36.96327031529435, "index_form_mm": 36.963864221406915}` → `numerically_verified`
- The calibration-tolerance ranking and the focus-margin ranking put the straight route at opposite ends: `{"calibration_first": "fan+0deg", "focus_last": "fan+0deg"}` → `numerically_verified`

![T137](../../artifacts/T137/margin-vs-length.svg)

*Physical validation:* `not_established`


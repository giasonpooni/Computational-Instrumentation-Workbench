# Geodesic, Jacobi and route-sensitivity experiments with explicit validity domains

*Generated draft from retained CIW lab reports. Not peer reviewed. Contains no hardware-measured findings.*

## Abstract

This draft summarizes 44 queued computational tasks. Every result below carries the evidence label assigned by `ciw.lab.evidence`; physical validation is not established for any of them.

## Methods

### T001 — Re-derive the geodesic equation for the plane, sphere, cylinder, saddle, torus, and variable-curvature surfaces.

*Hypothesis.* The metric, Christoffel symbols, geodesic equations and Gaussian curvature derived by hand (doc table) and symbolically from each chart's embedding (or intrinsic metric) coincide with the exact-derivative implementation in ciw.lab.surfaces, and the extrinsic and intrinsic curvatures agree.

*Model.* g_ij = X_i . X_j (or the declared intrinsic metric), Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il - d_l g_ij), u''^k = -Gamma^k_ij u'^i u'^j (first variation of length), K from the second fundamental form (LN - M^2)/det g, from the Brioschi formula (sympy) and from the Riemann tensor of finite-differenced Christoffel symbols.

### T002 — Build high-precision reference solutions for each surface.

*Hypothesis.* Every declared standard path has a reference end state accurate far beyond the integrators under test: closed forms on constant-curvature charts and a 34-digit extrapolated integration on the saddle, torus and gaussian bump.

*Model.* Great circle X(s) = cos(s/R) X0 + R sin(s/R) T0; helix and polar lines through the flat development; hyperbolic semicircles; transfer matrices cn_K, sn_K; otherwise the sympy-derived geodesic + Jacobi system integrated by Gragg-Bulirsch-Stoer in mpmath.

### T003 — Verify Euler, midpoint, RK4, and adaptive integrator orders.

*Hypothesis.* On charts with nonzero Christoffel symbols the global endpoint error of Euler, midpoint and RK4 scales like h^1, h^2 and h^4, and the Dormand-Prince error scales like (evaluations)^-5 and like the requested tolerance, which distinguishes it from a pair advancing with y4 ((evaluations)^-4, tol^0.8).

*Model.* Global error e(h) = C h^p + O(h^(p+1)) for a p-th order one-step method on a smooth ODE; for DP5(4) with per-step error control and local extrapolation, h ~ tol^(1/5) and e ~ h^5 ~ tol; advancing with y4 gives e ~ h^4 ~ tol^(4/5).

### T004 — Measure unit-speed drift without hidden renormalization.

*Hypothesis.* Without renormalization the speed invariant g(v, v) drifts at the global order of each method, and a non-unit initial speed is carried unchanged, showing that nothing resets it.

*Model.* g(v, v) is a first integral of the geodesic flow; for a p-th order method its global defect is O(h^p). A renormalizing step would force g(v, v) = 1 whatever the initial speed.

### T005 — Verify the Jacobi separation law across positive, zero, and negative curvature.

*Hypothesis.* Along a unit-speed geodesic of constant curvature K the heading Jacobi column is sn_K(s) and the lateral column cn_K(s), so separation oscillates (K > 0), grows linearly (K = 0) or exponentially (K < 0); the torus equators realize K = 1/(r(R+r)) and K = -1/(r(R-r)).

*Model.* j'' + K j = 0 with j(0) = 0, j'(0) = 1 (heading) and j(0) = 1, j'(0) = 0 (lateral); on a torus of radii R > r the equators theta = 0 and theta = pi are geodesics with K = cos(theta)/(r(R + r cos(theta))); on constant curvature d(gamma_eps(s), gamma_0(s)) = |eps| |j(s)| (1 + O(eps^2)), e.g. sin(d/2) = sin(eps/2) |sin s| on the unit sphere.

### T006 — Compare integrated Jacobi fields with finite-difference flow perturbations.

*Hypothesis.* The integrated lateral and heading Jacobi columns are the derivatives of the geodesic flow with respect to exact lateral displacement and heading rotation of the start.

*Model.* gamma_eps(s) = exp flow of the perturbed start; (gamma_eps - gamma_0)/eps = J + O(eps) and (gamma_eps - gamma_-eps)/(2 eps) = J + O(eps^2), with J = j N and g(J, N) = j.

### T007 — Measure the Wronskian and transfer-matrix determinant.

*Hypothesis.* det Phi(s) = 1 exactly; its numerical drift measures each method: Euler drifts at O(h) through the per-step factor 1 + h^2 K, and RK4 drifts at O(h^4) or better.

*Model.* Phi' = A(s) Phi with A = [[0, 1], [-K, 0]] and tr A = 0 (Liouville). Step matrices: Euler [[1, h], [-h K_n, 1]]; midpoint det = 1 + h^2 (K_mid - K_n)/2 + h^4 K_n K_mid/4; RK4 det - 1 has no h^1..h^5 terms for smooth K (sympy) and equals -h^6 K^3/72 + h^8 K^4/576 for constant K.

### T008 — Locate conjugate and focal points numerically.

*Hypothesis.* Zeros of the heading column are conjugate points and zeros of the lateral column are focal points; on constant K > 0 they sit at multiples of pi/sqrt(K) and half-way between, none exist where K <= 0, and on variable curvature none occurs before pi/sqrt(max K) and each lies between the first zeros of the comparison equations for the upper and lower curvature envelopes along the geodesic.

*Model.* j_head = sn_K, j_lat = cn_K on constant K; Sturm comparison: K <= K_max gives no conjugate point before pi/sqrt(K_max), and K <= 0 gives j_head >= s, j_lat >= 1; the Pruefer angle theta = atan2(j, j') obeys theta' = cos^2 theta + K sin^2 theta, which increases with K, so K_lo(s) <= K(s) <= K_hi(s) gives theta_lo <= theta <= theta_hi.

### T009 — Compare lateral-displacement and heading-perturbation columns separately.

*Hypothesis.* On a constant-K background a curvature change at arclength s moves j_lat(L) with weight sn(L-s) cn(s), which decreases along the path when K <= 0 or sqrt(K) L <= pi/2 (before the first focal distance), and j_head(L) with weight sn(L-s) sn(s), which is symmetric about mid-path for every K and vanishes at both ends; exactly, reversing the curvature profile leaves j_head(L) unchanged. The two columns therefore respond differently to where curvature sits along a path and can order paths differently.

*Model.* Endpoint normal displacement = j_lat(L) delta_perp + j_head(L) delta_alpha; delta j(L) = -int G(L, s) j(s) delta K(s) ds with G(L, s) = j_lat(s) j_head(L) - j_head(s) j_lat(L); reversal maps Phi(L) to D Phi(L)^-1 D; model spaces give j_head/j_lat = tan(sqrt(K)L)/sqrt(K), L, tanh(sqrt(-K)L)/sqrt(-K).

### T010 — Generate near-focus and post-focus counterexamples.

*Hypothesis.* Where the first-order separation eps j(s) vanishes (conjugate or focal points) the true separation is of higher order in eps, so the relative first-order error diverges there; past the zero the separation changes sign (image inversion) and the first-order prediction recovers as |j| grows again; symmetric configurations raise the order of the true separation.

*Model.* Unit-speed geodesics with j'' + K j = 0; d(s; eps) = eps j(s) + r(eps, s). At s* with j(s*) = 0, d(s*) = r(eps, s*). Torus(2, 1) outer equator: K = 1/3, s* = pi sqrt(3); a reflection theta -> -theta makes the normal offset odd in eps and the O(eps^2) along-track lag -eps^2 sin(2 w s)/(4 w) vanish at s*, so d(s*) = O(eps^3). Unit sphere: all great circles through a point refocus exactly at s = pi; the combined perturbation (lateral eps, heading eps) has j = eps (cos s + sin s) with zero at 3pi/4 where the exact chord is eps^2/2 + O(eps^3).

### T011 — Test coordinate-change invariance under nonlinear parameterizations.

*Hypothesis.* A geodesic and its Jacobi fields are geometric: the same initial point and unit tangent give the same embedded endpoint, length and transfer matrix in any chart, up to integration error; the size of that error, however, depends on the chart.

*Model.* Pullback metric g'(a) = J^T g(phi(a)) J with exact Hessian terms (Reparametrized); the initial tangent is t_a = J^{-1} t_u, so the geometric initial data coincide. RK4 local error ~ h^5 y^(5); a near fold u = c + mu a + a^3/3 makes the chart velocity ~ 1/mu over an arclength window ~ mu^(3/2), so the derivatives entering the error grow as mu decreases.

### T012 — Test frame-change invariance under rotations and tangent-basis changes.

*Hypothesis.* Geodesics and Jacobi fields are intrinsic: an ambient rotation changes only the embedded coordinates (which rotate exactly), and the choice of reference basis for headings is a relabeling; an orientation-reversing basis changes the meaning of '+eps' and of the normal together, so a consistently expressed first-order (Jacobi) separation is unchanged; the finite-eps separation is unchanged exactly only where it is odd in eps.

*Model.* Rotated(base, R): X' = R X, so X'_i . X'_j = X_i . X_j and the second fundamental form is unchanged; in exact arithmetic the chart ODE is identical. A basis with f1 = cos(beta) e1 + sin(beta) e2 and f2 = N(f1), the metric +90 degree rotation, is orthonormal, and heading h - beta in it is the unit tangent of heading h; a heading change +/- eps stated in it is the geometric heading perturbation, so (d(+eps) - d(-eps)) / (2 eps) = j_head + O(eps^2) with d measured along N by g(delta u, N). In the left-handed basis (e1, -e2) the heading -h + eps is the right-handed heading h - eps, and its +90 degree normal is -N. With d(eps) = J_eps . N = eps j + C2 eps^2 + ..., the left-handed run measured along -N is -d(-eps) = eps j - C2 eps^2 + ..., so (J_eps . (-N)) equals the right-handed J_eps . N to first order in eps; the eps^2 parts have opposite signs, so the finite separations agree exactly only where the separation is odd in eps, as on the unit sphere (sin(s) sin(eps)), and otherwise own/right - 1 = -2 C2 eps / j + O(eps^2). Measured along -N the left-handed separation is exactly minus its value along N, so its ratio to the right-handed one is minus the ratio along N by construction.

### T013 — Quantify the flat-cylinder/developable-surface limit.

*Hypothesis.* Intrinsic flatness (K = 0) makes Jacobi fields identical to the plane's even when the surface is curved in space, while chords (extrinsic) differ; along the outer equator of a torus with growing major radius the Jacobi deviation from flat vanishes like K L^3/6 ~ 1/R.

*Model.* Cylinder: the second fundamental form has only the phi-phi entry, so K = 0 and j_head = s exactly, but a helix of angle alpha has chord sqrt((2R sin(L cos(alpha)/(2R)))^2 + (L sin(alpha))^2). Torus(R, 1) outer equator: K = 1/(R + 1), j_head = sin(wL)/w, w = sqrt(K), L - j_head = K L^3/6 - K^2 L^5/120 + ...; the equator is a circle of radius R + 1 with chord deficit 2 (R + 1)(x - sin x), x = L/(2 (R + 1)), = L^3/(24 (R + 1)^2) + ...

### T014 — Test geodesic reversal and path truncation.

*Hypothesis.* The geodesic flow is reversible, so integrating forward, flipping velocities and Jacobi derivatives, and integrating again returns to the start up to the method's global error; truncate-and-continue on an identical grid is the same arithmetic as direct integration.

*Model.* With (u, v, j, j') -> (u, -v, j, -j') the flow over L is inverted. For a one-step method with local error C h^(p+1), the step with -h has local error C (-h)^(p+1); the composition cancels at order h^(p+1) when p is even, so the return error is O(h^(p+1)) for even p and O(h^p) for odd p. Linear check: RK4 R(z) R(-z) = 1 + z^6/72 + ..., midpoint 1 + z^4/4, Euler 1 - z^2.

### T015 — Explore long-horizon numerical drift.

*Hypothesis.* Non-symplectic integrators drift in the first integrals of the geodesic flow (speed, Clairaut constant, angular momentum); adaptive local-error control accumulates a linear secular drift, while a fixed step on these closed or quasi-periodic orbits can keep the speed drift bounded over long stretches; bounded speed error does not guarantee the right orbit.

*Model.* Energy g(v, v) = 1, torus Clairaut rho^2 phi' and sphere angular momentum M = X x X' are exact first integrals. On the unit sphere |M| is the speed and M/|M| the normal of the orbit plane, so a position error splits into a cross-track part, bounded by the tilt of M/|M|, and an along-track (phase) part. A speed error advances the point along its circle by the integrated excess arclength int (|X'| - 1) ds; a speed error growing ~ L therefore gives along-track error ~ L^2 (adaptive DP45). A tilt of the orbit plane growing ~ L gives cross-track error ~ L (fixed-step RK4, whose local error rotates the plane secularly while its speed error stays bounded). On Torus(2, 1) the orbit oscillates between |theta| <= arccos((c - R)/r); a Clairaut drift across the separatrix changes it into an orbit winding around the tube.

### T016 — Study stiff behavior on strongly negative-curvature surfaces.

*Hypothesis.* On K = -k^2 the Jacobi fields grow like e^(kL), so absolute errors are amplified by e^(kL) and a fixed relative accuracy needs steps growing with k; this is intrinsic instability of the flow (eigenvalues +k and -k of the Jacobi linearization), not classical stiffness, so A-stable implicit methods do not remove it. Concentrated negative curvature (Saddle with large c) does not produce exponential growth.

*Model.* j'' = k^2 j, j_head = sinh(kL)/k. A one-step method of order p with R(z) = e^z (1 + c z^(p+1) + ...) on the growing mode has relative error N |c| (kh)^(p+1), so N(tau) = L (L |c| k^(p+1) / tau)^(1/p) ~ k^((p+1)/p): RK4 c = -1/120 (k^(5/4)), implicit midpoint c = 1/12 (k^(3/2), pole at kh = 2 and negative factor beyond), 2-stage Gauss-Legendre (the (2, 2) Pade approximant) c = -1/720 (k^(5/4), no real pole). RK4 stability for the decaying mode needs kh <= 2.785. Saddle ridge y = 0: K = -c^2/(1 + c^2 x^2)^2 ~ -1/(4 s^2) away from the saddle point, where j'' = j/(4 s^2) has solutions |s|^((1 +/- sqrt(2))/2); matching the inbound and outbound power laws through the core of width 1/c gives j_head(L) ~ c^sqrt(2) (matched asymptotics, not a proof).

### T017 — Derive validity domains for the first-order approximation.

*Hypothesis.* The first-order remainder r(eps, s) = d(s) - eps |j(s)| is C2(s) eps^2 + C3(s) eps^3 + ...; the first-order prediction is within relative tolerance tau while |C2 eps + C3 eps^2| <= tau |j|, a domain that shrinks to zero where j vanishes unless the remainder vanishes there too.

*Model.* Unsigned separation d at matched arclength. Unit sphere, pure heading: d = 2 arcsin(|sin s| sin(eps/2)), so C2 = 0, C3 = -|sin s| cos^2 s / 24 and eps_max = sqrt(24 tau)/|cos s| (no collapse at s = pi). Hyperbolic plane: d = 2 asinh(sinh s sin(eps/2)), C2 = 0, eps_max = sqrt(24 tau)/cosh s. Generic paths: a normal eps^2 term C2(s*) != 0, so eps_max ~ tau |j'(s*)| |s - s*| / |C2(s*)| -> 0; symmetric paths: C2 = 0 and eps_max ~ sqrt(|s - s*|). Sphere lateral+heading: the eps^2/2 offset at s0 = 3pi/4 is along-track, orthogonal to eps j N, so d = sqrt((eps j)^2 + eps^4/4) near s0 and eps_max = 2 |j| sqrt(2 tau + tau^2) ~ 2 sqrt(2 tau) |j'(s0)| |s - s0|.

### T018 — Generate a report comparing intrinsic curvature effects with integrator error.

*Hypothesis.* A curvature effect in a numerical Jacobi field is meaningful only where it exceeds the integrator error; the working expectation was that weak curvature (small |K| L^3/6) is the hard case for coarse steps and low-order methods.

*Model.* Signal S = |j_head(L) - L| (zero on flat surfaces; |K| L^3/6 for small constant K). Truncation error E(h) ~ C h^p. For j'' + K j = 0 the Euler and midpoint errors are themselves proportional to K (the K-free part j = s is integrated exactly), so S/E is independent of K as K -> 0; the RK4 error enters at O(K^2 h^4), so S/E grows like 1/K. For constant K the Jacobi part of every explicit Runge-Kutta stage is linear, so a run's deviation is the method's step matrix to the power N plus rounding; evaluating that matrix in exact rational arithmetic separates the truncation part from the rounding part exactly. The limit is floating point: when S approaches ulp(L) = 4.4e-16 the computed deviation is rounded away. Resolved when S/E >= 10 for every finer step.

### T019 — Enumerate equivalent lattice representatives.

*Hypothesis.* Every SL(2,Z) change of basis of a lattice reduces, by exact Gauss/Lagrange reduction, to one canonical Gram form; the number of reduced bases equals the stabilizer count predicted by the position of tau in the closed fundamental domain; non-unimodular or orientation-reversing matrices are refused.

*Model.* Lattice Lambda = Z w1 + Z w2 carried by its Gram matrix G = B^T B; basis change B' = B M acts as G' = M^T G M; canonical form |2b| <= a <= c with b >= 0 on the boundary; tau = (b + i sqrt(det G)) / a.

### T020 — Classify geodesics by winding vector.

*Hypothesis.* Closed geodesics of a flat torus are exactly the straight lines in nonzero lattice directions: primitive (m, n) gives a closed geodesic traversed once with length |m w1 + n w2|, (k m', k n') its k-fold cover, and irrational directions never close although their return gaps shrink.

*Model.* Lattice-coordinate flow alpha = alpha0 + m t, beta = beta0 + n t on R^2 / Z^2 with metric Q(m, n) = a m^2 + 2 b m n + c n^2; predicted first return at t = 1/gcd(m, n) with displacement the primitive vector; Q(g m', g n') = g^2 Q(m', n'); intersection number |det(v, w)|; Binet: ||F_k phi|| = phi^-k and |F_k phi^-k - 1/sqrt 5| = phi^-2k / sqrt 5.

### T021 — Compare shortest path against least-sensitive path.

*Hypothesis.* On a flat torus the heading amplification of a route equals its length, so among the geodesics joining two points the shortest is exactly the least heading-sensitive.

*Model.* Geodesics p -> q on C / Lambda lift to segments p -> q + lambda; with K = 0 the Jacobi equation j'' = 0 gives j_head(s) = s and j_lat(s) = 1 along every route.

### T022 — Detect degenerate shortest representatives.

*Hypothesis.* Shortest representatives are tied exactly on the cut locus (the Voronoi boundary of the lattice): multiplicity 2 on edges and 3 or 4 at vertices, with sum over vertices of (k - 2) = 2 for every flat torus; binary64 cannot decide ties below its resolution, and a declared relative tolerance is needed so that exact ties at non-representable targets are not reported as unique.

*Model.* Squared lengths Q(z + lambda) are exact rationals for rational z and integer G; the cut locus of a point is a graph with V vertices of multiplicity k_v and E = sum k_v / 2 edges, and V - E + 1 = chi = 0.

### T023 — Compute sensitivity across all starting headings.

*Hypothesis.* The return distance r(theta; L) vanishes exactly at headings of primitive lattice vectors with |v| <= L, each zero is a V of slope |v| (closing error per unit heading error = loop length), so low-order (short) closed geodesics have the widest closure basins.

*Model.* r(theta; L) = min over lattice points lambda != 0 of dist(lambda, {s u(theta): s_min <= s <= L}); near theta_v = arg v, r = |v| sin|theta - theta_v|, basin {r < eps} has width 2 arcsin(eps/|v|).

### T024 — Add focus-margin-aware route ranking.

*Hypothesis.* Between two points of the torus of revolution the ranking of geodesic routes by length, by heading amplification |j_head(L)| and by focus margin (distance to the first conjugate point) disagree; on a flat torus there are no conjugate points (infinite margin).

*Model.* Geodesic + heading Jacobi system on Torus(R=2, r=1), chart (phi, theta); K = cos theta / (r (R + r cos theta)); focus margin = s_c - L for the first zero s_c > 0 of j_head along the extended geodesic (negative: the route passes a conjugate point and is not locally minimizing); targeting condition 1/|j_head(L)|.

### T025 — Generate Pareto fronts for length, amplification, and focus margin.

*Hypothesis.* Length, amplification and focus margin conflict, so the three-objective Pareto front of the torus routes has several members.

*Model.* Minimize (L, |j_head(L)|, -margin) with censored margins treated as +infinity; a dominates b when no worse in every objective and better in one. The shortest route is on the front by definition (nothing is shorter), so its membership is not a test.

### T026 — Test modular-reduction invariance.

*Hypothesis.* Length spectrum, area and systole are invariant under every SL(2,Z) change of basis and under reduction, exactly for integer Gram forms; winding labels transport by M^-1; orientation and index are not captured by the spectrum alone.

*Model.* G' = M^T G M, det G' = det G, Q'(M^-1 c) = Q(c); spectrum = multiset of Q over lattice vectors within R^2 = 60; systole^2 = first entry of the canonical form.

### T027 — Build polygonal translation-surface examples.

*Hypothesis.* The square-tiled L (3 squares) and the regular octagon with opposite sides glued are genus-2 translation surfaces with one vertex class; their horizontal cylinder decompositions are exactly predictable; non-translation pairings are refused.

*Model.* Surface = convex polygons + edge involution; chi = V - E + F; genus (2 - chi)/2; octagon of side 1 in Q(sqrt 2): area 2 + 2 sqrt 2, horizontal cylinders of circumference 1 + sqrt 2 (height 1) and 2 + sqrt 2 (height sqrt 2 / 2).

### T028 — Trace geodesics across glued edges.

*Hypothesis.* Straight-line flow crosses glued edges by the edge translations: on the square-tiled L every rational direction is completely periodic or ends in a saddle connection (exactly decided), and on the octagon a float trajectory reproduces the exact Q(sqrt 2) trajectory within a declared tolerance.

*Model.* Flow x + t d inside a polygon; exit edge by exact segment intersection; re-entry at x + (a' - b) on the partner edge; a hit on a cone point terminates (saddle connection); closure multiplier k with displacement k (p, q) in Z^2 and 1 <= k <= 3 on the 3-square surface (Veech dichotomy).

### T029 — Detect cone singularities.

*Hypothesis.* Vertex classes after gluing carry cone angles that sum corner angles; translation surfaces have angles 2 pi (k + 1); the octagon has one 6 pi point; Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds with chi known independently of the vertex classes.

*Model.* Union-find on corners (b ~ a', a ~ b' per glued pair), corner angles as exact rational multiples of pi; origami cone points = cycles of the commutator r u r^-1 u^-1 (angle 2 pi times length) and chi = -sum (c - 1) (Riemann-Hurwitz); declared chi for the polygon gluings (genus-2 octagon, tori, sphere). With chi = V - E + F from the same union-find, Gauss-Bonnet holds for every partition of the corners (sum theta_v = pi sum_f (n_f - 2)), so only the comparison with an independent chi tests the classes.

### T030 — Compare smooth and discrete geodesic approximations.

*Hypothesis.* Grid graph shortest paths do not converge to the Euclidean geodesic length under refinement: the relative error is set by direction (sqrt 2 - 1 = 41.4% at 45 deg for 4 neighbours, sqrt(4 - 2 sqrt 2) - 1 = 8.24% near 22.5 deg for 8 neighbours) and is independent of the spacing, whereas fast marching converges.

*Model.* Graph distance -> stencil norm |dx| + |dy| (4-nbr) or max + (sqrt 2 - 1) min (8-nbr); eikonal |grad T| = 1 solved by first-order fast marching (error O(h log 1/h) for a point source).

### T031 — Study route changes under small metric perturbations.

*Hypothesis.* Near a tie, an arbitrarily small constant metric perturbation g = I + eps h switches the shortest route discontinuously: the switch occurs at eps* = 4 delta for a target delta away from the tie, the heading jumps by about 127 deg, and the minimal length stays continuous.

*Model.* Flat torus Z^2 with g = I + eps h, h = [[0, 1], [1, 0]]; Q_eps(x) = |x|^2 + 2 eps x1 x2; routes a = (1/2 - delta, 1/4) and b = a - (1, 0) tie when eps* = (|b|^2 - |a|^2) / (a^T h a - b^T h b) = 4 delta.

### T032 — Create a counterexample library for “shortest means safest.”

*Hypothesis.* 'Shortest means safest' fails on curved surfaces: a shortest geodesic can have larger heading amplification, a smaller focus margin, a tie with another route, or a near-conjugate endpoint, while the flat torus (T021) and simply connected negatively curved surfaces admit no such witness.

*Model.* Heading amplification |j_head(L)|, targeting condition 1/|j_head(L)| and focus margin s_c - L from j'' + K j = 0 along each route; exact anchors: inner equator K = -1 (j_head = sinh s), outer equator K = 1/3 (conjugate at pi sqrt 3), unit sphere (j_head = sin s, conjugate at pi).

### T033 — Implement a reusable metric/Christoffel/curvature surface interface.

*Hypothesis.* Every catalogue surface, the reparametrized charts plane-polar and gaussian-bump-shear and a rigidly rotated torus satisfy the interface identities on their declared domains, and the Gauss equation holds: curvature recomputed from the metric alone equals the supplied K.

*Model.* g symmetric positive definite; Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il - d_l g_ij) symmetric in ij; d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il; d_l d_k g_ij symmetric in kl; Brioschi K(E, F, G and their first and second derivatives) = supplied K.

### T034 — Add automatic differentiation or symbolic derivative checks.

*Hypothesis.* At seeded points of every conformance surface, the hand-coded metric, metric derivatives, Christoffel symbols and Gaussian curvature equal those of its closed-form embedding (or metric) computed by an independent symbolic system (sympy differentiation, and sympy.diffgeom for the connection and curvature of seven surfaces) and by forward-mode automatic differentiation; for those seven surfaces the symbolic curvature equals the declared closed form exactly.

*Model.* Re-expressed X(u) per surface with exact rational parameters; g = X_i . X_j; dg by differentiation; sympy.diffgeom Gamma from metric_to_Christoffel_2nd and K = g_0m R^m_101 / det g from metric_to_Riemann_components; ciw-assembled Gamma and R_1212 / det g from sympy derivatives; dual-number K from Brioschi with exact second derivatives and from LN - M^2.

### T035 — Compare analytic and finite-difference metric derivatives.

*Hypothesis.* Central differences of the metric approach the analytic dg as h^2 until rounding, which grows as eps/h, takes over; the optimum lies near h* = (3 eps |g| / |d^3 g|)^(1/3) ~ eps^(1/3).

*Model.* D_h g = (g(u + h e_k) - g(u - h e_k)) / 2h = d_k g + h^2/6 d_k^3 g + O(h^4); rounding error ~ eps |g| / h; total minimized at h* with error ~ eps^(2/3).

### T036 — Add chart-transition support.

*Hypothesis.* An atlas of charts with exact transitions through the embedding integrates geodesics through or near a chart's coordinate singularity with an accuracy that does not depend on how close they pass, while a single chart loses accuracy or fails near its singular point. Shown for the two-chart polar atlas of the sphere (great circles through and near a pole) and for the Monge-plus-polar atlas of the Gaussian bump built from a ChartMap (geodesics through and near the apex).

*Model.* Atlas: charts X_c with closed-form inverses; u_B = X_B^{-1}(X_A(u_A)), v_B = g_B^{-1} J_B^T J_A v_A; regularity lambda_min / lambda_max of g (scale free); switch between RK4 steps when the active chart's regularity < 1/4. Sphere: chart A X = R(sin t cos p, sin t sin p, cos t), chart B = R_y(pi/2) X_A; regularity sin^2 theta = det g / R^4 in each chart and sin^2 theta_A + sin^2 theta_B = 1 + y^2/R^2 >= 1, so the better chart always has regularity >= 1/2. Graph z = f(x, y): Monge chart g = I + grad f grad f^T with regularity 1/(1 + |grad f|^2), and the polar chart (r, t) pulled back through PolarChart, singular at r = 0.

### T037 — Detect coordinate singularities.

*Hypothesis.* A scan into a candidate point, along approach loops that are preimages of geodesic circles, separates coordinate singularities (det g -> 0 or cond g -> infinity with bounded K and circumference ratio 1) from conical points (circumference deficit above 1e-6), curvature singularities (|K| ~ r^a with a <= -0.05) and boundaries at infinite distance (radial speed ~ r^b with b <= -1 + 1e-3); beyond these detection limits it misclassifies, and it leaves a metric blow-up at finite distance with bounded K unclassified even when that is a removable coordinate singularity. A pointwise guard refuses points whose metric condition number or |K| exceeds its declared bounds, with codes.

*Model.* Fit power laws r^a on r in [1e-8, 1e-5] for det g, cond(g), max|Gamma|, |K| and the radial speed |dX/dr|; a fit is clean when its max log residual is <= 0.05. Radial distance int r^b dr diverges iff b <= -1; circumference ratio C(r) / (2 pi rho(r)) -> 1 at smooth points and sin(alpha) at a cone apex.

### T038 — Build a triangle-mesh intrinsic geodesic solver.

*Hypothesis.* Unfolding across edges (straight in faces, equal angles at edges) yields exact straightest geodesics on developable meshes, and Steiner-graph paths are surface paths (every graph edge lies in one face), so their lengths are upper bounds on the polyhedral distance; the polyhedral distance itself is not computed here.

*Model.* Straightest geodesic: in each face a straight segment; at an edge the direction keeps its edge component and the magnitude of its perpendicular component (rotation about the edge). Distances: Dijkstra on the edge graph and on the graph of k Steiner points per edge (all pairs inside each face); vertex hits are refused.

### T039 — Measure convergence under mesh refinement.

*Hypothesis.* Straightest mesh geodesics converge to smooth geodesics; the length (metric) error is O(h^2) on inscribed meshes, the lateral error decreases at least like O(h) but not as a clean power law, and edge-graph distances do not converge.

*Model.* Inscribed icosphere: chord/arc metric error O(h^2). Lateral error of a straightest geodesic is driven by the imbalance of vertex curvature point masses on either side of the path (a discrepancy sum), so its order is not a single power law. Prism cylinder: development circumference 2nR sin(pi/n) gives L cos(alpha) (a / sin a - 1) ~ L cos(alpha) pi^2 / (6 n^2), plus a chord-position term atan(s tan a) - s a of relative size O(1/n). Edge graph at a valence-5 source: floor sqrt(5) - 2.

### T040 — Compare smooth and mesh Jacobi approximations.

*Hypothesis.* Mesh Jacobi fields and angle-defect curvature approximate the smooth ones only in the right order of limits and at vertices whose stars become regular.

*Model.* Smooth: j'' + K j = 0, j_head(s) = sin(s) for K = 1. Mesh: curvature is concentrated at vertices (cone points); two geodesics are rotated relative to each other only when a vertex lies between them, so the finite-difference field depends on delta/h. For vertices on a sphere the angle defect tends to K times the circumcentric (Voronoi) cell area, so defect/(A/3) tends to K Voronoi/(A/3): 3K / (4 cos^2(pi/n)) at regular valence-n stars, and a value other than K wherever the star stays irregular.

### T041 — Quantify mesh-quality effects.

*Hypothesis.* Under isotropic tangential jitter of the icosphere at fixed vertex count, worse triangle quality comes with larger curvature error; within the latitude-longitude family and across families quality metrics do not order the errors pairwise (and the barycentric estimator's cross-family ranking reverses with the mixed Voronoi area); the maximum radius ratio is positively rank-correlated with the Voronoi-area curvature and geodesic errors over the pooled meshes, an association carried by the jitter family that vanishes within the latitude-longitude family; the dihedral fold check is neither necessary nor sufficient for an inverted face; Hausdorff convergence does not imply convergence of area, distances, normals or mean curvature.

*Model.* Quality: minimum angle and radius ratio R_circ / (2 r_in). Curvature: angle defect over the barycentric area A/3 or the mixed Voronoi area. Schwarz lantern with m = q n^2 bands: face height sqrt((H/m)^2 + R^2 (1 - cos(pi/n))^2), so area and developed height tend to sqrt(1 + (pi^2 q R / 2H)^2) times their cylinder values while d_H = R (1 - cos(pi/n)) -> 0.

### T042 — Define refusal states for invalid or incomplete surface data.

*Hypothesis.* Each declared defect class in the mesh, trace and query catalogues (MESH_CODES, TRACE_CODES, QUERY_CODES) is detected before geometry is computed and reported under a stable name (inverted_face only for a mesh declared star-shaped about a centre), and valid meshes pass; defects outside the catalogue, including inverted faces of meshes without a declared centre whose bends stay below the fold threshold, are not claimed.

*Model.* Validation order: invalid_shape, empty_mesh, nonfinite_vertex, invalid_face_index, degenerate_face, non_manifold_edge, inconsistent_orientation, non_manifold_vertex, folded_face, inverted_face, unreferenced_vertex, disconnected_components, open_boundary. Tracing refusals: point_outside_face, invalid_direction, boundary_reached, vertex_hit, step_budget_exceeded. Query refusals: unreachable_target, boundary_vertex_curvature, mesh_too_large, invalid_strip.

### T043 — Add uncertainty on vertices, normals, and curvature.

*Hypothesis.* Gaussian vertex noise propagates to geodesic length and normals nearly linearly, while angle-defect curvature needs sigma << h^2 / R for linearization and becomes noisier as the mesh is refined; linearized per-vertex normal and curvature standard deviations under a declared vertex covariance match Monte Carlo at every vertex in the linear regime; the fixed-corridor distance stays meaningful only below a strip-dependent noise level that the vertex margin alone does not order.

*Model.* Linearization Var[f] = sigma^2 |grad f|^2 (or sum_j J_j C_j J_j^T for a declared per-vertex covariance C_j) with a central finite-difference Jacobian (step 1e-6) over the vertices that affect f; seeded Monte Carlo with the same noise. Marker distance = planar distance after unfolding a fixed face strip with barycentric markers.

### T044 — Test geometry uncertainty separately from sensor uncertainty.

*Hypothesis.* For a marker-distance observation on an uncertain surface, geometry and sensor variances add (law of total variance), can be separated by a nested design, and averaging sensor readings cannot remove the geometry part; which part dominates depends on the geometry noise model as well as on the sigmas.

*Model.* Residual r = y - d(V_nominal), y = d(V_nominal + eta) + eps, eta ~ N(0, sigma_g^2 I) per vertex coordinate (or sigma_g along each vertex normal), eps ~ N(0, sigma_s^2) independent. Var(r) = sigma_s^2 + Var_eta(d) ~ sigma_s^2 + sigma_g^2 |grad d|^2, with |grad_n d|^2 in place of |grad d|^2 for normal-only noise. Nested design: within-group variance estimates sigma_s^2, corrected between-group variance estimates the geometry part.

## Results

| Task | Finding | Value | Unit | Evidence | Report |
| --- | --- | --- | --- | --- | --- |
| T001 | Sympy-derived metrics, Christoffel symbols and geodesic equations match ciw.lab.surfaces on nine charts | {"cylinder": 1.1102230246251565e-16, "cylinder-polar": 2.220446049250313e-16, "gaussian-bump": 1.474514954580286e-17, "hyperbolic-plane": 2.048224298943313e-16 …(+5)} |  | `independently_verified` | `sha256:6e89fe0f06a5` |
| T001 | Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts | {"cylinder": 0.0, "cylinder-polar": 0.0, "gaussian-bump": 1.0408340855860843e-17, "hyperbolic-plane": 0.0 …(+5)} |  | `independently_verified` | `sha256:6e89fe0f06a5` |
| T001 | The hand-derived metrics, Christoffel symbols, geodesic equations and curvatures of the section doc, as transcribed in hand_geometry, match ciw.lab.surfaces on nine charts | {"cylinder.acceleration": 8.250512182813516e-18, "cylinder.christoffel": 2.349193846774299e-17, "cylinder.curvature": 0.0, "cylinder.metric": 1.1102230246251565e-16 …(+32)} |  | `numerically_verified` | `sha256:6e89fe0f06a5` |
| T001 | ciw.lab.surfaces Christoffel symbols are symmetric and metric-compatible and its exact metric derivatives match finite differences | {"cylinder.compatibility": 6.162975822039155e-33, "cylinder.metric_derivatives": 1.1102260975773557e-11, "cylinder.symmetry": 0.0, "cylinder-polar.compatibility": 4.440892098500626e-16 …(+23)} |  | `numerically_verified` | `sha256:6e89fe0f06a5` |
| T001 | Intrinsic curvature from finite-differenced Christoffel symbols matches the ciw Gaussian curvature | {"cylinder": 0.0, "cylinder-polar": 1.964112847374832e-12, "gaussian-bump": 4.406477890905869e-10, "hyperbolic-plane": 1.1751106088198071e-08 …(+5)} |  | `numerically_verified` | `sha256:6e89fe0f06a5` |
| T001 | The second-fundamental-form curvature (LN - M^2)/det g equals the intrinsic curvature on the six embedded charts (Theorema Egregium) | {"cylinder.extrinsic_vs_brioschi": 0.0, "cylinder.extrinsic_vs_closed_form": 0.0, "cylinder.extrinsic_vs_intrinsic": 0.0, "gaussian-bump.extrinsic_vs_brioschi": 1.734723475976807e-17 …(+14)} |  | `independently_verified` | `sha256:6e89fe0f06a5` |
| T001 | Nonzero Christoffel symbols do not imply curvature: the polar charts of the plane and cylinder are flat | {"cylinder-polar.max_abs_christoffel": 2.3849089821188505, "cylinder-polar.max_abs_intrinsic_curvature": 1.964112847374832e-12, "plane-polar.max_abs_christoffel": 2.4581872115706753, "plane-polar.max_abs_intrinsic_curvature": 8.742192596148665e-13} |  | `numerically_verified` | `sha256:6e89fe0f06a5` |
| T001 | An extrinsically curved surface can have identically zero Christoffel symbols: the cylinder in (phi, z) | {"max_abs_christoffel": 2.349193846774299e-17, "normal_curvature_phi": -1.0} |  | `numerically_verified` | `sha256:6e89fe0f06a5` |
| T002 | ciw Richardson RK4 end states match closed-form geodesics and transfer matrices on the six closed-form charts | {"cylinder": 1.8491463559123068e-14, "cylinder-polar": 2.5084292464429545e-14, "hyperbolic-plane": 2.871617124444836e-14, "plane": 2.220446049250313e-14 …(+2)} |  | `numerically_verified` | `sha256:5b78952a7bf8` |
| T002 | ciw Richardson RK4 matches a 34-digit integration of the sympy-derived equations on the saddle path | {"ciw_minus_reference": 9.222205069512407e-16, "reference_end_state[0]": 1.1281533540905775, "reference_end_state[1]": 0.85376046416804, "reference_end_state[2]": 0.6910137583919217 …(+5)} |  | `independently_verified` | `sha256:5b78952a7bf8` |
| T002 | ciw Richardson RK4 matches a 34-digit integration of the sympy-derived equations on the torus path | {"ciw_minus_reference": 6.378231276471524e-14, "reference_end_state[0]": 1.157398536397543, "reference_end_state[1]": 1.233412559238834, "reference_end_state[2]": 0.4050492375133295 …(+5)} |  | `independently_verified` | `sha256:5b78952a7bf8` |
| T002 | ciw Richardson RK4 matches a 34-digit integration of the sympy-derived equations on the gaussian-bump path | {"ciw_minus_reference": 2.3309462240877263e-14, "reference_end_state[0]": 1.2222127316295845, "reference_end_state[1]": 0.6630040047245069, "reference_end_state[2]": 0.9666549409931171 …(+5)} |  | `independently_verified` | `sha256:5b78952a7bf8` |
| T002 | Clairaut's integral rho^2 phi' is conserved along the torus reference path | {"ciw_richardson": 6.750155989720952e-14, "ciw_rk4_100_max": 3.3145450828442335e-09, "ciw_rk4_200_max": 2.0641754971961745e-10, "ciw_rk4_50_max": 5.339766140366464e-08 …(+1)} |  | `numerically_verified` | `sha256:5b78952a7bf8` |
| T003 | Explicit Euler global endpoint error converges at order 1 on every chart with nonzero Christoffel symbols | {"cylinder-polar": 1.0067496119764763, "gaussian-bump": 1.0027342687578236, "hyperbolic-plane": 1.007283300174659, "plane-polar": 1.0071050564304065 …(+3)} |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | Explicit midpoint global endpoint error converges at order 2 on every chart with nonzero Christoffel symbols | {"cylinder-polar": 1.9959681817660186, "gaussian-bump": 1.9815812770314443, "hyperbolic-plane": 2.0034284091119954, "plane-polar": 2.007631064911508 …(+3)} |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | Classical RK4 global endpoint error converges at order 4 on every chart with nonzero Christoffel symbols | {"cylinder-polar": 4.008735600037563, "gaussian-bump": 3.888017719262526, "hyperbolic-plane": 3.965312718796343, "plane-polar": 3.9528800226496705 …(+3)} |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | Adaptive Dormand-Prince error falls with function evaluations at a median effective order near 5 | 5.29791743787106 |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | Adaptive Dormand-Prince endpoint error is proportional to the requested tolerance (median over charts) | 0.9710731803196047 |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | The adaptive-order checks reject a Dormand-Prince variant that advances with its fourth-order solution | {"median_effective_order": 4.189750093080622, "median_tolerance_exponent": 0.7826906298237313} |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T003 | No convergence order is observable on flat Cartesian charts: every method is exact to rounding there | {"cylinder": 2.0945586572537882e-14, "plane": 2.1111760027289807e-14} |  | `numerically_verified` | `sha256:51f1f6e5cc68` |
| T004 | Unit-speed drift max\|g(v,v) - 1\| scales like h^p for Euler, midpoint and RK4 on every chart with nonzero Christoffel symbols | {"euler.cylinder-polar": 1.006764851696603, "euler.gaussian-bump": 1.0026927194942863, "euler.hyperbolic-plane": 1.0129999457858216, "euler.plane-polar": 0.9784061511020454 …(+17)} |  | `numerically_verified` | `sha256:2d58959c07c6` |
| T004 | A non-unit initial speed stays non-unit: g(v,v) remains 1.69 to integrator accuracy | {"hyperbolic-plane.adaptive": 1.9225732117433836e-11, "hyperbolic-plane.rk4": 4.0646286336709636e-10, "sphere.adaptive": 5.97679683522756e-11, "sphere.rk4": 2.1377283720980245e-08 …(+2)} |  | `numerically_verified` | `sha256:2d58959c07c6` |
| T004 | The integrator and geodesic/Jacobi right-hand-side code paths contain no state normalization | {"normalization_calls": 0, "exponential_growth.adaptive_1e-10": 4.561601168962278e-11, "exponential_growth.rk4_64": 1.548614416264312e-08} |  | `numerically_verified` | `sha256:2d58959c07c6` |
| T004 | Unit speed does not certify an accurate path: renormalized Euler keeps \|g - 1\| at rounding with a first-order endpoint error | {"plain_endpoint_error": 0.014891199556742403, "plain_max_speed_drift": 0.007682993302337793, "renormalized_endpoint_error": 0.013938653977328111, "renormalized_error_order": 1.0052159069590567 …(+2)} |  | `numerically_verified` | `sha256:2d58959c07c6` |
| T004 | Speed drift is at rounding on flat Cartesian charts, where every method is exact | {"cylinder": 2.220446049250313e-16, "plane": 0.0} |  | `numerically_verified` | `sha256:2d58959c07c6` |
| T005 | The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on positive, zero and negative curvature | {"cylinder": 2.803593496528173e-15, "hyperbolic-long": 1.2561964420119912e-09, "plane": 2.0441663628976168e-15, "sphere-great-circle": 2.6582168133337802e-09 …(+2)} |  | `numerically_verified` | `sha256:740e1c074f2c` |
| T005 | The full integrated transfer matrix (lateral column and both rates) follows the model-space law cn_K, sn_K | {"cylinder": 2.803593496528173e-15, "hyperbolic-long": 1.2561964420119912e-09, "plane": 2.0441663628976168e-15, "sphere-great-circle": 2.6582168133337802e-09 …(+2)} |  | `numerically_verified` | `sha256:740e1c074f2c` |
| T005 | Neighbouring closed-form geodesics on the sphere, the plane (seen in its polar chart) and the hyperbolic plane separate as \|sn_K\| (heading) and \|cn_K\| (lateral) per unit perturbation (K = 1, 0, -1) | {"hyperbolic-long.heading.error_at_smallest_eps": 0.0004218476128911917, "hyperbolic-long.heading.order": 1.987959844703613, "hyperbolic-long.lateral.error_at_smallest_eps": 0.00041768199824695123, "hyperbolic-long.lateral.order": 1.9878670145339832 …(+8)} |  | `numerically_verified` | `sha256:740e1c074f2c` |
| T005 | The torus equators are geodesics of constant curvature 1/(r(R+r)) and -1/(r(R-r)) whose Jacobi columns obey the model-space laws | {"torus-inner-equator.curvature": -1.0, "torus-inner-equator.curvature_drift": 0.0, "torus-inner-equator.max_error": 2.6915911645163793e-09, "torus-inner-equator.theta_drift": 0.0 …(+4)} |  | `numerically_verified` | `sha256:740e1c074f2c` |
| T005 | Residuals against the model-space law are fourth-order RK4 discretization error | {"hyperbolic-long": 3.982068546159354, "sphere-great-circle": 4.002782339723584, "torus-inner-equator": 3.982004180907906, "torus-outer-equator": 4.001618058911858} |  | `numerically_verified` | `sha256:740e1c074f2c` |
| T005 | ciw joint geodesic + Jacobi transfer matrices match the pinned CSG provider on six constant-curvature paths | {"cylinder.ciw_rk4_vs_csg_closed_form": 2.803593496528173e-15, "cylinder.ciw_rk4_vs_csg_rk4": 2.803593496528173e-15, "hyperbolic-long.ciw_rk4_vs_csg_closed_form": 1.2561964420119912e-09, "hyperbolic-long.ciw_rk4_vs_csg_rk4": 9.276187254982186e-16 …(+8)} |  | `independently_verified` | `sha256:740e1c074f2c` |
| T005 | Nearby real trajectories on a physical curved surface separate according to this Jacobi law | null |  | `not_established` | `sha256:740e1c074f2c` |
| T006 | Central finite differences of perturbed geodesics converge to the integrated Jacobi columns at second order | {"gaussian-bump.heading": 1.998409092931523, "gaussian-bump.lateral": 1.987230580205772, "sphere.heading": 2.001214536357786, "sphere.lateral": 1.9996977123619228 …(+2)} |  | `numerically_verified` | `sha256:feb2a7431684` |
| T006 | One-sided finite differences converge to the integrated Jacobi columns at first order | {"gaussian-bump.heading": 0.9483507350173194, "gaussian-bump.lateral": 0.9975089318172824, "sphere.heading": 0.9740682278971411, "sphere.lateral": 1.0243627244167715 …(+2)} |  | `numerically_verified` | `sha256:feb2a7431684` |
| T006 | Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation) | {"log10_best_eps": -7.0, "log10_error_ratio_1e-11_over_best": 3.942057165537953} |  | `numerically_verified` | `sha256:feb2a7431684` |
| T007 | Explicit Euler multiplies det Phi by exactly 1 + h^2 K(gamma_n) per step, so it is not area-preserving where K is nonzero | {"constant_curvature_prediction_error": 1.8084274401897394e-14, "per_step_factor_error": 4.824514648044523e-16, "variable_curvature_orders.bump-radial": 1.0229902664953436, "variable_curvature_orders.gaussian-bump": 0.9954741092263205 …(+3)} |  | `numerically_verified` | `sha256:c87c8c8bbcff` |
| T007 | Midpoint determinant drift is (h^2/4)(K(L) - K(0)) + O(h^3): second order on variable curvature, third order on constant curvature | {"boundary_ratio.bump-radial": 0.9986897812163655, "boundary_ratio.gaussian-bump": 0.9569206039988852, "boundary_ratio.saddle": 1.0035463186788571, "boundary_ratio.torus": 1.003113160361475 …(+15)} |  | `numerically_verified` | `sha256:c87c8c8bbcff` |
| T007 | RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable curvature | {"bump-radial": 4.989849579481997, "gaussian-bump": 4.992984430686957, "hyperbolic-long": 5.001956250322552, "saddle": 4.997545820568775 …(+5)} |  | `numerically_verified` | `sha256:c87c8c8bbcff` |
| T007 | Adaptive Dormand-Prince determinant drift decreases in proportion to the tolerance or faster (median over paths) | 1.027724743253913 |  | `numerically_verified` | `sha256:c87c8c8bbcff` |
| T007 | Where K = 0 every method preserves det Phi = 1 exactly, Euler included | 0.0 |  | `numerically_verified` | `sha256:c87c8c8bbcff` |
| T008 | Sphere conjugate points lie at pi R and 2 pi R and focal points at pi R/2 and 3 pi R/2 (R = 1 and R = 2) | {"R1.conjugate[0]": 3.1415926595845836, "R1.conjugate[1]": 6.283185319164188, "R1.focal[0]": 1.57079632979444, "R1.focal[1]": 4.712388989374272 …(+4)} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | On the torus outer equator conjugate points lie at pi sqrt(r(R+r)) multiples and focal points half-way | {"analytic_first_conjugate": 5.441398092702654, "conjugate[0]": 5.441398101104003, "conjugate[1]": 10.882796202214125, "focal[0]": 2.720699050555387 …(+1)} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | No conjugate or focal point occurs on the torus inner equator or the hyperbolic plane (K < 0) | {"hyperbolic-long.conjugate_count": 0, "hyperbolic-long.focal_count": 0, "hyperbolic-long.max_1_minus_j_lat": -0.00011250210937507887, "hyperbolic-long.max_s_minus_j_head": -5.625000000022973e-07 …(+4)} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | Sturm comparison bound holds on every seeded torus geodesic that reaches a conjugate point (none occurs before pi/sqrt(max K)) and rejects a heading column integrated with 2K | {"bound": 5.441398092702654, "control_margin": -0.9330514219119008, "margin": 2.34362035008761, "paths_with_conjugate_point": 2 …(+2)} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | Two-sided Sturm comparison with piecewise-constant curvature envelopes brackets the heading column on every seeded torus and bump geodesic and declared bump chord, and rejects the column integrated with 2K | {"chords_with_conjugate_point": 4, "largest_control_gap": -0.00323060846611678, "smallest_gap_above_lower_envelope": -5.463074437273008e-11, "smallest_gap_below_upper_envelope": 0.0 …(+18)} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | On variable curvature the first focal point is not half the first conjugate distance | {"conjugate": 7.785018442790264, "focal": 3.465336236529439} |  | `numerically_verified` | `sha256:466ee619217e` |
| T008 | ciw conjugate and focal points match the pinned CSG provider's focus events on constant-curvature paths | {"hyperbolic-long.closed_form_gap": 0.0, "hyperbolic-long.numeric_gap": 0.0, "sphere-great-circle.closed_form_gap": 2.620059724733892e-09, "sphere-great-circle.numeric_gap": 2.220446049250313e-16 …(+4)} |  | `independently_verified` | `sha256:466ee619217e` |
| T009 | Endpoint sensitivities to lateral offset (\|j_lat(L)\|) and heading error (\|j_head(L)\|) per path | {"bump-radial.heading": 2.9455288379948854, "bump-radial.lateral": 0.6771972872846138, "cylinder.heading": 3.0000000000000027, "cylinder.lateral": 1.0 …(+24)} |  | `numerically_verified` | `sha256:df2c19647cf9` |
| T009 | Lateral and heading sensitivities rank paths differently | {"discordant_pairs": 33, "kendall_tau": 0.26373626373626374, "witness_reversal_margin": 0.946371351997063} |  | `numerically_verified` | `sha256:df2c19647cf9` |
| T009 | On a flat background, to first order, curvature at arclength s moves j_lat(L) with weight L - s (early-weighted) and j_head(L) with weight s(L - s) (symmetric about mid-path) | {"first_order_relative_error": 5.6656251856357365e-05, "heading_early_late_asymmetry": 1.742387244899522e-11, "lateral_early_over_late": 4.936857582505496, "responses.0.5.heading": -0.0006418107507686344 …(+5)} |  | `numerically_verified` | `sha256:df2c19647cf9` |
| T009 | Reversing a geodesic leaves j_head(L) unchanged and exchanges j_lat(L) with j_head'(L) (transfer matrix D Phi(L)^-1 D) | {"gaussian-bump.heading_change": 2.078337502098293e-13, "gaussian-bump.prediction_error": 1.8096635301390052e-12, "saddle.heading_change": 3.441691376337985e-14, "saddle.prediction_error": 1.9320101074526974e-12 …(+4)} |  | `numerically_verified` | `sha256:df2c19647cf9` |
| T009 | The ranking of lateral versus heading start errors computed here predicts which error dominates the endpoint error of real tool or vehicle paths on physical curved parts | null |  | `not_established` | `sha256:df2c19647cf9` |
| T010 | On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2 | {"chord_exponent": 3.0003126899802814, "grid_relative_difference": 0.00015218413314643797, "signed_exponent": 3.000108734140718, "chord_at_s_star[0]": 8.502363791585313e-08 …(+3)} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | On a generic torus geodesic the separation at the conjugate point is O(eps^2) while eps j vanishes | {"chord_exponent": 2.023175167643266, "s_star": 6.051400589139297, "chord_at_s_star[0]": 3.82648454732879e-05, "chord_at_s_star[1]": 0.00015412037682610428 …(+2)} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | The relative first-order error diverges like 1/\|s - s*\| approaching the conjugate point | {"relative_error_at_s_star_minus_h_eps_0.04": 3.7958513020004787, "slope_equator_eps_0.04": -1.0119419484943362, "slope_generic_eps_0.01": -0.991200688074655} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | After the conjugate point the separation inverts sign and still follows eps j | {"equator.j_after": -1.224744871238089, "equator.j_before": 1.7320508075683134, "equator.ratio_after": 0.999927996957514, "equator.s_after": 6.801747615878317 …(+10)} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | Past the conjugate point the first-order prediction recovers: the relative error falls again like 1/\|s - s*\| | {"relative_error_at_1.25_s_star_eps_0.04": 0.018182546475165935, "relative_error_at_s_star_plus_h_eps_0.04": 3.7751136377575443, "slope_after_equator_eps_0.04": -0.9895857275963416, "slope_after_generic_eps_0.01": -1.010401090002985} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | Separation does not grow monotonically with length: it nearly vanishes at the conjugate point | 0.00015710984529684768 |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | On the unit sphere a pure heading perturbation refocuses exactly: the relative first-order error is uniform and does not diverge at the conjugate point | {"max_deviation_from_uniform": 7.656014666679312e-08, "relative_error_at_pi_minus_h": -1.6590023186879854e-05, "uniform_relative_error": -1.6666583333546647e-05} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | On the unit sphere the image inverts after the conjugate point (signed ratio -1) | -0.999999996944685 |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T010 | A combined lateral+heading perturbation on the sphere has true separation eps^2/2 where eps j vanishes | {"coefficient": 0.4999989583372256, "exponent": 1.9999395863581884, "s0": 2.356194490192345} |  | `numerically_verified` | `sha256:4ce28cbf27b4` |
| T011 | Converged geodesic endpoints, lengths and Jacobi transfer matrices agree in every chart | {"max_adaptive_error": 1.3217071881399534e-10, "max_length_error": 1.925495318744197e-10, "by_reference.analytic": 1.3217071881399534e-10, "by_reference.self_convergence": 1.3090595274434236e-10} |  | `numerically_verified` | `sha256:579fff6484c4` |
| T011 | Between N = 128 and 256 RK4 error falls at least like h^3.7 in every chart; smooth charts show order 4, near-fold charts are still pre-asymptotic (order above 4) at N = 256 | {"min_fold_order": 4.430712529325345, "min_order": 3.9497846937534242, "min_order_N_64_128": 3.676474289394472, "min_order_N_64_128_chart": "plane: near-fold mu=0.1" …(+17)} |  | `numerically_verified` | `sha256:579fff6484c4` |
| T011 | A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor | {"error_factor_mu_0.1_at_N_256.sphere": 1069563.9789352638, "error_factor_mu_0.1_at_N_256.torus": 98109.38296653116, "error_factor_mu_0.2_at_N_256.sphere": 8097.748798583593, "error_factor_mu_0.2_at_N_256.torus": 1245.2423102865773 …(+2)} |  | `numerically_verified` | `sha256:579fff6484c4` |
| T011 | The identity chart reproduces the base-chart integration bit for bit | 0.0 |  | `numerically_verified` | `sha256:579fff6484c4` |
| T012 | Ambient rotations leave chart trajectories and Jacobi fields unchanged to roundoff | 1.7763568394002505e-15 |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | Embedded endpoints rotate exactly with the ambient frame | 1.3732700395566711e-15 |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | Gaussian curvature from the rotated second fundamental form is unchanged | 1.9984014443252818e-15 |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | A reference tangent basis rotated in ciw's metric (heading measured from e1') gives the same start tangent, and heading perturbations stated in it separate as ciw's Jacobi field predicts | {"max_frame_residual": 4.440892098500626e-16, "max_jacobi_relative_gap": 2.1271427144972646e-08, "max_state_difference": 1.7763568394002505e-15, "max_tangent_difference": 2.220446049250313e-16} |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | A heading change +eps stated in a left-handed basis (e1, -e2) is the geometric perturbation -eps: along the right-handed normal its separation has the opposite sign | {"ratio_along_own_normal": 1.000000000002456, "ratio_along_right_handed_normal": -1.000000000002456, "right_handed_minus_exact": 8.431105640027692e-12} |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | On a non-symmetric surface the orientation-reversed separation agrees only to first order: own/right - 1 is proportional to eps (Torus(2, 1) generic path) | {"loglog_slope": 0.9972359056150474, "eps[0]": 0.001, "eps[1]": 0.01, "eps[2]": 0.04 …(+3)} |  | `numerically_verified` | `sha256:ac7f90630032` |
| T012 | An improper rotation (reflection) is refused as a frame change | "Frame change requires a proper rotation matrix" |  | `numerically_verified` | `sha256:ac7f90630032` |
| T013 | The cylinder is intrinsically flat: K from its second fundamental form vanishes along the helix and the Jacobi field integrated with that K is j_head(s) = s | {"helix_chart_error": 1.3322676295501878e-15, "j_head_minus_s_max": 2.6645352591003757e-15, "literal_K_plane_cylinder_jacobi_difference": 0.0, "second_form_max_abs_curvature": 0.0} |  | `numerically_verified` | `sha256:21d6fea668ab` |
| T013 | Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord | {"chord_cylinder": 2.5382081174896274, "chord_plane": 3.0000000000000018} |  | `numerically_verified` | `sha256:21d6fea668ab` |
| T013 | Torus outer-equator Jacobi deviation L - j_head(L) matches the closed form for every major radius | 1.7752341818777495e-09 |  | `numerically_verified` | `sha256:21d6fea668ab` |
| T013 | The Jacobi deviation from flat decays like 1/R: exponent -1 in R + 1 = 1/K, with the finite-R offset explained | {"closed_form_exponent": -0.9847201715660304, "fitted_exponent": -0.9847201717012537, "fitted_exponent_in_R_plus_1": -0.9974853377423354, "max_gap_to_K_L3_over_6": 0.011699011471468235 …(+2)} |  | `numerically_verified` | `sha256:21d6fea668ab` |
| T013 | The chord deficit vanishes faster (exponent -2 in R + 1) than the Jacobi deviation (-1) | {"exponent_in_R": -1.9743918815966817, "exponent_in_R_plus_1": -1.9999677621250946, "max_relative_error_vs_closed_form": 5.136381098225229e-09} |  | `numerically_verified` | `sha256:21d6fea668ab` |
| T013 | A physical cylinder or large-radius torus workpiece shows these separations | null |  | `not_established` | `sha256:21d6fea668ab` |
| T014 | Reversal error orders are 1 (Euler), 3 (midpoint) and 5 (RK4): even-order methods gain one order | {"hyperbolic-plane: euler": 1.0871829145034768, "hyperbolic-plane: midpoint": 3.0020086126315064, "hyperbolic-plane: rk4": 5.000151728589235, "sphere: euler": 1.0397463460131005 …(+5)} |  | `numerically_verified` | `sha256:dc1ec51d344d` |
| T014 | Adaptive forward-then-reversed integration returns to the start at tolerance level | 0.954654133522581 |  | `numerically_verified` | `sha256:dc1ec51d344d` |
| T014 | Truncating at L1 and continuing on the same dyadic grid reproduces direct integration bit for bit | {"euler": 0.0, "midpoint": 0.0, "rk4": 0.0} |  | `numerically_verified` | `sha256:dc1ec51d344d` |
| T014 | With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails | {"differs": true, "max_abs_difference": 4.440892098500626e-16} |  | `numerically_verified` | `sha256:dc1ec51d344d` |
| T014 | Adaptive restart at L1 reproduces direct adaptive integration only to tolerance level | 0.010709114151019605 |  | `numerically_verified` | `sha256:dc1ec51d344d` |
| T015 | Adaptive DP45 energy error grows linearly with length on the torus and the sphere | {"sphere": 0.9977233904023642, "torus": 0.9335193139363184} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Sphere position error grows like L for fixed-step RK4 and like L^2 for adaptive DP45 | {"adaptive": 2.0529904353388955, "rk4": 1.0151377851200762} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Sphere angular-momentum drift of fixed-step RK4 grows linearly with length and is a tilt of the orbit plane, not a change of its size | {"angular_momentum_exponent": 1.0073868099004275, "momentum_size_at_320": 2.3833006803641865e-06, "plane_tilt_at_320": 0.0005462862916566549, "plane_tilt_exponent": 1.0074120844845493 …(+1)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Fixed-step RK4 sphere position error is cross-track: the precessing orbit plane, not the speed error, sets it | {"along_over_integrated_speed_error": -0.16027306137215694, "along_track_at_320": -2.5966475792261697e-05, "cross_track_at_320": 0.0005408552664065206, "cross_track_over_plane_tilt": 0.9900582801855337 …(+3)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Adaptive DP45 sphere position error is along-track and equals its integrated speed error, which grows like L^2 | {"along_over_integrated_speed_error": 0.997749347217412, "along_track_at_320": -0.01020830012528686, "along_track_share": 0.9999822020375145, "cross_track_at_320": 0.00012167751069818315 …(+3)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Fixed-step RK4 energy error has a flat (oscillation-dominated) envelope up to L = 160 on the torus and L = 320 on the sphere; on the torus a secular term emerges between L = 160 and 320 | {"sphere_exponent_L_10_to_320": 0.006613658411833017, "sphere_growth_factor_10_to_320": 1.0248235813125994, "torus_exponent_L_10_to_160": 0.027506236135538223, "torus_growth_factor_10_to_160": 1.100021087555773 …(+1)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Torus Clairaut drift of adaptive DP45 grows linearly with length from L = 40 on | {"exponent_L_40_to_320": 0.9992597599612544, "local_slopes[0]": 1.332898467466863, "local_slopes[1]": 1.1607747969121782, "local_slopes[2]": 0.9282465031672579 …(+2)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Euler on the torus keeps a bounded energy error but changes the orbit type (Clairaut drift) | {"clairaut_drift_at_320": 2.2008846150507644, "clairaut_initial": 2.2008965407279346, "max_energy_error": 0.03499256018697694, "theta_max_abs": 285.03465063072775 …(+1)} |  | `numerically_verified` | `sha256:e0d8481def4a` |
| T015 | Euler on the sphere leaves the polar chart before the horizon | 14.0 | arclength | `numerically_verified` | `sha256:e0d8481def4a` |
| T016 | Jacobi fields on HyperbolicPlane(k) grow like sinh(kL)/k and adaptive integration resolves them | {"j_head_at_k_8": 555381.9076768042, "max_adaptive_relative_error": 2.61206217014763e-10} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T016 | Fixed-step RK4 relative error grows like k^5 and matches L k^5 h^4 / 120 | {"exponent": 4.940697853736755, "ratio_to_prediction[0]": 1.024383946988133, "ratio_to_prediction[1]": 0.9749753080569532, "ratio_to_prediction[2]": 0.949286444844628 …(+1)} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T016 | RK4 steps for relative accuracy 1e-6 grow like k^(5/4); DP45 accepted steps grow about linearly in k | {"adaptive_exponent": 0.8809837411874845, "rk4_exponent": 1.2392633266856508, "rk4_required_over_predicted[0]": 1.0121284973124025, "rk4_required_over_predicted[1]": 0.9991117860645851 …(+2)} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T016 | This is intrinsic exponential instability, not stiffness: the integrated Jacobi flow grows and decays at the same rate k and accuracy, not stability, sets the step | {"log_growth_slope_in_kL": 1.000894297561897, "measured_rates.1.decay": 0.9999999999962932, "measured_rates.1.growth": 1.0000000000033769, "measured_rates.2.decay": 1.9999999999933311 …(+9)} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T016 | A-stable implicit methods do not remove the growth of the required steps with k (implicit midpoint ~ k^(3/2), 2-stage Gauss-Legendre ~ k^(5/4)); implicit midpoint is qualitatively wrong beyond kh = 2 | {"gauss_legendre_2.exponent": 1.2369214159922024, "gauss_legendre_2.over_rk4_steps[0]": 0.6521739130434783, "gauss_legendre_2.over_rk4_steps[1]": 0.6481481481481481, "gauss_legendre_2.over_rk4_steps[2]": 0.6535433070866141 …(+12)} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T016 | Saddle(c): peak \|K\| = c^2 but Jacobi growth is polynomial in c; the local exponent of j_head(L) decreases toward sqrt(2) | {"fitted_exponent_c_16_to_16384": 1.4258879402441527, "heuristic_exponent": 1.4142135623730951, "j_head[0]": 3.167908032458897, "j_head[1]": 14.26337708828688 …(+13)} |  | `numerically_verified` | `sha256:59100f16fdb0` |
| T017 | On a generic torus geodesic C2(s*) != 0 and the validity domain shrinks to zero linearly at the conjugate point | {"C2_at_s_star": 1.5193096777532757, "collapse_slope": 0.007384139832327847, "collapse_slope_predicted": 0.007385655090193269, "eps_max_at_s_star": 8.585102871929361e-13 …(+9)} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | The predicted validity boundary holds: new integrations at eps_max/2 and 2 eps_max fall on either side of tau and match the C2/C3 remainder model | {"s": 5.446260530225366, "double.eps": 0.0082412918164584, "double.j": 0.665874702922783, "double.model_relative_error": 0.020107072762936197 …(+11)} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | Unit sphere, pure heading: C2 = 0, C3 = -\|sin s\| cos^2 s / 24, and the domain does not shrink at s = pi | {"max_C2_over_j": 1.5663757687783023e-08, "max_C3_relative_error": 8.275677306457396e-05, "max_eps_max_relative_error": 4.083325472370447e-05, "eps_max_near_pi.0.999": 0.4899060501928133 …(+1)} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | Hyperbolic plane, pure heading: C2 = 0 and eps_max = sqrt(24 tau)/cosh(s) shrinks exponentially | {"max_C2_over_C3_eps_min": 0.014169729258414752, "max_eps_max_relative_error": 0.00345180239911258, "eps_max[0]": 0.47498610579044526, "eps_max[1]": 0.43446806911245356 …(+4)} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | Sphere lateral+heading perturbation: C2(s0) = 1/2 at the first-order zero s0 = 3pi/4, and eps_max shrinks to zero linearly there | {"C2_at_s0": 0.5000140349820008, "collapse_slope": 0.40127711021012086, "collapse_slope_predicted": 0.40099875311526834, "eps_max_at_0.9_s0": 0.1008281905308577 …(+1)} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | Torus outer equator: C2 vanishes along the whole path (reflection symmetry); the remainder is cubic | {"eps_max_at_s_star": 3.5376386015910856e-07, "max_abs_C2": 5.117970796264466e-06, "remainder_exponent_at_s_star": 3.000317287330155} |  | `numerically_verified` | `sha256:13faf54511fe` |
| T017 | These validity domains certify first-order path corrections as safe on real machines | null |  | `not_established` | `sha256:13faf54511fe` |
| T018 | RK4 resolves every truncation-dominated curvature signal from N = 16 (h = 1/8) | {"min_ratio_rk4_N_ge_16": 117432.0196008736, "min_ratio_steps": 16, "min_ratio_surface": "hyperbolic k=1", "weakest_signal": 0.00013333066669196647 …(+4)} |  | `numerically_verified` | `sha256:4e8942228f27` |
| T018 | Weak curvature is harder to resolve for Euler and midpoint only down to K ~ 1e-2: for K <= 1e-2 their ratios no longer depend on K, and the RK4 ratio grows like 1/K | {"rk4_ratio_K_1e-4_over_K_1e-2_at_N16": 98.0027758672259, "ratio_K_1e-2_over_K_1_at_N16.euler": 0.6394864803160453, "ratio_K_1e-2_over_K_1_at_N16.midpoint": 0.41185603435685014, "ratio_K_1e-8_over_K_1e-2_at_N16.euler": 0.9964328188585644 …(+1)} |  | `numerically_verified` | `sha256:4e8942228f27` |
| T018 | Below the floating-point resolution of L no step size resolves the curvature signal, although the methods' truncation errors alone would resolve it | {"max_ratio_K_1e-16": 1.0, "min_truncation_only_ratio_K_1e-16_at_N128": 42.890052356020796, "nonzero_computed_deviations_K_1e-16": 0} |  | `numerically_verified` | `sha256:4e8942228f27` |
| T018 | Resolvability ratios improve like h^-p with p = 1, 2, 4 (sphere R = 1) | {"euler": 1.0388062197182897, "midpoint": 1.904785224238742, "rk4": 3.888721475066301} |  | `numerically_verified` | `sha256:4e8942228f27` |
| T018 | Flat surfaces (K from the second fundamental form) show no spurious curvature signal for any method or step | {"max_abs_deviation": 0.0, "max_abs_path_curvature": 0.0} |  | `numerically_verified` | `sha256:4e8942228f27` |
| T018 | Curvature signals resolvable here would be resolvable in measured sensor data | null |  | `not_established` | `sha256:4e8942228f27` |
| T019 | Every enumerated SL(2,Z) basis and every random word reduces exactly to the same canonical Gram form | {"reductions": 3048, "failures.area": 0, "failures.automorphism": 0, "failures.inverse": 0 …(+2)} |  | `numerically_verified` | `sha256:22cc2c1239b4` |
| T019 | Number of reduced bases equals the predicted stabilizer count (2 generic, 4 boundary or square, 12 hexagonal) | {"arc-boundary": 4, "generic": 2, "hexagonal": 12, "rectangular": 2 …(+2)} |  | `numerically_verified` | `sha256:22cc2c1239b4` |
| T019 | Integer matrices with det != 1 are refused as SL(2,Z) basis changes | {"det -1 (orientation reversal)": "BASIS_CHANGE_REVERSES_ORIENTATION", "det 2 (index-2 sublattice)": "BASIS_CHANGE_NOT_UNIMODULAR", "non-integer": "BASIS_CHANGE_NOT_INTEGER"} |  | `numerically_verified` | `sha256:22cc2c1239b4` |
| T019 | Float Gauss reduction agrees with the pinned FTR fold_to_fundamental_domain on reduced tau and the reducing SL(2,Z) matrix (up to -I) | {"cases": 12, "matrix_mismatches": 0, "max_tau_difference": 1.1304854779871473e-12} |  | `independently_verified` | `sha256:22cc2c1239b4` |
| T020 | Every winding with \|m\|, \|n\| <= 6 first returns to its start at t = 1/gcd, displaced by its primitive vector after (\|m\| + \|n\|)/gcd edge crossings, and returns gcd times by t = 1 (a gcd-fold cover of the primitive loop) | {"classes": 168, "failures": 0} |  | `numerically_verified` | `sha256:59df99d5453f` |
| T020 | For all 120 pairs of the 16 primitive classes with 0 <= m <= 3, \|n\| <= 3 the transverse intersections number \|det(v, w)\| | {"failures": 0, "pairs": 120} |  | `numerically_verified` | `sha256:59df99d5453f` |
| T020 | The lattice-point count within radius R (including the origin, and its primitive part) equals a brute-force count over a box proven to contain the disk | {"lattice_points_with_origin": 229, "primitive": 142, "radius": 20} |  | `numerically_verified` | `sha256:59df99d5453f` |
| T020 | The golden-slope geodesic has positive return gaps phi^-k at Fibonacci returns, with q * gap -> 1/sqrt 5 | {"gap[0]": 0.3819660112501051, "gap[1]": 0.2360679774997898, "gap[2]": 0.1458980337503153, "gap[3]": 0.09016994374947451 …(+38)} |  | `independently_verified` | `sha256:59df99d5453f` |
| T020 | A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction | {"log2_denominator": 49, "slope_denominator": 562949953421312} |  | `analytic` | `sha256:59df99d5453f` |
| T020 | Closed-geodesic lengths \|m w1 + n w2\|, edge-crossing counts and unit area agree with the pinned FTR loop_length, trace_closed_geodesic and normalized_lattice | {"crossing_mismatches": 0, "lengths": 160, "max_relative_difference": 2.1329254204086564e-16} |  | `independently_verified` | `sha256:59df99d5453f` |
| T021 | On the test flat torus every route's heading amplification equals its length (ciw.lab.jacobi), so the length and amplification orders coincide | {"discordant_pairs": 0, "max_relative_j_head_error": 4.780060277391527e-16, "routes": 19} |  | `numerically_verified` | `sha256:f561f7dfab1b` |
| T021 | For every flat torus and every pair of points the least-sensitive geodesic is a shortest one | true |  | `analytic` | `sha256:f561f7dfab1b` |
| T021 | Shortest equals least heading-sensitive whenever j_head(L) is one strictly increasing function of L for all routes (constant K <= 0); with K > 0 somewhere, or curvature that differs between routes, the equivalence can fail | true |  | `analytic` | `sha256:f561f7dfab1b` |
| T021 | The shortest route on a physical flat workpiece is the safest route to execute | null |  | `not_established` | `sha256:f561f7dfab1b` |
| T022 | Square-torus half-period targets have exactly 2 (edge) or 4 (centre) shortest representatives | {"(0, 1/2)": 2, "(1/2, 0)": 2, "(1/2, 1/2)": 4, "(1/3, 1/5)": 1} |  | `numerically_verified` | `sha256:b0522096c3f9` |
| T022 | Multiplicity census on the 12 x 12 rational grid equals the predicted cut-locus counts | {"1": 121, "2": 22, "4": 1} |  | `numerically_verified` | `sha256:b0522096c3f9` |
| T022 | On six lattices the cut-locus vertices satisfy sum over vertices of (k_v - 2) = 2 (Euler characteristic 0) | {"arc-boundary": 2, "generic": 2, "hexagonal": 2, "rectangular": 2 …(+2)} |  | `numerically_verified` | `sha256:b0522096c3f9` |
| T022 | Rounding the target to binary64 changes its shortest-representative multiplicity (exact 1, binary64 2) | {"binary64": 2, "epsilon": "2^-60", "exact": 1, "exact_of_rounded_target": 2} |  | `numerically_verified` | `sha256:b0522096c3f9` |
| T022 | At exact cut-locus ties that binary64 cannot represent, raw float comparison undercounts some multiplicities while a relative tolerance of 1e-9 recovers all of them | {"binary64_undercounts": 4, "ties": 6, "tolerance_undercounts": 0} |  | `numerically_verified` | `sha256:b0522096c3f9` |
| T023 | Zeros of the return distance (tau = 0.31 + 1.07i, L = 3) are exactly the primitive lattice directions with \|v\| <= L | {"found": 18, "predicted": 18} |  | `numerically_verified` | `sha256:3bed11da32c2` |
| T023 | Near each closing heading the closing error grows at rate \|v\| (the loop length) per radian | {"max_relative_slope_error": 5.891816425508636e-12} |  | `numerically_verified` | `sha256:3bed11da32c2` |
| T023 | Closure basins are widest for the shortest (lowest-order) primitive classes | {"discordant_pairs": 0, "unresolved_pairs": 0, "widest[0].basin_width": 0.04137927381071043, "widest[0].length": 0.9667364890456636 …(+18)} |  | `numerically_verified` | `sha256:3bed11da32c2` |
| T023 | A physical heading sensor closes the shortest loop within the computed heading tolerance | null |  | `not_established` | `sha256:3bed11da32c2` |
| T024 | Every fan-search route p -> q on Torus(2, 1) ends on a lift of q with the Wronskian and the Clairaut integral conserved to tolerance | {"[0].amplification": 1.8341504196036555, "[0].focus_margin": 1.2453092693335792, "[0].heading": 0.7663176018658682, "[0].length": 5.846451377518235 …(+77)} |  | `independently_verified` | `sha256:35e76c8f288e` |
| T024 | The torus route set is unchanged when the fan density doubles (1440 to 2880 headings) | {"differences": 0, "headings[0]": 1440, "headings[1]": 2880, "routes[0]": 9 …(+1)} |  | `numerically_verified` | `sha256:35e76c8f288e` |
| T024 | Rankings by length, amplification and focus margin disagree: the shortest route is neither the least amplifying nor in the best focus-margin group | {"by_amplification[0]": 2, "by_amplification[1]": 1, "by_amplification[2]": 0, "by_amplification[3]": 4 …(+23)} |  | `numerically_verified` | `sha256:35e76c8f288e` |
| T024 | A flat torus has no conjugate points: j_head(s) = s > 0 (infinite focus margin) | 1.0 |  | `numerically_verified` | `sha256:35e76c8f288e` |
| T024 | Ranking routes by focus margin selects a route that is safe to execute on a physical part | null |  | `not_established` | `sha256:35e76c8f288e` |
| T025 | The three-objective front has several members, and a second computation (numpy dominance matrix; sort-and-sweep for the two-objective fronts) reproduces every front | {"routes": 9, "front[0]": 0, "front[1]": 1, "front[2]": 2 …(+8)} |  | `numerically_verified` | `sha256:767a36c022d2` |
| T025 | A Pareto-optimal route is safe to execute on a physical part | null |  | `not_established` | `sha256:767a36c022d2` |
| T026 | Length spectrum (R^2 <= 60), area and systole are exactly invariant under every tested SL(2,Z) change of basis and under reduction | {"matrices": 368, "failures.area": 0, "failures.spectrum": 0, "failures.systole": 0 …(+1)} |  | `numerically_verified` | `sha256:ca637e94b6a3` |
| T026 | Area-one float spectra (first 30 lengths) agree within 1e-8 under all 368 basis changes, a conditioning-limited agreement rather than unit roundoff | 1.1122324405657753e-10 |  | `numerically_verified` | `sha256:ca637e94b6a3` |
| T026 | Transporting winding labels with M instead of M^-1 breaks length invariance | 6428 |  | `numerically_verified` | `sha256:ca637e94b6a3` |
| T026 | The length spectrum does not determine the oriented shape: a mirror image is isospectral but not SL(2,Z)-equivalent | {"same_canonical": false, "same_spectrum": true, "canonical[0]": "5", "canonical[1]": "-2" …(+4)} |  | `numerically_verified` | `sha256:ca637e94b6a3` |
| T026 | A det-2 integer matrix changes area and spectrum (an index-2 sublattice, not a basis change) | {"area_sq_ratio": "4", "original_systole_sq": "5", "same_spectrum": false, "systole_sq": "7" …(+3)} |  | `numerically_verified` | `sha256:ca637e94b6a3` |
| T026 | Closed-geodesic lengths before and after the fold agree with the pinned FTR length_pair and its winding transport | {"cases": 12, "max_relative_difference": 5.279494431636665e-13, "winding_mismatches": 0} |  | `independently_verified` | `sha256:ca637e94b6a3` |
| T027 | L-shape and regular octagon are genus-2 translation surfaces with one vertex class | {"H(1,1) origami.area": "4", "H(1,1) origami.edges": 8, "H(1,1) origami.euler_characteristic": -2, "H(1,1) origami.faces": 4 …(+53)} |  | `numerically_verified` | `sha256:6384edc15de5` |
| T027 | Horizontal cylinders: L-shape circumferences equal the r-cycle lengths, octagon strips have circumferences 2 + sqrt 2 and 1 + sqrt 2 with areas summing to the surface area | {"l_shape[0]": 2.0, "l_shape[1]": 1.0, "octagon[0]": 3.414213562373095, "octagon[1]": 2.414213562373095} |  | `numerically_verified` | `sha256:6384edc15de5` |
| T027 | Pairings that are not translations, or glue unequal edges, are refused as translation surfaces | {"mismatched edge lengths": "EDGE_LENGTH_MISMATCH", "octagon adjacent pairing": "GLUING_NOT_TRANSLATION", "pillowcase": "GLUING_NOT_TRANSLATION"} |  | `numerically_verified` | `sha256:6384edc15de5` |
| T028 | All 96 tested rational trajectories on the L-shape close (multiplier 1-3) or end in a saddle connection | {"closed": 90, "saddle_connections": 6, "undecided": 0} |  | `numerically_verified` | `sha256:24878313ca60` |
| T028 | Trajectories hitting a cone point are terminated as saddle connections, exactly or within the declared float tolerance | {"exact_l_shape": "SADDLE_CONNECTION", "exact_octagon": "SADDLE_CONNECTION", "float_octagon": "NEAR_VERTEX_WITHIN_TOLERANCE"} |  | `numerically_verified` | `sha256:24878313ca60` |
| T028 | A float start within the declared tolerance of a polygon edge is refused as not interior | {"float_start": "START_NOT_INTERIOR"} |  | `numerically_verified` | `sha256:24878313ca60` |
| T028 | Float octagon flow reproduces the exact Q(sqrt 2) trajectory (same crossings, closure time) | {"crossings": 20, "time_difference": 1.7763568394002505e-15} |  | `numerically_verified` | `sha256:24878313ca60` |
| T028 | A generic float octagon trajectory follows the exact Q(sqrt 2) trace of its binary64 start and direction for 400 crossings and stays farther than the tolerance from every vertex | {"crossings": 400, "max_position_deviation": 1.199040866595169e-14, "min_along_edge_clearance": 0.00027425327282185557, "min_euclidean_vertex_distance": 0.0002425740721023586} |  | `numerically_verified` | `sha256:24878313ca60` |
| T029 | Cone angles: octagon and L-shape one 6 pi point, H(1,1) two 4 pi points, pillowcase four pi points, square and hexagon tori none | {"H(1,1) origami[0]": "4", "H(1,1) origami[1]": "4", "L-shape[0]": "6", "hexagon[0]": "2" …(+7)} |  | `numerically_verified` | `sha256:462f94dfa91b` |
| T029 | Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds exactly on every surface with chi known independently of the vertex classes (declared topology; Riemann-Hurwitz for origamis) | {"H(1,1) origami.chi": -2, "H(1,1) origami.defect_over_pi": "0", "L-shape.chi": -2, "L-shape.defect_over_pi": "0" …(+8)} |  | `numerically_verified` | `sha256:462f94dfa91b` |
| T029 | Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points | ["2", "2"] |  | `numerically_verified` | `sha256:462f94dfa91b` |
| T030 | 4- and 8-neighbour grid shortest paths do not converge to Euclidean length under refinement | {"ratio4_45deg[0]": 1.414213562373095, "ratio4_45deg[1]": 1.4142135623730965, "ratio4_45deg[2]": 1.4142135623730943, "ratio8_22_62deg[0]": 1.0823898316819596 …(+2)} |  | `independently_verified` | `sha256:4823bcb02484` |
| T030 | Worst-direction metrication error is sqrt 2 - 1 (4-nbr, 45 deg) and sqrt(4 - 2 sqrt 2) - 1 (8-nbr, 22.5 deg) | {"4": 0.4142135623730949, "8": 0.08239220029239402} |  | `numerically_verified` | `sha256:4823bcb02484` |
| T030 | Fast marching converges to Euclidean length under refinement | {"observed_order": 0.7120027708331585, "max_relative_error.120": 0.0191733644126737, "max_relative_error.30": 0.05144778819856555, "max_relative_error.60": 0.03184500861733186} |  | `numerically_verified` | `sha256:4823bcb02484` |
| T030 | Grid-planned path lengths predict distances travelled by a physical vehicle or tool | null |  | `not_established` | `sha256:4823bcb02484` |
| T031 | The shortest route switches at eps* = 4 delta, which vanishes as the unperturbed gap vanishes | {"eps_star": "1/250", "scaling[0].delta": "1/10", "scaling[0].eps_star": "2/5", "scaling[0].ratio": "4" …(+9)} |  | `independently_verified` | `sha256:1af0ba792b50` |
| T031 | A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about 127 deg while the minimal length changes continuously | {"heading_jump_deg": 126.86975096836686, "length_jump": 8.944644025454807e-08} |  | `numerically_verified` | `sha256:1af0ba792b50` |
| T031 | A metric calibrated from physical measurements is accurate enough to decide between near-tied routes | null |  | `not_established` | `sha256:1af0ba792b50` |
| T032 | Torus inner equator: the shortest route has larger heading amplification than a longer route | {"alternative.amplification": 2.2424499641265205, "alternative.focus_margin": null, "alternative.heading": -0.6654954777289914, "alternative.length": 7.4048964512362 …(+8)} |  | `independently_verified` | `sha256:bb5b68cba56d` |
| T032 | Torus outer equator: the shortest route has a smaller focus margin than a longer route | {"alternative.amplification": 29.49806945310811, "alternative.focus_margin": null, "alternative.heading": -1.3844466400297355, "alternative.length": 6.723214266466979 …(+8)} |  | `independently_verified` | `sha256:bb5b68cba56d` |
| T032 | Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique | {"headings[0]": -0.932959589373497, "headings[1]": 0.932959589373497, "lengths[0]": 6.30866770803647, "lengths[1]": 6.308667708036468} |  | `numerically_verified` | `sha256:bb5b68cba56d` |
| T032 | Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point | {"alternative.amplification": 6.488420201199406, "alternative.focus_margin": -1.9973831828910869, "alternative.heading": 0.0, "alternative.length": 5.8779306681414605 …(+8)} |  | `independently_verified` | `sha256:bb5b68cba56d` |
| T032 | On the unit sphere the minimizing arc between points at separation pi - delta ends delta before its conjugate point, so minimizing geodesics have no positive lower bound on focus margin | {"delta[0]": 0.1, "delta[1]": 0.01, "delta[2]": 0.001, "focus_margin[0]": 0.1000000000875092 …(+8)} |  | `numerically_verified` | `sha256:bb5b68cba56d` |
| T032 | Search on the saddle (K < 0, simply connected) finds exactly one route, consistent with Cartan-Hadamard uniqueness, so this surface admits no witness | {"routes": 1} |  | `numerically_verified` | `sha256:bb5b68cba56d` |
| T032 | Every T032 route search gives the same route set at twice the fan density (720 to 1440 headings) | {"bump.differences": 0, "bump.routes[0]": 3, "bump.routes[1]": 3, "saddle.differences": 0 …(+11)} |  | `numerically_verified` | `sha256:bb5b68cba56d` |
| T032 | A route from this library is safe (or unsafe) to execute on a physical part or vehicle | null |  | `not_established` | `sha256:bb5b68cba56d` |
| T033 | Brioschi curvature from the metric alone equals the supplied Gaussian curvature at every sampled point | 7.569957621233518e-11 | normalized residual | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | Metrics are symmetric positive definite at every sampled point of the declared domains | {"max_asymmetry": 0.0, "min_eigenvalue_ratio": 0.1076492566503344, "points_refused_by_core_check": 0} |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | Christoffel symbols are symmetric in their lower indices at every sampled point | 0.0 |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | The connection is metric compatible at every sampled point: d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il | 1.6035737915386125e-16 |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | Supplied metric derivatives agree with fourth-order differences of the metric at every sampled point | 8.102692869746462e-12 |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | Differences of the exact metric derivatives have symmetric mixed partials (dg is a gradient field) | 5.882921077369268e-13 |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | A rigid rotation leaves the torus metric unchanged at every sampled point | 3.665815015959401e-16 |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | The conformance suite rejects every seeded defect mutant | {"mutants": 7, "undetected": 0} |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | A curvature-misscaled sphere passes every identity except the Gauss equation | {"gauss_residual": 0.167, "failed[0]": "gauss_equation"} |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | Metric compatibility cannot detect wrong metric derivatives | {"compatibility_residual_passes": true, "derivative_consistency_residual": 1.0} |  | `numerically_verified` | `sha256:1812f90750bf` |
| T033 | The conformance suite certifies surfaces reconstructed from physical measurements | null |  | `not_established` | `sha256:1812f90750bf` |
| T034 | sympy-differentiated metric and metric derivatives match the ciw surface interface on every conformance surface | 4.226085531613157e-16 | normalized residual | `independently_verified` | `sha256:dcb93c9d2c9f` |
| T034 | sympy.diffgeom Christoffel symbols match the ciw surface interface on seven surfaces | 1.817349289946462e-16 | normalized residual | `independently_verified` | `sha256:dcb93c9d2c9f` |
| T034 | sympy.diffgeom Riemann curvature R_1212 / det g matches the supplied Gaussian curvature on seven surfaces | 6.977634294120269e-16 | normalized residual | `independently_verified` | `sha256:dcb93c9d2c9f` |
| T034 | sympy.diffgeom curvature of seven surfaces simplifies exactly to the closed forms restated from ciw.lab.surfaces | {"mismatches": 0, "surfaces": 7} |  | `independently_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Christoffel symbols and curvature assembled in ciw code from sympy derivatives match the interface on every conformance surface | 5.551115123125783e-16 | normalized residual | `numerically_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Nested dual-number derivatives of re-expressed embeddings match the metric, dg and Christoffel symbols | 4.4259596565437094e-16 | normalized residual | `numerically_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Dual-number curvature (Brioschi with exact second derivatives, and LN - M^2) matches the supplied K | 1.0416744990940514e-15 | normalized residual | `numerically_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Dual numbers reproduce closed-form first, mixed and third derivatives without perturbation confusion | 3.552713678800501e-15 |  | `numerically_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see | 0.0608397 |  | `numerically_verified` | `sha256:dcb93c9d2c9f` |
| T034 | Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from physical measurements | null |  | `not_established` | `sha256:dcb93c9d2c9f` |
| T035 | Central-difference error of the analytic metric derivatives falls as h^2 on the truncation branch | {"gaussian-bump": 1.99998, "hyperbolic-plane": 2.00003, "sphere": 2.0, "torus": 1.99999} |  | `numerically_verified` | `sha256:404808e46324` |
| T035 | Rounding error of central differences grows as 1/h for small steps | {"gaussian-bump": -1.026, "hyperbolic-plane": -1.03, "sphere": -1.013, "torus": -0.9783} |  | `numerically_verified` | `sha256:404808e46324` |
| T035 | The optimal step lies within a factor 4 of the predicted h* = (3 eps \|g\| / \|d^3 g\|)^(1/3) | {"gaussian-bump": -4.75, "hyperbolic-plane": -5.75, "sphere": -5.25, "torus": -5.0} | log10(relative step) | `numerically_verified` | `sha256:404808e46324` |
| T035 | The median central-difference error curve bottoms out at or below the predicted minimum error | 4e-11 | normalized error | `numerically_verified` | `sha256:404808e46324` |
| T035 | At every sampled point the best central difference agrees with the analytic metric derivatives to within twice that point's predicted minimum error | 7.4e-11 | normalized error | `numerically_verified` | `sha256:404808e46324` |
| T035 | Smaller finite-difference steps can be far less accurate | 6.71 | log10 error ratio | `numerically_verified` | `sha256:404808e46324` |
| T035 | Quadratic metrics have no truncation branch and a constant metric differences to exactly zero | {"plane_max_error": 0.0, "quadratic_error_at_1e-2": 6.5e-15} |  | `numerically_verified` | `sha256:404808e46324` |
| T035 | The optimal-step law derived here applies to derivatives of measured surface samples | null |  | `not_established` | `sha256:404808e46324` |
| T036 | Atlas integration of the great circle through the north pole matches the exact great circle | 4.86e-08 | length | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | Atlas accuracy is independent of the distance of closest approach to a pole | 1.21 |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | Atlas integration keeps fourth-order convergence across chart switches | [4.16, 3.962] |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | Chart transitions are exact: round trip, speed preservation and difference Jacobian | {"jacobian": 3.7175822615951626e-10, "roundtrip": 6.497413668604471e-16, "speed": 5.248858339920839e-16} |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | The better chart of the atlas always has det g / R^4 >= 1/2, so switching never chatters | {"covering_min": 0.501614, "min_active_det": 0.254484, "min_det_after_switch": 0.753766} |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | A single polar chart fails or loses accuracy on great circles passing near its pole | {"cases": 8, "degraded": 7, "failed": 4} |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | At 400 RK4 steps chart A alone crosses both poles on the exact meridian to within 1e-10 | {"log10_closest_step_to_pole": -2.275, "log10_error": -13.13, "log10_max_abs_momentum": -13.13, "log10_max_abs_v_phi": -9.607} | log10 of each quantity | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | On the exact meridian chart A alone fails or loses accuracy whenever a step point lands within 1e-4 of a pole (350 to 450 RK4 steps) | {"failed": 1, "log10_max_other_error": -9.923, "log10_min_near_pole_error": -7.678, "near_pole": 5 …(+1)} | log10 length | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | The Monge-plus-polar atlas of a graph surface has exact transitions and covers the plane with regularity above the analytic Monge bound | {"covering_min": 0.915791, "jacobian": 3.021773264554725e-10, "monge_bound": 0.915776, "roundtrip": 6.004449063730082e-16 …(+1)} |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | The graph atlas integrates geodesics through and near the Gaussian-bump apex, where its polar chart alone fails or loses accuracy | {"atlas_error": 4.05e-09, "cases": 6, "min_switches": 1, "monge_alone_error": 6.11e-11 …(+2)} |  | `numerically_verified` | `sha256:0ac0bef69712` |
| T036 | Chart-switching geodesic integration is ready for tool paths over physical parts | null |  | `not_established` | `sha256:0ac0bef69712` |
| T037 | The sphere pole in the polar chart is a coordinate singularity: det g -> 0 while K stays 1 | {"classification": "coordinate_singularity", "regularity_in_chart_B": 1.0, "exponents.christoffel": -1.0, "exponents.condition": -2.0 …(+3)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The scan classifies 9 of the 10 declared approaches as their true type, with no false positive at regular points | {"correct": 9, "classes.cone-apex": "conical_singularity", "classes.conformal-0.9-boundary": "curvature_singularity", "classes.hyperbolic-boundary": "infinite_distance_boundary" …(+7)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The scan misses the removable coordinate singularity of the plane in the cube-root chart | {"condition_exponent": -1.33333, "det_exponent": -1.33333, "max_abs_curvature": 0.0, "observed": "unclassified" …(+2)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The graph z = r^(3/2) has a curvature singularity although its Monge metric is regular | {"K_times_r": 1.1249999, "christoffel_exponent": -2.32e-06, "curvature_exponent": -1.0, "det_exponent": 2.32e-06} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | Christoffel blow-up and curvature blow-up are independent near singular points | {"power_graph.christoffel": -2.316e-06, "power_graph.curvature": -1.0, "sphere_pole.christoffel": -1.0, "sphere_pole.curvature": 0.0} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | A cone apex and the polar-chart origin share every pointwise exponent; only the circumference ratio separates them | {"angle_deficit": 3.1415927, "cone_ratio": 0.5, "polar_ratio": 1.0, "signature_difference": 8.88e-16} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The hyperbolic chart boundary y -> 0 is at infinite distance, not a singular point | {"det_exponent": -4.0, "radial_speed_exponent": -1.0} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The boundary of g = y^-1.8 I lies at finite distance and carries a curvature singularity | {"classification": "curvature_singularity", "curvature_exponent": -0.2, "radial_speed_exponent": -0.9} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | Cases just beyond each classification threshold are misclassified | {"near_critical_speed_exponent": -0.9995, "slow_curvature_exponent": -0.02, "small_deficit_ratio_defect": 5e-07, "cases.cone-small-deficit-apex.observed": "coordinate_singularity" …(+5)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | Cartesian loops around the sphere pole in the polar chart make a coordinate singularity read as a conical point | {"circumference_ratio": 0.831686, "classification": "conical_singularity", "exponents.christoffel": -1.0, "exponents.condition": -2.0 …(+3)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The pointwise guard and the core check refuse degenerate, blown-up, nonfinite and out-of-chart points with computed codes, and accept curvature below the declared bound | {"cone apex r = 0 (guard)": "degenerate_metric", "hyperbolic y = -1 (guard, core outside_chart propagated)": "outside_chart", "nonfinite coordinates (guard)": "nonfinite_point", "power graph p = 1.8 at rho = 1e-8, \|K\| below the bound (guard)": "accepted" …(+4)} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The pointwise guard propagates a refusal code that a surface raises at its own apex unchanged | {"power graph apex through the guard": "curvature_singularity"} |  | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The core Surface.check accepts sphere-chart points with metric condition number above 1e10 | 11.5 | log10 condition number | `numerically_verified` | `sha256:623001d9a3a0` |
| T037 | The singularity classification applies to scanned physical parts | null |  | `not_established` | `sha256:623001d9a3a0` |
| T038 | Straightest geodesics on sheared planar meshes coincide with straight lines | 8.881784197001252e-16 | normalized length | `numerically_verified` | `sha256:ed6578dce5b8` |
| T038 | Traced prism-cylinder geodesics match the exact planar development of the mesh | 7.993605777301127e-15 | normalized length | `numerically_verified` | `sha256:ed6578dce5b8` |
| T038 | The unfolded face-strip distance equals the traced length on every completed sphere trace | {"max_abs_difference": 3.1086244689504383e-15, "trace_statuses[0]": "completed"} |  | `numerically_verified` | `sha256:ed6578dce5b8` |
| T038 | Heap Dijkstra edge-graph distances agree with a dense Floyd-Warshall recomputation (and scipy.sparse.csgraph when installed) | 8.881784197001252e-16 | normalized length | `independently_verified` | `sha256:ed6578dce5b8` |
| T038 | Nested Steiner-graph distances never increase with k, and every Steiner-graph edge lies inside one mesh face | {"edges_checked": 115740, "edges_outside_faces": 0, "max_increase_with_k": 0.0, "vertex0_to_last.0": 1.4344305272278777 …(+3)} |  | `numerically_verified` | `sha256:ed6578dce5b8` |
| T038 | Steiner distances between traced endpoints stay above the traced length and approach it as k grows | {"min_gap": 0.0017535599433380344, "mean_gap_by_k.1": 0.03011309798124127, "mean_gap_by_k.3": 0.01293801727738277, "mean_gap_by_k.7": 0.005299587716630953} | normalized length | `numerically_verified` | `sha256:ed6578dce5b8` |
| T038 | Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the scanned physical surface | null |  | `not_established` | `sha256:ed6578dce5b8` |
| T039 | Traced-geodesic length defect on icospheres converges at second order | {"order": 2.002181989260691, "local_orders[0]": 2.062967989312907, "local_orders[1]": 1.9898549346031695, "local_orders[2]": 1.983955634901557 …(+2)} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | Traced-geodesic endpoint error on icospheres decreases at least at first order, with irregular pairwise local orders | {"cross_track_order": 1.5642050157521836, "endpoint_order": 1.6288067358430567, "finest_mean_endpoint": 0.00011617510426007931, "cross_track_local_orders[0]": 1.3698994063012953 …(+12)} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | Prism-cylinder helix endpoint error equals the closed-form development and chord-position prediction, and n^2 error tends to L cos(alpha) pi^2 / 6 | {"max_abs_minus_exact": 4.0653244659516474e-15, "order": 2.0237175858011835, "local_orders[0]": 2.229405542631075, "local_orders[1]": 1.9097179986891355 …(+7)} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | Edge-graph Dijkstra distance from a valence-5 vertex keeps a relative-error floor that tends to sqrt(5) - 2 | {"aitken_extrapolation": 0.23612512058243176, "prediction": 0.2360679774997898, "levels[0]": 1, "levels[1]": 2 …(+6)} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | Steiner graphs with a fixed number of points per edge keep a relative-error floor | {"k1_max_abs_relative_error[0]": 0.030675797123964732, "k1_max_abs_relative_error[1]": 0.02781558056741229, "k1_max_abs_relative_error[2]": 0.044510672419889596, "k1_max_abs_relative_error[3]": 0.04920837664486166 …(+4)} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | Heat-method distance error decreases under refinement at first order | {"order": 1.0078547915405713, "local_orders[0]": 1.0786428316398602, "local_orders[1]": 0.939894799461312} |  | `numerically_verified` | `sha256:91ed8a53192e` |
| T039 | The observed convergence orders transfer to meshes reconstructed from real scans | null |  | `not_established` | `sha256:91ed8a53192e` |
| T040 | The smooth ciw.lab.jacobi heading column equals sin(L) on the unit sphere | 6.80930867247298e-11 |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | Finite-difference mesh Jacobi fields with a 0.1 rad heading offset converge to the smooth field | {"order": 1.532560194023188, "levels[0]": 2, "levels[1]": 3, "levels[2]": 4 …(+14)} |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L) while no vertex lies between the paired geodesics | {"error_vs_smooth": 1.0907025731743183, "identical_face_sequences": 30, "max_flat_deviation": 4.96358509849415e-10, "pairs": 30 …(+15)} |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K | {"order_to_limit": 2.0011096480659107, "predicted_limit": 1.1458980337503153, "local_orders_to_limit[0]": 2.0026513212346653, "local_orders_to_limit[1]": 2.000662792609324 …(+8)} |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | With the mixed Voronoi area the valence-5 curvature estimate converges | [0.06991069920332804, 0.01703436697063121, 0.004231854874286567, 0.001056307420673086 …(+3)] |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | Barycentric angle-defect curvature does not converge pointwise at valence-6 icosphere vertices on the icosahedral mirror planes | {"base_edge_max_error[0]": 0.0019319256495926584, "base_edge_max_error[1]": 0.002431873259649331, "base_edge_max_error[2]": 0.0025567564634368933, "base_edge_max_error[3]": 0.002587970803474504 …(+16)} |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | Angle-defect curvature on regular torus grids converges at second order to the sign-changing K | {"order": 1.9334209531237276, "local_orders[0]": 1.8412826683401746, "local_orders[1]": 1.9587870732947334, "local_orders[2]": 1.9895863910151574 …(+4)} |  | `numerically_verified` | `sha256:c3b92a553066` |
| T040 | Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part | null |  | `not_established` | `sha256:c3b92a553066` |
| T041 | Seed-averaged curvature RMS error grows and minimum angle falls with tangential jitter | {"[0].amplitude": 0.0, "[0].curvature_rms": 0.02042617452826786, "[0].geodesic_mean": 0.012334617469521443, "[0].max_radius_ratio": 1.0307631773587616 …(+24)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | For the three declared seeds per amplitude, the dihedral fold check refuses exactly the jittered meshes that have an inverted face | {"inverted_faces[0]": 0, "inverted_faces[1]": 0, "inverted_faces[2]": 0, "inverted_faces[3]": 0 …(+44)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Over 60 further seeds per amplitude the dihedral fold check accepts some jittered meshes with an inverted face | {"rates[0].amplitude": 0.1, "rates[0].clean_accepted": 60, "rates[0].clean_refused": 0, "rates[0].inverted_accepted": 0 …(+34)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | A tangential jitter of 0.2 h inverts a face for some but not all of 60 further seeds | {"inverted_fraction.0.1": 0.0, "inverted_fraction.0.15": 0.03333333333333333, "inverted_fraction.0.2": 0.65, "inverted_fraction.0.3": 1.0} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Over 60 further seeds per amplitude the dihedral fold check refuses no jittered mesh without an inverted face | {"rates[0].amplitude": 0.1, "rates[0].clean_accepted": 60, "rates[0].clean_refused": 0, "rates[0].inverted_accepted": 0 …(+20)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | A mesh with a smaller minimum angle can have a smaller geodesic error | {"regular.geodesic_mean": 0.012334617469521443, "regular.mesh": "icosphere-3", "regular.min_angle_deg": 54.0995585363326, "witness.geodesic_mean": 0.009929414081371678 …(+2)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Across mesh families a much smaller minimum angle can come with a smaller barycentric-area angle-defect RMS error | {"regular.curvature_max": 0.14785072359127138, "regular.curvature_rms": 0.02042617452826786, "regular.max_radius_ratio": 1.0307631773587616, "regular.mesh": "icosphere-3" …(+8)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | With the mixed Voronoi area the regular icosphere has a smaller curvature RMS error than every latitude-longitude sphere of the same vertex count | {"icosphere": 0.004777035900366827, "latitude_longitude.uv-sphere-10x64": 0.011105733411546024, "latitude_longitude.uv-sphere-20x32": 0.006270019710208169, "latitude_longitude.uv-sphere-20x32-t0.08": 0.011487716381650736 …(+2)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Within the latitude-longitude family a mesh with smaller minimum angle and larger radius ratio can have smaller curvature errors under both area choices | {"better_quality.curvature_max": 0.2195004380023058, "better_quality.curvature_rms": 0.02720448926919287, "better_quality.max_radius_ratio": 2.873046712440642, "better_quality.mesh": "uv-sphere-40x16" …(+8)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Maximum radius ratio is positively rank-correlated with the mixed-Voronoi curvature error and the geodesic error across the pooled valid 642-vertex meshes, but not within the latitude-longitude family | {"meshes": 15, "pairs": 105, "discordant_pairs.geodesic_mean": 20, "discordant_pairs.voronoi_curvature_rms": 18 …(+16)} |  | `independently_verified` | `sha256:b274c37e5765` |
| T041 | Schwarz lantern meshes converge in Hausdorff distance but not in area | {"limit_area_ratio": 1.5880859697781753, "area_ratio[0]": 1.3867744389205312, "area_ratio[1]": 1.5356721533539741, "area_ratio[2]": 1.5748478593251627 …(+12)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Schwarz lantern intrinsic height does not converge to the cylinder height | {"height": 1.0, "traced_height[0]": 1.5403191234385416, "traced_height[1]": 1.575863999874121, "traced_height[2]": 1.5850127822359787 …(+2)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Lantern angle defects vanish while total absolute mean curvature grows like n^2 and normal tilt converges to atan(pi^2 q R / 2H) instead of 0 | {"growth_exponent_in_n": 2.0066161652004584, "limit_tilt_deg": 50.97283111818916, "max_interior_angle_defect_curvature": 6.420162261836559e-11, "smooth_total_abs_mean_curvature": 3.141592653589793 …(+10)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | A strongly pleated lantern (m = n^2) is refused as folded although no face normal points towards the axis | {"min_normal_radial": 0.20107436521303879, "n": 8, "q": 1.0, "issues[0]": "folded_face"} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | Straightest geodesics on planar meshes are exact at any tested triangle quality, while edge-graph distance error changes with the edge directions | {"max_trace_error": 8.881784197001252e-16, "max_graph_excess[0]": 0.08231174597004576, "max_graph_excess[1]": 0.04475790710240424, "max_graph_excess[2]": 0.02739739461930646 …(+5)} |  | `numerically_verified` | `sha256:b274c37e5765` |
| T041 | A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology | null |  | `not_established` | `sha256:b274c37e5765` |
| T042 | Every declared surface-data defect is refused with its named code | {"aimed-at-vertex": "vertex_hit", "bowtie-vertex": "non_manifold_vertex", "curvature-at-boundary-vertex": "boundary_vertex_curvature", "distance-across-components": "unreachable_target" …(+21)} |  | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | Valid control meshes pass validation without issues | 0 |  | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | A mesh with several defects reports all of them in declared order | ["nonfinite_vertex", "inconsistent_orientation", "unreferenced_vertex"] |  | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | A geodesic stopped at a boundary retains its partial length | 0.5700000000000001 | normalized length | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | Curvature and normals evaluated on an unvalidated zero-area face are nonfinite | {"nonfinite_curvatures": 1, "nonfinite_normals": 3} |  | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | Without a declared centre the validator accepts a jittered icosphere with an inverted face | {"amplitude": 0.2, "inverted_faces": 1, "seed": 20261940, "structural_issues": []} |  | `numerically_verified` | `sha256:530d8c99f29c` |
| T042 | The refusal catalogue covers every defect present in real scanned surface data | null |  | `not_established` | `sha256:530d8c99f29c` |
| T043 | Linearized vertex-noise propagation matches Monte Carlo for the marker distance along a fixed face corridor | {"gradient_norm": 1.0515093665234345, "ratios[0]": 1.0049536213813617, "ratios[1]": 1.0024880461627421, "ratios[2]": 0.975339008040431 …(+5)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | The declared marker segment stays in its face corridor for sigma <= 1e-3 but leaves it in over 10% of samples at sigma = 1e-2 | {"start_index": 5, "vertex_margin": 0.05280540132339888, "left_fraction_by_sigma.0.0001": 0.0, "left_fraction_by_sigma.0.001": 0.0 …(+2)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | The noise level at which a fixed face corridor fails depends on the strip, and the declared strip, which has the largest vertex margin of the six, stays in its corridor for sigma <= 1e-3 | {"sigmas[0]": 0.0001, "sigmas[1]": 0.001, "sigmas[2]": 0.003, "sigmas[3]": 0.01 …(+48)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | A strip with a smaller vertex margin than the declared strip leaves its face corridor less often at every tested sigma >= 3e-3 | {"declared.start_index": 5, "declared.vertex_margin": 0.05280540132339888, "declared.left_fraction[0]": 0.0, "declared.left_fraction[1]": 0.0 …(+12)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Linearized vertex-noise propagation matches Monte Carlo for vertex normals at every tested sigma, with no normal sign flips | {"flipped_normals": 0, "min_normal_dot": 0.9816869074782865, "ratios.vertex normal at valence-5 vertex 0[0]": 0.9917660952871518, "ratios.vertex normal at valence-5 vertex 0[1]": 0.9870253780900908 …(+6)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Linearized propagation matches Monte Carlo for angle-defect curvature when sigma <= 1e-3 | {"angle-defect curvature at valence-5 vertex 0[0]": 1.021280945024692, "angle-defect curvature at valence-5 vertex 0[1]": 1.0212754353510558, "angle-defect curvature at valence-5 vertex 0[2]": 1.0726645278141003, "angle-defect curvature at valence-5 vertex 0[3]": 1.6334652716776885 …(+4)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2 | {"min_ratio": 1.3859890674673723, "sigma_over_h_squared": 0.4401516158412972, "bias[0]": 0.6359997385538216, "bias[1]": -0.021354373765355206} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Sensitivity to vertex noise scales as h^-2 for curvature, h^-1 for normals and h^0 for the marker distance of one declared geodesic | {"local_slopes.angle-defect curvature at valence-5 vertex 0[0]": -2.0333906874577137, "local_slopes.angle-defect curvature at valence-5 vertex 0[1]": -2.008374282747887, "local_slopes.angle-defect curvature at valence-5 vertex 0[2]": -2.0020951666806215, "local_slopes.angle-defect curvature at valence-6 vertex far from valence 5[0]": -1.7973091893267128 …(+24)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | The marker-distance sensitivity is carried by the marker-face vertices at order h^0 while the interior-strip part decreases under refinement | {"interior_slope": 0.3900111336592245, "marker_face_slope": -0.07379389366227951, "interior_local_slopes[0]": 0.29333524873033456, "interior_local_slopes[1]": 0.3607603289270139 …(+13)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Under fixed vertex noise the curvature error grows as the mesh is refined | {"discretization_error[0]": 0.008929308868696362, "discretization_error[1]": 0.0034959528570253084, "discretization_error[2]": 0.0010150870472893647, "discretization_error[3]": 0.00021411616684008372 …(+8)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Linearized per-vertex standard deviations of vertex normals and angle-defect curvature match Monte Carlo at every vertex of icosphere-3 under isotropic and normal-dominant vertex covariances | {"two_vertex_cross_check_max_rel": 1.3169465518103607e-12, "vertices": 642, "models.isotropic.covariance": "isotropic", "models.isotropic.max_abs_z_curvature": 2.9886996985935155 …(+20)} |  | `numerically_verified` | `sha256:32b6b6a9b161` |
| T043 | Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner | null |  | `not_established` | `sha256:32b6b6a9b161` |
| T044 | Residual variance equals geometry variance plus sensor variance in every scenario | {"corridor_left_fraction": 0.0, "fresh_ratio[0]": 0.9950280512988768, "fresh_ratio[1]": 1.006972816193483, "fresh_ratio[2]": 0.9908944486612336 …(+9)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | Nested variance components are consistent with the declared sensor variance and the linearized geometry variance within sampling error | {"between_ratio[0]": 1.0489297857740798, "between_ratio[1]": 0.8875594450358201, "between_ratio[2]": -0.3309768590036692, "between_ratio[3]": 0.9661441782995915 …(+20)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | Under isotropic vertex noise with barycentric markers, geometry uncertainty dominates the baseline marker-distance residual | {"gain": 1.0515093665234345, "baseline.crossover_sensor_sigma": 0.0010515093665234346, "baseline.geometry_share": 0.8155896045575417, "baseline.sigma_geometry": 0.001 …(+25)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | Tangential vertex displacement, mostly of the marker-face vertices, carries most of the marker-distance gain \|grad d\|^2 | {"marker_fraction": 0.9270970105582156, "tangential_fraction": 0.9121248755438055, "gain_split.interior": 0.28391335006097834, "gain_split.interior_normal": 0.2834028088486743 …(+6)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | Under normal-only (shape) vertex noise of the same sigma the sensor dominates the baseline residual | {"crossover_sensor_sigma": 0.00031170668909770053, "gain": 0.31170668909770055, "geometry_share": 0.27987315173062316, "geometry_variance_linear": 9.716106002825055e-08 …(+5)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | Averaging repeated sensor readings leaves the geometry variance as a floor | {"geometry_floor": 1.1056719478865147e-06, "geometry_share[0]": 0.2165575773712226, "geometry_share[1]": 0.5250922153359593, "geometry_share[2]": 0.8155896045575417 …(+13)} |  | `numerically_verified` | `sha256:7aa1e7740cc3` |
| T044 | A real marker-distance sensor on a real scanned part has this geometry and sensor variance split | null |  | `not_established` | `sha256:7aa1e7740cc3` |
| T044 | The geometry sigma of 1e-3 is the accuracy of a real scanned surface | null |  | `not_established` | `sha256:7aa1e7740cc3` |

## Limitations

- T005: Nearby real trajectories on a physical curved surface separate according to this Jacobi law — not established.
- T009: The ranking of lateral versus heading start errors computed here predicts which error dominates the endpoint error of real tool or vehicle paths on physical curved parts — not established.
- T013: A physical cylinder or large-radius torus workpiece shows these separations — not established.
- T017: These validity domains certify first-order path corrections as safe on real machines — not established.
- T018: Curvature signals resolvable here would be resolvable in measured sensor data — not established.
- T021: The shortest route on a physical flat workpiece is the safest route to execute — not established.
- T023: A physical heading sensor closes the shortest loop within the computed heading tolerance — not established.
- T024: Ranking routes by focus margin selects a route that is safe to execute on a physical part — not established.
- T025: A Pareto-optimal route is safe to execute on a physical part — not established.
- T030: Grid-planned path lengths predict distances travelled by a physical vehicle or tool — not established.
- T031: A metric calibrated from physical measurements is accurate enough to decide between near-tied routes — not established.
- T032: A route from this library is safe (or unsafe) to execute on a physical part or vehicle — not established.
- T033: The conformance suite certifies surfaces reconstructed from physical measurements — not established.
- T034: Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from physical measurements — not established.
- T035: The optimal-step law derived here applies to derivatives of measured surface samples — not established.
- T036: Chart-switching geodesic integration is ready for tool paths over physical parts — not established.
- T037: The singularity classification applies to scanned physical parts — not established.
- T038: Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the scanned physical surface — not established.
- T038 is partial: Trace declared geodesics; compare with exact developments; re-derive lengths by a second ciw implementation (strip layout from edge lengths); compare Dijkstra with a dense Floyd-Warshall and, when …
- T039: The observed convergence orders transfer to meshes reconstructed from real scans — not established.
- T040: Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part — not established.
- T041: A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology — not established.
- T042: The refusal catalogue covers every defect present in real scanned surface data — not established.
- T043: Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner — not established.
- T044: A real marker-distance sensor on a real scanned part has this geometry and sensor variance split — not established.
- T044: The geometry sigma of 1e-3 is the accuracy of a real scanned surface — not established.

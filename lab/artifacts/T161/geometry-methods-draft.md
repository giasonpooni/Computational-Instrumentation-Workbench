# Geodesic, Jacobi and route-sensitivity experiments with explicit validity domains

*Generated draft from retained CIW lab reports. Not peer reviewed. Contains no physical measurement.*

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

*Hypothesis.* Zeros of the heading column are conjugate points and zeros of the lateral column are focal points; on constant K > 0 they sit at multiples of pi/sqrt(K) and half-way between, none exist where K <= 0, and on variable curvature none occurs before pi/sqrt(max K).

*Model.* j_head = sn_K, j_lat = cn_K on constant K; Sturm comparison: K <= K_max gives no conjugate point before pi/sqrt(K_max), and K <= 0 gives j_head >= s, j_lat >= 1.

### T009 — Compare lateral-displacement and heading-perturbation columns separately.

*Hypothesis.* On a constant-K background a curvature change at arclength s moves j_lat(L) with weight sn(L-s) cn(s), which decreases along the path when K <= 0 or sqrt(K) L <= pi/2 (before the first focal distance), and j_head(L) with weight sn(L-s) sn(s), which is symmetric about mid-path for every K and vanishes at both ends; exactly, reversing the curvature profile leaves j_head(L) unchanged. The two columns therefore respond differently to where curvature sits along a path and can order paths differently.

*Model.* Endpoint normal displacement = j_lat(L) delta_perp + j_head(L) delta_alpha; delta j(L) = -int G(L, s) j(s) delta K(s) ds with G(L, s) = j_lat(s) j_head(L) - j_head(s) j_lat(L); reversal maps Phi(L) to D Phi(L)^-1 D; model spaces give j_head/j_lat = tan(sqrt(K)L)/sqrt(K), L, tanh(sqrt(-K)L)/sqrt(-K).

### T010 — Generate near-focus and post-focus counterexamples.

*Hypothesis.* Where the first-order separation eps j(s) vanishes (conjugate or focal points) the true separation is of higher order in eps, so the relative first-order error diverges there; past the zero the separation changes sign (image inversion); symmetric configurations raise the order of the true separation.

*Model.* Unit-speed geodesics with j'' + K j = 0; d(s; eps) = eps j(s) + r(eps, s). At s* with j(s*) = 0, d(s*) = r(eps, s*). Torus(2, 1) outer equator: K = 1/3, s* = pi sqrt(3); a reflection theta -> -theta makes the normal offset odd in eps and the O(eps^2) along-track lag -eps^2 sin(2 w s)/(4 w) vanish at s*, so d(s*) = O(eps^3). Unit sphere: all great circles through a point refocus exactly at s = pi; the combined perturbation (lateral eps, heading eps) has j = eps (cos s + sin s) with zero at 3pi/4 where the exact chord is eps^2/2 + O(eps^3).

### T011 — Test coordinate-change invariance under nonlinear parameterizations.

*Hypothesis.* A geodesic and its Jacobi fields are geometric: the same initial point and unit tangent give the same embedded endpoint, length and transfer matrix in any chart, up to integration error; the size of that error, however, depends on the chart.

*Model.* Pullback metric g'(a) = J^T g(phi(a)) J with exact Hessian terms (Reparametrized); the initial tangent is t_a = J^{-1} t_u, so the geometric initial data coincide. RK4 local error ~ h^5 y^(5); a near fold u = c + mu a + a^3/3 makes the chart velocity ~ 1/mu over an arclength window ~ mu^(3/2), so the derivatives entering the error grow as mu decreases.

### T012 — Test frame-change invariance under rotations and tangent-basis changes.

*Hypothesis.* Geodesics and Jacobi fields are intrinsic: an ambient rotation changes only the embedded coordinates (which rotate exactly), and the choice of reference basis for headings is a relabeling; an orientation-reversing basis changes the meaning of '+eps' and of the normal together, so a consistently expressed first-order (Jacobi) separation is unchanged; the finite-eps separation is unchanged exactly only where it is odd in eps.

*Model.* Rotated(base, R): X' = R X, so X'_i . X'_j = X_i . X_j and the second fundamental form is unchanged; in exact arithmetic the chart ODE is identical. A basis (e1', e2') rotated by beta with heading h - beta yields the same unit tangent. In the left-handed basis (e1, -e2) the heading -h + eps is the right-handed heading h - eps, and its +90 degree normal is -N. With d(eps) = J_eps . N = eps j + C2 eps^2 + ..., the left-handed run measured along -N is -d(-eps) = eps j - C2 eps^2 + ..., so (J_eps . (-N)) equals the right-handed J_eps . N to first order in eps; the eps^2 parts have opposite signs, so the finite separations agree exactly only where the separation is odd in eps, as on the unit sphere (sin(s) sin(eps)), and otherwise own/right - 1 = -2 C2 eps / j + O(eps^2).

### T013 — Quantify the flat-cylinder/developable-surface limit.

*Hypothesis.* Intrinsic flatness (K = 0) makes Jacobi fields identical to the plane's even when the surface is curved in space, while chords (extrinsic) differ; along the outer equator of a torus with growing major radius the Jacobi deviation from flat vanishes like K L^3/6 ~ 1/R.

*Model.* Cylinder: the second fundamental form has only the phi-phi entry, so K = 0 and j_head = s exactly, but a helix of angle alpha has chord sqrt((2R sin(L cos(alpha)/(2R)))^2 + (L sin(alpha))^2). Torus(R, 1) outer equator: K = 1/(R + 1), j_head = sin(wL)/w, w = sqrt(K), L - j_head = K L^3/6 - K^2 L^5/120 + ...; the equator is a circle of radius R + 1 with chord deficit 2 (R + 1)(x - sin x), x = L/(2 (R + 1)), = L^3/(24 (R + 1)^2) + ...

### T014 — Test geodesic reversal and path truncation.

*Hypothesis.* The geodesic flow is reversible, so integrating forward, flipping velocities and Jacobi derivatives, and integrating again returns to the start up to the method's global error; truncate-and-continue on an identical grid is the same arithmetic as direct integration.

*Model.* With (u, v, j, j') -> (u, -v, j, -j') the flow over L is inverted. For a one-step method with local error C h^(p+1), the step with -h has local error C (-h)^(p+1); the composition cancels at order h^(p+1) when p is even, so the return error is O(h^(p+1)) for even p and O(h^p) for odd p. Linear check: RK4 R(z) R(-z) = 1 + z^6/72 + ..., midpoint 1 + z^4/4, Euler 1 - z^2.

### T015 — Explore long-horizon numerical drift.

*Hypothesis.* Non-symplectic integrators drift in the first integrals of the geodesic flow (speed, Clairaut constant, angular momentum); adaptive local-error control accumulates a linear secular drift, while a fixed step on these closed or quasi-periodic orbits can keep the drift bounded over long stretches; bounded speed error does not guarantee the right orbit.

*Model.* Energy g(v, v) = 1, torus Clairaut rho^2 phi' and sphere angular momentum X x X' are exact first integrals. A constant speed error gives a constant frequency error on the sphere, hence position error ~ L; a speed error growing ~ L gives position error ~ L^2. On Torus(2, 1) the orbit oscillates between |theta| <= arccos((c - R)/r); a Clairaut drift across the separatrix changes it into an orbit winding around the tube.

### T016 — Study stiff behavior on strongly negative-curvature surfaces.

*Hypothesis.* On K = -k^2 the Jacobi fields grow like e^(kL), so absolute errors are amplified by e^(kL) and a fixed relative accuracy needs steps growing with k; this is intrinsic instability of the flow (eigenvalues +k and -k of the Jacobi linearization), not classical stiffness, so A-stable implicit methods do not remove it. Concentrated negative curvature (Saddle with large c) does not produce exponential growth.

*Model.* j'' = k^2 j, j_head = sinh(kL)/k. A one-step method of order p with R(z) = e^z (1 + c z^(p+1) + ...) on the growing mode has relative error N |c| (kh)^(p+1), so N(tau) = L (L |c| k^(p+1) / tau)^(1/p) ~ k^((p+1)/p): RK4 c = -1/120 (k^(5/4)), implicit midpoint c = 1/12 (k^(3/2), pole at kh = 2 and negative factor beyond), 2-stage Gauss-Legendre (the (2, 2) Pade approximant) c = -1/720 (k^(5/4), no real pole). RK4 stability for the decaying mode needs kh <= 2.785. Saddle ridge y = 0: K = -c^2/(1 + c^2 x^2)^2 ~ -1/(4 s^2) away from the saddle point, where j'' = j/(4 s^2) has solutions |s|^((1 +/- sqrt(2))/2); matching the inbound and outbound power laws through the core of width 1/c gives j_head(L) ~ c^sqrt(2) (matched asymptotics, not a proof).

### T017 — Derive validity domains for the first-order approximation.

*Hypothesis.* The first-order remainder r(eps, s) = d(s) - eps |j(s)| is C2(s) eps^2 + C3(s) eps^3 + ...; the first-order prediction is within relative tolerance tau while |C2 eps + C3 eps^2| <= tau |j|, a domain that shrinks to zero where j vanishes unless the remainder vanishes there too.

*Model.* Unsigned separation d at matched arclength. Unit sphere, pure heading: d = 2 arcsin(|sin s| sin(eps/2)), so C2 = 0, C3 = -|sin s| cos^2 s / 24 and eps_max = sqrt(24 tau)/|cos s| (no collapse at s = pi). Hyperbolic plane: d = 2 asinh(sinh s sin(eps/2)), C2 = 0, eps_max = sqrt(24 tau)/cosh s. Generic paths: C2(s*) != 0 so eps_max ~ tau |j'(s*)| |s - s*| / |C2(s*)| -> 0; symmetric paths: C2 = 0 and eps_max ~ sqrt(|s - s*|).

### T018 — Generate a report comparing intrinsic curvature effects with integrator error.

*Hypothesis.* A curvature effect in a numerical Jacobi field is meaningful only where it exceeds the integrator error; the working expectation was that weak curvature (small |K| L^3/6) is the hard case for coarse steps and low-order methods.

*Model.* Signal S = |j_head(L) - L| (zero on flat surfaces; |K| L^3/6 for small constant K). Truncation error E(h) ~ C h^p. For j'' + K j = 0 the Euler and midpoint errors are themselves proportional to K (the K-free part j = s is integrated exactly), so S/E is independent of K as K -> 0; the RK4 error enters at O(K^2 h^4), so S/E grows like 1/K. The limit is floating point: when S approaches ulp(L) = 4.4e-16 the computed deviation is rounded away. Resolved when S/E >= 10 for every finer step.

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

*Hypothesis.* A two-chart polar atlas of the sphere with exact transitions integrates great circles through or near a pole with an accuracy that does not depend on how close they pass, while a single polar chart loses accuracy or fails near its poles.

*Model.* Chart A: X = R(sin t cos p, sin t sin p, cos t); chart B = R_y(pi/2) X_A, poles on A's equator. det g / R^4 = sin^2 theta in each chart and sin^2 theta_A + sin^2 theta_B = 1 + y^2/R^2 >= 1, so the better chart always has det g / R^4 >= 1/2. Transitions: u_B = X_B^{-1}(X_A(u_A)), v_B = g_B^{-1} J_B^T J_A v_A.

### T037 — Detect coordinate singularities.

*Hypothesis.* A scan into a candidate point, along approach loops that are preimages of geodesic circles, separates coordinate singularities (det g -> 0 or cond g -> infinity with bounded K and circumference ratio 1) from conical points (circumference deficit above 1e-6), curvature singularities (|K| ~ r^a with a <= -0.05) and boundaries at infinite distance (radial speed ~ r^b with b <= -1 + 1e-3); beyond these detection limits it misclassifies. A pointwise guard refuses points whose metric condition number or |K| exceeds its declared bounds, with codes.

*Model.* Fit power laws r^a on r in [1e-8, 1e-5] for det g, cond(g), max|Gamma|, |K| and the radial speed |dX/dr|; a fit is clean when its max log residual is <= 0.05. Radial distance int r^b dr diverges iff b <= -1; circumference ratio C(r) / (2 pi rho(r)) -> 1 at smooth points and sin(alpha) at a cone apex.

### T038 — Build a triangle-mesh intrinsic geodesic solver.

*Hypothesis.* Unfolding across edges (straight in faces, equal angles at edges) yields exact straightest geodesics on developable meshes, and graph distances bound polyhedral distances from above.

*Model.* Straightest geodesic: in each face a straight segment; at an edge the direction keeps its edge component and the magnitude of its perpendicular component (rotation about the edge). Distances: Dijkstra on the edge graph and on the graph of k Steiner points per edge (all pairs inside each face); vertex hits are refused.

### T039 — Measure convergence under mesh refinement.

*Hypothesis.* Straightest mesh geodesics converge to smooth geodesics; the length (metric) error is O(h^2) on inscribed meshes, the lateral error decreases at least like O(h) but not as a clean power law, and edge-graph distances do not converge.

*Model.* Inscribed icosphere: chord/arc metric error O(h^2). Lateral error of a straightest geodesic is driven by the imbalance of vertex curvature point masses on either side of the path (a discrepancy sum), so its order is not a single power law. Prism cylinder: development circumference 2nR sin(pi/n) gives L cos(alpha) (a / sin a - 1) ~ L cos(alpha) pi^2 / (6 n^2), plus a chord-position term atan(s tan a) - s a of relative size O(1/n). Edge graph at a valence-5 source: floor sqrt(5) - 2.

### T040 — Compare smooth and mesh Jacobi approximations.

*Hypothesis.* Mesh Jacobi fields and angle-defect curvature approximate the smooth ones only in the right order of limits and at vertices whose stars become regular.

*Model.* Smooth: j'' + K j = 0, j_head(s) = sin(s) for K = 1. Mesh: curvature is concentrated at vertices (cone points); two geodesics are rotated relative to each other only when a vertex lies between them, so the finite-difference field depends on delta/h. For vertices on a sphere the angle defect tends to K times the circumcentric (Voronoi) cell area, so defect/(A/3) tends to K Voronoi/(A/3): 3K / (4 cos^2(pi/n)) at regular valence-n stars, and a value other than K wherever the star stays irregular.

### T041 — Quantify mesh-quality effects.

*Hypothesis.* Under isotropic tangential jitter of the icosphere at fixed vertex count, worse triangle quality comes with larger curvature error; within the latitude-longitude family and across families quality metrics do not order the errors pairwise (and the barycentric estimator's cross-family ranking reverses with the mixed Voronoi area), although the maximum radius ratio ranks the Voronoi-area curvature and geodesic errors over all tested meshes; Hausdorff convergence does not imply convergence of area, distances, normals or mean curvature.

*Model.* Quality: minimum angle and radius ratio R_circ / (2 r_in). Curvature: angle defect over the barycentric area A/3 or the mixed Voronoi area. Schwarz lantern with m = q n^2 bands: face height sqrt((H/m)^2 + R^2 (1 - cos(pi/n))^2), so area and developed height tend to sqrt(1 + (pi^2 q R / 2H)^2) times their cylinder values while d_H = R (1 - cos(pi/n)) -> 0.

### T042 — Define refusal states for invalid or incomplete surface data.

*Hypothesis.* Each declared defect class in the mesh, trace and query catalogues (MESH_CODES, TRACE_CODES, QUERY_CODES) is detected before geometry is computed and reported under a stable name, and valid meshes pass; defects outside the catalogue are not claimed.

*Model.* Validation order: invalid_shape, empty_mesh, nonfinite_vertex, invalid_face_index, degenerate_face, non_manifold_edge, inconsistent_orientation, non_manifold_vertex, folded_face, unreferenced_vertex, disconnected_components, open_boundary. Tracing refusals: point_outside_face, invalid_direction, boundary_reached, vertex_hit, step_budget_exceeded. Query refusals: unreachable_target, boundary_vertex_curvature, mesh_too_large, invalid_strip.

### T043 — Add uncertainty on vertices, normals, and curvature.

*Hypothesis.* Gaussian vertex noise propagates to geodesic length and normals nearly linearly, while angle-defect curvature needs sigma << h^2 / R for linearization and becomes noisier as the mesh is refined; the fixed-corridor distance stays meaningful only below a strip-dependent noise level.

*Model.* Linearization Var[f] = sigma^2 |grad f|^2 with a central finite-difference Jacobian (step 1e-6) over the vertices that affect f; seeded Monte Carlo with the same isotropic noise. Marker distance = planar distance after unfolding a fixed face strip with barycentric markers.

### T044 — Test geometry uncertainty separately from sensor uncertainty.

*Hypothesis.* For a marker-distance observation on an uncertain surface, geometry and sensor variances add (law of total variance), can be separated by a nested design, and averaging sensor readings cannot remove the geometry part; which part dominates depends on the geometry noise model as well as on the sigmas.

*Model.* Residual r = y - d(V_nominal), y = d(V_nominal + eta) + eps, eta ~ N(0, sigma_g^2 I) per vertex coordinate (or sigma_g along each vertex normal), eps ~ N(0, sigma_s^2) independent. Var(r) = sigma_s^2 + Var_eta(d) ~ sigma_s^2 + sigma_g^2 |grad d|^2, with |grad_n d|^2 in place of |grad d|^2 for normal-only noise. Nested design: within-group variance estimates sigma_s^2, corrected between-group variance estimates the geometry part.

## Results

| Task | Finding | Value | Evidence | Report |
| --- | --- | --- | --- | --- |
| T001 | Sympy-derived metrics, Christoffel symbols and geodesic equations match ciw.lab.surfaces on nine charts | (object of 9 entries; see the source report)  | `independently_verified` | `sha256:1eab2f596a90` |
| T001 | Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts | (object of 9 entries; see the source report)  | `independently_verified` | `sha256:1eab2f596a90` |
| T001 | The hand-derived metrics, Christoffel symbols, geodesic equations and curvatures of the section doc match ciw.lab.surfaces on nine charts | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:1eab2f596a90` |
| T001 | ciw.lab.surfaces Christoffel symbols are symmetric and metric-compatible and its exact metric derivatives match finite differences | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:1eab2f596a90` |
| T001 | Intrinsic curvature from finite-differenced Christoffel symbols matches the ciw Gaussian curvature | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:1eab2f596a90` |
| T001 | The second-fundamental-form curvature (LN - M^2)/det g equals the intrinsic curvature on the six embedded charts (Theorema Egregium) | (object of 6 entries; see the source report)  | `independently_verified` | `sha256:1eab2f596a90` |
| T001 | Nonzero Christoffel symbols do not imply curvature: the polar charts of the plane and cylinder are flat | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:1eab2f596a90` |
| T001 | An extrinsically curved surface can have identically zero Christoffel symbols: the cylinder in (phi, z) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:1eab2f596a90` |
| T002 | ciw Richardson RK4 end states match closed-form geodesics and transfer matrices on the six closed-form charts | (object of 6 entries; see the source report)  | `independently_verified` | `sha256:45c80a7493b1` |
| T002 | ciw Richardson RK4 matches the 34-digit mpmath reference on the saddle path | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:45c80a7493b1` |
| T002 | ciw Richardson RK4 matches the 34-digit mpmath reference on the torus path | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:45c80a7493b1` |
| T002 | ciw Richardson RK4 matches the 34-digit mpmath reference on the gaussian-bump path | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:45c80a7493b1` |
| T002 | Clairaut's integral rho^2 phi' is conserved along the torus reference path | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:45c80a7493b1` |
| T003 | Explicit Euler global endpoint error converges at order 1 on every curved chart | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | Explicit midpoint global endpoint error converges at order 2 on every curved chart | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | Classical RK4 global endpoint error converges at order 4 on every curved chart | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | Adaptive Dormand-Prince error falls with function evaluations at a median effective order near 5 | 5.394386914005626  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | Adaptive Dormand-Prince endpoint error is proportional to the requested tolerance (median over charts) | 0.9710731803196047  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | The adaptive-order checks reject a Dormand-Prince variant that advances with its fourth-order solution | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T003 | No convergence order is observable on flat Cartesian charts: every method is exact to rounding there | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2d7c8c621fc0` |
| T004 | Unit-speed drift max|g(v,v) - 1| scales like h^p for Euler, midpoint and RK4 on every curved chart | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9bc03c016d01` |
| T004 | A non-unit initial speed stays non-unit: g(v,v) remains 1.69 to integrator accuracy | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9bc03c016d01` |
| T004 | The integrator and geodesic/Jacobi right-hand-side code paths contain no state normalization | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9bc03c016d01` |
| T004 | Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding with a first-order endpoint error | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:9bc03c016d01` |
| T004 | Speed drift is at rounding on flat Cartesian charts, where every method is exact | {"cylinder": 2.220446049250313e-16, "plane": 0.0}  | `numerically_verified` | `sha256:9bc03c016d01` |
| T005 | The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on positive, zero and negative curvature | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:67bfd975889c` |
| T005 | The full integrated transfer matrix (lateral column and both rates) follows the model-space law cn_K, sn_K | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:67bfd975889c` |
| T005 | Neighbouring closed-form geodesics on the sphere and hyperbolic plane separate as |sn_K| (heading) and |cn_K| (lateral) per unit perturbation | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:67bfd975889c` |
| T005 | The torus equators are geodesics of constant curvature 1/(r(R+r)) and -1/(r(R-r)) whose Jacobi columns obey the model-space laws | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:67bfd975889c` |
| T005 | Residuals against the model-space law are fourth-order RK4 discretization error | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:67bfd975889c` |
| T005 | ciw joint geodesic + Jacobi transfer matrices match the pinned CSG provider on six constant-curvature paths | (object of 6 entries; see the source report)  | `independently_verified` | `sha256:67bfd975889c` |
| T005 | Nearby real trajectories on a physical curved surface separate according to this Jacobi law | null  | `not_established` | `sha256:67bfd975889c` |
| T006 | Central finite differences of perturbed geodesics converge to the integrated Jacobi columns at second order | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b183a28a42de` |
| T006 | One-sided finite differences converge to the integrated Jacobi columns at first order | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b183a28a42de` |
| T006 | Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:b183a28a42de` |
| T007 | Explicit Euler multiplies det Phi by exactly 1 + h^2 K(gamma_n) per step, so it is not area-preserving where K is nonzero | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:194626340cfc` |
| T007 | Midpoint determinant drift is (h^2/4)(K(L) - K(0)) + O(h^3): second order on variable curvature, third order on constant curvature | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:194626340cfc` |
| T007 | RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable curvature | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:194626340cfc` |
| T007 | Adaptive Dormand-Prince determinant drift decreases in proportion to the tolerance or faster (median over paths) | 1.027724743253913  | `numerically_verified` | `sha256:194626340cfc` |
| T007 | Where K = 0 every method preserves det Phi = 1 exactly, Euler included | 0.0  | `numerically_verified` | `sha256:194626340cfc` |
| T008 | Sphere conjugate points lie at pi R and 2 pi R and focal points at pi R/2 and 3 pi R/2 (R = 1 and R = 2) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:100e9c385322` |
| T008 | On the torus outer equator conjugate points lie at pi sqrt(r(R+r)) multiples and focal points half-way | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:100e9c385322` |
| T008 | No conjugate or focal point occurs on the torus inner equator or the hyperbolic plane (K < 0) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:100e9c385322` |
| T008 | Sturm comparison bound holds on every seeded torus geodesic and declared bump chord that reaches a conjugate point: none occurs before pi/sqrt(max K) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:100e9c385322` |
| T008 | On variable curvature the first focal point is not half the first conjugate distance | {"conjugate": 7.785018442790264, "focal": 3.465336236529439}  | `numerically_verified` | `sha256:100e9c385322` |
| T008 | ciw conjugate and focal points match the pinned CSG provider's focus events on constant-curvature paths | (object of 4 entries; see the source report)  | `independently_verified` | `sha256:100e9c385322` |
| T009 | Endpoint sensitivities to lateral offset (|j_lat(L)|) and heading error (|j_head(L)|) per path | (object of 16 entries; see the source report)  | `numerically_verified` | `sha256:9097e67f0675` |
| T009 | Lateral and heading sensitivities rank paths differently | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9097e67f0675` |
| T009 | On a flat background, to first order, curvature at arclength s moves j_lat(L) with weight L - s (early-weighted) and j_head(L) with weight s(L - s) (symmetric about mid-path) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:9097e67f0675` |
| T009 | Reversing a geodesic leaves j_head(L) unchanged and exchanges j_lat(L) with j_head'(L) (transfer matrix D Phi(L)^-1 D) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:9097e67f0675` |
| T009 | Which starting error dominates the endpoint error of real tool or vehicle paths on physical curved parts | null  | `not_established` | `sha256:9097e67f0675` |
| T010 | On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | On a generic torus geodesic the separation at the conjugate point is O(eps^2) while eps j vanishes | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | The relative first-order error diverges like 1/|s - s*| approaching the conjugate point | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | After the conjugate point the separation inverts sign and still follows eps j | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | Separation does not grow monotonically with length: it nearly vanishes at the conjugate point | 0.00015710984529684768  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | On the unit sphere a pure heading perturbation refocuses exactly: the relative first-order error is uniform and does not diverge at the conjugate point | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | On the unit sphere the image inverts after the conjugate point (signed ratio -1) | -0.999999996944685  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T010 | A combined lateral+heading perturbation on the sphere has true separation eps^2/2 where eps j vanishes | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1fd19f9ceb83` |
| T011 | Converged geodesic endpoints, lengths and Jacobi transfer matrices agree in every chart | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c12e4c2a1f76` |
| T011 | RK4 error falls at least like h^3.7 in every chart; smooth charts show order 4, near-fold charts are still pre-asymptotic (order above 4) at N = 256 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:c12e4c2a1f76` |
| T011 | A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c12e4c2a1f76` |
| T011 | The identity chart reproduces the base-chart integration bit for bit | 0.0  | `numerically_verified` | `sha256:c12e4c2a1f76` |
| T012 | Ambient rotations leave chart trajectories and Jacobi fields unchanged to roundoff | 1.7763568394002505e-15  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | Embedded endpoints rotate exactly with the ambient frame | 1.3732700395566711e-15  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | Gaussian curvature from the rotated second fundamental form is unchanged | 1.9984014443252818e-15  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | Rotating the reference tangent basis (heading measured from e1') leaves every result unchanged | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | A heading change +eps stated in a left-handed basis (e1, -e2) is the geometric perturbation -eps: along the right-handed normal its separation has the opposite sign | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | On the unit sphere, where the separation sin(s) sin(eps) is odd in eps, signed separations are invariant under an orientation-reversing basis change when the perturbation and the normal are both expressed in the new basis | 1.000000000002456  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | On a non-symmetric surface the orientation-reversed separation agrees only to first order: own/right - 1 is proportional to eps (Torus(2, 1) generic path) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:84483eaf043c` |
| T012 | An improper rotation (reflection) is refused as a frame change | "Frame change requires a proper rotation matrix"  | `numerically_verified` | `sha256:84483eaf043c` |
| T013 | Plane and cylinder runs give bitwise identical Jacobi columns (both supply K = 0 to the same Jacobi arithmetic) | 0.0  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | The cylinder is intrinsically flat: K from its second fundamental form vanishes along the helix and the Jacobi field integrated with that K is j_head(s) = s | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | Torus outer-equator Jacobi deviation L - j_head(L) matches the closed form for every major radius | 1.7752341818777495e-09  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | The Jacobi deviation from flat decays like 1/R: exponent -1 in R + 1 = 1/K, with the finite-R offset explained | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | The chord deficit vanishes faster (exponent -2 in R + 1) than the Jacobi deviation (-1) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c4df3021c1be` |
| T013 | A physical cylinder or large-radius torus workpiece shows these separations | null  | `not_established` | `sha256:c4df3021c1be` |
| T014 | Reversal error orders are 1 (Euler), 3 (midpoint) and 5 (RK4): even-order methods gain one order | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:100f23e45d17` |
| T014 | Adaptive forward-then-reversed integration returns to the start at tolerance level | 0.954654133522581  | `numerically_verified` | `sha256:100f23e45d17` |
| T014 | Truncating at L1 and continuing on the same dyadic grid reproduces direct integration bit for bit | {"euler": 0.0, "midpoint": 0.0, "rk4": 0.0}  | `numerically_verified` | `sha256:100f23e45d17` |
| T014 | With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:100f23e45d17` |
| T014 | Adaptive restart at L1 reproduces direct adaptive integration only to tolerance level | 0.010709114151019605  | `numerically_verified` | `sha256:100f23e45d17` |
| T015 | Adaptive DP45 energy error grows linearly with length on the torus and the sphere | {"sphere": 0.9977233904023642, "torus": 0.9335193139363184}  | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T015 | Sphere position error grows like L for fixed-step RK4 and like L^2 for adaptive DP45 | {"adaptive": 2.0529904353388955, "rk4": 1.0151377851200762}  | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T015 | Fixed-step RK4 energy error has a flat (oscillation-dominated) envelope up to L = 160 on the torus and L = 320 on the sphere; on the torus a secular term emerges between L = 160 and 320 | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T015 | Torus Clairaut drift of adaptive DP45 grows linearly with length | 1.0777649788802994  | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T015 | Euler on the torus keeps a bounded energy error but changes the orbit type (Clairaut drift) | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T015 | Euler on the sphere leaves the polar chart before the horizon | 14.0 arclength | `numerically_verified` | `sha256:9f3e3df9bd0e` |
| T016 | Jacobi fields on HyperbolicPlane(k) grow like sinh(kL)/k and adaptive integration resolves them | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T016 | Fixed-step RK4 relative error grows like k^5 and matches L k^5 h^4 / 120 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T016 | RK4 steps for relative accuracy 1e-6 grow like k^(5/4); DP45 accepted steps grow about linearly in k | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T016 | This is intrinsic exponential instability, not stiffness: the Jacobi eigenvalues are +k and -k and accuracy, not stability, sets the step | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T016 | A-stable implicit methods do not remove the growth of the required steps with k (implicit midpoint ~ k^(3/2), 2-stage Gauss-Legendre ~ k^(5/4)); implicit midpoint is qualitatively wrong beyond kh = 2 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T016 | Saddle(c): peak |K| = c^2 but Jacobi growth is polynomial in c; the local exponent of j_head(L) decreases toward sqrt(2) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:62484ad079cb` |
| T017 | On a generic torus geodesic C2(s*) != 0 and the validity domain shrinks to zero at the conjugate point | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | The predicted validity boundary holds: new integrations at eps_max/2 and 2 eps_max give the predicted relative errors tau/2 and 2 tau | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | Unit sphere, pure heading: C2 = 0, C3 = -|sin s| cos^2 s / 24, and the domain does not shrink at s = pi | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | Hyperbolic plane, pure heading: C2 = 0 and eps_max = sqrt(24 tau)/cosh(s) shrinks exponentially | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | Sphere lateral+heading perturbation: C2(s0) = 1/2 at the first-order zero s0 = 3pi/4, eps_max -> 0 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | Torus outer equator: C2 vanishes along the whole path (reflection symmetry); the remainder is cubic | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d9112a850399` |
| T017 | These validity domains certify first-order path corrections as safe on real machines | null  | `not_established` | `sha256:d9112a850399` |
| T018 | RK4 resolves every truncation-limited curvature signal from N = 16 (h = 1/8) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:5be956b775d7` |
| T018 | Weak curvature is harder to resolve for Euler and midpoint only down to K ~ 1e-2: for K <= 1e-2 their ratios no longer depend on K, and the RK4 ratio grows like 1/K | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:5be956b775d7` |
| T018 | Below the floating-point resolution of L no step size resolves the curvature signal | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:5be956b775d7` |
| T018 | Resolvability ratios improve like h^-p with p = 1, 2, 4 (sphere R = 1) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:5be956b775d7` |
| T018 | Flat surfaces (K from the second fundamental form) show no spurious curvature signal for any method or step | {"max_abs_deviation": 0.0, "max_abs_path_curvature": 0.0}  | `numerically_verified` | `sha256:5be956b775d7` |
| T018 | Curvature signals resolvable here would be resolvable in measured sensor data | null  | `not_established` | `sha256:5be956b775d7` |
| T019 | Every enumerated SL(2,Z) basis and every random word reduces exactly to the same canonical Gram form | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:7962325e01bd` |
| T019 | Number of reduced bases equals the predicted stabilizer count (2 generic, 4 boundary or square, 12 hexagonal) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:7962325e01bd` |
| T019 | Integer matrices with det != 1 are refused as SL(2,Z) basis changes | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:7962325e01bd` |
| T019 | Float Gauss reduction agrees with the pinned FTR fold_to_fundamental_domain on reduced tau and the reducing SL(2,Z) matrix (up to -I) | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:7962325e01bd` |
| T020 | Every winding with |m|, |n| <= 6 first returns to its start at t = 1/gcd, displaced by its primitive vector after (|m| + |n|)/gcd edge crossings, and returns gcd times by t = 1 (a gcd-fold cover of the primitive loop) | {"classes": 168, "failures": 0}  | `numerically_verified` | `sha256:2c73387fb872` |
| T020 | For all 120 pairs of the 16 primitive classes with 0 <= m <= 3, |n| <= 3 the transverse intersections number |det(v, w)| | {"failures": 0, "pairs": 120}  | `numerically_verified` | `sha256:2c73387fb872` |
| T020 | The lattice-point count within radius R (including the origin, and its primitive part) equals a brute-force count over a box proven to contain the disk | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:2c73387fb872` |
| T020 | The golden-slope geodesic has positive return gaps phi^-k at Fibonacci returns, with q * gap -> 1/sqrt 5 | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:2c73387fb872` |
| T020 | A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2c73387fb872` |
| T020 | Closed-geodesic lengths |m w1 + n w2|, edge-crossing counts and unit area agree with the pinned FTR loop_length, trace_closed_geodesic and normalized_lattice | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:2c73387fb872` |
| T021 | On the test flat torus every route's heading amplification equals its length (ciw.lab.jacobi), so the length and amplification orders coincide | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b3e99b5f1b85` |
| T021 | For every flat torus and every pair of points the least-sensitive geodesic is a shortest one | true  | `analytic` | `sha256:b3e99b5f1b85` |
| T021 | Shortest equals least heading-sensitive whenever j_head(L) is one strictly increasing function of L for all routes (constant K <= 0); with K > 0 somewhere, or curvature that differs between routes, the equivalence can fail | true  | `analytic` | `sha256:b3e99b5f1b85` |
| T021 | The shortest route on a physical flat workpiece is the safest route to execute | null  | `not_established` | `sha256:b3e99b5f1b85` |
| T022 | Square-torus half-period targets have exactly 2 (edge) or 4 (centre) shortest representatives | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:3495b96c60c3` |
| T022 | Multiplicity census on the 12 x 12 rational grid equals the predicted cut-locus counts | {"1": 121, "2": 22, "4": 1}  | `numerically_verified` | `sha256:3495b96c60c3` |
| T022 | On six lattices the cut-locus vertices satisfy sum over vertices of (k_v - 2) = 2 (Euler characteristic 0) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:3495b96c60c3` |
| T022 | Rounding the target to binary64 changes its shortest-representative multiplicity (exact 1, binary64 2) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:3495b96c60c3` |
| T022 | At exact cut-locus ties that binary64 cannot represent, raw float comparison undercounts some multiplicities while a relative tolerance of 1e-9 recovers all of them | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:3495b96c60c3` |
| T023 | Zeros of the return distance (tau = 0.31 + 1.07i, L = 3) are exactly the primitive lattice directions with |v| <= L | {"found": 18, "predicted": 18}  | `numerically_verified` | `sha256:b842bcb4d6fb` |
| T023 | Near each closing heading the closing error grows at rate |v| (the loop length) per radian | {"max_relative_slope_error": 5.891816425508636e-12}  | `numerically_verified` | `sha256:b842bcb4d6fb` |
| T023 | Closure basins are widest for the shortest (lowest-order) primitive classes | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b842bcb4d6fb` |
| T023 | A physical heading sensor closes the shortest loop within the computed heading tolerance | null  | `not_established` | `sha256:b842bcb4d6fb` |
| T024 | Every fan-search route p -> q on Torus(2, 1) ends on a lift of q with the Wronskian and the Clairaut integral conserved to tolerance | (list of 9 entries; see the source report)  | `independently_verified` | `sha256:6e59e3fa4595` |
| T024 | The torus route set is unchanged when the fan density doubles (1440 to 2880 headings) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6e59e3fa4595` |
| T024 | Rankings by length, amplification and focus margin disagree: the shortest route is neither the least amplifying nor in the best focus-margin group | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6e59e3fa4595` |
| T024 | A flat torus has no conjugate points: j_head(s) = s > 0 (infinite focus margin) | 1.0  | `numerically_verified` | `sha256:6e59e3fa4595` |
| T024 | Ranking routes by focus margin selects a route that is safe to execute on a physical part | null  | `not_established` | `sha256:6e59e3fa4595` |
| T025 | The three-objective front has several members, and every route off the front is dominated by a front member | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:3974b5d19da3` |
| T025 | A Pareto-optimal route is safe to execute on a physical part | null  | `not_established` | `sha256:3974b5d19da3` |
| T026 | Length spectrum (R^2 <= 60), area and systole are exactly invariant under every tested SL(2,Z) change of basis and under reduction | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:bc49b3ccf657` |
| T026 | Area-one float spectra agree to rounding under the same basis changes | 1.1122324405657753e-10  | `numerically_verified` | `sha256:bc49b3ccf657` |
| T026 | Transporting winding labels with M instead of M^-1 breaks length invariance | 6428  | `numerically_verified` | `sha256:bc49b3ccf657` |
| T026 | The length spectrum does not determine the oriented shape: a mirror image is isospectral but not SL(2,Z)-equivalent | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:bc49b3ccf657` |
| T026 | A det-2 integer matrix changes area and spectrum (an index-2 sublattice, not a basis change) | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:bc49b3ccf657` |
| T026 | Closed-geodesic lengths before and after the fold agree with the pinned FTR length_pair and its winding transport | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:bc49b3ccf657` |
| T027 | L-shape and regular octagon are genus-2 translation surfaces with one vertex class | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:173384af382c` |
| T027 | Horizontal cylinders: L-shape circumferences equal the r-cycle lengths, octagon strips have circumferences 2 + sqrt 2 and 1 + sqrt 2 with areas summing to the surface area | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:173384af382c` |
| T027 | Pairings that are not translations, or glue unequal edges, are refused as translation surfaces | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:173384af382c` |
| T028 | All 96 tested rational trajectories on the L-shape close (multiplier 1-3) or end in a saddle connection | {"closed": 90, "saddle_connections": 6, "undecided": 0}  | `numerically_verified` | `sha256:479a23855e94` |
| T028 | Trajectories hitting a cone point are terminated as saddle connections, exactly or within the declared float tolerance | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:479a23855e94` |
| T028 | Float octagon flow reproduces the exact Q(sqrt 2) trajectory (same crossings, closure time) | {"crossings": 20, "time_difference": 1.7763568394002505e-15}  | `numerically_verified` | `sha256:479a23855e94` |
| T028 | A generic float octagon trajectory stays farther than the tolerance from every vertex for 400 crossings | {"crossings": 400, "min_clearance": 0.00027425327282185557}  | `numerically_verified` | `sha256:479a23855e94` |
| T029 | Cone angles: octagon and L-shape one 6 pi point, H(1,1) two 4 pi points, pillowcase four pi points, square and hexagon tori none | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:87fb82bc6a7a` |
| T029 | Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds exactly on every surface with chi known independently of the vertex classes (declared topology; Riemann-Hurwitz for origamis) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:87fb82bc6a7a` |
| T029 | Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points | ["2", "2"]  | `numerically_verified` | `sha256:87fb82bc6a7a` |
| T030 | 4- and 8-neighbour grid shortest paths do not converge to Euclidean length under refinement | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:d594f334f1b9` |
| T030 | Worst-direction metrication error is sqrt 2 - 1 (4-nbr, 45 deg) and sqrt(4 - 2 sqrt 2) - 1 (8-nbr, 22.5 deg) | {"4": 0.4142135623730949, "8": 0.08239220029239402}  | `numerically_verified` | `sha256:d594f334f1b9` |
| T030 | Fast marching converges to Euclidean length under refinement | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:d594f334f1b9` |
| T030 | Grid-planned path lengths predict distances travelled by a physical vehicle or tool | null  | `not_established` | `sha256:d594f334f1b9` |
| T031 | The shortest route switches at eps* = 4 delta, which vanishes as the unperturbed gap vanishes | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:cc2aecfbcbf2` |
| T031 | A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about 127 deg while the minimal length changes continuously | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:cc2aecfbcbf2` |
| T031 | A metric calibrated from physical measurements is accurate enough to decide between near-tied routes | null  | `not_established` | `sha256:cc2aecfbcbf2` |
| T032 | Torus inner equator: the shortest route has larger heading amplification than a longer route | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:47dd2fb2c004` |
| T032 | Torus outer equator: the shortest route has a smaller focus margin than a longer route | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:47dd2fb2c004` |
| T032 | Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:47dd2fb2c004` |
| T032 | Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:47dd2fb2c004` |
| T032 | On the unit sphere the minimizing arc between points at separation pi - delta ends delta before its conjugate point, so minimizing geodesics have no positive lower bound on focus margin | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:47dd2fb2c004` |
| T032 | Search on the saddle (K < 0, simply connected) finds exactly one route, consistent with Cartan-Hadamard uniqueness, so this surface admits no witness | {"routes": 1}  | `numerically_verified` | `sha256:47dd2fb2c004` |
| T032 | Every T032 route search gives the same route set at twice the fan density (720 to 1440 headings) | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:47dd2fb2c004` |
| T032 | A route from this library is safe (or unsafe) to execute on a physical part or vehicle | null  | `not_established` | `sha256:47dd2fb2c004` |
| T033 | Brioschi curvature from the metric alone equals the supplied Gaussian curvature at every sampled point | 7.569957621233518e-11 normalized residual | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | Metrics are symmetric positive definite at every sampled point of the declared domains | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | Christoffel symbols are symmetric in their lower indices at every sampled point | 0.0  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | The connection is metric compatible at every sampled point: d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il | 1.6035737915386125e-16  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | Supplied metric derivatives agree with fourth-order differences of the metric at every sampled point | 8.102692869746462e-12  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | Differences of the exact metric derivatives have symmetric mixed partials (dg is a gradient field) | 5.882921077369268e-13  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | A rigid rotation leaves the torus metric unchanged at every sampled point | 3.665815015959401e-16  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | The conformance suite rejects every seeded defect mutant | {"mutants": 7, "undetected": 0}  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | A curvature-misscaled sphere passes every identity except the Gauss equation | {"failed": ["gauss_equation"], "gauss_residual": 0.167}  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | Metric compatibility cannot detect wrong metric derivatives | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:f4c75df5c628` |
| T033 | The conformance suite certifies surfaces reconstructed from physical measurements | null  | `not_established` | `sha256:f4c75df5c628` |
| T034 | sympy-differentiated metric and metric derivatives match the ciw surface interface on every conformance surface | 4.226085531613157e-16 normalized residual | `independently_verified` | `sha256:c9f341a5aa4f` |
| T034 | sympy.diffgeom Christoffel symbols match the ciw surface interface on seven surfaces | 1.817349289946462e-16 normalized residual | `independently_verified` | `sha256:c9f341a5aa4f` |
| T034 | sympy.diffgeom Riemann curvature R_1212 / det g matches the supplied Gaussian curvature on seven surfaces | 6.977634294120269e-16 normalized residual | `independently_verified` | `sha256:c9f341a5aa4f` |
| T034 | sympy.diffgeom curvature of seven surfaces simplifies exactly to the closed forms restated from ciw.lab.surfaces | {"mismatches": 0, "surfaces": 7}  | `independently_verified` | `sha256:c9f341a5aa4f` |
| T034 | Christoffel symbols and curvature assembled in ciw code from sympy derivatives match the interface on every conformance surface | 5.551115123125783e-16 normalized residual | `numerically_verified` | `sha256:c9f341a5aa4f` |
| T034 | Nested dual-number derivatives of re-expressed embeddings match the metric, dg and Christoffel symbols | 4.4259596565437094e-16 normalized residual | `numerically_verified` | `sha256:c9f341a5aa4f` |
| T034 | Dual-number curvature (Brioschi with exact second derivatives, and LN - M^2) matches the supplied K | 1.0416744990940514e-15 normalized residual | `numerically_verified` | `sha256:c9f341a5aa4f` |
| T034 | Dual numbers reproduce closed-form first, mixed and third derivatives without perturbation confusion | 3.552713678800501e-15  | `numerically_verified` | `sha256:c9f341a5aa4f` |
| T034 | Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see | 0.0608397  | `numerically_verified` | `sha256:c9f341a5aa4f` |
| T034 | Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from physical measurements | null  | `not_established` | `sha256:c9f341a5aa4f` |
| T035 | Central-difference error of the analytic metric derivatives falls as h^2 on the truncation branch | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | Rounding error of central differences grows as 1/h for small steps | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | The optimal step lies within a factor 4 of the predicted h* = (3 eps |g| / |d^3 g|)^(1/3) | (object of 4 entries; see the source report) log10(relative step) | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | The median central-difference error curve bottoms out at or below the predicted minimum error | 4e-11 normalized error | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | At every sampled point the best central difference agrees with the analytic metric derivatives to within twice that point's predicted minimum error | 7.4e-11 normalized error | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | Smaller finite-difference steps can be far less accurate | 6.71 log10 error ratio | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | Quadratic metrics have no truncation branch and a constant metric differences to exactly zero | {"plane_max_error": 0.0, "quadratic_error_at_1e-2": 6.5e-15}  | `numerically_verified` | `sha256:1a9b800121df` |
| T035 | The optimal-step law derived here applies to derivatives of measured surface samples | null  | `not_established` | `sha256:1a9b800121df` |
| T036 | Atlas integration of the great circle through the north pole matches the exact great circle | 4.86e-08 length | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | Atlas accuracy is independent of the distance of closest approach to a pole | 1.21  | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | Atlas integration keeps fourth-order convergence across chart switches | [4.16, 3.962]  | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | Chart transitions are exact: round trip, speed preservation and difference Jacobian | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | The better chart of the atlas always has det g / R^4 >= 1/2, so switching never chatters | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | A single polar chart fails or loses accuracy on great circles passing near its pole | {"cases": 8, "degraded": 7, "failed": 4}  | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | Along the exact meridian (v_phi = 0 exactly) chart A alone crosses the pole accurately | 7.4e-14 length | `numerically_verified` | `sha256:76249f968a6d` |
| T036 | Chart-switching geodesic integration is ready for tool paths over physical parts | null  | `not_established` | `sha256:76249f968a6d` |
| T037 | The sphere pole in the polar chart is a coordinate singularity: det g -> 0 while K stays 1 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | Every declared approach is classified as expected, with no false positive at regular points | (object of 10 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The graph z = r^(3/2) has a curvature singularity although its Monge metric is regular | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | Christoffel blow-up and curvature blow-up are independent near singular points | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | A cone apex and the polar-chart origin share every pointwise exponent; only the circumference ratio separates them | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The hyperbolic chart boundary y -> 0 is at infinite distance, not a singular point | {"det_exponent": -4.0, "radial_speed_exponent": -1.0}  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The boundary of g = y^-1.8 I lies at finite distance and carries a curvature singularity | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | Cases just beyond each classification threshold are misclassified | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | Cartesian loops around the sphere pole in the polar chart make a coordinate singularity read as a conical point | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The pointwise guard and the core check refuse degenerate, blown-up, nonfinite and out-of-chart points with computed codes, and accept curvature below the declared bound | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | Singular surfaces declare their apex refusal codes, and the guard propagates them unchanged | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The core Surface.check accepts sphere-chart points with metric condition number above 1e10 | 11.5 log10 condition number | `numerically_verified` | `sha256:6bd65810c556` |
| T037 | The singularity classification applies to scanned physical parts | null  | `not_established` | `sha256:6bd65810c556` |
| T038 | Straightest geodesics on sheared planar meshes coincide with straight lines | 8.881784197001252e-16 normalized length | `numerically_verified` | `sha256:e5c15b794845` |
| T038 | Traced prism-cylinder geodesics match the exact planar development of the mesh | 7.993605777301127e-15 normalized length | `numerically_verified` | `sha256:e5c15b794845` |
| T038 | The unfolded face-strip distance equals the traced length on every completed sphere trace | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:e5c15b794845` |
| T038 | Heap Dijkstra edge-graph distances agree with a dense Floyd-Warshall recomputation (and scipy.sparse.csgraph when installed) | 8.881784197001252e-16 normalized length | `independently_verified` | `sha256:e5c15b794845` |
| T038 | Nested Steiner-graph distances never increase with k and never fall below the chord | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:e5c15b794845` |
| T038 | Steiner distances between traced endpoints stay above the traced length and approach it as k grows | (object of 2 entries; see the source report) normalized length | `numerically_verified` | `sha256:e5c15b794845` |
| T038 | Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the scanned physical surface | null  | `not_established` | `sha256:e5c15b794845` |
| T039 | Traced-geodesic length defect on icospheres converges at second order | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | Traced-geodesic endpoint error on icospheres decreases at least at first order, with irregular pairwise local orders | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | Prism-cylinder helix endpoint error equals the closed-form development and chord-position prediction, and n^2 error tends to L cos(alpha) pi^2 / 6 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | Edge-graph Dijkstra distance from a valence-5 vertex keeps a relative-error floor that tends to sqrt(5) - 2 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | Steiner graphs with a fixed number of points per edge keep a relative-error floor | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | Heat-method distance error decreases under refinement at first order | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:59cff6c7f169` |
| T039 | The observed convergence orders transfer to meshes reconstructed from real scans | null  | `not_established` | `sha256:59cff6c7f169` |
| T040 | The smooth ciw.lab.jacobi heading column equals sin(L) on the unit sphere | 6.80930867247298e-11  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Finite-difference mesh Jacobi fields with a 0.1 rad heading offset converge to the smooth field | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L) while no vertex lies between the paired geodesics | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | With the mixed Voronoi area the valence-5 curvature estimate converges | (list of 7 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Barycentric angle-defect curvature does not converge pointwise at valence-6 icosphere vertices on the icosahedral mirror planes | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Discrete Gauss-Bonnet holds: angle defects sum to 4 pi on icospheres and 0 on tori | 3.963052108701959e-11  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Angle-defect curvature on regular torus grids converges at second order to the sign-changing K | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:fa79b72df7ce` |
| T040 | Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part | null  | `not_established` | `sha256:fa79b72df7ce` |
| T041 | Seed-averaged curvature RMS error grows and minimum angle falls with tangential jitter | (list of 4 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Jittered meshes are refused as folded exactly when a face is inverted, which happens at every jitter of 0.2 h or more | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | A mesh with a smaller minimum angle can have a smaller geodesic error | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Across mesh families a much smaller minimum angle can come with a smaller barycentric-area angle-defect RMS error | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | With the mixed Voronoi area the regular icosphere has a smaller curvature RMS error than every latitude-longitude sphere of the same vertex count | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Within the latitude-longitude family a mesh with smaller minimum angle and larger radius ratio can have smaller curvature errors under both area choices | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Maximum radius ratio ranks the mixed-Voronoi curvature error and the geodesic error across the valid 642-vertex meshes | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:8df4df9d5d56` |
| T041 | Schwarz lantern meshes converge in Hausdorff distance but not in area | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Schwarz lantern intrinsic height does not converge to the cylinder height | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Lantern angle defects vanish while total absolute mean curvature grows like n^2 and normal tilt converges to atan(pi^2 q R / 2H) instead of 0 | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | A strongly pleated lantern (m = n^2) is refused as folded | {"issues": ["folded_face"], "n": 8, "q": 1.0}  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | Straightest geodesics on planar meshes are exact at any tested triangle quality, while edge-graph distance error changes with the edge directions | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:8df4df9d5d56` |
| T041 | A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology | null  | `not_established` | `sha256:8df4df9d5d56` |
| T042 | Every declared surface-data defect is refused with its named code | (object of 23 entries; see the source report)  | `numerically_verified` | `sha256:f9b726f71c20` |
| T042 | Valid control meshes pass validation without issues | 0  | `numerically_verified` | `sha256:f9b726f71c20` |
| T042 | A mesh with several defects reports all of them in declared order | (list of 3 entries; see the source report)  | `numerically_verified` | `sha256:f9b726f71c20` |
| T042 | A geodesic stopped at a boundary retains its partial length | 0.5700000000000001 normalized length | `numerically_verified` | `sha256:f9b726f71c20` |
| T042 | Curvature and normals evaluated on an unvalidated zero-area face are nonfinite | {"nonfinite_curvatures": 1, "nonfinite_normals": 3}  | `numerically_verified` | `sha256:f9b726f71c20` |
| T042 | The refusal catalogue covers every defect present in real scanned surface data | null  | `not_established` | `sha256:f9b726f71c20` |
| T043 | Linearized vertex-noise propagation matches Monte Carlo for the marker distance along a fixed face corridor | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | The declared marker segment stays in its face corridor for sigma <= 1e-3 but leaves it in over 10% of samples at sigma = 1e-2 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | The noise level at which a fixed face corridor fails depends on the strip, and the declared strip is the most robust of the six | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | Linearized vertex-noise propagation matches Monte Carlo for vertex normals at every tested sigma, with no normal sign flips | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | Linearized propagation matches Monte Carlo for angle-defect curvature when sigma <= 1e-3 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | Sensitivity to vertex noise scales as h^-2 for curvature, h^-1 for normals and h^0 for the marker distance of one declared geodesic | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | The marker-distance sensitivity is carried by the marker-face vertices at order h^0 while the interior-strip part decreases under refinement | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | Under fixed vertex noise the curvature error grows as the mesh is refined | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:430e19747482` |
| T043 | Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner | null  | `not_established` | `sha256:430e19747482` |
| T044 | Nested Monte Carlo sums of squares decompose exactly into between and within parts | 3.1907705800688217e-16  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Residual variance equals geometry variance plus sensor variance in every scenario | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Nested variance components are consistent with the declared sensor variance and the linearized geometry variance within sampling error | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Under isotropic vertex noise with barycentric markers, geometry uncertainty dominates the baseline marker-distance residual | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Tangential vertex displacement, mostly of the marker-face vertices, carries most of the marker-distance gain |grad d|^2 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Under normal-only (shape) vertex noise of the same sigma the sensor dominates the baseline residual | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | Averaging repeated sensor readings leaves the geometry variance as a floor | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:6ac07a8e4d53` |
| T044 | A real marker-distance sensor on a real scanned part has this geometry and sensor variance split | null  | `not_established` | `sha256:6ac07a8e4d53` |
| T044 | The geometry sigma of 1e-3 is the accuracy of a real scanned surface | null  | `not_established` | `sha256:6ac07a8e4d53` |

## Limitations

- T005: Nearby real trajectories on a physical curved surface separate according to this Jacobi law — not established.
- T009: Which starting error dominates the endpoint error of real tool or vehicle paths on physical curved parts — not established.
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
- T039: The observed convergence orders transfer to meshes reconstructed from real scans — not established.
- T040: Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part — not established.
- T041: A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology — not established.
- T042: The refusal catalogue covers every defect present in real scanned surface data — not established.
- T043: Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner — not established.
- T044: A real marker-distance sensor on a real scanned part has this geometry and sensor variance split — not established.
- T044: The geometry sigma of 1e-3 is the accuracy of a real scanned surface — not established.

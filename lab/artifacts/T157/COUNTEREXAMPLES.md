# Counterexample catalogue

Generated from retained lab reports. Each entry refutes the quoted general statement.

## T001: A chart with nonzero Christoffel symbols describes a curved surface

- Finding: Nonzero Christoffel symbols do not imply curvature: the polar charts of the plane and cylinder are flat
- Evidence status: `numerically_verified`
- Witness: `{"chart": "plane-polar", "christoffel": "Gamma^r_tt = -r, Gamma^t_rt = 1/r", "gaussian_curvature": 0.0}`

## T001: A surface that bends in space has nonzero Christoffel symbols in every chart

- Finding: An extrinsically curved surface can have identically zero Christoffel symbols: the cylinder in (phi, z)
- Evidence status: `numerically_verified`
- Witness: `{"chart": "cylinder (phi, z)", "normal_curvature_phi": -1.0}`

## T003: An effective order near 5 is observed whatever solution a DP5(4) pair advances

- Finding: The adaptive-order checks reject a Dormand-Prince variant that advances with its fourth-order solution
- Evidence status: `numerically_verified`
- Witness: `{"median_effective_order": 4.189750093080622, "median_tolerance_exponent": 0.7826906298237313, "variant": "advance with y4"}`

## T003: Every integrator exhibits its nominal convergence order on every surface

- Finding: No convergence order is observable on flat Cartesian charts: every method is exact to rounding there
- Evidence status: `numerically_verified`
- Witness: `{"reason": "Gamma vanishes, so Euler, midpoint and RK4 reproduce u(s) = …", "charts[0]": "plane", "charts[1]": "cylinder", "max_endpoint_error.cylinder": 2.0945586572537882e-14 …(+1)}`

## T004: A computed geodesic whose speed stays exactly 1 is accurate

- Finding: Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding with a first-order endpoint error
- Evidence status: `numerically_verified`
- Witness: `{"endpoint_error": 0.013938653977328111, "method": "Euler with per-step speed renormalization", "steps": 128, "surface": "sphere"}`

## T006: A smaller finite-difference step always gives a more accurate Jacobi estimate

- Finding: Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation)
- Evidence status: `numerically_verified`
- Witness: `{"best_eps": 1e-07, "best_error": 1.2829856288476549e-08, "column": "heading", "eps": 1e-11 …(+2)}`

## T007: The RK4 transfer-matrix determinant drifts at the method's global order h^4

- Finding: RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable curvature
- Evidence status: `numerically_verified`
- Witness: `{"per_step_defect": "-h^6 K^3/72 + O(h^7) (constant K); O(h^6) for smooth K(s)", "orders.bump-radial": 4.989849579481997, "orders.gaussian-bump": 4.992984430686957, "orders.hyperbolic-long": 5.001956250322552 …(+6)}`

## T007: Explicit Euler never preserves the transfer-matrix determinant

- Finding: Where K = 0 every method preserves det Phi = 1 exactly, Euler included
- Evidence status: `numerically_verified`
- Witness: `{"max_drift": 0.0, "per_step_factor": "1 + h^2 K = 1", "charts[0]": "plane", "charts[1]": "cylinder"}`

## T008: The first focal point lies at half the first conjugate distance

- Finding: On variable curvature the first focal point is not half the first conjugate distance
- Evidence status: `numerically_verified`
- Witness: `{"first_conjugate": 7.785018442790264, "first_focal": 3.465336236529439, "heading": 0.8753827096614808, "surface": "torus" …(+2)}`

## T009: Ranking paths by sensitivity to heading error gives the same order as ranking by sensitivity to lateral offset

- Finding: Lateral and heading sensitivities rank paths differently
- Evidence status: `numerically_verified`
- Witness: `{"length": 3.0, "heading.torus-inner-to-outer": 2.8459238929397412, "heading.torus-outer-to-inner": 3.907448965394896, "lateral.torus-inner-to-outer": 1.8031835771642162 …(+3)}`

## T010: The separation at a conjugate point is of exact order eps^2

- Finding: On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2
- Evidence status: `numerically_verified`
- Witness: `{"chord_exponent": 3.0003126899802814, "path": "outer equator, heading 0", "s_star": 5.441398092702653, "surface": "Torus(2, 1)" …(+4)}`

## T010: The first-order separation eps j is accurate along the whole path once eps is small

- Finding: The relative first-order error diverges like 1/|s - s*| approaching the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.04, "path": "start (0, 0.3), heading 0.5", "relative_error": 3.7958513020004787, "s": 6.036272087666448 …(+6)}`

## T010: Neighbouring geodesics stay on the side of the base geodesic they start on (the signed separation keeps its sign along the path)

- Finding: After the conjugate point the separation inverts sign and still follows eps j
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.01, "surface": "Torus(2, 1)", "paths.equator.s_after": 6.801747615878317, "paths.equator.s_before": 2.7206990463513265 …(+8)}`

## T010: Once the first-order prediction has failed at a conjugate point it stays invalid beyond it

- Finding: Past the conjugate point the first-order prediction recovers: the relative error falls again like 1/|s - s*|
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.04, "path": "start (0, 0.3), heading 0.5", "relative_error_after_h": 3.7751136377575443, "relative_error_late": 0.018182546475165935 …(+4)}`

## T010: The separation of neighbouring geodesics grows monotonically with length

- Finding: Separation does not grow monotonically with length: it nearly vanishes at the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"chord_ratio": 0.00015710984529684768, "eps": 0.02, "path": "outer equator", "s_half": 2.7206990463513265 …(+2)}`

## T010: The relative first-order error diverges at every conjugate point

- Finding: On the unit sphere a pure heading perturbation refocuses exactly: the relative first-order error of the embedded chord is uniform (2 sin(eps/2)/eps - 1, about -eps^2/24) and does not diverge at the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.02, "perturbation": "pure heading", "relative_error": -1.6590023186879854e-05, "s": 3.129320807286708 …(+1)}`

## T010: The image of a family of geodesics keeps its orientation beyond a conjugate point

- Finding: On the unit sphere the image inverts after the conjugate point (signed ratio -1)
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.02, "perturbation": "pure heading", "s_after": 3.9269908169872414, "s_before": 2.356194490192345 …(+4)}`

## T011: A change of chart that preserves the geometry leaves the fixed-step integration error unchanged

- Finding: A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor
- Evidence status: `numerically_verified`
- Witness: `{"chart": "u_axis = c + mu a + a^3/3, mu = 0.1 (det J >= 0.1)", "factor_sphere": 1069563.9789352638, "factor_torus": 98109.38296653116, "plane_error_base": 1.1102230246251565e-14 …(+5)}`

## T012: Finite-eps signed separations are invariant under an orientation-reversing basis change expressed consistently

- Finding: On a non-symmetric surface the orientation-reversed separation agrees only to first order: own/right - 1 is proportional to eps (Torus(2, 1) generic path)
- Evidence status: `numerically_verified`
- Witness: `{"path": "Torus(2, 1) from (0, 0.3), heading 0.5, L = 3", "eps[0]": 0.001, "eps[1]": 0.01, "eps[2]": 0.04 …(+3)}`

## T013: Surfaces with identical Jacobi fields (intrinsic geometry) have identical chords between corresponding points

- Finding: Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord
- Evidence status: `numerically_verified`
- Witness: `{"chord_cylinder": 2.5382081174896274, "chord_plane": 3.0000000000000018, "heading": 0.6, "length": 3.0 …(+4)}`

## T013: Intrinsic (Jacobi) and extrinsic (chord) signatures of curvature vanish at the same rate in the flat limit

- Finding: The chord deficit vanishes faster (exponent -2 in R + 1) than the Jacobi deviation (-1)
- Evidence status: `numerically_verified`
- Witness: `{"chord_exponent_in_R_plus_1": -1.9999677621250946, "jacobi_exponent_in_R_plus_1": -0.9974853377423354}`

## T014: Forward-then-reversed integration with a method of order p returns to the start with error proportional to h^p

- Finding: Reversal error orders are 1 (Euler), 3 (midpoint) and 5 (RK4): even-order methods gain one order
- Evidence status: `numerically_verified`
- Witness: `{"mean_orders.euler": 1.049562088626885, "mean_orders.midpoint": 3.0007881457739116, "mean_orders.rk4": 4.998833708326354, "surfaces[0]": "sphere" …(+2)}`

## T014: Truncate-and-continue reproduces fixed-step integration bit for bit for any truncation length

- Finding: With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails
- Evidence status: `numerically_verified`
- Witness: `{"L1": 1.13, "L2": 3.0, "N1": 113, "N2": 300 …(+6)}`

## T014: The forward-then-reversed return error of an integrator measures its global error

- Finding: Symmetric implicit methods (implicit midpoint, 2-stage Gauss-Legendre; fixed-point stage solve to 1e-13) return from forward-then-reversed integration at rounding level at every step size
- Evidence status: `numerically_verified`
- Witness: `{"forward_position_error": 4.261941548118264e-07, "method": "gauss-legendre-2", "return_error": 3.352873534367973e-14, "steps": 20 …(+1)}`

## T015: The fixed-step RK4 position error on the sphere is the phase error of its speed error (a constant speed error gives position error ~ L)

- Finding: Fixed-step RK4 sphere position error is cross-track: the precessing orbit plane, not the speed error, sets it
- Evidence status: `numerically_verified`
- Witness: `{"L": 320.0, "along_over_integrated_speed_error": -0.16027306137215694, "along_track_at_320": -2.5966475792261697e-05, "cross_track_at_320": 0.0005408552664065206 …(+6)}`

## T015: The energy error of a non-symplectic fixed-step integrator grows linearly with length at every horizon

- Finding: Fixed-step RK4 energy error has a flat (oscillation-dominated) envelope up to L = 160 on the torus and L = 320 on the sphere; on the torus a secular term emerges between L = 160 and 320
- Evidence status: `numerically_verified`
- Witness: `{"exponent": 0.027506236135538223, "h": 0.125, "method": "rk4", "surface": "Torus(2, 1)" …(+10)}`

## T015: A bounded energy (speed) error implies a qualitatively correct long-horizon geodesic

- Finding: Euler on the torus keeps a bounded energy error but changes the orbit type (Clairaut drift)
- Evidence status: `numerically_verified`
- Witness: `{"h": 0.125, "max_energy_error": 0.03499256018697694, "method": "euler", "surface": "Torus(2, 1)" …(+2)}`

## T016: An implicit (A-stable) integrator removes the growth of the step count with k on strongly negatively curved surfaces

- Finding: A-stable implicit methods do not remove the growth of the required steps with k (implicit midpoint ~ k^(3/2), 2-stage Gauss-Legendre ~ k^(5/4)); implicit midpoint is qualitatively wrong beyond kh = 2
- Evidence status: `numerically_verified`
- Witness: `{"beyond_pole_kh": 2.2857142857142856, "gauss_legendre_2_steps[0]": 15, "gauss_legendre_2_steps[1]": 35, "gauss_legendre_2_steps[2]": 83 …(+29)}`

## T016: Jacobi growth is exponential in sqrt(peak |K|) times the length

- Finding: Saddle(c): peak |K| = c^2 but Jacobi growth is polynomial in c; the local exponent of j_head(L) decreases toward sqrt(2)
- Evidence status: `numerically_verified`
- Witness: `{"c": 16384.0, "log_j_head": 14.554722514358353, "peak_abs_curvature": 268435456.0, "sqrt_peak_times_L": 32768.0}`

## T016: An integrator with a six times smaller error constant on the constant-curvature Jacobi equation is correspondingly more accurate for the geodesic itself

- Finding: The constant-curvature Jacobi error constants do not carry over to the nonlinear hyperbolic geodesic: Gauss-Legendre's end-point and speed errors are not 1/6 of RK4's
- Evidence status: `numerically_verified`
- Witness: `{"endpoint_error_ratio": 0.703619247447021, "j_head_error_ratio": 0.16771886729645658, "k": 1.0, "max_speed_error_ratio": 0.6953013007698434 …(+1)}`

## T016: Gauss-Legendre's constant-curvature advantage over RK4 (j_head error 1/6 of RK4's) holds on variable-curvature geodesics

- Finding: On ridge and oblique saddle-surface geodesics (full nonlinear system) Gauss-Legendre and RK4 both converge at order 4, but their j_head error ratio depends on the geodesic instead of being the constant-curvature 1/6, while Gauss-Legendre's speed error is several times smaller at equal steps
- Evidence status: `numerically_verified`
- Witness: `{"geodesic": "c = 16 oblique", "j_head_error_ratio": 2.4526153875369117, "curvature_range[0]": -0.558213912126984, "curvature_range[1]": -0.07160818589836329}`

## T016: Over long horizons RK4's speed error drifts while a symmetric integrator's stays bounded

- Finding: On escaping geodesics (hyperbolic plane, oblique saddle geodesic) RK4's speed error does not drift either: it saturates like the symmetric methods', so boundedness there does not distinguish them
- Evidence status: `numerically_verified`
- Witness: `{"hyperbolic-plane.rk4_envelope": 2.4755903290096803e-05, "hyperbolic-plane.rk4_growth_after_one_eighth": 1.0, "saddle-oblique.rk4_envelope": 3.417308181563605e-05, "saddle-oblique.rk4_growth_after_one_eighth": 1.0}`

## T017: The validity domain of the first-order approximation shrinks to zero at every conjugate point

- Finding: Unit sphere, pure heading: C2 = 0, C3 = -|sin s| cos^2 s / 24, and the domain does not shrink at s = pi
- Evidence status: `numerically_verified`
- Witness: `{"perturbation": "pure heading", "surface": "unit sphere", "tau": 0.01, "eps_max_near_pi.0.999": 0.4899060501928133 …(+1)}`

## T018: Weaker intrinsic curvature is harder to resolve at a fixed step size

- Finding: Weak curvature is harder to resolve for Euler and midpoint only down to K ~ 1e-2: for K <= 1e-2 their ratios no longer depend on K, and the RK4 ratio grows like 1/K
- Evidence status: `numerically_verified`
- Witness: `{"truncation_dominated_ratios_at_N16.euler.sphere R=1": 8.733790907567972, "truncation_dominated_ratios_at_N16.euler.sphere R=10": 5.585141207296921, "truncation_dominated_ratios_at_N16.euler.sphere R=100": 5.565415886526588, "truncation_dominated_ratios_at_N16.euler.sphere R=1e4": 5.565217996909997 …(+7)}`

## T018: Refining the step size always makes a nonzero curvature effect resolvable

- Finding: Below the floating-point resolution of L no step size resolves the curvature signal, although the methods' truncation errors alone would resolve it
- Evidence status: `numerically_verified`
- Witness: `{"computed_deviation": 0.0, "signal": 1.3333333333333334e-16, "surface": "Sphere(1e8), K = 1e-16", "steps[0]": 4 …(+5)}`

## T019: Any integer change of basis generates the same lattice

- Finding: Integer matrices with det != 1 are refused as SL(2,Z) basis changes
- Evidence status: `numerically_verified`
- Witness: `{"canonical[0]": "5", "canonical[1]": "2", "canonical[2]": "7", "gram[0]": 5 …(+9)}`

## T020: A floating-point geodesic direction can be irrational (non-closing)

- Finding: A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction
- Evidence status: `analytic`
- Witness: `{"closes_after_alpha_turns": 562949953421312, "exact_fraction": "910872158600853/2^49", "float_phi": 1.618033988749895}`

## T022: Converting a target to binary64 preserves its shortest-representative multiplicity

- Finding: Rounding the target to binary64 changes its shortest-representative multiplicity (exact 1, binary64 2)
- Evidence status: `numerically_verified`
- Witness: `{"binary64_multiplicity": 2, "exact_multiplicity": 1, "target": "(1/2 + 2^-60, 1/4)", "binary64_target[0]": 0.5 …(+1)}`

## T022: Floating-point distance comparison finds every shortest representative

- Finding: At exact cut-locus ties that binary64 cannot represent, raw float comparison undercounts some multiplicities while a relative tolerance of 1e-9 recovers all of them
- Evidence status: `numerically_verified`
- Witness: `{"binary64": 2, "exact": 3, "lattice": "generic", "tolerance": 3 …(+2)}`

## T024: The shortest route also minimizes amplification and maximizes focus margin s_c - L

- Finding: Rankings by length, amplification |j_head(L)| and focus margin s_c - L (s_c the first zero of j_head) disagree: the shortest route is neither the least amplifying nor in the best focus-margin group
- Evidence status: `numerically_verified`
- Witness: `{"best_margin_group[0].amplification": 35.84781929262236, "best_margin_group[0].focus_margin": null, "best_margin_group[0].heading": 1.328869465787598, "best_margin_group[0].length": 6.354490781532511 …(+50)}`

## T026: Winding labels transform with the same matrix as the basis

- Finding: Transporting winding labels with M instead of M^-1 breaks length invariance
- Evidence status: `numerically_verified`
- Witness: `{"correct_rule": "c' = M^-1 c", "lattice": "generic", "mismatches": 6428, "q_g_of_winding": "5" …(+18)}`

## T026: Equal length spectra imply SL(2,Z)-equivalent oriented lattices

- Finding: The length spectrum does not determine the oriented shape: a mirror image is isospectral but not SL(2,Z)-equivalent
- Evidence status: `numerically_verified`
- Witness: `{"same_canonical": false, "same_spectrum": true, "canonical[0]": "5", "canonical[1]": "-2" …(+4)}`

## T029: Every polygon vertex of a glued surface is a cone singularity

- Finding: Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points
- Evidence status: `numerically_verified`
- Witness: `{"surface": "regular hexagon, opposite sides glued", "vertex_classes": 2, "cone_angles_over_pi[0]": "2", "cone_angles_over_pi[1]": "2"}`

## T030: Grid shortest-path lengths converge to geodesic length as the grid is refined

- Finding: 4- and 8-neighbour grid shortest paths do not converge to Euclidean length under refinement
- Evidence status: `independently_verified`
- Witness: `{"direction": "45 deg", "ratio": 1.414213562373095, "stencil": "4-neighbour", "grids[0]": 30 …(+2)}`

## T031: A small metric perturbation changes the shortest route only slightly

- Finding: A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about 127 deg while the minimal length changes continuously
- Evidence status: `numerically_verified`
- Witness: `{"delta": "1/1000", "eps_after": 0.0041, "eps_before": 0.0039, "eps_star": "1/250" …(+8)}`

## T032: The shortest geodesic between two points has the least heading amplification

- Finding: Torus inner equator: the shortest route has larger heading amplification than a longer route
- Evidence status: `independently_verified`
- Witness: `{"alternative.amplification": 2.2424499641265205, "alternative.focus_margin": null, "alternative.heading": -0.6654954777289914, "alternative.length": 7.4048964512362 …(+16)}`

## T032: The shortest geodesic has the largest focus margin s_c - L (farthest from its first conjugate point)

- Finding: Torus outer equator: the shortest route has a smaller focus margin s_c - L (s_c the first zero of j_head) than a longer route
- Evidence status: `independently_verified`
- Witness: `{"alternative.amplification": 29.49806945310811, "alternative.focus_margin": null, "alternative.heading": -1.3844466400297355, "alternative.length": 6.723214266466979 …(+16)}`

## T032: The shortest route between two points is unique

- Finding: Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique
- Evidence status: `numerically_verified`
- Witness: `{"first.amplification": 3.893697160114145, "first.focus_margin": 2.0712536359131057, "first.heading": -0.932959589373497, "first.length": 6.30866770803647 …(+16)}`

## T032: Low heading amplification certifies a robust (locally minimizing) route

- Finding: Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point
- Evidence status: `independently_verified`
- Witness: `{"alternative.amplification": 6.488420201199406, "alternative.focus_margin": -1.9973831828910869, "alternative.heading": 0.0, "alternative.length": 5.8779306681414605 …(+15)}`

## T032: Minimizing geodesics on the unit sphere have focus margin s_c - L bounded below by a positive constant

- Finding: On the unit sphere the minimizing arc between points at separation pi - delta ends delta before its conjugate point, so minimizing geodesics have no positive lower bound on the focus margin s_c - L
- Evidence status: `numerically_verified`
- Witness: `{"kind": "near-conjugate conditioning", "surface": "unit sphere", "delta[0]": 0.1, "delta[1]": 0.01 …(+7)}`

## T033: Metric symmetry, positive definiteness, Christoffel symmetry, metric compatibility and derivative consistency together certify a surface implementation

- Finding: A curvature-misscaled sphere passes every identity except the Gauss equation
- Evidence status: `numerically_verified`
- Witness: `{"gauss_residual": 0.167, "mutant": "misscaled-curvature (sphere R = 2 returning K = 1/R)", "failed[0]": "gauss_equation"}`

## T033: A metric-compatible connection certifies the metric derivatives

- Finding: Metric compatibility cannot detect wrong metric derivatives
- Evidence status: `numerically_verified`
- Witness: `{"mutant": "saddle with dg negated", "failed[0]": "derivative_consistency", "failed[1]": "gauss_equation"}`

## T034: Finite, index-symmetric hand-coded metric derivatives are correct

- Finding: Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see
- Evidence status: `numerically_verified`
- Witness: `{"mutant": "gaussian-bump with f_xy dropped", "normalized_error": 0.0608}`

## T035: Decreasing the finite-difference step always improves agreement with the analytic derivative

- Finding: Smaller finite-difference steps can be far less accurate
- Evidence status: `numerically_verified`
- Witness: `{"log10_error_at_1e-12": -3.993, "log10_error_at_h_opt": -10.7, "log10_h_opt": -5.25, "surface": "sphere"}`

## T035: Every smooth metric shows an O(h^2) truncation branch in central-difference error

- Finding: Quadratic metrics have no truncation branch and a constant metric differences to exactly zero
- Evidence status: `numerically_verified`
- Witness: `{"error_at_1e-2": 3.4e-15, "surface": "saddle, E = 1 + c^2 x^2 (degree-2 metric)"}`

## T036: Fixed-step RK4 in a single polar chart integrates every great circle as accurately as a chart-switching atlas at the same step count

- Finding: A single polar chart fails or loses accuracy on great circles passing near its pole
- Evidence status: `numerically_verified`
- Witness: `{"log10_delta_0.1_atlas_error": -7.246, "log10_delta_0.1_single_error": -5.141, "log10_delta_failed[0]": -2, "log10_delta_failed[1]": -3 …(+2)}`

## T036: Fixed-step integration in a single polar chart across its pole always fails

- Finding: From a declared angular-momentum seed of 1e-12, chart A alone crosses both poles of the meridian at 400 RK4 steps to within 1e-6
- Evidence status: `numerically_verified`
- Witness: `{"error": 1.03344e-07, "seed": 1e-12, "steps": 400}`

## T036: A single-chart integration that crosses a pole accurately at one step count stays accurate at nearby step counts

- Finding: Across declared seeds from 1e-15 to 1e-10, chart A alone fails on the meridian whenever an RK4 stage point lands within 0.9 d* of a pole and never beyond 1.5 d*, d* = (L0 h^4)^(1/5), but no single multiple of d* separates the outcomes at every seed (350 to 450 RK4 steps at 1e-12, the step counts within 2 d* at the other half-decade seeds)
- Evidence status: `numerically_verified`
- Witness: `{"error_at_400": 1.03344e-07, "failed_steps[0]": 355, "failed_steps[1]": 377, "failed_steps[2]": 399 …(+2)}`

## T037: The scan detects every coordinate singularity, that is every det g -> 0 or cond g -> infinity at finite distance with bounded K

- Finding: The scan misses the removable coordinate singularity of the plane in the cube-root chart
- Evidence status: `numerically_verified`
- Witness: `{"approach": "plane-cube-root-chart", "observed": "unclassified", "true": "coordinate_singularity"}`

## T037: A nondegenerate metric chart implies bounded Gaussian curvature

- Finding: The graph z = r^(3/2) has a curvature singularity although its Monge metric is regular
- Evidence status: `numerically_verified`
- Witness: `{"K_at_1e-8": 112500000.0, "det_g_at_1e-8": 1.0, "surface": "z = r^(3/2)"}`

## T037: Christoffel-symbol blow-up indicates a curvature singularity

- Finding: Christoffel blow-up and curvature blow-up are independent near singular points
- Evidence status: `numerically_verified`
- Witness: `{"sphere_pole_K": 1.0, "sphere_pole_max_gamma_at_1e-8": 100000000.0}`

## T037: Pointwise metric and curvature invariants distinguish a removable coordinate singularity from a conical point

- Finding: A cone apex and the polar-chart origin share every pointwise exponent; only the circumference ratio separates them
- Evidence status: `numerically_verified`
- Witness: `{"angle_deficit": 3.14, "conical": "cone alpha = pi/6 apex", "removable": "plane in polar chart at r = 0"}`

## T037: The approach scan classifies every singular point correctly

- Finding: Cases just beyond each classification threshold are misclassified
- Evidence status: `numerically_verified`
- Witness: `{"cone-small-deficit-apex.observed": "coordinate_singularity", "cone-small-deficit-apex.true": "conical_singularity", "conformal-0.9995-boundary.observed": "infinite_distance_boundary", "conformal-0.9995-boundary.true": "curvature_singularity" …(+2)}`

## T037: The scan classifies a degenerate point independently of the approach loops chosen by the caller

- Finding: Cartesian loops around the sphere pole in the polar chart make a coordinate singularity read as a conical point
- Evidence status: `numerically_verified`
- Witness: `{"cartesian_loops": "conical_singularity", "point": "sphere north pole, chart A", "polar_loops": "coordinate_singularity"}`

## T038: A straightest geodesic shorter than pi on a mesh inscribed in the unit sphere is a shortest path between its endpoints

- Finding: Traced straightest geodesics are never shorter than the exact distance between their endpoints, yet on every icosphere level 1-4 some of length 2 (below pi) are not shortest paths, by an excess that falls with refinement
- Evidence status: `numerically_verified`
- Witness: `{"exact": 1.9999625298889687, "excess": 3.747011103172326e-05, "level": 4, "start": 1 …(+1)}`

## T038: A straightest geodesic through a vertex is the limit of the straightest geodesics that pass it on either side

- Finding: At a saddle vertex every end direction between the one-sided limits of the geodesics passing it is reached by a shortest path through the vertex, at a cone vertex none is, and the Polthier-Schmies continuation bisects the two limits
- Evidence status: `numerically_verified`
- Witness: `{"continued_polar": 2.356194490192345, "total_angle_over_pi": 1.5, "one_sided_polar[0]": 1.570798612509183, "one_sided_polar[1]": 3.1415903678755073}`

## T038: The first cut point of a straightest geodesic on a convex polyhedral surface is where it crosses the cut ray of one vertex, as if that vertex carried all the curvature

- Finding: A straightest geodesic can stop being shortest before every single vertex's isolated-cone prediction, when the digon between it and the other shortest path encloses several vertices
- Evidence status: `numerically_verified`
- Witness: `{"cut_point": 1.1709509040694328, "level": 2, "prediction": 1.3189570176010945, "start": 5 …(+2)}`

## T039: Edge-graph shortest paths converge to the geodesic distance under mesh refinement

- Finding: Edge-graph Dijkstra distance from a valence-5 vertex keeps a relative-error floor that tends to sqrt(5) - 2
- Evidence status: `numerically_verified`
- Witness: `{"mesh": "icosphere levels 1-4, source vertex 0 (valence 5)", "max_signed_relative_error[0]": 0.1448504685578944, "max_signed_relative_error[1]": 0.21078482135841448, "max_signed_relative_error[2]": 0.22958417381401675 …(+1)}`

## T039: Refining the mesh alone makes a Steiner-graph distance with fixed k converge to the geodesic distance

- Finding: Steiner graphs with a fixed number of points per edge keep a relative-error floor
- Evidence status: `numerically_verified`
- Witness: `{"k": 3, "max_abs_relative_error[0]": 0.030675797123964732, "max_abs_relative_error[1]": 0.008227705748911074, "max_abs_relative_error[2]": 0.011039725230062247 …(+1)}`

## T040: On a fixed mesh the finite-difference Jacobi field converges to the smooth Jacobi field as the perturbation tends to zero

- Finding: At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L) while no vertex lies between the paired geodesics
- Evidence status: `numerically_verified`
- Witness: `{"delta": 1e-05, "j_fd": 2.0, "j_smooth": 0.9092974268256817, "levels[0]": 2 …(+4)}`

## T040: The angle defect over one third of the incident area converges pointwise to the Gaussian curvature under refinement

- Finding: Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K
- Evidence status: `numerically_verified`
- Witness: `{"finest_value": 1.1459056232841356, "limit": 1.1458980337503153, "mesh": "icosphere, 12 valence-5 vertices"}`

## T040: Angle-defect curvature with the barycentric area converges pointwise at the valence-6 vertices of icosphere refinements

- Finding: Barycentric angle-defect curvature does not converge pointwise at valence-6 icosphere vertices on the icosahedral mirror planes
- Evidence status: `numerically_verified`
- Witness: `{"base_edge_max_error[0]": 0.0019319256495926584, "base_edge_max_error[1]": 0.002431873259649331, "base_edge_max_error[2]": 0.0025567564634368933, "base_edge_max_error[3]": 0.002587970803474504 …(+8)}`

## T041: The dihedral fold check (folded_face) refuses every jittered icosphere that has an inverted face

- Finding: Over 60 further seeds per amplitude the dihedral fold check accepts some jittered meshes with an inverted face
- Evidence status: `numerically_verified`
- Witness: `{"amplitude": 0.2, "face": 854, "normal_radial": -0.3410445645964844, "seed": 20261944 …(+6)}`

## T041: Tangential jitter of 0.2 h or more inverts a face of the level-3 icosphere for every seed

- Finding: A tangential jitter of 0.2 h inverts a face for some but not all of 60 further seeds
- Evidence status: `numerically_verified`
- Witness: `{"amplitude": 0.2, "meshes": 60, "without_inverted_face": 21}`

## T041: A larger minimum angle implies a smaller geodesic error

- Finding: A mesh with a smaller minimum angle can have a smaller geodesic error
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "icosphere-3", "worse_quality": "icosphere-3-jitter0.05-s20260939"}`

## T041: Minimum angle orders the barycentric angle-defect curvature error across mesh families

- Finding: Across mesh families a much smaller minimum angle can come with a smaller barycentric-area angle-defect RMS error
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "icosphere-3", "note": "the icosphere's RMS is dominated by its 12 valence-5 …", "worse_quality": "uv-sphere-20x32"}`

## T041: Within one mesh family better triangle quality (larger minimum angle, smaller radius ratio) implies smaller curvature error

- Finding: Within the latitude-longitude family a mesh with smaller minimum angle and larger radius ratio can have smaller curvature errors under both area choices
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "uv-sphere-40x16", "worse_quality": "uv-sphere-10x64"}`

## T041: Hausdorff convergence of a mesh to a surface implies convergence of area

- Finding: Schwarz lantern meshes converge in Hausdorff distance but not in area
- Evidence status: `numerically_verified`
- Witness: `{"area_ratio": 1.5872559353818663, "bands": 1024, "hausdorff": 0.001204543794827706, "n": 64 …(+1)}`

## T041: Hausdorff convergence of a mesh implies convergence of its geodesic distances

- Finding: Schwarz lantern intrinsic height does not converge to the cylinder height
- Evidence status: `numerically_verified`
- Witness: `{"H": 1.0, "n": 64, "traced_height": 1.5878935490351203}`

## T041: Hausdorff convergence of a mesh implies convergence of its normals and curvature

- Finding: Lantern angle defects vanish while total absolute mean curvature grows like n^2 and normal tilt converges to atan(pi^2 q R / 2H) instead of 0
- Evidence status: `numerically_verified`
- Witness: `{"n": 64, "tilt_deg": 50.96720303293561, "total_abs_mean_curvature": 11440.883001930979}`

## T041: The dihedral fold check refuses a mesh only when a face is inverted

- Finding: A strongly pleated lantern (m = n^2) is refused as folded although no face normal points towards the axis
- Evidence status: `numerically_verified`
- Witness: `{"min_normal_radial": 0.20107436521303879, "n": 8, "q": 1.0}`

## T042: Curvature and normal formulas can be evaluated safely on unvalidated meshes

- Finding: Curvature and normals evaluated on an unvalidated zero-area face are nonfinite
- Evidence status: `numerically_verified`
- Witness: `{"defect": "zero-area face", "mesh": "square plus a collinear face"}`

## T042: Structural validation, including the dihedral fold check, refuses every mesh with an inverted face

- Finding: Without a declared centre the validator accepts a jittered icosphere with an inverted face
- Evidence status: `numerically_verified`
- Witness: `{"amplitude": 0.2, "seed": 20261940}`

## T043: A fixed face corridor (fixed mesh combinatorics) represents the perturbed marker geodesic at every tested vertex-noise level

- Finding: The declared marker segment stays in its face corridor for sigma <= 1e-3 but leaves it in over 10% of samples at sigma = 1e-2
- Evidence status: `numerically_verified`
- Witness: `{"left_fraction": 0.39525, "sigma": 0.01}`

## T043: The fixed-corridor marker distance is valid for sigma <= 1e-3 on icosphere-3 whichever declared geodesic carries the markers

- Finding: The noise level at which a fixed face corridor fails depends on the strip, and the declared strip, which has the largest vertex margin of the six, stays in its corridor for sigma <= 1e-3
- Evidence status: `numerically_verified`
- Witness: `{"left_fraction": 0.3715, "sigma": 0.001, "start_index": 0, "vertex_margin": 0.002659028956720544}`

## T043: The strip with the largest vertex margin leaves its fixed face corridor least often at every tested noise level

- Finding: A strip with a smaller vertex margin than the declared strip leaves its face corridor less often at every tested sigma >= 3e-3
- Evidence status: `numerically_verified`
- Witness: `{"declared_start_index": 5, "more_robust_start_index": 1, "declared_left_fraction[0]": 0.02825, "declared_left_fraction[1]": 0.41075 …(+4)}`

## T043: First-order (linearized) propagation of vertex noise is adequate for angle-defect curvature at sigma = 1e-2 on icosphere-3

- Finding: First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2
- Evidence status: `numerically_verified`
- Witness: `{"h": 0.1507297051948821, "min_ratio": 1.3859890674673723, "sigma": 0.01}`

## T043: Refining the mesh reduces the error of angle-defect curvature

- Finding: Under fixed vertex noise the curvature error grows as the mesh is refined
- Evidence status: `numerically_verified`
- Witness: `{"sigma": 0.001, "coarse.discretization_error": 0.008929308868696362, "coarse.h": 0.2993320753185587, "coarse.level": 2 …(+7)}`

## T044: Whether geometry or sensor noise dominates the marker-distance residual is fixed by sigma_g and sigma_s alone

- Finding: Under normal-only (shape) vertex noise of the same sigma the sensor dominates the baseline residual
- Evidence status: `numerically_verified`
- Witness: `{"isotropic_geometry_share": 0.8155896045575417, "normal_only_geometry_share": 0.27987315173062316, "sigma_geometry": 0.001, "sigma_sensor": 0.0005}`

## T044: Averaging repeated measurements drives the observation error to zero

- Finding: Averaging repeated sensor readings leaves the geometry variance as a floor
- Evidence status: `numerically_verified`
- Witness: `{"geometry_floor": 1.1056719478865147e-06, "repeats": 64, "variance": 1.1753748253402434e-06}`

## T046: c = s - kappa(0)^2 s^3/24 + O(s^5) along every surface geodesic

- Finding: A torus geodesic with varying curvature keeps the s^4 term kappa0 kappa0'/24 at the start point
- Evidence status: `numerically_verified`
- Witness: `{"heading_rad": 0.6, "kappa0": 0.4967476850397676, "kappa0_prime": -0.222669223718571, "s4_coefficient": -0.0046087696213116995 …(+3)}`

## T046: For constant space curvature kappa, c = 2 sin(kappa s/2)/kappa

- Finding: Constant curvature alone does not give the circle chord: a cylinder helix with torsion deviates at order s^5 by kappa^2 tau^2/720
- Evidence status: `numerically_verified`
- Witness: `{"alpha_deg": 45.0, "gap": 2.839002859589268e-05, "kappa": 0.5000000000000001, "s": 0.8 …(+2)}`

## T047: The chord-versus-geodesic correction is a property of the surface alone, independent of path direction

- Finding: Axial rulings (alpha = 90 deg) have zero chord correction; circumferential paths have the largest coefficient 1/(24 R^2)
- Evidence status: `numerically_verified`
- Witness: `{"alpha_deg_max_correction": 0, "alpha_deg_zero_correction": 90, "coefficient_at_0_deg_R1": 0.04166666666657247, "surface": "cylinder"}`

## T047: On an intrinsically flat surface (K = 0) a chord equals the geodesic distance

- Finding: Zero Gaussian curvature does not make chord and geodesic distance equal
- Evidence status: `numerically_verified`
- Witness: `{"alpha_deg": 0, "chord": 0.09995833854135666, "s": 0.1, "surface": "cylinder R = 1"}`

## T049: A common focal-length error rescales every measured distance by one factor

- Finding: A common focal-length error leaves same-depth chords unchanged and scales only the depth component
- Evidence status: `numerically_verified`
- Witness: `{"circumferential_relative_change": 0.00018211500785203505, "focal_error_px": 5.0, "rig": "rectified", "ruling_relative_change": 1.5543122344752192e-15}`

## T050: The Brown-Conrady radial model can be inverted everywhere in the image

- Finding: Strong barrel distortion (k1 = -0.6) folds inside the image, so the radial model is not invertible there
- Evidence status: `numerically_verified`
- Witness: `{"distorted_radius": 0.481525, "fold_radius_distorted": 0.4969039949999533, "image_corner_radius": 0.5333333333333333, "k1": -0.6 …(+2)}`

## T050: The centre of a circular marker's image ellipse is the projection of the marker's centre

- Finding: Under full perspective the image-ellipse centre of a tilted circular marker is displaced from the projected marker centre by rho^2 (X t_z - Z t_xy) / (Z (Z^2 - rho^2 t_z)); the displacement vanishes for fronto-parallel markers and under a weak-perspective (affine) projection
- Evidence status: `numerically_verified`
- Witness: `{"camera": "left", "marker": 10, "marker_radius_m": 0.008, "offset_px[0]": -0.1782213150095231 …(+1)}`

## T051: Integer pixel rounding always adds 1/12 px^2 to the noise variance

- Finding: Without a random grid phase the quantization variance is not 1/12: an integer-aligned coordinate with sigma = 0.1 px has almost no total error, because rounding cancels the Gaussian noise, so the error variance is far below both sigma^2 + 1/12 and sigma^2
- Evidence status: `numerically_verified`
- Witness: `{"sample_variance": 0.0, "sigma_px": 0.1, "true_coordinate_px": 512.0}`

## T051: With a uniform grid phase, integer-rounded pixel errors give chord variance (sigma^2 + 1/12) sum J^2

- Finding: A grid phase shared by all markers of a camera correlates their rounding errors: the chord variance follows J Sigma J^T with the sawtooth covariance, and without Gaussian noise (sigma = 0) the independent-error law (sigma^2 + 1/12) sum J^2 is refuted
- Evidence status: `numerically_verified`
- Witness: `{"alpha_deg": 45.0, "arc_m": 0.06, "grid_phase": "shared by all markers of a camera", "predicted_departure": -0.44041853644450124 …(+2)}`

## T052: Fitting reading = a x + c without a direction term gives unbiased scale and bias under backlash

- Finding: Least squares that ignores backlash biases the offset estimate by the projection of the play error onto [x, 1]
- Evidence status: `numerically_verified`
- Witness: `{"backlash_m": 2e-05, "bias_error_over_backlash": 0.49788561310110585, "naive_bias_z": 40.156837606636714}`

## T053: The mean orientation error from a constant gyro bias grows as |bias| t

- Finding: On a body rotating about z, transverse gyro bias produces a bounded orientation error 2 |b_perp| / |omega| instead of |b_perp| t
- Evidence status: `numerically_verified`
- Witness: `{"max_transverse_rad": 0.0006365184325471376, "stationary_transverse_rad": 0.009999999999999929, "bias_rad_per_s[0]": 0.002, "bias_rad_per_s[1]": 0.0 …(+4)}`

## T055: Zero-filled gaps can be recognized from the values alone

- Finding: Values alone recognize zero fills only in favourable signals: a neighbour-median detector finds every fill 50 sigma from zero, but on a stationary quantized encoder axis a fill equals a genuine zero-count reading
- Evidence status: `numerically_verified`
- Witness: `{"dropped": 46, "fills_equal_to_genuine_readings": 26, "identical_stream_gap_m": 0.0, "seed": 55 …(+1)}`

## T055: The drop rate alone determines how much estimates degrade

- Finding: Hold-estimate error follows q E[age] + r; bursts at the same drop rate raise it
- Evidence status: `numerically_verified`
- Witness: `{"bernoulli_mse": 2.6377978571622868e-05, "burst_mse": 9.379806297833248e-05, "drop_rate": 0.2}`

## T057: The smoothed estimate is closer to truth than the filtered estimate at every sample

- Finding: Smoothing does not reduce the error of every individual sample
- Evidence status: `numerically_verified`
- Witness: `{"fraction_worse": 0.28528333333333333, "seed": 57}`

## T061: A covariance check on one bench run detects a 10% misstatement for every sensor

- Finding: At this sample size a 10% understatement of every declared variance is detected with probability above 0.999 per variance for the 10-20 Hz streams but only about half the time for the 2 Hz tracker (exact chi-square power); replicate tracker noise streams reproduce that power
- Evidence status: `numerically_verified`
- Witness: `{"detected_this_run": false, "detection_power": 0.4901452071629382, "samples": 2000, "sensor": "tracker" …(+2)}`

## T062: Ignoring correlation between sensor noises is harmless

- Finding: Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES exceeds the 99% upper bound at nearly every tick
- Evidence status: `numerically_verified`
- Witness: `{"common_mode_covariance": 0.09, "grand_mean_nees": 5.514214143152381, "nominal": 4.0}`

## T062: A passing mean-NIS chi-square test shows the measurement noise model is correct

- Finding: The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); among innovation-based tests, which need no ground truth, only the whitened-innovation covariance test exposes the missing cross-correlation
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean_nis": 3.9174506536208624, "whitened_cross_term": 0.6662952145647711}`

## T062: Assuming correlation between sensor noises where there is none is a conservative, harmless choice

- Finding: Assuming a common-mode correlation that the noise does not have makes the filter underconfident (run-averaged NEES below the 99% lower bound at nearly every tick) and less accurate than the block-diagonal filter, while its NIS exceeds the 99% upper bound at nearly every tick: the reverse mismatch is neither harmless nor invisible
- Evidence status: `numerically_verified`
- Witness: `{"common_mode_assumed": 0.09, "grand_mean_nees": 3.1372673894736733, "grand_mean_nis": 7.722661371152557, "nominal": 4.0 …(+2)}`

## T063: Rotating a measurement into another frame without rotating its covariance is harmless

- Finding: Rotating a position measurement by 35 degrees without rotating its covariance inflates the mean Mahalanobis distance to tr(P^-1 R P R^T) and multiplies the 99% gate rejection rate
- Evidence status: `numerically_verified`
- Witness: `{"mean_d2": 14.49496148224193, "nominal_mean_d2": 2.0, "rejection_rate_at_99pct_gate": 0.42499, "rotation_deg": 35.0}`

## T063: A unit change of the measurement vector alone leaves gating decisions unchanged

- Finding: Converting a state to millimetres while keeping its covariance in metres multiplies every squared Mahalanobis distance by exactly 10^6
- Evidence status: `numerically_verified`
- Witness: `{"consistent_mean_d2": 1.9955687303281424, "mean_d2": 1995568.7303281422}`

## T064: Lateral position uncertainty grows monotonically with distance travelled

- Finding: Lateral variance collapses at the sphere's conjugate point s = pi: the heading contribution vanishes because j_head(pi) = 0, leaving only the initial lateral variance
- Evidence status: `numerically_verified`
- Witness: `{"s": 3.141592653589793, "surface": "unit sphere", "variance_ratio_to_peak": 0.009048433205893418}`

## T064: A first-order (Phi P Phi^T) covariance is accurate for any heading uncertainty below 0.1 rad

- Finding: The heading standard deviation at which the first-order variance is 10% too large is about ten times smaller on the hyperbolic plane at s = 3 than on the sphere near its conjugate point (s = 0.9 pi)
- Evidence status: `numerically_verified`
- Witness: `{"relative_variance_error": 0.5892623400956292, "s": 3.0, "sigma_heading": 0.1, "surface": "hyperbolic plane"}`

## T065: Successive filtered estimates can be averaged as independent samples

- Finding: Treating 20 successive filtered outputs as independent underestimates the variance of their average by the exact factor V_20 / (P/20), and a naive 95% interval covers far less often
- Evidence status: `numerically_verified`
- Witness: `{"n": 20, "naive_95_coverage": 0.5095463774037217, "variance_ratio": 8.078252159913767}`

## T066: Innovations can be tested against the raw sensor covariance

- Finding: Normalizing innovations by the raw sensor covariance R inflates NIS to tr(R^-1 S) and fails the chi-square test at nearly every tick
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean_nis": 3.3625270796684887, "nominal": 2.0}`

## T066: Post-fit residuals have the sensor covariance R

- Finding: Normalizing post-fit residuals by the raw R deflates them to tr(S^-1 R) < 2, which would hide an inconsistent filter
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean": 1.2273555630758988, "nominal": 2.0}`

## T067: The false-rejection rate of a gated filter equals the nominal 1 - p

- Finding: Closed loop, a gated filter rejects valid readings more often than 1 - p at p = 0.9: a rejected reading signals a large prior error that the filter then keeps
- Evidence status: `numerically_verified`
- Witness: `{"p": 0.9, "rate": 0.135175, "rate_after_a_rejection": 0.29975705475612036, "wilson[0]": 0.12964829573860664 …(+1)}`

## T067: A chi-square gate on raw-R Mahalanobis distance has false-rejection rate 1 - p

- Finding: Gating the NIS computed with the raw sensor covariance R at the 99% quantile rejects valid readings at more than twice the nominal 1% rate
- Evidence status: `numerically_verified`
- Witness: `{"nominal": 0.01, "rate": 0.0366}`

## T068: A chi-square gate protects a filter from gross outliers

- Finding: Cold-start lock-out: a gross outlier in the first reading passes the gate under the broad prior, and the corrupted state then rejects runs of valid readings; every lock-out run starts this way
- Evidence status: `numerically_verified`
- Witness: `{"longest_valid_rejection_streak": 36, "rmse_gated": 1.9753980773555264, "rmse_oracle": 0.13815641328637332, "run": 150}`

## T068: Gating never degrades the estimate when the data are clean

- Finding: On clean data the 99% gate raises false alarms at about 1% and increases the mean squared error: the rejected valid readings are the ones that would have corrected a large prior error
- Evidence status: `numerically_verified`
- Witness: `{"paired_z": 9.577530243653909, "rmse_gated": 0.13882950665782604, "rmse_ungated": 0.13535119430297377}`

## T068: A Mahalanobis gate removes injected outliers and so improves the estimate

- Finding: Subtle 0.3 m outliers pass the gate at the predicted low detection rate; for them gating costs more accuracy than the outliers do
- Evidence status: `numerically_verified`
- Witness: `{"detection_rate": 0.08105263157894736, "magnitude_m": 0.3, "rmse.gated": 0.1426001142269842, "rmse.oracle": 0.13794394236831456 …(+1)}`

## T069: A zero placeholder for a missing reading is harmless

- Finding: Zero-filling missing readings by hand drags the estimate toward the origin and destroys consistency by the amount the exact joint moments predict
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean_nees": 2305.6002151702023, "nominal": 4.0, "rmse_m": 3.7711939615873082}`

## T070: A stale sensor clock always shows up as a bias in the filter innovations

- Finding: With the stale camera as the only position sensor the lag is invisible to the innovations: the lagged constant-velocity path is itself a constant-velocity path, so the estimate is biased by about -tau E[v] while the mean test passes
- Evidence status: `numerically_verified`
- Witness: `{"sensors": "camera only", "camera_mean_z[0]": 0.4228861059098964, "camera_mean_z[1]": -0.5395726400825746, "position_error_mean_m[0]": -0.09866380307289671 …(+1)}`

## T071: A passing NIS test shows that the sensor frames agree

- Finding: Near the rotation centre (ticks 1-20) the same mismatch goes undetected by the per-tick NIS test at this sample size, where its predicted inflation is only about 2%; the run-level grand-mean NIS there agrees with the exact prediction and comes close to the family bound against the nominal value, so a pooled test nearly flags what the per-tick test misses
- Evidence status: `numerically_verified`
- Witness: `{"early_grand_nis": 4.1307324105019, "early_z_vs_nominal": 3.323742365144154, "fraction_inside": 0.95, "rotation_deg": 2.0 …(+1)}`

## T071: A frame mismatch always produces NIS inflation

- Finding: A rotated sensor fused alone keeps NIS consistent (its readings form a rotated constant-velocity path) while NEES grows: the estimate is confidently in the wrong frame
- Evidence status: `numerically_verified`
- Witness: `{"final_nees": 343.25789304896057, "late_grand_nis": 1.9946921697683178, "sensors": "rotated tracker alone"}`

## T072: Readings taken after calibration expiry can be fused like the others

- Finding: Under a declared post-expiry drift of 1 cm per tick, refusing expired readings keeps NEES consistent (with a growing covariance) while fusing them makes the filter overconfident by the amount the exact joint moments predict
- Evidence status: `numerically_verified`
- Witness: `{"drift_m_per_tick": 0.01, "nominal": 4.0, "post_expiry_grand_nees": 14.024327327919162}`

## T073: A coasting track stays statistically consistent for any gap because its covariance grows

- Finding: Under an unmodelled 0.2 rad/s turn during the gap, the expected NEES of the coasting track exceeds the 99% chi-square(4) quantile after a finite gap even though its covariance keeps growing; a radius rule protects against this only if its radius is below the ellipse radius reached by then
- Evidence status: `numerically_verified`
- Witness: `{"gap_ticks": 44, "omega_rad_s": 0.2, "radius_at_inconsistency_m": 4.590603032028965}`

## T077: A retained bundle identity binds every provenance record stored in the bundle

- Finding: The energy replay bundle identity excludes its replay receipt and verification
- Evidence status: `numerically_verified`
- Witness: `{"bundle": "bundle:B1", "function": "src/ciw/telemetry.py:_bundle_digest", "removed": "replay_receipts", "replaced": "verification"}`

## T077: Reopening a workspace checks every retained provider runtime identity against CIW's pin

- Finding: A telemetry bundle whose GSIE runtime revision is forged, with every unkeyed digest over it recomputed, reopens, and a replay on the bound stack refuses it
- Evidence status: `numerically_verified`
- Witness: `{"edit": "runtimes.gsie.revision replaced by a revision that is not …", "reopen": "accepted", "replay": "Telemetry replay runtime identity mismatch", "recomputed[0]": "bundle_digest" …(+4)}`

## T077: A telemetry replay compares every retained runtime identity field except the host paths

- Finding: A telemetry bundle whose GSIE runtime identity carries a forged adapter_version and an injected key, with every unkeyed digest over it recomputed, reopens, and the runtime identity comparison a replay makes before executing accepts it
- Evidence status: `numerically_verified`
- Witness: `{"reopen": "accepted", "replay_comparison": "accepted", "compared_fields[0]": "revision", "compared_fields[1]": "source_tree" …(+14)}`

## T077: A saved workspace carries no host path of an operator-bound provider checkout

- Finding: A saved telemetry workspace retains the host paths of the bound provider checkouts and interpreter in its runtime identities
- Evidence status: `numerically_verified`
- Witness: `{"restored_binding": "none: the reopened session offers no telemetry operation", "fields[0]": "runtimes.<role>.repository_root", "fields[1]": "runtimes.<role>.python_executable", "roles[0]": "ppda" …(+3)}`

## T079: Canonical-content identity treats numerically equal JSON numbers as equal content

- Finding: CIW canonical comparison is type-sensitive (1 != 1.0): a consistent int-for-float rewrite is refused by the stale log seal and, once resealed, retained as distinct evidence rather than aliased
- Evidence status: `numerically_verified`
- Witness: `{"ciw_canonical_equal": false, "edit": "unit diagonal of runtime.workload.solver_settings and …", "python_equal": true, "unsealed": "Retained log digest differs" …(+3)}`

## T080: A retained oscillator result stays bound to the execution occurrence that produced it

- Finding: Surviving mutant alias.swap-pairing: two statistics occurrences whose execution and result identities and creation times are exchanged consistently reopen re-paired
- Evidence status: `numerically_verified`
- Witness: `{"description": "R1 and R2 exchange executions consistently: result …", "name": "alias.swap-pairing", "observed": "accepted", "recompute": "local" …(+5)}`

## T080: Saved execution and result selection revisions are checked against a retained selection history

- Finding: Surviving mutant revision.gap: a workspace whose selection revision jumps to 1000, with records claiming revision 999, reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "selection revision 1000 with execution and result claiming …", "name": "revision.gap", "observed": "accepted", "recompute": "local" …(+3)}`

## T081: Reopen re-analysis protects retained energy results from numerical forgery

- Finding: Surviving mutant energy-source.resealed: a resealed edit of the retained source log, with every derived record rebuilt, reopens with a different gross energy
- Evidence status: `numerically_verified`
- Witness: `{"description": "first measurement counter sample lowered by 50 mJ in the …", "name": "energy-source.resealed", "observed": "accepted", "recompute": "full" …(+6)}`

## T081: Unkeyed record seals detect every edit to a retained numerical result

- Finding: Surviving mutant oscillator-stats.resealed: an in-bounds statistics edit with a recomputed seal reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "statistics mean moved to the midpoint of [minimum …", "name": "oscillator-stats.resealed", "observed": "accepted", "recompute": "local" …(+3)}`

## T081: Saved statistics satisfy |mean| <= rms <= max(|min|, |max|)

- Finding: Surviving mutant oscillator-stats.impossible-moments: resealed statistics that no sample set can have reopen
- Evidence status: `numerically_verified`
- Witness: `{"description": "R1 mean set to its maximum and rms to half of it (|mean| > …", "name": "oscillator-stats.impossible-moments", "observed": "accepted", "recompute": "local" …(+15)}`

## T081: Every retained oscillator result is sealed against edits

- Finding: Surviving mutant oscillator-stats.legacy: an edit to an unsealed legacy result reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "legacy (unsealed) statistics mean moved to the midpoint", "name": "oscillator-stats.legacy", "observed": "accepted", "recompute": "none" …(+3)}`

## T082: A retained execution occurrence binds its creation time

- Finding: Surviving mutant fresh.created-at-shift: a backdated execution and result pair reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "created_at backdated identically in execution and result …", "name": "fresh.created-at-shift", "observed": "accepted", "recompute": "local" …(+2)}`

## T083: A replay receipt cannot be moved from its replay bundle onto another retained bundle

- Finding: Surviving mutant receipt.transplanted-full: a replay receipt moved onto an original sibling and rebuilt from the two bundles reopens there, and the replay reopens without it
- Evidence status: `numerically_verified`
- Witness: `{"description": "receipt moved from the replay B1 onto the original sibling …", "name": "receipt.transplanted-full", "observed": "accepted", "recompute": "full" …(+5)}`

## T083: A replay receipt can only be retained on a bundle produced by that replay

- Finding: Surviving mutant receipt.fabricated: a receipt written onto a never-replayed original, claiming it replays its sibling, reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "a new receipt written onto the never-replayed original B0b …", "name": "receipt.fabricated", "observed": "accepted", "recompute": "full" …(+5)}`

## T083: A replay bundle cannot be retained without its replay receipt

- Finding: Surviving mutant receipt.deleted-resealed: a replay bundle reopens without its replay receipt once the unkeyed catalog receipt seal is recomputed, listed with no receipt like an original execution
- Evidence status: `numerically_verified`
- Witness: `{"description": "replay_receipts removed from the replay bundle; the catalog …", "name": "receipt.deleted-resealed", "observed": "accepted", "recompute": "full" …(+5)}`

## T083: A workspace saved with a replay-receipt seal cannot reopen without it

- Finding: Surviving mutant receipt.deleted-seal-removed: a replay bundle reopens without its replay receipt when the catalog receipt seal is removed too, as from a workspace saved before the seal existed
- Evidence status: `numerically_verified`
- Witness: `{"description": "replay_receipts removed from the replay bundle and …", "name": "receipt.deleted-seal-removed", "observed": "accepted", "recompute": "none" …(+5)}`

## T084: Unkeyed record seals detect every replay-provenance forgery

- Finding: Surviving mutant receipt-source.sibling-execution-resealed: a receipt re-pointed, with its verification subject, at a sibling execution of the same bytes reopens once the catalog receipt seal is recomputed
- Evidence status: `numerically_verified`
- Witness: `{"description": "receipt source and verification subject moved together to a …", "name": "receipt-source.sibling-execution-resealed", "observed": "accepted", "recompute": "full" …(+4)}`

## T085: A retained replay cannot be dated before its source bundle

- Finding: Surviving mutant receipt-replayed.reidentified-bundle: a replay re-sessioned and dated before its source, with recomputed digests, reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "replay bundle dated 2001, before its source, with a new …", "name": "receipt-replayed.reidentified-bundle", "observed": "accepted", "recompute": "full" …(+5)}`

## T086: Sealed operation records refuse verification-subject claims outside their schema

- Finding: Surviving mutant oscillator-subject.injected: a sealed result carrying an injected verification subject naming its sibling result reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "subject_ref naming the sibling result R2 injected into a …", "name": "oscillator-subject.injected", "observed": "accepted", "recompute": "local" …(+3)}`

## T087: Sealed operation records refuse verification-method claims outside their schema

- Finding: Surviving mutant oscillator-method.injected: a sealed result carrying an injected verification_method reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "verification_method field injected into a sealed result …", "name": "oscillator-method.injected", "observed": "accepted", "recompute": "local" …(+3)}`

## T088: Sealed operation records refuse independence claims outside their schema

- Finding: Surviving mutant oscillator-independent.injected: sealed records carrying an injected independent: true reopen
- Evidence status: `numerically_verified`
- Witness: `{"description": "independent: true injected into a sealed execution and …", "name": "oscillator-independent.injected", "observed": "accepted", "recompute": "local" …(+3)}`

## T089: Sealed operation records refuse admission claims outside their schema

- Finding: Surviving mutant oscillator-admission.injected: a sealed result carrying an injected state_admission reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "state_admission: admitted injected into a sealed result …", "name": "oscillator-admission.injected", "observed": "accepted", "recompute": "local" …(+3)}`

## T090: Unkeyed record seals detect a forged provider runtime identity

- Finding: Surviving mutant oscillator-runtime.both: a provider runtime forged identically in execution and result reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "runtime forged identically in execution and result; both …", "name": "oscillator-runtime.both", "observed": "accepted", "recompute": "local" …(+5)}`

## T090: Reopening a workspace detects a forged analysis runtime identity

- Finding: Surviving mutant energy-runtime.all-bundles: a consistently forged analysis code digest reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "code_sha256 forged in every energy bundle; every digest …", "name": "energy-runtime.all-bundles", "observed": "accepted", "recompute": "full" …(+5)}`

## T090: Reopening a workspace detects forged runtime dependency versions

- Finding: Surviving mutant energy-runtime.python-version: forged dependency versions reopen
- Evidence status: `numerically_verified`
- Witness: `{"description": "python_version forged as 3.99.0 in every energy bundle …", "name": "energy-runtime.python-version", "observed": "accepted", "recompute": "full" …(+5)}`

## T091: Reopening a saved workspace performs no numerical recomputation

- Finding: Reopening recomputes the retained energy analysis to validate content
- Evidence status: `numerically_verified`
- Witness: `{"ciw.energy_records.analyze calls": 7, "new execution occurrences": 0}`

## T092: Reopen validation of a retained numerical-heat bundle establishes that its values are the pinned SCR heat-kernel output

- Finding: A retained numerical-heat bundle whose values no provider computed passes reopen validation
- Evidence status: `numerically_verified`
- Witness: `{"bundle_id": "sha256:954c39d69990ad4cce8c3bc853d4a79ac19d2dc4a756c4c484bec …", "runtime_repository_root": "/fabricated/not-a-provider-checkout", "reference_values[0]": 0, "reference_values[1]": 16 …(+8)}`

## T093: Every refused request leaves the session's in-memory state unchanged

- Finding: A refused recording operation is retained as a refused execution record
- Evidence status: `numerically_verified`
- Witness: `{"new_files": 1, "request": "operation.execute ciw.lab-unregistered.v1", "changed[0]": "executions"}`

## T095: Every CIW JSON reader refuses nonfinite numbers

- Finding: session.read_json accepts an overflowing number as infinity while the exchange and workbench parsers refuse it
- Evidence status: `numerically_verified`
- Witness: `{"fixture": "overflow.json", "parsed_value": "inf", "reader": "ciw.session.read_json"}`

## T095: session.read_json refuses every malformed workspace file with a ValueError

- Finding: session.read_json does not refuse a 20000-deep array with a ValueError
- Evidence status: `numerically_verified`
- Witness: `{"fixture": "deep-nesting.json", "observed": "RecursionError"}`

## T095: Session.from_workspace refuses every malformed workspace with a ValueError

- Finding: Session.from_workspace raises AttributeError, not a ValueError refusal, on a workspace holding only its version
- Evidence status: `numerically_verified`
- Witness: `{"error_type": "AttributeError", "fixture": "incomplete-workspace.json", "content.workspace_version": 3}`

## T095: Saved workspaces with fields outside the schema are refused on reopen

- Finding: Session.from_workspace accepts an unknown top-level field and drops it on re-save
- Evidence status: `numerically_verified`
- Witness: `{"dropped_on_resave": true, "field": "lab_unexpected_field", "reopened": true}`

## T095: Workbench source errors name the source, not a bound runtime

- Finding: Workbench source.add reports malformed source JSON with text that names a bound runtime
- Evidence status: `numerically_verified`
- Witness: `{"fixture": "duplicate-key.json", "message": "MALFORMED_RESPONSE: The bound runtime did not return …", "request": "source.add (energy-accuracy)"}`

## T096: Every exchange artifact identity is bound to its content

- Finding: Observation-batch identities are caller-declared: mutated batches pass exchange._identity
- Evidence status: `numerically_verified`
- Witness: `{"identity_status": "caller_declared_reference", "schema": "notation.instrument.observation-batch.v1"}`

## T096: The CIW candidate boundary refuses any ESM response field it does not recognise

- Finding: validate_response accepts unknown extra fields in an ESM response
- Evidence status: `numerically_verified`
- Witness: `{"reason": "the boundary checks bindings only; ESM owns its record …", "fields[0]": "lab_admission_override", "fields[1]": "candidate.lab_admitted"}`

## T100: CIW energy records can distinguish a genuinely acquired log from a relabelled synthetic fixture

- Finding: A resealed relabel of the synthetic energy fixture under a fresh occurrence is accepted by the workbench and classified as a physical-domain measurement
- Evidence status: `numerically_verified`
- Witness: `{"classification": "physical_domain_measurement", "hardware_provenance": "retained_operator_record_not_authenticated", "origin": "physical_measurement (declared)", "run_id": "energy-run-22222222222222222222222222222222"}`

## T100: CIW's retained records and their classification keep provider-backed results visibly distinct from fabricated ones

- Finding: A fabricated numerical-heat bundle sealed with CIW's pinned SCR revision and source tree reopens and is labelled provider_backed by the workspace classifier, as a provider result is
- Evidence status: `numerically_verified`
- Witness: `{"adapter_version": "fabricated-by-ciw-lab", "bundle_id": "sha256:60f6b1a3bc1e43ffe35c1786663264183e5e3788356702dceb044 …", "engine_source_binding": "operator_asserted_not_attested", "repository_root": "/fabricated/not-a-provider-checkout" …(+9)}`

## T101: The float64 resolution floor bounds the rounding error of the decrease form at every matrix scale

- Finding: PLSR's resolution of a subnormal plant is zero while forming its decrease matrix rounds an exactly indefinite declared form to a negative definite one
- Evidence status: `numerically_verified`
- Witness: `{"resolution": 0.0, "A_hex[0][0]": "-0x0.0000000000002p-1022", "A_hex[0][1]": "0x0.0000000000005p-1022", "A_hex[1][0]": "0x0.0p+0" …(+15)}`

## T103: OUTSIDE_LEVEL_SET is returned whenever V(x) exceeds the declared level

- Finding: The PLSR level gate misses exceedances when s^2 underflows: V > level is certified
- Evidence status: `numerically_verified`
- Witness: `{"A": "-I", "P": "2^500 I", "documented_exceeded": false, "e": -700 …(+7)}`

## T103: The level gate is decided exactly through the power-of-two scaling

- Finding: The PLSR level gate reports OUTSIDE_LEVEL_SET for V below the level when s^2 overflows
- Evidence status: `numerically_verified`
- Witness: `{"A": "-I", "P": "2^-1060 I", "documented_exceeded": true, "e": 512 …(+7)}`

## T103: Every finite in-box sample yields a runtime-status-v1 code

- Finding: A finite in-box theta whose A(theta) overflows raises an input error instead of NUMERICAL_OVERFLOW
- Evidence status: `numerically_verified`
- Witness: `{"A0": "-I", "A1": "2I", "theta": 1e+308, "box[0]": -1e+308 …(+3)}`

## T103: value_out_of_range is set exactly when V or x^T M x is not representable in binary64

- Finding: PLSR sets value_out_of_range and reports V = 0 for states whose exact V and x^T M x are representable binary64 subnormals, because s^2 underflows before the product is formed
- Evidence status: `numerically_verified`
- Witness: `{"code": "CERTIFIED_WITH_MARGIN", "decrease": -0.0, "error_units": null, "exact_representable": true …(+4)}`

## T105: The declared box bounds and their binary64 neighbours keep their SI box decision under either conversion formula

- Finding: A binary64 neighbour just outside the declared stiffness box is admitted in some unit system when the sample is converted as k / (1 / c) and the bound as k * c
- Evidence status: `numerically_verified`
- Witness: `{"cases[0]": "um, ms, N/um|k just below 8", "k_hex.k just above 12": "0x1.8000000000001p+3", "k_hex.k just below 8": "0x1.fffffffffffffp+2"}`

## T105: Converting the box and the sample with the same formula preserves box membership

- Finding: A parameter just above the SI bound is admitted after multiplying bound and sample by 1e-3
- Evidence status: `numerically_verified`
- Witness: `{"bound": 64576.90184335635, "hex": "0x1.f881cdbe69934p+15", "codes.SI": "OUTSIDE_PARAMETER_BOX", "codes.x1e-3": "CERTIFIED_WITH_MARGIN"}`

## T105: Mathematically equal unit conversions give the same box decision

- Finding: A parameter exactly on the SI bound is refused when bound and sample are converted by the two mathematically equal formulas k * 0.001 and k / 1000
- Evidence status: `numerically_verified`
- Witness: `{"bound": 2787.074437234679, "hex": "0x1.5c6261ca3211ap+11", "codes.SI": "CERTIFIED_WITH_MARGIN", "codes.x1e-3": "OUTSIDE_PARAMETER_BOX"}`

## T105: The same physical plant in different units gets the same PLSR verdict

- Finding: The light-damping plant's verdict depends on the unit system although its exact decrease form is negative definite in all of them
- Evidence status: `numerically_verified`
- Witness: `{"codes.2^-10 m, 2^-7 s, 2^10 N/m": "CERTIFIED_WITH_MARGIN", "codes.m, ms, N/m": "NUMERICAL_INCONCLUSIVE", "codes.m, s, N/m": "CERTIFIED_WITH_MARGIN", "codes.mm, s, N/mm": "CERTIFIED_WITH_MARGIN" …(+6)}`

## T106: An in-box sample at which P is not positive definite yields CERTIFICATE_NOT_POSITIVE, which RUNTIME-STATUS-v1 at the pinned commit describes as 'P is not positive definite here, so there is no certificate to evaluate'

- Finding: An affine certificate that is singular at an in-box theta raises ValueError instead of returning CERTIFICATE_NOT_POSITIVE
- Evidence status: `numerically_verified`
- Witness: `{"A0": "-I", "A1": "0", "P0": "I", "P1": "diag(0, 1)" …(+7)}`

## T107: Near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than CERTIFIED_WITH_MARGIN (candidate hypothesis formulated for T107, not a quoted specification)

- Finding: At required_margin 0 near-boundary spectra within two resolutions of zero receive CERTIFIED_WITH_MARGIN, and every such certificate is exactly sound
- Evidence status: `numerically_verified`
- Witness: `{"case": 3, "code_at_required_margin_0": "CERTIFIED_WITH_MARGIN", "exact_bin": "[-2, -1) res", "exact_class": "negative_definite" …(+14)}`

## T109: PLSR's solve_lyapunov refuses only plants for which no valid quadratic certificate is available in float64

- Finding: solve_lyapunov refuses an exactly Hurwitz plant for which an exactly valid quadratic certificate exists and PLSR's own verdict certifies it
- Evidence status: `numerically_verified`
- Witness: `{"certificate": "scipy.linalg.solve_continuous_lyapunov@1.16.2", "certificate_condition": 496874100465.51154, "name": "Jordan n=4, lambda=2^-6", "solver_gate": "Lyapunov residual" …(+1)}`

## T111: A negative sampled scalar decrease at every tested state implies a negative definite decrease form

- Finding: The scalar route sees decrease at every sampled state of an indefinite form that PLSR reports DECREASE_NOT_DEFINITE
- Evidence status: `numerically_verified`
- Witness: `{"A": "diag(-0.5, 5e-7)", "M": "diag(-1, 1e-6)", "P": "I", "samples": 64 …(+1)}`

## T114: A Lyapunov P solved at the nominal model with Q = I certifies the declared +-30 % inertia interval

- Finding: With Q = I the nominal-model P does not cover the declared inertia interval
- Evidence status: `numerically_verified`
- Witness: `{"J[0]": 0.0014, "J[1]": 0.00155, "J[2]": 0.0017000000000000001, "J[3]": 0.00185}`

## T119: Gross energy divided by executed solves is an energy per accepted numerical result

- Finding: Dividing gross energy by executed solves reports a finite energy per result for the under-target fixture although no result is accepted
- Evidence status: `numerically_verified`
- Witness: `{"accepted_solves": 0, "fixture": "under-target", "naive_j_per_solve": 0.05}`

## T120: Halving floating-point precision reaches every accuracy target at no greater operation count

- Finding: Lowering precision to float32 cannot reach a 1e-7 endpoint accuracy at any step count up to 2048, while float64 reaches it
- Evidence status: `numerically_verified`
- Witness: `{"float32_min_error": 5.826263982645739e-07, "float64_steps": 128, "target": 1e-07}`

## T120: Halving floating-point precision reaches every accuracy target of the Gaussian VI workload at no greater operation count

- Finding: Lowering the common Gaussian VI workload to float32 cannot reach a 1e-14 nat KL target within 256 iterations, while float64 reaches it
- Evidence status: `numerically_verified`
- Witness: `{"float32_floor_kl": 2.429755366417848e-14, "float64_iteration": 69, "target_kl_nats": 1e-14}`

## T121: The sign of a float32 reduction (a pass/fail decision at threshold 0) does not depend on the reduction order

- Finding: Reduction order alone flips the sign of a float32 sum whose exact value is +0.25 (the one-thread sequential CPU fold against the GPU-style orders)
- Evidence status: `numerically_verified`
- Witness: `{"exact_sum": 0.25, "found_by": "first flipping seed of 1..399 comparing sequential …", "seed": 201, "sums.atomic-0": 0.2431640625 …(+13)}`

## T121: float64 reductions are order-robust for sign decisions

- Finding: Reduction order alone flips the sign of a float64 sum of float64-native cancellation data whose exact value is +0.25 (the sequential fold against the GPU-style orders)
- Evidence status: `numerically_verified`
- Witness: `{"exact_sum": 0.25, "scale": 1099511627776.0, "seed": 1, "sums.atomic-0": 0.2529296875 …(+13)}`

## T121: Atomic completion order cannot change a float32 pass/fail decision on identical inputs

- Finding: Atomic completion order alone changes a float32 pass/fail test |S - 0.25| <= 0.01 on identical inputs
- Evidence status: `numerically_verified`
- Witness: `{"exact_sum": 0.25, "tolerance": 0.01, "failing.atomic-5": 0.232421875, "failing.atomic-7": 0.23779296875 …(+6)}`

## T121: Accumulating the same block partials yields the same float32 result whatever the atomic completion order

- Finding: Emulated atomicAdd completion orders of identical float32 block partials give distinct sums
- Evidence status: `numerically_verified`
- Witness: `{"distinct_results[0]": 4102183.5, "distinct_results[1]": 4102183.75, "distinct_results[2]": 4102184.0, "distinct_results[3]": 4102184.25}`

## T121: A max reduction built from a comparison select is bitwise order-invariant for every IEEE input

- Finding: A comparison-select maximum (a if a >= b else b, the rule numpy documents for np.maximum) returns its first operand for +0.0 and -0.0, so the sign of the result depends on operand order
- Evidence status: `numerically_verified`
- Witness: `{"orders[0]": "(+0.0, -0.0)", "orders[1]": "(-0.0, +0.0)", "signbits[0]": false, "signbits[1]": true}`

## T121: A fixed-order reduction gives the same binary64 result whether or not the compiler contracts its multiply-adds

- Finding: Fused multiply-add contraction of the common workload's in-kernel reductions changes its binary64 outputs, so a bitwise CPU/GPU comparison detects a contracting build
- Evidence status: `numerically_verified`
- Witness: `{"iterations": 38, "max_ulp": 6, "variant": "fma-second"}`

## T122: Gradient descent on the variational free energy converges for every positive step size

- Finding: A mean step 1.2 times the stability bound makes KL grow instead of converge
- Evidence status: `numerically_verified`
- Witness: `{"alpha": 0.563720425245574, "stable_bound": 0.46976702103797835}`

## T122: Coordinate normalization changes only units, not bounded convergence

- Finding: Unit scales leave the same problem unconverged after 512 iterations (condition number 1.3e3)
- Evidence status: `numerically_verified`
- Witness: `{"declared_iterations": 93, "unit_final_kl_nats": 3.9917424034797877}`

## T123: Adding variational free energy to physical energy yields a unit-independent quantity

- Finding: An untyped sum of free energy and physical energy changes when the energy unit changes
- Evidence status: `analytic`
- Witness: `{"energy": "0.2 J = 200 mJ", "free_energy_nats": 8.40639441534792}`

## T124: A valid log digest shows that the retained readings are unmodified hardware output

- Finding: A log whose counter readings were doubled and then resealed passes validation
- Evidence status: `numerically_verified`
- Witness: `{"log_digest": "sha256:d93ef3111f855922cfa1fd87f8fe04d6b18b63b70836e93b48a47 …", "mutation": "every energy_mj doubled, then energy_records.seal"}`

## T124: The declared origin of a retained energy log authenticates a physical measurement

- Finding: Relabelling a synthetic fixture as physical_measurement makes its analysis eligible for physical comparison
- Evidence status: `numerically_verified`
- Witness: `{"origin": "physical_measurement", "run_id": "energy-run-11111111111111111111111111111111"}`

## T127: On an intrinsically flat (developable) part the straight chord between two markers equals their surface distance, as on the flat plate

- Finding: A marker chord differs from the surface distance on a developable (intrinsically flat) part
- Evidence status: `numerically_verified`
- Witness: `{"chord_mm": 141.42135623730948, "dphi_deg": 90, "geodesic_mm": 157.07963267948966, "radius_mm": 100.0}`

## T129: A window chosen from the osculating circle keeps the quadratic-fit curvature bias within tolerance on any convex profile

- Finding: The osculating-circle window rule under-predicts the coupon-crest curvature bias
- Evidence status: `numerically_verified`
- Witness: `{"bias_per_mm": 0.002450826969304161, "profile": "Gaussian crest h = 10 mm, sigma = 20 mm", "quartic_ratio_gauss_over_circle": 3.999999999999999, "window_mm": 27.325202042558928}`

## T129: Sampling a smaller neighbourhood (more local fit) always improves the curvature estimate

- Finding: A smaller fitting window at fixed spacing can make the curvature estimate worse
- Evidence status: `numerically_verified`
- Witness: `{"rms_error_per_mm": 0.0034813081403234844, "spacing_mm": 0.6457054880813021, "window_mm": 4.554200340426489}`

## T129: A dome inside the declared forming tolerances is described by the nominal Gaussian model, so its parameters alone fix the prediction

- Finding: The fit residual flags an elliptical as-built dome inside the declared width tolerance (chi2 model test)
- Evidence status: `numerically_verified`
- Witness: `{"chi2_per_dof": 19.979819127048078, "sigma_x_mm": 20.5, "sigma_y_mm": 19.5, "threshold": 1.0858002377726916}`

## T132: A helix programmed in machine coordinates (phi, z) is insensitive to mandrel radius error because it is a geodesic on every cylinder

- Finding: Programming a helix in machine angles transfers mandrel radius error into lateral drift
- Evidence status: `numerically_verified`
- Witness: `{"course_mm": 1000.0, "fibre_angle_deg": 45.0, "radius_error_mm": 0.2, "radius_mm": 100.0}`

## T133: A constant winding angle (as on a cylinder) is a geodesic, slip-free path on every mandrel of revolution

- Finding: A constant winding angle is not geodesic on the torus mandrel
- Evidence status: `numerically_verified`
- Witness: `{"heading_from_parallel_deg": 50, "max_slippage_ratio": 0.44362847281391804, "mandrel.chart": "(phi, theta)", "mandrel.major": 150.0 …(+2)}`

## T134: The standoff (offset) tool path of a smooth surface path is itself a smooth path the robot can follow at constant speed

- Finding: A spray standoff beyond the concave radius of curvature folds the tool-centre-point path
- Evidence status: `numerically_verified`
- Witness: `{"min_concave_radius_mm": 94.49847176098345, "route": "coupon nominal route across the dome rim", "standoff_mm": 120.0}`

## T135: Geodesic scan rows launched at the swath spacing cover a curved coupon as completely as they cover a flat plate

- Finding: Geodesic rows at the swath spacing leave gaps on the domed coupon
- Evidence status: `numerically_verified`
- Witness: `{"coverage": 0.9603546538097882, "max_j_lat": 1.5555839572007397, "plan": "geodesic x1", "plate_coverage": 1.0}`

## T136: Allocating spec / (2 max |j|) to each error source keeps the exactly perturbed route within the spec

- Finding: The first-order tolerance allocation exceeds the spec at a tolerance corner on some off-axis routes
- Evidence status: `numerically_verified`
- Witness: `{"route": "fan+15deg", "worst_corner_ratio": 1.0091492341266857}`

## T137: The shortest route between a station and an edge is also the one farthest, relative to its length, from a focal or conjugate point (largest focal clearance ratio)

- Finding: The shortest candidate route has the lowest focal clearance ratio
- Evidence status: `numerically_verified`
- Witness: `{"focus_margin_lower_bound_mm": 297.82031552321394, "focus_margin_mm": null, "length_mm": 202.1796844767861, "nearest_focus_kind": "focal" …(+4)}`

## T138: A physically correct model passes E_n <= 1 against its open-loop prediction when U covers only the instrument and the dome tolerances

- Finding: A 2-sigma start offset of the declared jig makes a correct model fail E_n <= 1 unless the start pose is budgeted or measured
- Evidence status: `numerically_verified`
- Witness: `{"declared_jig_sigma_mm": 0.05196152422706632, "max_en_without_execution": 1.8788917732749837, "realized_offset_mm": 2.1039230484541327}`

## T140: The uncertainty of a curved-surface prediction compared with a photogrammetric measurement is limited by the instrument

- Finding: The coupon focal-distance prediction is geometry-limited, not instrument-limited, under the declared dome tolerances, instrument and start-pose uncertainties
- Evidence status: `numerically_verified`
- Witness: `{"quantity": "coupon focal distance", "components_mm.execution": 0.8324762391006113, "components_mm.geometry": 4.388650263504834, "components_mm.instrument": 1.2903303761332685 …(+1)}`

## T141: evidence.finding labels a claim from its basis and domain alone, so an acceptance statement filed in a computational domain with a passing check is established (the loophole T141 recorded before evidence.screen_authority_claim)

- Finding: evidence.finding and validate_finding refuse an acceptance or rejection statement filed in a computational or physical domain because of its wording
- Evidence status: `numerically_verified`
- Witness: `{"claim": "Coupon lot accepted for production", "domain": "computational_pipeline", "refused_by": "ciw.lab.evidence.screen_authority_claim"}`

## T141: The lab API cannot mark production acceptance

- Finding: A paraphrased acceptance statement outside both screened vocabularies, filed in a computational domain with a passing check, is still labelled by its checks
- Evidence status: `numerically_verified`
- Witness: `{"claim": "Coupon lot fit for shipment to the customer", "domain": "computational_pipeline", "label": "numerically_verified", "screens_passed[0]": "ciw.lab.evidence.screen_authority_claim" …(+1)}`

## T144: Numerical providers are invoked only through pinned executables

- Finding: Some process spawns run a PATH-resolved executable without comparing it to a pinned identity
- Evidence status: `numerically_verified`
- Witness: `{"executable": "rustc from shutil.which('rustc')", "module": "ciw.lab.implementation_targets_serial", "recorded": "rustc -vV release and commit (provenance, compared with …"}`

## T144: A text search for 'subprocess' identifies the process-spawning modules

- Finding: Text search for 'subprocess' finds modules that spawn no process
- Evidence status: `numerically_verified`
- Witness: `{"module": "ciw.acquired_dataset", "reason": "mentions ciw.adapters.subprocess or quotes subprocess in …"}`

## T146: CIW already has one canonical JSON byte encoding

- Finding: ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for non-ASCII text
- Evidence status: `numerically_verified`
- Witness: `{"identities_sha256": "4e2c3d77419efa08ed3f0637d7db5152bd5b0196396a90b37d4c177a0b52 …", "telemetry_sha256": "e28c9c5bcbf4f143f4aa9c0a207150257ba4ebb473f776aa735b58367ef6 …", "vector": "unicode-bmp"}`

## T146: Distinct Python values have distinct CIW content identities

- Finding: ciw.core.identities.canonical_json gives {1: 'x'} and {'1': 'x'} the same content identity
- Evidence status: `numerically_verified`
- Witness: `{"canonical": "{\"1\":\"x\"}", "values[0]": "{1: 'x'}", "values[1]": "{'1': 'x'}"}`

## T146: Existing CIW canonicalizers enforce the cross-language specification

- Finding: Python canonicalizers accept values the specification refuses
- Evidence status: `numerically_verified`
- Witness: `{"vector": "integer-2^53", "accepted_by[0]": "ciw.core.identities.canonical_json", "accepted_by[1]": "ciw.telemetry.canonical"}`

## T146: CIW canonical JSON bytes equal RFC 8785 JCS bytes

- Finding: CIW canonical JSON differs from RFC 8785 (JCS) numbers and key order
- Evidence status: `numerically_verified`
- Witness: `{"1.0[0]": "1.0", "1.0[1]": "1", "1e+16[0]": "1e+16", "1e+16[1]": "10000000000000000" …(+2)}`

## T146: Rust's shortest float formatting yields the CPython repr digits

- Finding: Rust's own shortest float formatting breaks exact decimal ties away from the specification
- Evidence status: `numerically_verified`
- Witness: `{"rust_shortest": "1.0000000000000003e15", "specification": "1000000000000000.2", "value": "1e15 + 0.25"}`

## T147: The float32 tolerance policy detects every dropped partial product larger than its row bound

- Finding: The float32 policy misses dropped partial products up to about its bound, as often as the product distribution predicts
- Evidence status: `numerically_verified`
- Witness: `{"above_bound.column": 538, "above_bound.magnitude": 0.0005776939797215164, "above_bound.ratio": 1.0001610045362923, "above_bound.row": 14 …(+6)}`

## T148: Pairwise summation is order-independent

- Finding: Fixed-order pairwise summation is reproducible for one order but not permutation-invariant
- Evidence status: `numerically_verified`
- Witness: `{"dataset": "uniform n=1024 PCG64(148)", "distinct_results": 4}`

## T148: Kahan compensated summation is accurate whenever Neumaier's is

- Finding: Kahan summation loses the sum [1, 1e100, 1, -1e100] that Neumaier summation keeps
- Evidence status: `numerically_verified`
- Witness: `{"kahan": 0.0, "neumaier": 2.0, "input[0]": 1.0, "input[1]": 1e+100 …(+2)}`

## T148: Neumaier summation error is at most 2u|S| + 4n u^2 sum|x|

- Finding: The bound 2u|S| + 4n u^2 sum|x| does not bound Neumaier summation
- Evidence status: `numerically_verified`
- Witness: `{"input": "[1] + 1000 x [0.7u(1 + 2^-20)] + [-1]", "n": 1002, "ratio": 12.312657557436786}`

## T149: Appending CRC-32 in either byte order keeps the 32-bit burst guarantee

- Finding: A big-endian CRC trailer lets a 32-bit burst across the payload/CRC boundary escape
- Evidence status: `numerically_verified`
- Witness: `{"frame": "encode_frame(123456, 987654321, 7, [1, -2, 3, 2**31 - 1])", "trailer": "big-endian", "flipped_bits[0]": 321, "flipped_bits[1]": 326 …(+15)}`

## T149: CRC-32 detects every 32-bit burst whatever order the link sends bits in

- Finding: The burst guarantee holds only in the LSB-first bit order of the reflected CRC
- Evidence status: `numerically_verified`
- Witness: `{"numbering": "bit p = bit 7 - p % 8 of byte p // 8", "flipped_bits[0]": 228, "flipped_bits[1]": 230, "flipped_bits[2]": 235 …(+16)}`

## T151: Rolling back to the immediately previous bitstream is always compatible

- Finding: Rolling back to the previous bitstream version can be incompatible
- Evidence status: `numerically_verified`
- Witness: `{"board": "revC", "from": "1.4.1", "host": "1.0", "to": "1.2.0"}`

## T152: Summing seq - previous - 1 in arrival order counts lost frames

- Finding: Differencing sequence numbers in arrival order miscounts losses under reordering
- Evidence status: `numerically_verified`
- Witness: `{"arrival_order": 1258, "in_order": 119, "true": 120}`

## T152: Sequence differences without modular arithmetic count every loss in a sequence-ordered stream

- Finding: Non-modular differencing misses the loss at the 32-bit wrap
- Evidence status: `numerically_verified`
- Witness: `{"in_order": 119, "true": 120, "missed_at_wrap[0]": 4294967295}`

## T153: An in-process Python flag is an actuator authority boundary

- Finding: A frozen in-process policy object can be mutated
- Evidence status: `numerically_verified`
- Witness: `{"mutation": "object.__setattr__(policy, 'enabled', True)", "outcome": "flag changed; the gate still refused because it re-verifies …"}`

## T154: A frozen dataclass status field keeps every control output a proposal

- Finding: A frozen control proposal's status can be forced in memory, and the forced object is refused
- Evidence status: `numerically_verified`
- Witness: `{"mutation": "object.__setattr__(proposal, 'status', 'command')", "outcome": "status changed; record() and to_command re-check it and …"}`

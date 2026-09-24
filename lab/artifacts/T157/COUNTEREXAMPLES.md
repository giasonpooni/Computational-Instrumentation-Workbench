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
- Witness: `{"median_effective_order": 4.195900523492557, "median_tolerance_exponent": 0.7686254902067906, "variant": "advance with y4"}`

## T003: Every integrator exhibits its nominal convergence order on every surface

- Finding: No convergence order is observable on flat Cartesian charts: every method is exact to rounding there
- Evidence status: `numerically_verified`
- Witness: `{"charts": ["plane", "cylinder"], "max_endpoint_error": {"cylinder": 2.0945586572537882e-14, "plane": 2.1111760027289807e-14}, "reason": "Gamma vanishes, so Euler, midpoint and RK4 reproduce u(s) = u0 + s v0"}`

## T004: A computed geodesic whose speed stays exactly 1 is accurate

- Finding: Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding with a first-order endpoint error
- Evidence status: `numerically_verified`
- Witness: `{"endpoint_error": 0.013938653977328111, "method": "Euler with per-step speed renormalization", "steps": 128, "surface": "sphere"}`

## T006: A smaller finite-difference step always gives a more accurate Jacobi estimate

- Finding: Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation)
- Evidence status: `numerically_verified`
- Witness: `{"best_eps": 1e-07, "best_error": 1.2829856288476549e-08, "column": "heading", "eps": 1e-11, "error": 0.00011227393838841859, "surface": "sphere"}`

## T007: The RK4 transfer-matrix determinant drifts at the method's global order h^4

- Finding: RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable curvature
- Evidence status: `numerically_verified`
- Witness: `(object of 2 entries; see the source report)`

## T007: Explicit Euler never preserves the transfer-matrix determinant

- Finding: Where K = 0 every method preserves det Phi = 1 exactly, Euler included
- Evidence status: `numerically_verified`
- Witness: `{"charts": ["plane", "cylinder"], "max_drift": 0.0, "per_step_factor": "1 + h^2 K = 1"}`

## T008: The first focal point lies at half the first conjugate distance

- Finding: On variable curvature the first focal point is not half the first conjugate distance
- Evidence status: `numerically_verified`
- Witness: `{"first_conjugate": 7.785018442790264, "first_focal": 3.465336236529439, "heading": 0.8753827096614808, "surface": "torus", "u0": [3.316172108544377, 0.24686558661589375]}`

## T009: Ranking paths by sensitivity to heading error gives the same order as ranking by sensitivity to lateral offset

- Finding: Lateral and heading sensitivities rank paths differently
- Evidence status: `numerically_verified`
- Witness: `{"heading": {"torus-inner-to-outer": 2.8459238929397412, "torus-outer-to-inner": 3.907448965394896}, "lateral": {"torus-inner-to-outer": 1.8031835771642162, "torus-outer-to-inner": 0.8568122251671532}, "length": 3.0, "paths": ["torus-outer-to-inner", "torus-inner-to-outer"]}`

## T010: The separation at a conjugate point is of exact order eps^2

- Finding: On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2
- Evidence status: `numerically_verified`
- Witness: `{"chord_exponent": 3.0003126899802814, "eps": [0.005, 0.01, 0.02, 0.04], "path": "outer equator, heading 0", "s_star": 5.441398092702653, "surface": "Torus(2, 1)"}`

## T010: The first-order separation eps j is accurate along the whole path once eps is small

- Finding: The relative first-order error diverges like 1/|s - s*| approaching the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.04, "invalid_window_half_width_by_eps": {"0.005": 0.006820113209327356, "0.01": 0.013734488574677014, "0.02": 0.027850914979647465, "0.04": 0.057270023587832335}, "path": "start (0, 0.3), heading 0.5", "relative_error": 3.7958513020004787, "s": 6.036272087666448, "surface": "Torus(2, 1)", "window_exponent": 1.0229659350471423}`

## T010: The separation of neighbouring geodesics grows monotonically with length

- Finding: Separation does not grow monotonically with length: it nearly vanishes at the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"chord_ratio": 0.00015710984529684768, "eps": 0.02, "path": "outer equator", "s_half": 2.7206990463513265, "s_star": 5.441398092702653, "surface": "Torus(2, 1)"}`

## T010: The relative first-order error diverges at every conjugate point

- Finding: On the unit sphere a pure heading perturbation refocuses exactly: the relative first-order error is uniform and does not diverge at the conjugate point
- Evidence status: `numerically_verified`
- Witness: `{"eps": 0.02, "perturbation": "pure heading", "relative_error": -1.6590023186879854e-05, "s": 3.129320807286708, "surface": "unit sphere"}`

## T011: A change of chart that preserves the geometry leaves the fixed-step integration error unchanged

- Finding: A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor
- Evidence status: `numerically_verified`
- Witness: `{"adaptive_step_ratio": {"plane": 59.0, "sphere": 3.9571428571428573, "torus": 3.953125}, "chart": "u_axis = c + mu a + a^3/3, mu = 0.1 (det J >= 0.1)", "factor_sphere": 1069563.9789352638, "factor_torus": 98109.38296653116, "plane_error_base": 1.1102230246251565e-14, "plane_error_fold": 1.1860356830650787e-05, "steps": 256}`

## T012: Finite-eps signed separations are invariant under an orientation-reversing basis change expressed consistently

- Finding: On a non-symmetric surface the orientation-reversed separation agrees only to first order: own/right - 1 is proportional to eps (Torus(2, 1) generic path)
- Evidence status: `numerically_verified`
- Witness: `{"eps": [0.001, 0.01, 0.04], "own_over_right_minus_1": [-0.00047633809261071747, -0.004752691591994318, -0.018846216359292622], "path": "Torus(2, 1) from (0, 0.3), heading 0.5, L = 3"}`

## T013: Surfaces with identical Jacobi fields (intrinsic geometry) have identical chords between corresponding points

- Finding: Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord
- Evidence status: `numerically_verified`
- Witness: `{"chord_cylinder": 2.5382081174896274, "chord_plane": 3.0000000000000018, "heading": 0.6, "length": 3.0, "start": [0.2, 0.1], "surfaces": ["Plane", "Cylinder(1)"]}`

## T013: Intrinsic (Jacobi) and extrinsic (chord) signatures of curvature vanish at the same rate in the flat limit

- Finding: The chord deficit vanishes faster (exponent -2 in R + 1) than the Jacobi deviation (-1)
- Evidence status: `numerically_verified`
- Witness: `{"chord_exponent_in_R_plus_1": -1.9999677621250946, "jacobi_exponent_in_R_plus_1": -0.9974853377423354}`

## T014: Forward-then-reversed integration with a method of order p returns to the start with error proportional to h^p

- Finding: Reversal error orders are 1 (Euler), 3 (midpoint) and 5 (RK4): even-order methods gain one order
- Evidence status: `numerically_verified`
- Witness: `{"mean_orders": {"euler": 1.049562088626885, "midpoint": 3.0007881457739116, "rk4": 4.998833708326354}, "surfaces": ["sphere", "torus", "hyperbolic-plane"]}`

## T014: Truncate-and-continue reproduces fixed-step integration bit for bit for any truncation length

- Finding: With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails
- Evidence status: `numerically_verified`
- Witness: `{"L1": 1.13, "L2": 3.0, "N1": 113, "N2": 300, "differing_final_components": 6, "h_direct": 0.01, "h_first": 0.009999999999999998, "h_rest": 0.01, "max_abs_difference": 4.440892098500626e-16}`

## T015: The energy error of a non-symplectic fixed-step integrator grows linearly with length at every horizon

- Finding: Fixed-step RK4 energy error has a flat (oscillation-dominated) envelope up to L = 160 on the torus and L = 320 on the sphere; on the torus a secular term emerges between L = 160 and 320
- Evidence status: `numerically_verified`
- Witness: `{"energy_envelope": [5.034257781755258e-07, 5.034257781755258e-07, 5.034257781755258e-07, 5.034257781755258e-07, 5.537789720122532e-07], "exponent": 0.027506236135538223, "h": 0.125, "lengths": [10.0, 20.0, 40.0, 80.0, 160.0], "method": "rk4", "surface": "Torus(2, 1)"}`

## T015: A bounded energy (speed) error implies a qualitatively correct long-horizon geodesic

- Finding: Euler on the torus keeps a bounded energy error but changes the orbit type (Clairaut drift)
- Evidence status: `numerically_verified`
- Witness: `{"h": 0.125, "max_energy_error": 0.03499256018697694, "method": "euler", "surface": "Torus(2, 1)", "theta_max_abs": 285.03465063072775, "theta_turning": 1.368523292357925}`

## T016: An implicit (A-stable) integrator removes the growth of the step count with k on strongly negatively curved surfaces

- Finding: A-stable implicit methods do not remove the growth of the required steps with k (implicit midpoint ~ k^(3/2), 2-stage Gauss-Legendre ~ k^(5/4)); implicit midpoint is qualitatively wrong beyond kh = 2
- Evidence status: `numerically_verified`
- Witness: `(object of 7 entries; see the source report)`

## T016: Jacobi growth is exponential in sqrt(peak |K|) times the length

- Finding: Saddle(c): peak |K| = c^2 but Jacobi growth is polynomial in c; the local exponent of j_head(L) decreases toward sqrt(2)
- Evidence status: `numerically_verified`
- Witness: `{"c": 16384.0, "log_j_head": 14.554722514358353, "peak_abs_curvature": 268435456.0, "sqrt_peak_times_L": 32768.0}`

## T017: The validity domain of the first-order approximation shrinks to zero at every conjugate point

- Finding: Unit sphere, pure heading: C2 = 0, C3 = -|sin s| cos^2 s / 24, and the domain does not shrink at s = pi
- Evidence status: `numerically_verified`
- Witness: `{"eps_max_near_pi": {"0.999": 0.4899060501928133, "1.001": 0.48990604983305625}, "perturbation": "pure heading", "surface": "unit sphere", "tau": 0.01}`

## T018: Weaker intrinsic curvature is harder to resolve at a fixed step size

- Finding: Weak curvature is harder to resolve for Euler and midpoint only down to K ~ 1e-2: for K <= 1e-2 their ratios no longer depend on K, and the RK4 ratio grows like 1/K
- Evidence status: `numerically_verified`
- Witness: `(object of 1 entries; see the source report)`

## T018: Refining the step size always makes a nonzero curvature effect resolvable

- Finding: Below the floating-point resolution of L no step size resolves the curvature signal
- Evidence status: `numerically_verified`
- Witness: `{"computed_deviation": 0.0, "signal": 1.3333333333333334e-16, "steps": [4, 8, 16, 32, 64, 128], "surface": "Sphere(1e8), K = 1e-16"}`

## T019: Any integer change of basis generates the same lattice

- Finding: Integer matrices with det != 1 are refused as SL(2,Z) basis changes
- Evidence status: `numerically_verified`
- Witness: `{"canonical": ["5", "2", "7"], "gram": [5, 2, 7], "image_canonical": ["7", "3", "19"], "matrix": [[2, 0], [0, 1]]}`

## T020: A floating-point geodesic direction can be irrational (non-closing)

- Finding: A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction
- Evidence status: `numerically_verified`
- Witness: `{"closes_after_alpha_turns": 562949953421312, "exact_fraction": "910872158600853/2^49", "float_phi": 1.618033988749895}`

## T022: Converting a target to binary64 preserves its shortest-representative multiplicity

- Finding: Rounding the target to binary64 changes its shortest-representative multiplicity (exact 1, binary64 2)
- Evidence status: `numerically_verified`
- Witness: `{"binary64_multiplicity": 2, "binary64_target": [0.5, 0.25], "exact_multiplicity": 1, "target": "(1/2 + 2^-60, 1/4)"}`

## T022: Floating-point distance comparison finds every shortest representative

- Finding: At exact cut-locus ties that binary64 cannot represent, raw float comparison undercounts some multiplicities while a relative tolerance of 1e-9 recovers all of them
- Evidence status: `numerically_verified`
- Witness: `{"binary64": 2, "exact": 3, "lattice": "generic", "target": ["21/62", "25/62"], "tolerance": 3}`

## T024: The shortest route also minimizes amplification and maximizes focus margin

- Finding: Rankings by length, amplification and focus margin disagree: the shortest route is neither the least amplifying nor in the best focus-margin group
- Evidence status: `numerically_verified`
- Witness: `(object of 3 entries; see the source report)`

## T026: Winding labels transform with the same matrix as the basis

- Finding: Transporting winding labels with M instead of M^-1 breaks length invariance
- Evidence status: `numerically_verified`
- Witness: `{"correct_rule": "c' = M^-1 c", "mismatches": 6428}`

## T026: Equal length spectra imply SL(2,Z)-equivalent oriented lattices

- Finding: The length spectrum does not determine the oriented shape: a mirror image is isospectral but not SL(2,Z)-equivalent
- Evidence status: `numerically_verified`
- Witness: `{"canonical": ["5", "-2", "7"], "gram": ["5", "-2", "7"], "same_canonical": false, "same_spectrum": true}`

## T029: Every polygon vertex of a glued surface is a cone singularity

- Finding: Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points
- Evidence status: `numerically_verified`
- Witness: `{"cone_angles_over_pi": ["2", "2"], "surface": "regular hexagon, opposite sides glued", "vertex_classes": 2}`

## T030: Grid shortest-path lengths converge to geodesic length as the grid is refined

- Finding: 4- and 8-neighbour grid shortest paths do not converge to Euclidean length under refinement
- Evidence status: `independently_verified`
- Witness: `{"direction": "45 deg", "grids": [30, 60, 120], "ratio": 1.414213562373095, "stencil": "4-neighbour"}`

## T031: A small metric perturbation changes the shortest route only slightly

- Finding: A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about 127 deg while the minimal length changes continuously
- Evidence status: `numerically_verified`
- Witness: `{"delta": "1/1000", "eps_after": 0.0041, "eps_before": 0.0039, "eps_star": "1/250", "heading_after_deg": 153.48071221535733, "heading_before_deg": 26.610961246990474, "heading_jump_deg": 126.86975096836686, "length_jump": 8.944644025454807e-08, "translates_after": [[-1, 0]], "translates_before": [[0, 0]]}`

## T032: The shortest geodesic between two points has the least heading amplification

- Finding: Torus inner equator: the shortest route has larger heading amplification than a longer route
- Evidence status: `independently_verified`
- Witness: `(object of 5 entries; see the source report)`

## T032: The shortest geodesic has the largest focus margin (farthest from conjugate points)

- Finding: Torus outer equator: the shortest route has a smaller focus margin than a longer route
- Evidence status: `independently_verified`
- Witness: `(object of 5 entries; see the source report)`

## T032: The shortest route between two points is unique

- Finding: Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique
- Evidence status: `numerically_verified`
- Witness: `(object of 5 entries; see the source report)`

## T032: Low heading amplification certifies a robust (locally minimizing) route

- Finding: Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point
- Evidence status: `independently_verified`
- Witness: `(object of 5 entries; see the source report)`

## T032: Minimizing geodesics on the unit sphere have focus margin bounded below by a positive constant

- Finding: On the unit sphere the minimizing arc between points at separation pi - delta ends delta before its conjugate point, so minimizing geodesics have no positive lower bound on focus margin
- Evidence status: `numerically_verified`
- Witness: `{"delta": [0.1, 0.01, 0.001], "focus_margin": [0.1000000000875092, 0.010000000098338013, 0.0010000000994705438], "kind": "near-conjugate conditioning", "surface": "unit sphere", "targeting_condition": [10.016686123180598, 100.00166570580501, 1000.0000672123366]}`

## T033: Metric symmetry, positive definiteness, Christoffel symmetry, metric compatibility and derivative consistency together certify a surface implementation

- Finding: A curvature-misscaled sphere passes every identity except the Gauss equation
- Evidence status: `numerically_verified`
- Witness: `{"failed": ["gauss_equation"], "gauss_residual": 0.167, "mutant": "misscaled-curvature (sphere R = 2 returning K = 1/R)"}`

## T033: A metric-compatible connection certifies the metric derivatives

- Finding: Metric compatibility cannot detect wrong metric derivatives
- Evidence status: `numerically_verified`
- Witness: `{"failed": ["derivative_consistency", "gauss_equation"], "mutant": "saddle with dg negated"}`

## T034: Finite, index-symmetric hand-coded metric derivatives are correct

- Finding: Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see
- Evidence status: `numerically_verified`
- Witness: `{"mutant": "gaussian-bump with f_xy dropped", "normalized_error": 0.0608}`

## T035: Decreasing the finite-difference step always improves agreement with the analytic derivative

- Finding: Smaller finite-difference steps can be far less accurate
- Evidence status: `numerically_verified`
- Witness: `{"error_at_1e-12": 0.000102, "error_at_h_opt": 1.97e-11, "h_opt": 5.623413251903491e-06, "surface": "sphere"}`

## T035: Every smooth metric shows an O(h^2) truncation branch in central-difference error

- Finding: Quadratic metrics have no truncation branch and a constant metric differences to exactly zero
- Evidence status: `numerically_verified`
- Witness: `{"error_at_1e-2": 3.4e-15, "surface": "saddle, E = 1 + c^2 x^2 (degree-2 metric)"}`

## T036: Fixed-step RK4 in a single polar chart integrates every great circle as accurately as a chart-switching atlas at the same step count

- Finding: A single polar chart fails or loses accuracy on great circles passing near its pole
- Evidence status: `numerically_verified`
- Witness: `{"delta_0.1_atlas_error": 5.68e-08, "delta_0.1_single_error": 7.23e-06, "delta_failed": [0.01, 0.001, 0.0001, 1e-06]}`

## T036: Single-chart integration exactly through a coordinate pole always fails

- Finding: Along the exact meridian (v_phi = 0 exactly) chart A alone crosses the pole accurately
- Evidence status: `numerically_verified`
- Witness: `{"delta": 0.0, "single_chart_error": 7.4e-14, "single_chart_error_by_delta": {"1e-08": 0.1, "1e-10": 1.1e-05, "1e-12": 1e-07}}`

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
- Witness: `{"cone-small-deficit-apex": {"observed": "coordinate_singularity", "true": "conical_singularity"}, "conformal-0.9995-boundary": {"observed": "infinite_distance_boundary", "true": "curvature_singularity"}, "power-graph-1.99-apex": {"observed": "regular", "true": "curvature_singularity"}}`

## T037: The scan classifies a degenerate point independently of the approach loops chosen by the caller

- Finding: Cartesian loops around the sphere pole in the polar chart make a coordinate singularity read as a conical point
- Evidence status: `numerically_verified`
- Witness: `{"cartesian_loops": "conical_singularity", "point": "sphere north pole, chart A", "polar_loops": "coordinate_singularity"}`

## T039: Edge-graph shortest paths converge to the geodesic distance under mesh refinement

- Finding: Edge-graph Dijkstra distance from a valence-5 vertex keeps a relative-error floor that tends to sqrt(5) - 2
- Evidence status: `numerically_verified`
- Witness: `{"max_signed_relative_error": [0.1448504685578944, 0.21078482135841448, 0.22958417381401675, 0.23443674333601439], "mesh": "icosphere levels 1-4, source vertex 0 (valence 5)"}`

## T039: Refining the mesh alone makes a Steiner-graph distance with fixed k converge to the geodesic distance

- Finding: Steiner graphs with a fixed number of points per edge keep a relative-error floor
- Evidence status: `numerically_verified`
- Witness: `{"k": 3, "max_abs_relative_error": [0.030675797123964732, 0.008227705748911074, 0.011039725230062247, 0.013758700097505683]}`

## T040: On a fixed mesh the finite-difference Jacobi field converges to the smooth Jacobi field as the perturbation tends to zero

- Finding: At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L) while no vertex lies between the paired geodesics
- Evidence status: `numerically_verified`
- Witness: `{"delta": 1e-05, "j_fd": 2.0, "j_smooth": 0.9092974268256817, "levels": [2, 3, 4, 5, 6]}`

## T040: The angle defect over one third of the incident area converges pointwise to the Gaussian curvature under refinement

- Finding: Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K
- Evidence status: `numerically_verified`
- Witness: `{"finest_value": 1.1459056232841356, "limit": 1.1458980337503153, "mesh": "icosphere, 12 valence-5 vertices"}`

## T040: Angle-defect curvature with the barycentric area converges pointwise at the valence-6 vertices of icosphere refinements

- Finding: Barycentric angle-defect curvature does not converge pointwise at valence-6 icosphere vertices on the icosahedral mirror planes
- Evidence status: `numerically_verified`
- Witness: `{"base_edge_max_error": [0.0019319256495926584, 0.002431873259649331, 0.0025567564634368933, 0.002587970803474504], "levels": [4, 5, 6, 7], "median_max_error": [0.0026966059689821353, 0.002135126501182638, 0.001994888688580332, 0.001959837496334327]}`

## T041: A larger minimum angle implies a smaller geodesic error

- Finding: A mesh with a smaller minimum angle can have a smaller geodesic error
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "icosphere-3", "worse_quality": "icosphere-3-jitter0.05-s20260939"}`

## T041: Minimum angle orders the barycentric angle-defect curvature error across mesh families

- Finding: Across mesh families a much smaller minimum angle can come with a smaller barycentric-area angle-defect RMS error
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "icosphere-3", "note": "the icosphere's RMS is dominated by its 12 valence-5 vertices (T040); with the mixed Voronoi area the icosphere is better", "worse_quality": "uv-sphere-20x32"}`

## T041: Within one mesh family better triangle quality (larger minimum angle, smaller radius ratio) implies smaller curvature error

- Finding: Within the latitude-longitude family a mesh with smaller minimum angle and larger radius ratio can have smaller curvature errors under both area choices
- Evidence status: `numerically_verified`
- Witness: `{"better_quality": "uv-sphere-40x16", "worse_quality": "uv-sphere-10x64"}`

## T041: Hausdorff convergence of a mesh to a surface implies convergence of area

- Finding: Schwarz lantern meshes converge in Hausdorff distance but not in area
- Evidence status: `numerically_verified`
- Witness: `{"area_ratio": 1.5872559353818663, "bands": 1024, "hausdorff": 0.001204543794827706, "n": 64, "q": 0.25}`

## T041: Hausdorff convergence of a mesh implies convergence of its geodesic distances

- Finding: Schwarz lantern intrinsic height does not converge to the cylinder height
- Evidence status: `numerically_verified`
- Witness: `{"H": 1.0, "n": 64, "traced_height": 1.5878935490351203}`

## T041: Hausdorff convergence of a mesh implies convergence of its normals and curvature

- Finding: Lantern angle defects vanish while total absolute mean curvature grows like n^2 and normal tilt converges to atan(pi^2 q R / 2H) instead of 0
- Evidence status: `numerically_verified`
- Witness: `{"n": 64, "tilt_deg": 50.96720303293561, "total_abs_mean_curvature": 11440.883001930979}`

## T042: Curvature and normal formulas can be evaluated safely on unvalidated meshes

- Finding: Curvature and normals evaluated on an unvalidated zero-area face are nonfinite
- Evidence status: `numerically_verified`
- Witness: `{"defect": "zero-area face", "mesh": "square plus a collinear face"}`

## T043: A fixed face corridor (fixed mesh combinatorics) represents the perturbed marker geodesic at every tested vertex-noise level

- Finding: The declared marker segment stays in its face corridor for sigma <= 1e-3 but leaves it in over 10% of samples at sigma = 1e-2
- Evidence status: `numerically_verified`
- Witness: `{"left_fraction": 0.39525, "sigma": 0.01}`

## T043: The fixed-corridor marker distance is valid for sigma <= 1e-3 on icosphere-3 whichever declared geodesic carries the markers

- Finding: The noise level at which a fixed face corridor fails depends on the strip, and the declared strip is the most robust of the six
- Evidence status: `numerically_verified`
- Witness: `{"left_fraction": 0.3715, "sigma": 0.001, "start_index": 0, "vertex_margin": 0.002659028956720544}`

## T043: First-order (linearized) propagation of vertex noise is adequate for angle-defect curvature at sigma = 1e-2 on icosphere-3

- Finding: First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2
- Evidence status: `numerically_verified`
- Witness: `{"h": 0.1507297051948821, "min_ratio": 1.3859890674673723, "sigma": 0.01}`

## T043: Refining the mesh reduces the error of angle-defect curvature

- Finding: Under fixed vertex noise the curvature error grows as the mesh is refined
- Evidence status: `numerically_verified`
- Witness: `{"coarse": {"discretization_error": 0.008929308868696362, "h": 0.2993320753185587, "level": 2, "noise_sd": 0.04704511699628665, "total_rms_error": 0.04764173912603594}, "fine": {"discretization_error": 0.00021411616684008372, "h": 0.03776637041792051, "level": 5, "noise_sd": 3.640366766542064, "total_rms_error": 3.639522243174912}, "sigma": 0.001}`

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
- Witness: `{"heading_rad": 0.6, "kappa0": 0.4967476850397676, "kappa0_prime": -0.222669223718571, "s4_coefficient": -0.0046087696213116995, "surface": "torus major 2, minor 1", "u0": [0.0, 0.7853981633974483]}`

## T046: For constant space curvature kappa, c = 2 sin(kappa s/2)/kappa

- Finding: Constant curvature alone does not give the circle chord: a cylinder helix with torsion deviates at order s^5 by kappa^2 tau^2/720
- Evidence status: `numerically_verified`
- Witness: `{"alpha_deg": 45.0, "gap": 2.839002859589268e-05, "kappa": 0.5000000000000001, "s": 0.8, "surface": "cylinder R = 1", "tau": 0.5}`

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
- Witness: `{"distorted_radius": 0.481525, "fold_radius_distorted": 0.4969039949999533, "image_corner_radius": 0.5333333333333333, "k1": -0.6, "r_other": 0.6355619579575096, "r_true": 0.85}`

## T051: Integer pixel rounding always adds 1/12 px^2 to the noise variance

- Finding: Without a random grid phase the quantization variance is not 1/12: an integer-aligned coordinate with sigma = 0.1 px has almost no rounding error
- Evidence status: `numerically_verified`
- Witness: `{"sample_variance": 0.0, "sigma_px": 0.1, "true_coordinate_px": 512.0}`

## T052: Fitting reading = a x + c without a direction term gives unbiased scale and bias under backlash

- Finding: Least squares that ignores backlash biases the offset estimate by the projection of the play error onto [x, 1]
- Evidence status: `numerically_verified`
- Witness: `{"backlash_m": 2e-05, "bias_error_over_backlash": 0.49788561310110585, "naive_bias_z": 40.156837606636714}`

## T053: The mean orientation error from a constant gyro bias grows as |bias| t

- Finding: On a body rotating about z, transverse gyro bias produces a bounded orientation error 2 |b_perp| / |omega| instead of |b_perp| t
- Evidence status: `numerically_verified`
- Witness: `{"bias_rad_per_s": [0.002, 0.0, 0.001], "max_transverse_rad": 0.0006365184325471376, "rotation_rad_per_s": [0.0, 0.0, 6.283185307179586], "stationary_transverse_rad": 0.009999999999999929}`

## T055: Zero-filled gaps can be recognized from the values alone

- Finding: Values alone recognize zero fills only in favourable signals: a neighbour-median detector finds every fill 50 sigma from zero, but on a stationary quantized encoder axis a fill equals a genuine zero-count reading
- Evidence status: `numerically_verified`
- Witness: `{"dropped": 46, "fills_equal_to_genuine_readings": 26, "identical_stream_gap_m": 0.0, "seed": 55, "signal": "stationary encoder axis, 0.5 um vibration quantized to 1 um counts"}`

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
- Witness: `{"detected_this_run": false, "detection_power": 0.4901452071629382, "samples": 2000, "sensor": "tracker", "z_critical": 4.133588386895672, "z_this_run": 3.48247611344711}`

## T062: Ignoring correlation between sensor noises is harmless

- Finding: Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES exceeds the 99% upper bound at nearly every tick
- Evidence status: `numerically_verified`
- Witness: `{"common_mode_covariance": 0.09, "grand_mean_nees": 5.514214143152381, "nominal": 4.0}`

## T062: A passing mean-NIS chi-square test shows the measurement noise model is correct

- Finding: The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); only the full whitened-innovation covariance test exposes the missing cross-correlation
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean_nis": 3.9174506536208624, "whitened_cross_term": 0.6662952145647711}`

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
- Witness: `{"p": 0.9, "rate": 0.135175, "rate_after_a_rejection": 0.29975705475612036, "wilson": [0.12964829573860664, 0.1408991591539061]}`

## T067: A chi-square gate on raw-R Mahalanobis distance has false-rejection rate 1 - p

- Finding: Gating the NIS computed with the raw sensor covariance R at the 99% quantile rejects valid readings at more than twice the nominal 1% rate
- Evidence status: `numerically_verified`
- Witness: `{"nominal": 0.01, "rate": 0.0366}`

## T068: Gating that wins in most runs improves the mean error

- Finding: Over all runs with gross outliers, gating lowers the mean squared error in most runs (sign test) but its mean improvement over fusing every reading is not statistically significant, because the cold-start lock-out runs lose heavily
- Evidence status: `numerically_verified`
- Witness: `{"gated_better_runs": 374, "paired_z": -0.7022936860857409, "rmse_gated": 0.17795991041645262, "rmse_ungated": 0.1964153397312212, "runs": 400}`

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
- Witness: `{"detection_rate": 0.08105263157894736, "magnitude_m": 0.3, "rmse": {"gated": 0.1426001142269842, "oracle": 0.13794394236831456, "ungated": 0.1384335410857886}}`

## T069: A zero placeholder for a missing reading is harmless

- Finding: Zero-filling missing readings by hand drags the estimate toward the origin and destroys consistency by the amount the exact joint moments predict
- Evidence status: `numerically_verified`
- Witness: `{"grand_mean_nees": 2305.6002151702023, "nominal": 4.0, "rmse_m": 3.7711939615873082}`

## T070: A stale sensor clock always shows up as a bias in the filter innovations

- Finding: With the stale camera as the only position sensor the lag is invisible to the innovations: the lagged constant-velocity path is itself a constant-velocity path, so the estimate is biased by about -tau E[v] while the mean test passes
- Evidence status: `numerically_verified`
- Witness: `{"camera_mean_z": [0.4228861059098964, -0.5395726400825746], "position_error_mean_m": [-0.09866380307289671, -0.03973045449682786], "sensors": "camera only"}`

## T071: A passing NIS test shows that the sensor frames agree

- Finding: Near the rotation centre (ticks 1-20) the same mismatch goes undetected by the per-tick NIS test at this sample size; its predicted inflation there is only about 2%
- Evidence status: `numerically_verified`
- Witness: `{"fraction_inside": 0.95, "rotation_deg": 2.0, "ticks": "1-20"}`

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

## T077: Every retained CIW operation result carries a replay-stable numerical-result identity

- Finding: Oscillator operation results carry no replay-stable numerical-result identity
- Evidence status: `numerically_verified`
- Witness: `{"fields": ["channel", "created_at", "data", "evidence_id", "execution_id", "interval_s", "operation_id", "parameters", "record_digest", "recording_file", "result_id", "role", "run_id", "runtime", "schema", "selection_revision", "verification_id", "verification_status"], "operation": "statistics.v1", "records": ["result:R1", "result:R2"]}`

## T077: A retained bundle identity binds every provenance record stored in the bundle

- Finding: The energy replay bundle identity excludes its replay receipt and verification
- Evidence status: `numerically_verified`
- Witness: `{"bundle": "bundle:B1", "function": "src/ciw/telemetry.py:_bundle_digest", "removed": "replay_receipts", "replaced": "verification"}`

## T079: Canonical-content identity treats numerically equal JSON numbers as equal content

- Finding: CIW canonical comparison is type-sensitive (1 != 1.0): a consistent int-for-float rewrite is refused by the stale log seal and, once resealed, retained as distinct evidence rather than aliased
- Evidence status: `numerically_verified`
- Witness: `{"ciw_canonical_equal": false, "edit": "unit diagonal of runtime.workload.solver_settings and plan.solver initial_covariance written as 1 instead of 1.0", "python_equal": true, "resealed_identities_differ_from_baseline": {"experiment_digest": true, "log_digest": true, "numerical_result_id": true}, "unsealed": "Retained log digest differs"}`

## T080: Saved execution and result selection revisions are checked against a retained selection history

- Finding: Surviving mutant revision.gap: a workspace whose selection revision jumps to 1000, with records claiming revision 999, reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "selection revision 1000 with execution and result claiming revision 999; resealed", "name": "revision.gap", "observed": "accepted", "post_reopen": {"execution_selection_revision": 999, "selection_revision": 1000}, "recompute": "local", "target": "oscillator selection history"}`

## T081: Reopen re-analysis protects retained energy results from numerical forgery

- Finding: Surviving mutant energy-source.resealed: a resealed edit of the retained source log, with every derived record rebuilt, reopens with a different gross energy
- Evidence status: `numerically_verified`
- Witness: `(object of 6 entries; see the source report)`

## T081: Unkeyed record seals detect every edit to a retained numerical result

- Finding: Surviving mutant oscillator-stats.resealed: an in-bounds statistics edit with a recomputed seal reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "statistics mean moved to the midpoint of [minimum, maximum]; resealed", "name": "oscillator-stats.resealed", "observed": "accepted", "post_reopen": {"retained_mean_differs_from_computed": true, "retained_mean_is_midpoint": true}, "recompute": "local", "target": "oscillator result data"}`

## T081: Saved statistics satisfy |mean| <= rms <= max(|min|, |max|)

- Finding: Surviving mutant oscillator-stats.impossible-moments: resealed statistics that no sample set can have reopen
- Evidence status: `numerically_verified`
- Witness: `(object of 6 entries; see the source report)`

## T081: Every retained oscillator result is sealed against edits

- Finding: Surviving mutant oscillator-stats.legacy: an edit to an unsealed legacy result reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "legacy (unsealed) statistics mean moved to the midpoint", "name": "oscillator-stats.legacy", "observed": "accepted", "post_reopen": {"retained_mean_differs_from_computed": true, "retained_mean_is_midpoint": true}, "recompute": "none", "target": "legacy oscillator result data"}`

## T082: A retained execution occurrence binds its creation time

- Finding: Surviving mutant fresh.created-at-shift: a backdated execution and result pair reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "created_at backdated identically in execution and result; resealed", "name": "fresh.created-at-shift", "observed": "accepted", "post_reopen": {"execution_created_at": "2001-01-01T00:00:00+00:00"}, "recompute": "local", "target": "oscillator execution and result"}`

## T083: A replay bundle cannot be retained without its replay receipt

- Finding: Surviving mutant receipt.deleted: a replay bundle reopens without its replay receipt, indistinguishable from an original execution
- Evidence status: `numerically_verified`
- Witness: `{"description": "replay_receipts removed from the replay bundle", "name": "receipt.deleted", "observed": "accepted", "post_reopen": {"bundles_with_receipts": 0, "replay_bundle_listed": true}, "recompute": "none", "target": "energy replay bundle"}`

## T084: Unkeyed record seals detect every replay-provenance forgery

- Finding: Surviving mutant receipt-source.sibling-execution: a receipt re-pointed, with its verification subject, at a sibling execution of the same bytes reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "receipt source and verification subject moved together to a sibling execution of the same source bytes; verification rebuilt", "name": "receipt-source.sibling-execution", "observed": "accepted", "post_reopen": {"receipt_source": "bundle:B0b", "replay_bundle": "bundle:B1", "verification_subject": "bundle:B0b"}, "recompute": "local", "target": "energy replay receipt"}`

## T085: A retained replay cannot be dated before its source bundle

- Finding: Surviving mutant receipt-replayed.reidentified-bundle: a replay re-sessioned and dated before its source, with recomputed digests, reopens
- Evidence status: `numerically_verified`
- Witness: `(object of 6 entries; see the source report)`

## T087: Sealed operation records refuse verification-method claims outside their schema

- Finding: Surviving mutant oscillator-method.injected: a sealed result carrying an injected verification_method reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "verification_method field injected into a sealed result; resealed", "name": "oscillator-method.injected", "observed": "accepted", "post_reopen": {"result.get": {"verification_method": "independent_reimplementation", "verification_status": "not_verified"}}, "recompute": "local", "target": "oscillator result"}`

## T088: Sealed operation records refuse independence claims outside their schema

- Finding: Surviving mutant oscillator-independent.injected: sealed records carrying an injected independent: true reopen
- Evidence status: `numerically_verified`
- Witness: `{"description": "independent: true injected into a sealed execution and result; resealed", "name": "oscillator-independent.injected", "observed": "accepted", "post_reopen": {"result.get": {"independent": true, "verification_status": "not_verified"}}, "recompute": "local", "target": "oscillator execution and result"}`

## T089: Sealed operation records refuse admission claims outside their schema

- Finding: Surviving mutant oscillator-admission.injected: a sealed result carrying an injected state_admission reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "state_admission: admitted injected into a sealed result; resealed", "name": "oscillator-admission.injected", "observed": "accepted", "post_reopen": {"result.get": {"state_admission": "admitted", "verification_status": "not_verified"}}, "recompute": "local", "target": "oscillator result"}`

## T090: Unkeyed record seals detect a forged provider runtime identity

- Finding: Surviving mutant oscillator-runtime.both: a provider runtime forged identically in execution and result reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "runtime forged identically in execution and result; both resealed", "name": "oscillator-runtime.both", "observed": "accepted", "post_reopen": {"execution_runtime": {"provider": "lab.forged-provider", "version": "99"}, "result_runtime": {"provider": "lab.forged-provider", "version": "99"}}, "recompute": "local", "target": "oscillator execution and result"}`

## T090: Reopening a workspace detects a forged analysis runtime identity

- Finding: Surviving mutant energy-runtime.all-bundles: a consistently forged analysis code digest reopens
- Evidence status: `numerically_verified`
- Witness: `{"description": "code_sha256 forged in every energy bundle; every digest recomputed", "name": "energy-runtime.all-bundles", "observed": "accepted", "post_reopen": {"replay_after_reopen": {"message": "Retained energy analysis binding differs", "outcome": "refused"}, "retained_runtime_forged": {"code_sha256": true, "python_version": false}}, "recompute": "full", "target": "energy bundle runtimes"}`

## T090: Reopening a workspace detects forged runtime dependency versions

- Finding: Surviving mutant energy-runtime.python-version: forged dependency versions reopen
- Evidence status: `numerically_verified`
- Witness: `(object of 6 entries; see the source report)`

## T091: Reopening a saved workspace performs no numerical recomputation

- Finding: Reopening recomputes the retained energy analysis to validate content
- Evidence status: `numerically_verified`
- Witness: `{"ciw.energy_records.analyze calls": 7, "new execution occurrences": 0}`

## T092: Reopen validation of a retained numerical-heat bundle establishes that its values are the pinned SCR heat-kernel output

- Finding: A retained numerical-heat bundle whose values no provider computed passes reopen validation
- Evidence status: `numerically_verified`
- Witness: `{"bundle_id": "sha256:954c39d69990ad4cce8c3bc853d4a79ac19d2dc4a756c4c484beca68652b2e43", "reference_values": [0, 16, 24, 16, 0], "retained_values": [0, 1, 2, 3, 0], "runtime_repository_root": "/fabricated/not-a-provider-checkout"}`

## T093: Every refused request leaves the session's in-memory state unchanged

- Finding: A refused recording operation is retained as a refused execution record
- Evidence status: `numerically_verified`
- Witness: `{"changed": ["executions"], "new_files": 1, "request": "operation.execute ciw.lab-unregistered.v1"}`

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
- Witness: `{"content": {"workspace_version": 3}, "error_type": "AttributeError", "fixture": "incomplete-workspace.json"}`

## T095: Saved workspaces with fields outside the schema are refused on reopen

- Finding: Session.from_workspace accepts an unknown top-level field and drops it on re-save
- Evidence status: `numerically_verified`
- Witness: `{"dropped_on_resave": true, "field": "lab_unexpected_field", "reopened": true}`

## T095: Workbench source errors name the source, not a bound runtime

- Finding: Workbench source.add reports malformed source JSON with text that names a bound runtime
- Evidence status: `numerically_verified`
- Witness: `{"fixture": "duplicate-key.json", "message": "MALFORMED_RESPONSE: The bound runtime did not return finite, unambiguous JSON", "request": "source.add (energy-accuracy)"}`

## T096: Every exchange artifact identity is bound to its content

- Finding: Observation-batch identities are caller-declared: mutated batches pass exchange._identity
- Evidence status: `numerically_verified`
- Witness: `{"identity_status": "caller_declared_reference", "schema": "notation.instrument.observation-batch.v1"}`

## T096: The CIW candidate boundary refuses any ESM response field it does not recognise

- Finding: validate_response accepts unknown extra fields in an ESM response
- Evidence status: `numerically_verified`
- Witness: `{"fields": ["lab_admission_override", "candidate.lab_admitted"], "reason": "the boundary checks bindings only; ESM owns its record schema"}`

## T100: CIW energy records can distinguish a genuinely acquired log from a relabelled synthetic fixture

- Finding: A resealed relabel of the synthetic energy fixture under a fresh occurrence is accepted by the workbench and classified as a physical-domain measurement
- Evidence status: `numerically_verified`
- Witness: `{"classification": "physical_domain_measurement", "hardware_provenance": "retained_operator_record_not_authenticated", "origin": "physical_measurement (declared)", "run_id": "energy-run-22222222222222222222222222222222"}`

## T101: The float64 resolution floor bounds the rounding error of the decrease form at every matrix scale, so an exactly indefinite declared form is never certified

- Finding: A subnormal plant whose declared decrease form is exactly indefinite drives the PLSR resolution to zero; PLSR's code on it is recorded
- Evidence status: `numerically_verified`
- Witness: `(object of 7 entries; see the source report)`

## T102: Power-of-two scaling of A and P never changes a PLSR verdict code while all quantities stay in the binary64 normal range

- Finding: Outside LAPACK's scaling window power-of-two rescaling of A and P never moves a PLSR code to a certificate that the exact class contradicts
- Evidence status: `numerically_verified`
- Witness: `{"abc": [300, 300, 0], "base": "CERTIFIED_WITH_MARGIN", "base_ratio": 1.0000000051989002, "exact_class": "negative_definite", "kappa": -1.0000000051989002, "scaled": "NUMERICAL_INCONCLUSIVE", "scaled_ratio": 0.9999999999999996, "window": "outside"}`

## T102: Power-of-two scaling of A never changes a PLSR verdict code

- Finding: Scaling the witness by 2^-1074 drives its resolution to zero while its unit-scale code is DECREASE_NOT_DEFINITE; the scaled code is recorded
- Evidence status: `numerically_verified`
- Witness: `{"A_unit": [[-2.0, 5.0], [0.0, -3.0]], "P": "I", "scale": "2^-1074", "scaled_code": "CERTIFIED_WITH_MARGIN", "unit_code": "DECREASE_NOT_DEFINITE", "x": [1.0, 0.0]}`

## T103: OUTSIDE_LEVEL_SET is returned whenever V(x) exceeds the declared level

- Finding: The PLSR level gate misses exceedances when s^2 underflows: V > level is certified
- Evidence status: `numerically_verified`
- Witness: `{"A": "-I", "P": "2^500 I", "documented_exceeded": false, "e": -700, "exact_exceeded": true, "log2_V": -900, "log2_level": -901, "p": 500, "plsr_code": "CERTIFIED_WITH_MARGIN", "plsr_exceeded": false, "x": "(2^-700, 0)"}`

## T103: The level gate is decided exactly through the power-of-two scaling

- Finding: The PLSR level gate reports OUTSIDE_LEVEL_SET for V below the level when s^2 overflows
- Evidence status: `numerically_verified`
- Witness: `{"A": "-I", "P": "2^-1060 I", "documented_exceeded": true, "e": 512, "exact_exceeded": false, "log2_V": -36, "log2_level": -35, "p": -1060, "plsr_code": "OUTSIDE_LEVEL_SET", "plsr_exceeded": true, "x": "(2^512, 0)"}`

## T103: Every finite in-box sample yields a runtime-status-v1 code

- Finding: A finite in-box theta whose A(theta) overflows raises an input error instead of NUMERICAL_OVERFLOW
- Evidence status: `numerically_verified`
- Witness: `{"A0": "-I", "A1": "2I", "box": [-1e+308, 1e+308], "error": {"message": "A must be finite", "type": "ValueError"}, "theta": 1e+308}`

## T103: No binary64 input near the representable limits yields a false certificate

- Finding: A subnormal plant whose declared decrease form is exactly indefinite drives the PLSR resolution to zero; PLSR's code on it is recorded
- Evidence status: `numerically_verified`
- Witness: `(object of 7 entries; see the source report)`

## T104: A P accepted by PLSR's QuadraticCertificate is exactly positive definite

- Finding: No verdict certifies with a P that quadratic() accepts although it is exactly indefinite
- Evidence status: `numerically_verified`
- Witness: `{"P_hex": [["0x1.67f6ca5722b82p-1", "0x1.d3e0cbd54bf46p-2"], ["0x1.d3e0cbd54bf46p-2", "0x1.30126b51ba8ffp-2"]], "e1_code": "NUMERICAL_INCONCLUSIVE", "exact_det_sign": -1, "weak_direction_code": "NUMERICAL_INCONCLUSIVE"}`

## T105: Converting the box and the sample with the same formula preserves box membership

- Finding: A parameter just above the SI bound is admitted after multiplying bound and sample by 1e-3
- Evidence status: `numerically_verified`
- Witness: `{"bound": 64576.90184335635, "codes": {"SI": "OUTSIDE_PARAMETER_BOX", "x1e-3": "CERTIFIED_WITH_MARGIN"}, "hex": "0x1.f881cdbe69934p+15"}`

## T105: Mathematically equal unit conversions give the same box decision

- Finding: A parameter exactly on the SI bound is refused when bound and sample are converted by the two mathematically equal formulas k * 0.001 and k / 1000
- Evidence status: `numerically_verified`
- Witness: `{"bound": 2787.074437234679, "codes": {"SI": "CERTIFIED_WITH_MARGIN", "x1e-3": "OUTSIDE_PARAMETER_BOX"}, "hex": "0x1.5c6261ca3211ap+11"}`

## T105: The same physical plant in different units gets the same PLSR verdict

- Finding: The light-damping plant's verdict depends on the unit system although its exact decrease form is negative definite in all of them
- Evidence status: `numerically_verified`
- Witness: `(object of 2 entries; see the source report)`

## T107: Near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than CERTIFIED_WITH_MARGIN (T107 specification)

- Finding: At required_margin 0 near-boundary spectra within two resolutions of zero receive CERTIFIED_WITH_MARGIN, and every such certificate is exactly sound
- Evidence status: `numerically_verified`
- Witness: `(object of 9 entries; see the source report)`

## T109: PLSR's solve_lyapunov refuses only plants for which no valid quadratic certificate is available in float64

- Finding: solve_lyapunov refuses an exactly Hurwitz plant for which an exactly valid quadratic certificate exists and PLSR's own verdict certifies it
- Evidence status: `numerically_verified`
- Witness: `{"certificate": "scipy.linalg.solve_continuous_lyapunov@1.16.2", "certificate_condition": 496874100465.51154, "name": "Jordan n=4, lambda=2^-6", "solver_error": "Lyapunov residual 5.875e-04 exceeds tolerance 1.000e-06; the equation is not resolvable in float64 at this scale", "verdict_with_certificate": "CERTIFIED_WITH_MARGIN"}`

## T109: The sign of the floating-point spectral abscissa decides Hurwitz stability

- Finding: numpy.linalg.eigvals misplaces the exact eigenvalue -lambda of every defective test matrix by far more than machine precision; whether the sign flips is recorded
- Evidence status: `numerically_verified`
- Witness: `{"A": [[-1.000244140625, 1.0, 0.0, 0.0, 0.0], [0.0, 0.999755859375, 1.0, 0.0, 0.0], [1.0, -1.0, 0.999755859375, 1.0, 0.0], [-2.0, 1.0, -3.0, -2.000244140625, 1.0], [-1.0, 0.0, -1.0, 0.0, 0.999755859375]], "exact_spectrum": -0.000244140625, "name": "Jordan n=5, lambda=2^-12", "numpy_abscissa": 0.00018266938715315225}`

## T111: A negative sampled scalar decrease at every tested state implies a negative definite decrease form

- Finding: The scalar route sees decrease at every sampled state of an indefinite form that PLSR reports DECREASE_NOT_DEFINITE
- Evidence status: `numerically_verified`
- Witness: `{"A": "diag(-0.5, 5e-7)", "M": "diag(-1, 1e-6)", "P": "I", "samples": 64, "seed": 1111}`

## T114: A Lyapunov P solved at the nominal model with Q = I certifies the declared +-30 % inertia interval

- Finding: With Q = I the nominal-model P does not cover the declared inertia interval
- Evidence status: `numerically_verified`
- Witness: `{"J": [0.0014, 0.00155, 0.0017000000000000001, 0.00185]}`

## T119: Gross energy divided by executed solves is an energy per accepted numerical result

- Finding: Dividing gross energy by executed solves reports a finite energy per result for the under-target fixture although no result is accepted
- Evidence status: `numerically_verified`
- Witness: `{"accepted_solves": 0, "fixture": "under-target", "naive_j_per_solve": 0.05}`

## T120: Halving floating-point precision reaches every accuracy target at no greater operation count

- Finding: Lowering precision to float32 cannot reach a 1e-7 endpoint accuracy at any step count up to 2048, while float64 reaches it
- Evidence status: `numerically_verified`
- Witness: `{"float32_min_error": 5.826263982645739e-07, "float64_steps": 128, "target": 1e-07}`

## T121: The sign of a float32 reduction (a pass/fail decision at threshold 0) does not depend on the reduction order

- Finding: Reduction order alone flips the sign of a float32 sum whose exact value is +0.25 (the one-thread sequential CPU fold against the GPU-style orders)
- Evidence status: `numerically_verified`
- Witness: `(object of 4 entries; see the source report)`

## T121: float64 reductions are order-robust for sign decisions

- Finding: Reduction order alone flips the sign of a float64 sum of float64-native cancellation data whose exact value is +0.25 (the sequential fold against the GPU-style orders)
- Evidence status: `numerically_verified`
- Witness: `(object of 4 entries; see the source report)`

## T121: Atomic completion order cannot change a float32 pass/fail decision on identical inputs

- Finding: Atomic completion order alone changes a float32 pass/fail test |S - 0.25| <= 0.01 on identical inputs
- Evidence status: `numerically_verified`
- Witness: `{"exact_sum": 0.25, "failing": {"atomic-5": 0.232421875, "atomic-7": 0.23779296875}, "passing": {"atomic-0": 0.2431640625, "atomic-1": 0.24072265625, "atomic-2": 0.24609375, "atomic-3": 0.242919921875, "atomic-4": 0.2421875, "atomic-6": 0.24462890625}, "tolerance": 0.01}`

## T121: Accumulating the same block partials yields the same float32 result whatever the atomic completion order

- Finding: Emulated atomicAdd completion orders of identical float32 block partials give distinct sums
- Evidence status: `numerically_verified`
- Witness: `{"distinct_results": [4102183.5, 4102183.75, 4102184.0, 4102184.25]}`

## T121: Max reductions are bitwise order-invariant for all IEEE inputs

- Finding: The IEEE maximum of +0.0 and -0.0 depends on operand order (numpy max)
- Evidence status: `numerically_verified`
- Witness: `{"orders": ["[0.0, -0.0]", "[-0.0, 0.0]"], "signbits": [true, false]}`

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
- Evidence status: `numerically_verified`
- Witness: `{"energy": "0.2 J = 200 mJ", "free_energy_nats": 8.40639441534792}`

## T124: A valid log digest shows that the retained readings are unmodified hardware output

- Finding: A log whose counter readings were doubled and then resealed passes validation
- Evidence status: `numerically_verified`
- Witness: `{"log_digest": "sha256:d93ef3111f855922cfa1fd87f8fe04d6b18b63b70836e93b48a4776ef8054688", "mutation": "every energy_mj doubled, then energy_records.seal"}`

## T124: The declared origin of a retained energy log authenticates a physical measurement

- Finding: Relabelling a synthetic fixture as physical_measurement makes its analysis eligible for physical comparison
- Evidence status: `numerically_verified`
- Witness: `{"origin": "physical_measurement", "run_id": "energy-run-11111111111111111111111111111111"}`

## T127: On an intrinsically flat (developable) part the straight chord between two markers equals their surface distance, as on the flat plate

- Finding: A marker chord differs from the surface distance on a developable (intrinsically flat) part
- Evidence status: `numerically_verified`
- Witness: `{"chord_mm": 141.4213562373095, "dphi_deg": 90, "geodesic_mm": 157.07963267948966, "radius_mm": 100.0}`

## T129: A window chosen from the osculating circle keeps the quadratic-fit curvature bias within tolerance on any convex profile

- Finding: The osculating-circle window rule under-predicts the coupon-crest curvature bias
- Evidence status: `numerically_verified`
- Witness: `{"bias_per_mm": 0.002450826969304161, "profile": "Gaussian crest h = 10 mm, sigma = 20 mm", "quartic_ratio_gauss_over_circle": 3.999999999999999, "window_mm": 27.325202042558928}`

## T129: Sampling a smaller neighbourhood (more local fit) always improves the curvature estimate

- Finding: A smaller fitting window at fixed spacing can make the curvature estimate worse
- Evidence status: `numerically_verified`
- Witness: `{"rms_error_per_mm": 0.0034813081403234844, "spacing_mm": 0.6457054880813021, "window_mm": 4.554200340426489}`

## T132: A helix programmed in machine coordinates (phi, z) is insensitive to mandrel radius error because it is a geodesic on every cylinder

- Finding: Programming a helix in machine angles transfers mandrel radius error into lateral drift
- Evidence status: `numerically_verified`
- Witness: `{"course_mm": 1000.0, "fibre_angle_deg": 45.0, "radius_error_mm": 0.2, "radius_mm": 100.0}`

## T133: A constant winding angle (as on a cylinder) is a geodesic, slip-free path on every mandrel of revolution

- Finding: A constant winding angle is not geodesic on the torus mandrel
- Evidence status: `numerically_verified`
- Witness: `{"heading_from_parallel_deg": 50, "mandrel": {"chart": "(phi, theta)", "major": 150.0, "minor": 50.0, "name": "torus"}, "max_slippage_ratio": 0.44362847281391804}`

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

## T137: The shortest route between a station and an edge is also the safest route (largest distance to a focal or conjugate point)

- Finding: The shortest candidate route has the worst focus margin
- Evidence status: `numerically_verified`
- Witness: `{"length_mm": 202.1796844767861, "nearest_focus_kind": "focal", "nearest_focus_mm": 153.40715534052734, "next_longer": "fan+5deg", "route": "fan+0deg"}`

## T138: A physically correct model passes E_n <= 1 against its open-loop prediction when U covers only the instrument and the dome tolerances

- Finding: A 2-sigma start offset of the declared jig makes a correct model fail E_n <= 1 unless the start pose is budgeted or measured
- Evidence status: `numerically_verified`
- Witness: `{"declared_jig_sigma_mm": 0.05, "max_en_without_execution": 1.8079673521997977, "realized_offset_mm": 2.1}`

## T140: The uncertainty of a curved-surface prediction compared with a photogrammetric measurement is limited by the instrument

- Finding: The coupon focal-distance prediction is geometry-limited, not instrument-limited
- Evidence status: `numerically_verified`
- Witness: `{"components_mm": {"execution": 0.8324762391006113, "geometry": 4.388650263504834, "instrument": 1.2903303761332685, "solver": 9.235394221226064e-07}, "quantity": "coupon focal distance"}`

## T141: The lab API cannot mark production acceptance

- Finding: evidence.finding establishes an acceptance statement when its author files it in a computational domain
- Evidence status: `numerically_verified`
- Witness: `{"caught_by": "manufacturing_records.screen_acceptance_language", "claim": "Coupon lot accepted for production", "domain": "computational_pipeline", "label": "numerically_verified"}`

## T144: Numerical providers are invoked only through pinned executables

- Finding: Some process spawns run a PATH-resolved executable without comparing it to a pinned identity
- Evidence status: `numerically_verified`
- Witness: `{"executable": "rustc from shutil.which('rustc')", "module": "ciw.lab.implementation_targets_serial", "recorded": "rustc -vV release and commit (provenance, compared with nothing)"}`

## T144: A text search for 'subprocess' identifies the process-spawning modules

- Finding: Text search for 'subprocess' finds modules that spawn no process
- Evidence status: `numerically_verified`
- Witness: `{"module": "ciw.acquired_dataset", "reason": "mentions ciw.adapters.subprocess or quotes subprocess in text but has no spawn call"}`

## T146: CIW already has one canonical JSON byte encoding

- Finding: ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for non-ASCII text
- Evidence status: `numerically_verified`
- Witness: `{"identities_sha256": "4e2c3d77419efa08ed3f0637d7db5152bd5b0196396a90b37d4c177a0b525362", "telemetry_sha256": "e28c9c5bcbf4f143f4aa9c0a207150257ba4ebb473f776aa735b58367ef62061", "vector": "unicode-bmp"}`

## T146: Distinct Python values have distinct CIW content identities

- Finding: ciw.core.identities.canonical_json gives {1: 'x'} and {'1': 'x'} the same content identity
- Evidence status: `numerically_verified`
- Witness: `{"canonical": "{\"1\":\"x\"}", "values": ["{1: 'x'}", "{'1': 'x'}"]}`

## T146: Existing CIW canonicalizers enforce the cross-language specification

- Finding: Python canonicalizers accept values the specification refuses
- Evidence status: `numerically_verified`
- Witness: `{"accepted_by": ["ciw.core.identities.canonical_json", "ciw.telemetry.canonical"], "vector": "integer-2^53"}`

## T146: CIW canonical JSON bytes equal RFC 8785 JCS bytes

- Finding: CIW canonical JSON differs from RFC 8785 (JCS) numbers and key order
- Evidence status: `numerically_verified`
- Witness: `{"1.0": ["1.0", "1"], "1e+16": ["1e+16", "10000000000000000"], "keys": ["Ａ", "😀"]}`

## T146: Rust's shortest float formatting yields the CPython repr digits

- Finding: Rust's own shortest float formatting breaks exact decimal ties away from the specification
- Evidence status: `numerically_verified`
- Witness: `{"rust_shortest": "1.0000000000000003e15", "specification": "1000000000000000.2", "value": "1e15 + 0.25"}`

## T147: The float32 tolerance policy detects every dropped partial product larger than its row bound

- Finding: The float32 policy misses dropped partial products up to about its bound
- Evidence status: `numerically_verified`
- Witness: `{"above_bound": {"column": 538, "magnitude": 0.0005776939797215164, "ratio": 1.0001610045362923, "row": 14, "row_tolerance": 0.0005776009833430312}, "largest_undetected": {"column": 117, "magnitude": 0.000589098664931953, "ratio": 0.992232575276228, "row": 109, "row_tolerance": 0.0005937102647209033}}`

## T148: Pairwise summation is order-independent

- Finding: Fixed-order pairwise summation is reproducible for one order but not permutation-invariant
- Evidence status: `numerically_verified`
- Witness: `{"dataset": "uniform n=1024 PCG64(148)", "distinct_results": 4}`

## T148: Kahan compensated summation is accurate whenever Neumaier's is

- Finding: Kahan summation loses the sum [1, 1e100, 1, -1e100] that Neumaier summation keeps
- Evidence status: `numerically_verified`
- Witness: `{"input": [1.0, 1e+100, 1.0, -1e+100], "kahan": 0.0, "neumaier": 2.0}`

## T148: Neumaier summation error is at most 2u|S| + 4n u^2 sum|x|

- Finding: The bound 2u|S| + 4n u^2 sum|x| does not bound Neumaier summation
- Evidence status: `numerically_verified`
- Witness: `{"input": "[1] + 1000 x [0.7u(1 + 2^-20)] + [-1]", "n": 1002, "ratio": 12.312657557436786}`

## T149: Appending CRC-32 in either byte order keeps the 32-bit burst guarantee

- Finding: A big-endian CRC trailer lets a 32-bit burst across the payload/CRC boundary escape
- Evidence status: `numerically_verified`
- Witness: `{"flipped_bits": [321, 326, 327, 328, 330, 331, 333, 334, 336, 337, 344, 345, 346, 347, 348, 349, 352], "frame": "encode_frame(123456, 987654321, 7, [1, -2, 3, 2**31 - 1])", "trailer": "big-endian"}`

## T149: CRC-32 detects every 32-bit burst whatever order the link sends bits in

- Finding: The burst guarantee holds only in the LSB-first bit order of the reflected CRC
- Evidence status: `numerically_verified`
- Witness: `{"flipped_bits": [228, 230, 235, 236, 237, 238, 240, 241, 242, 244, 247, 248, 249, 251, 253, 255, 256, 257, 258], "numbering": "bit p = bit 7 - p % 8 of byte p // 8"}`

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
- Witness: `{"in_order": 119, "missed_at_wrap": [4294967295], "true": 120}`

## T153: An in-process Python flag is an actuator authority boundary

- Finding: A frozen in-process policy object can be mutated
- Evidence status: `numerically_verified`
- Witness: `{"mutation": "object.__setattr__(policy, 'enabled', True)", "outcome": "flag changed; the gate still refused because it re-verifies the authorization on every write"}`

## T154: A frozen dataclass status field keeps every control output a proposal

- Finding: A frozen control proposal's status can be forced in memory, and the forced object is refused
- Evidence status: `numerically_verified`
- Witness: `{"mutation": "object.__setattr__(proposal, 'status', 'command')", "outcome": "status changed; record() and to_command re-check it and refuse (proposal_status_tampered)"}`

# A non-upgrading evidence-label discipline for computational experiments

*Generated draft from retained CIW lab reports. Not peer reviewed. Contains no physical measurement.*

## Abstract

This draft summarizes 51 queued computational tasks. Every result below carries the evidence label assigned by `ciw.lab.evidence`; physical validation is not established for any of them.

## Methods

### T077 — Build an identity matrix for: source evidence; operation; execution; result; numerical result; verification; bundle; replay; external producer IDs.

*Hypothesis.* Each CIW identity on the offline paths is a content or byte hash, a fresh event UUID, a caller- or provider-declared value, or absent, and its binding, replay and reopen behaviour follow from how it is derived.

*Model.* Derivation classes used in the matrix: content_hash, byte_hash, content_hash_of_fresh_occurrence and unkeyed_seal (SHA-256 over canonical JSON or raw bytes); fresh_event_uuid (uuid4 draws); caller_declared, caller_declared_name, constant_name and provider_declared (copied, not derived); code_hash_plus_declared_versions (a code digest beside self-reported versions); absent. Content identities are sha256 over canonical JSON (sort_keys, compact separators); byte identities are sha256 over raw bytes; event identities are uuid4 draws (collision probability ~2^-122 per pair).

### T078 — Test exact source-byte retention.

*Hypothesis.* The workbench retains the exact bytes supplied for each source: live, inside every native bundle, and after save and reopen.

*Model.* Let b be the supplied bytes and r the retained bytes (base64-decoded). Prediction: r == b, sha256(r) == evidence_id and len(r) == byte_count for every source; non-canonical base64 is refused, never normalized.

### T079 — Test whitespace-preserving evidence retention.

*Hypothesis.* Byte variants of one log that differ only in whitespace, key order or float spelling are retained as distinct evidence with distinct byte-level identities, while every content-level identity is equal; CIW's canonical comparison is type-sensitive, so rewriting 1.0 as 1 is a content change.

*Model.* Byte-level identities are functions of the bytes b (sha256(b)); content-level identities are functions of canon(parse(b)) with CIW's canonical JSON, which prints 1 and 1.0 differently. For variants with canon(parse(b_i)) equal and b_i distinct, byte-level identities are pairwise distinct and content-level identities coincide.

### T080 — Test operation/execution/result identity separation.

*Hypothesis.* Operation, execution and result identities have different kinds (name, fresh occurrence, result occurrence); forged aliases between execution and result identities are refused on reopen, while the selection revision a record claims is not checked against any history.

*Model.* operation_id is a constant per operation; execution_id and result_id are independent uuid4 draws per occurrence (energy result_id = sha256 over a record that includes execution_ref); numerical_result_id excludes the occurrence.

### T081 — Test stable numerical-result identity.

*Hypothesis.* The energy numerical_result_id is a content identity of the analysed data, and that data includes the retained log's identity (data.log_digest, origin and device_uuid), so it is equal across every occurrence of a canonically identical log (original, sibling, replay, replay after reopen, replay of a replay, separate session in the same process and code) and changes under a resealed metadata-only edit. Re-analysis on reopen refuses numerical edits that leave the retained source bytes unchanged, but not a forger who rewrites and reseals the source log itself. Oscillator results lack a numerical identity and their statistics are only bounds-checked.

*Model.* numerical_result_id = sha256(canon({operation_id, data})) with data = analyze(parse(bytes)) deterministic in-process; data.log_digest = sha256(canon(log without log_digest)). For statistics, |mean| <= rms <= max(|min|, |max|) holds for every sample set (Cauchy-Schwarz and the maximum bound).

### T082 — Test fresh replay execution and result identities.

*Hypothesis.* Every execution, result, verification, bundle and replay occurrence receives a fresh identity, and a retained workspace that reuses an occurrence is refused; creation times are bound only between the two records of one occurrence.

*Model.* Occurrence identities are uuid4 draws or digests over records containing them; for n occurrences the probability of any uuid4 collision is at most n(n-1) / 2^123.

### T083 — Test replay receipt binding.

*Hypothesis.* A replay receipt binds the source bundle digest, the replayed bundle digest, the verification subject, the fresh reproduction step, the runtime digest and a not_performed admission, but the replay bundle's own identity does not bind the receipt.

*Model.* replay_id = sha256(receipt \ replay_id); receipt.verification.subject_ref = source_bundle_digest; bundle_digest = sha256(bundle \ {bundle_digest, verification, replay_receipts}).

### T084 — Mutate replay receipt source digest.

*Hypothesis.* Predicted kills: a source digest edited without resealing (stale replay_id); resealed to a forged digest (the energy workflow rebuilds the receipt verification with subject = source and refuses the stale subject); re-pointed together with its subject at a digest that is not retained, at the replay itself, or at a bundle of another source log (workbench._validate_links requires a retained source bundle of the same kind, source_id and upstream). Predicted survivor: receipt-source.sibling-execution, because a sibling execution of the same source bytes satisfies every one of those checks and every digest is unkeyed.

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T085 — Mutate replay receipt replayed digest.

*Hypothesis.* Predicted kills: the four replayed_bundle_digest edits, resealed or not, because reopen requires it to equal the containing bundle's own digest. Predicted survivor: receipt-replayed.reidentified-bundle, because created_at and session_id sit inside the unkeyed bundle_digest, the forger recomputes that digest, the verification and the receipt together, and no check compares a replay's creation time with its source's.

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T086 — Mutate verification subject.

*Hypothesis.* Predicted kills: a receipt verification subject edited with or without recomputed ids (reopen rebuilds the expected verification with subject = receipt source); a bundle verification subject other than the bundle's own digest; oscillator verification fields (protocol v1 fixes them); an ESM candidate naming another bundle; an exchange subject edit without a recomputed verification_id. No survivor of its own is predicted: the subject can move only together with the receipt source, which is T084's receipt-source.sibling-execution (cross-referenced here, not re-run).

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T087 — Mutate verification method.

*Hypothesis.* Predicted kills: the three energy verification method edits, because reopen rebuilds the expected verification with the fixed method fresh_analysis_of_same_retained_measurement. Predicted survivor: oscillator-method.injected, because sealed oscillator records do not refuse extra keys and their seal is unkeyed.

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T088 — Mutate independence flag.

*Hypothesis.* Predicted kills: independent true in an energy receipt or bundle verification (reopen fixes independent: false) and in an ESM inspection or candidate (the validator requires false). Predicted survivor: oscillator-independent.injected (open key set, unkeyed seal). exchange._identity is predicted to accept a recomputed artifact claiming independence, by design: it checks content only and reports content_recomputed_not_authenticated.

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T089 — Mutate admission status.

*Hypothesis.* Predicted kills: receipt admission other than not_performed; energy verification or result authority other than the fixed not_performed record; an oscillator verification_status other than not_verified; an ESM response or candidate claiming admission. Predicted survivor: oscillator-admission.injected (open key set, unkeyed seal).

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T090 — Mutate provider runtime identity.

*Hypothesis.* Predicted kills: oscillator runtime edits that break the seal, differ between execution and result, or empty the identity; energy runtime edits that leave bundle_digest stale, differ between a replay and its source (the receipt verification binds one runtime digest), or break the code_sha256 format. Predicted survivors: oscillator-runtime.both (a provider-declared identity checked only for equality between the two records) and energy-runtime.all-bundles and energy-runtime.python-version (reopen format-checks the retained runtime; only a new replay compares it with the current analysis identity).

*Model.* Each record seal or identity d = sha256(canon(record \ d)) is unkeyed, so any holder can recompute d after an edit. Reopen kills a mutant only if some check compares the edited field with an independently recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.

### T091 — Test save/reopen without provider access.

*Hypothesis.* A CIW workspace saved with retained oscillator results and an energy-accuracy original and replay can be reopened with no provider binding and without reaching any execution entry point.

*Model.* Reopen = validate(saved JSON) then construct; the guard replaces 22 execution entry points (workflow steps, sessions, provider adapters, subprocesses, recording operations) with a refusing recorder.

### T092 — Test replay refusal without provider binding.

*Hypothesis.* Without a host-side trusted binding, CIW refuses every replay or execution of a provider workflow with a named error, and no client or saved value can supply the binding.

*Model.* Workbench._reserve(kind) requires kind in trusted bindings; bindings are process configuration set only by Workbench.bind_workflow, never by a protocol request or a saved workspace. Kinds that consume an upstream bundle are checked for that bundle first.

### T093 — Test unchanged workspace after refusal.

*Hypothesis.* A request that CIW refuses changes neither the session state a save would write nor the session directory.

*Model.* State = digests of selection, results, executions, retained catalog, catalog revision, byte and reservation counters, identity claims, candidates and bindings; plus the session directory listing and the re-saved workspace content without its saved_at timestamp.

### T094 — Add golden retained bundles.

*Hypothesis.* Workspaces saved by CIW earlier (golden fixtures) still reopen and validate with the current code, with no provider binding and no execution.

*Model.* Golden = exact saved bytes recorded in GOLDEN_MANIFEST; validation = Session.from_workspace (structure, identities, commitments, energy-analysis recomputation) under the execution guard.

### T095 — Add malformed exchange fixtures.

*Hypothesis.* Each malformed exchange input (duplicate keys, NaN/Infinity, overflow, wrong schema, oversize, truncated bytes, invalid UTF-8, wrong types, extra fields, forged identity) is refused by the relevant CIW validator.

*Model.* Validators: Workbench source.add (energy-accuracy), ciw.session.read_json, ciw.exchange.inspect_exchange (parsing stage, no SET checkout), ciw.exchange._identity, Session.from_workspace, Workbench.restore.

### T096 — Add provider-free conformance tests.

*Hypothesis.* The provider-free parts of the exchange and ESM candidate boundaries accept valid records and refuse every mutation of a bound field, without any provider checkout.

*Model.* exchange._identity: id = 'sha256:' + SHA-256(schema NUL canonical_json(record without id)); validate_response: equality of boundary fields with the explicit request and the selected bundle.

### T097 — Add exact SET/SCR/PPDA integration tests when checkouts are available.

*Hypothesis.* The pinned SCR provider, driven through CIW's shared numerical-heat workflow, returns the declared integer heat field, replays with the same numerical identity, and its reopened workspace refuses replay unbound.

*Model.* u_i <- u_i + trunc((u_{i-1} - 2 u_i + u_{i+1}) / 4), fixed ends (SCR heat descriptor, ciw.declared_workload).

### T098 — Record provider HEAD, tree, source hash, lockfile, and engine digest.

*Hypothesis.* Every bound provider checkout is clean and at a CIW pin; its working bytes reproduce its Git tree; its lockfiles and the engine used in this run are recorded.

*Model.* Git object ids: blob = H('blob' len NUL bytes), tree = H('tree' len NUL sorted(mode name NUL id)); tracked digest = SHA-256 over 'mode kind sha256(bytes) path' lines in ls-tree order.

### T099 — Verify `cargo build --locked`.

*Hypothesis.* The pinned SCR execution engine builds with cargo build --release --locked --offline, leaves the checkout and Cargo.lock unchanged, is reproducible across target directories and executes the heat kernel correctly.

*Model.* Cargo resolution fixed by crates/Cargo.lock (no external crates); determinism = equal binary digests.

### T100 — Keep synthetic, provider-backed, and physical results visibly distinct.

*Hypothesis.* Retained lab reports and CIW's own records keep synthetic, provider-backed and physical results visibly distinct, and relabelling is refused wherever the record can detect it.

*Model.* Allowed labels per domain: physical -> {not_established, hardware_measured, independently_verified with acquisition}; authority -> {not_established}; computational -> never hardware_measured.

### T101 — Test the resolution floor over many matrix scales.

*Hypothesis.* For the family A = s[[-eps, 1], [-1, -eps]], P = I, PLSR's resolution floor is the documented Higham/Weyl/LAPACK bound and scales exactly with s, so the inconclusive threshold stays at eps* wherever every quantity is a normal binary64 number (this family has a diagonal decrease form, where eigvalsh is exact; T102 tests general near-threshold forms); the bound is not derived for subnormal arithmetic, where it can underflow below the actual rounding error.

*Model.* Continuous decrease form M = A^T P + P A (symmetrised); resolution = n*(2 gamma_{n+3} n max|A| max|P|) + n^3 u max|M|; certified iff -max eig(M) > resolution. For A = s[[-eps, 1], [-1, -eps]], P = I the threshold is eps* = 4 gamma_5/(1 - 8u) = 2.2204e-15 for every s with normal arithmetic.

### T102 — Test power-of-two homogeneous scaling.

*Hypothesis.* Replacing (A, P, x) by (2^a A, 2^b P, 2^c x) multiplies M, max eig(M) and the resolution by 2^(a+b) exactly, so no verdict code and no margin ratio changes while every quantity stays normal and LAPACK does not rescale internally; outside that window rounding differs and near-threshold codes may move, but never against the exact class of the declared form.

*Model.* Continuous time: M(2^a A, 2^b P) = 2^(a+b) M(A, P) and resolution likewise (homogeneous of degree one in each of max|A|, max|P|, max|M|); PLSR divides x by a power of two before evaluating, so c never enters. Discrete time admits only (P, x) scaling. LAPACK dsyevd rescales by a non-power-of-two factor when max|M| lies outside [2^-485, 2^485]. Exact power-of-two scaling preserves the exact class of M.

### T103 — Test overflow and underflow state evaluation.

*Hypothesis.* States anywhere in binary64 are classified with the code unchanged through PLSR's power-of-two state scaling (components more than about 2^1022 below the largest one become subnormal in the scaled state and vanish beyond about 2^1075, which a code decided relative to |x|^2 does not see); matrices whose arithmetic leaves binary64 return NUMERICAL_OVERFLOW or an input refusal; no near-limit input yields a false certificate or a missed level-set exceedance.

*Model.* V(x) = x^T P x and x^T M x are homogeneous of degree two in x, so PLSR evaluates at x / 2^e with the unit state in [1, 2). The level gate decides V > level as scaled_value > level / s^2 and treats an infinite s^2 as 'exceeded' and a zero s^2 as 'not exceeded'. Exact truth: V = 2^(p + 2e) for P = 2^p I and x = 2^e e1.

### T104 — Test semidefinite and skew-symmetric edge cases.

*Hypothesis.* PLSR never certifies a decrease form that is only semidefinite or indefinite: skew-symmetric plants with P = I give an exactly zero form and must be NUMERICAL_INCONCLUSIVE; defective (Jordan) plants are certified with P = I exactly when lambda > 1/2; a semidefinite Q is refused by the solver.

*Model.* Skew A: A^T + A = 0 exactly; with any P, trace(A^T P + P A) = 0 so the form is never negative definite. Jordan A = [[-l, 1], [0, -l]], P = I: M = [[-2l, 1], [1, -2l]] is negative definite iff l > 1/2 and singular at l = 1/2. Orthogonal discrete A: A^T A - I = 0 up to rounding.

### T105 — Test parameter-box comparisons across unit scales.

*Hypothesis.* The same physical parameter box and plant expressed in different units give the same PLSR verdicts: box membership is preserved by a monotone conversion and the sign of the decrease form is congruence-invariant.

*Model.* Mass-spring-damper m = 2 kg, c = 3 N s/m, stiffness k in [8, 12] N/m: A(k) = A0 + k A1, common P = [[6, 0.75], [0.75, 1]] (exactly negative definite decrease at both vertices, hence on the box). A unit change is x' = T x, t' = t/tau, k' = c k: A' = tau T A T^-1, P' = T^-1 P T^-1. Inertia is invariant; eigenvalues, max|A| and max|P| (hence margin and resolution) are not.

### T106 — Test every runtime status transition.

*Hypothesis.* Each of the nine runtime-status-v1 codes is reachable with declared inputs, codes change along one-parameter paths exactly as the documented decision order predicts, and the runtime refuses the five host-owned codes.

*Model.* Decision order: outside box -> NUMERICAL_OVERFLOW -> CERTIFICATE_NOT_POSITIVE (min eig P <= 0 or V < 0) -> OUTSIDE_LEVEL_SET -> NOT_CERTIFIED (x^T M x > res |x|^2) -> CERTIFIED_WITH_MARGIN / MARGIN_LOW (margin > res, MARGIN_LOW iff margin <= required margin) -> DECREASE_NOT_DEFINITE (max eig M > res) -> NUMERICAL_INCONCLUSIVE.

### T107 — Verify numerical inconclusive behavior.

*Hypothesis.* PLSR never gives a certifying code to a near-boundary decrease form that is not exactly negative definite; beyond two resolutions from zero it always resolves the sign; under a declared margin of three resolutions no case within two resolutions is CERTIFIED_WITH_MARGIN. The specification's stronger statement -- that near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than CERTIFIED_WITH_MARGIN -- is tested at required_margin 0 as a candidate counterexample.

*Model.* If the resolution bounds the float64 error e of max eig(M) (|e| <= res), then exact lambda < -2 res gives margin > res (certified), exact lambda >= 2 res gives max eig > res (not definite), and exact lambda >= 0 can never give margin > res. With required margin 3 res, certification needs margin > 3 res, impossible for exact lambda >= -2 res.

### T108 — Verify required-margin monotonicity.

*Hypothesis.* Increasing required_margin never turns a failing verdict into a passing one: CERTIFIED_WITH_MARGIN and meets_required_margin are nonincreasing in the declared margin, non-certifying codes and inequality_certified do not depend on it, and the switch to MARGIN_LOW happens exactly at required_margin = margin.

*Model.* The declared margin r enters the decision order only in the branch margin > res, as MARGIN_LOW iff margin <= r; meets_required_margin = margin > max(r, res). Both are monotone in r by construction; negative or non-finite r must be refused.

### T109 — Add adversarial eigenvalue cases.

*Hypothesis.* On highly non-normal, exactly defective and clustered Hurwitz plants PLSR either certifies soundly (the exact decrease form of its P is negative definite and P is exactly positive definite) or refuses; its Lyapunov solutions agree with an independent solver to within conditioning; its solver may refuse plants that do have a valid quadratic certificate; floating-point eigenvalues may misjudge stability where the exact certificate route does not.

*Model.* Lyapunov: A Hurwitz iff A^T P + P A = -Q has P > 0 for Q > 0; a certified P bounds transients by ||exp(At)|| <= sqrt(cond P). Non-normal A = [[-1, K], [0, -2]]: P = I certifies iff K < 2 sqrt 2. Defective A = T J T^-1 with integer unimodular T (T Ti = I checked exactly) has exact spectrum {-lambda}; eigenvalue perturbation of an n-Jordan block is O(eps^(1/n)). A backward-stable solve has forward error of order n^2 u cond(P).

### T110 — Test discrete-time versus continuous-time interpretation.

*Hypothesis.* PLSR distinguishes x' = A x from x+ = A x: the same matrix is certified in a time convention exactly when it is stable in that convention (Hurwitz versus Schur), and the two interpretations give different verdicts whenever the stability quadrants differ.

*Model.* Continuous decrease A^T P + P A, resolution 2 gamma n^2 a p + n^3 u |M|; discrete decrease A^T P A - P, resolution n(2 gamma n^2 a^2 p + u p) + n^3 u |M|. Lyapunov: a positive definite solution of the respective equation with Q = I exists iff A is Hurwitz, respectively Schur.

### T111 — Compare scalar quadratic and matrix-eigenvalue routes.

*Hypothesis.* The PLSR matrix route (Lyapunov solve, then the sign of max eig of the decrease form beyond the resolution) agrees with independent references -- numpy eigenvalues and SciPy's Bartels-Stewart Lyapunov solvers -- while the scalar quadratic route (the sign of x^T M x at sampled states) cannot certify definiteness and misses thin positive cones.

*Model.* Rayleigh: x^T M x <= max eig(M) |x|^2 for every x, with equality only on the top eigenvector, so sampled negativity never implies negative definiteness. An unstable A has v*(A + A^T)v = 2 Re(lambda)|v|^2 > 0 for an eigenvector v, so P = I can never certify it. For n = 1 the decrease form is 2 a p (continuous), so the verdict must follow sign(a) whenever 2|a|p exceeds the resolution.

### T112 — Define a separate disturbance-aware/ISS research branch.

*Hypothesis.* For x' = A x + B w with |w| <= w_bar, the quadratic ISS-Lyapunov bound sqrt(V(t)) <= max(sqrt(V(0)), 2 ||P^(1/2) B|| w_bar / c) holds on every bounded disturbance; it is conservative against the sharp reachable-set supremum in two dimensions and approached in one dimension; this belongs to a separate research branch, not to the PLSR runtime.

*Model.* V = x^T P x, A^T P + P A = -Q: V' <= -c V + 2 sqrt(V) beta with c = min eig(P^-1 Q), beta = ||P^(1/2) B|| w_bar (Cauchy-Schwarz in the P inner product), so W = sqrt(V) obeys W' <= -(c/2) W + beta. The sharp supremum from x(0) = 0 is max over unit u of w_bar int_0^inf |u^T L^T e^(A s) B| ds with P = L L^T (support function of the reachable set).

### T113 — Connect filtered residuals without putting sensors inside the kernel.

*Hypothesis.* A host-side adapter can turn filtered residual statistics into plsr-sample-v1 samples (a schema tag and numbers only) while sensor identity, units, calibration and timing stay in a host envelope; host-owned statuses are decided before the kernel and never reach it; forwarding theta_hat together with both ends of its three-standard-error interval keeps a near-bound point estimate from being accepted alone; and the kernel's codes on forwarded samples follow the declared model.

*Model.* Declared discrete map A(theta) = I + h [[0, 1], [-(4 + theta), -0.4]], h = 0.01 s, theta in [-0.5, 0.5], common P from the nominal discrete Lyapunov equation (exactly valid at both vertices, hence on the box by convexity). An EKF on (x, v, theta) yields x_hat, theta_hat, its standard error se and the innovation NIS; mean NIS above 1 + 6 sqrt(2/N) is MODEL_MISMATCH. The adapter forwards theta_hat and theta_hat +- 3 se; the host accepts a window only if the kernel certifies all three.

### T114 — Prepare a non-production servo-axis pilot specification.

*Hypothesis.* A non-production servo-axis pilot can be specified so that the Lyapunov monitor's scope, data, abort criteria and lack of authority are explicit, every runtime abort trigger can actually fire for the declared monitor configuration, and the offline certificate check passes at every grid inertia before any powered test.

*Model.* Axis J theta'' = -b theta' + Kt u with PD state feedback designed for 20 Hz, damping 0.7 at nominal J; exact ZOH at Ts = 1 ms; closed loop A_cl(J) = Phi(J) - Gamma(J) K on a 9-point grid over J +- 30 %; common P from the discrete Lyapunov equation at nominal J. Online: PLSR on the nominal model with a declared level c such that {V <= c} lies inside the operating envelope; for a fixed model the decrease form's sign is state-independent, so the code depends on the state only through the level gate.

### T142 — Identify kernels suitable for Rust.

*Hypothesis.* Python dispatch overhead, not arithmetic, dominates the geodesic/Jacobi kernels, so the fused transfer loop is the best Rust target; the Kalman update and a standalone RK4 step are poor targets.

*Model.* Scalar restatements on a counting number type give exact flop counts; RK4 on an n-vector adds 13n+3 flops to four right-hand sides; interpreter dispatches are calls issued from ciw frames (profile hook); arithmetic intensity = flops per byte of state, stage vectors and matrices read or written per call.

### T143 — Identify interfaces requiring C++ industrial libraries.

*Hypothesis.* OPC UA, EtherCAT monitoring, vendor camera SDKs, PCL/Open3D and OpenCASCADE need C/C++ libraries, and each can sit behind a pinned subprocess boundary that exchanges retained bytes, leaving evidence code in Python.

*Model.* Design inventory: interface -> (native libraries, reason, boundary, direction, write path, fieldbus role, identity pins); validation rules: boundary = pinned_subprocess, direction in {read_only, geometry_exchange}, write path absent or disabled, fieldbus role passive_tap (a master originates output process data), identity pins concrete: refused are blank pins, the whole-pin labels current, main, master, trunk, head, dev, develop and x, the words latest, nightly, snapshot, stable, any, unknown, tbd, n/a and na anywhere, HEAD anywhere, the characters * ? < > = ~ ^ and N.x wildcards.

### T144 — Keep Python as orchestration and evidence layer.

*Hypothesis.* The evidence and identity layer (ciw.lab.evidence, ciw.lab.report, ciw.core.identities and their ciw imports) is pure standard-library Python; process spawns use argument vectors without a shell; native loading is confined to declared hardware probes. Whether spawned providers are pinned is not decidable from source and is tested only by a heuristic plus counterexample search.

*Model.* Directed import graph over parsed modules; transitive closure from the evidence roots; structural rules {closure stdlib-only, no native/spawn in closure, native loading within allowlist, no shell (shell=True, os.system/popen, asyncio.create_subprocess_shell), no compiled extensions}; heuristic rule: a spawning module names an identity token in identifiers or non-docstring strings.

### T145 — Use Julia for symbolic, optimization, and exploratory work.

*Hypothesis.* Julia can carry symbolic, optimization and exploratory work behind the pinned CIW to SCR boundary; until a pinned Julia environment exists, SymPy demonstrates the symbolic role on the geometry core.

*Model.* Pin procedure from docs/JULIA_SP1.md; symbolic Christoffel symbols Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il - d_l g_ij) and K = -(1/2W)[d_phi(G_phi/W) + d_theta(E_theta/W)], W = sqrt(EG), for the torus embedding.

### T146 — Define one canonical serialization across languages.

*Hypothesis.* One canonical JSON encoding (sorted keys, no whitespace, UTF-8, shortest round-trip binary64 with decimal ties to even in CPython repr layout, safe integers, refusals) is reproducible byte for byte by a separately written implementation in another language (Rust), and CIW's existing Python encoders can be measured against it.

*Model.* Encoding E: JSON values -> bytes per ciw.canonical-json.v1; identity = sha256(E(v)). Floats: the fewest digits k for which a k-digit decimal round-trips (monotone in k, found by bisection with exact integers); the nearer of the two k-digit neighbours, even digit on a tie; fixed form iff -4 < decpt <= 16.

### T147 — Compare CPU and GPU outputs.

*Hypothesis.* A comparison harness with an analytic tolerance policy separates legitimate reduction-order and precision differences from faults larger than the policy bound plus the candidate's own deviation; faults up to about the bound can escape. Exercised CPU-against-CPU because no GPU is present.

*Model.* For a sum of terms each passing through k roundings, |computed - exact| <= gamma_k * sum|a_j x_j| with gamma_k = k u/(1 - k u), u = 2^-53 (float64) or 2^-24 (float32, inputs rounded). The policy tolerance for two outputs is the sum of their bounds; bitwise mode compares bit patterns. A dropped term p is detected when |p| > tolerance + |candidate - reference| (triangle inequality); smaller faults may escape.

### T148 — Add deterministic reduction policies.

*Hypothesis.* Only a correctly rounded (exact-accumulation) sum is bitwise independent of summation order; fixed-tree pairwise sums are reproducible only for a fixed order; compensated sums are accurate but not order-invariant, and plain Kahan fails on large cancelling terms.

*Model.* Rigorous bounds (u = 2^-53, S = sum|x|, s = exact sum): sequential gamma_{n-1} S; pairwise gamma_{ceil(log2 n)} S; Neumaier u|s| + gamma_{n-1}^2 S (its compensation terms are those of Sum2); exact rounding u|s|. Kahan: 2u S + O(n u^2) S, checked with a declared allowance 4 n u^2 S. Exact sums by integer accumulation at scale 2^1074.

### T149 — Define FPGA telemetry-only interfaces.

*Hypothesis.* A read-only frame (header, sequence, timestamp, clock id, raw payload, CRC-32) with a single telemetry frame type and no host-to-device field lets the host refuse every corrupted frame and every command or write path before any payload is used, provided the CRC is appended in the order that keeps the frame one codeword.

*Model.* Frame = 28-byte big-endian header | 4*c bytes int32 payload | CRC-32/IEEE (reflected) appended little-endian. Bit p is bit p % 8 of byte p // 8 (LSB first, the reflected CRC's polynomial order). An error pattern e escapes iff its syndrome sum is zero; if the 32 single-bit syndromes of every 32-bit window are linearly independent over GF(2), no burst of length <= 32 escapes. Other patterns escape with probability about 2^-32.

### T150 — Record bitstream identity and toolchain.

*Hypothesis.* An identity record over canonical JSON can bind a bitstream's sha256 to its exact toolchain version and installation digest, constraint files and source tree, so that a change to any of them is detected when the changed artifact is checked against the record.

*Model.* record_sha256 = sha256(E(record without record_sha256)), E = ciw.canonical-json.v1; constraints_sha256 = sha256(E({file: sha256})); source tree = sha256(E({path: sha256})); toolchain version must match [0-9]+(\.[0-9]+){1,3}([-_.](rc|beta|alpha)[0-9]+)?\Z and the toolchain carries installation_sha256; every digest field is 64 lowercase hex digits, sizes are nonnegative integers and the synthetic flag is a boolean.

### T151 — Add FPGA rollback and compatibility metadata.

*Hypothesis.* A versioned compatibility matrix (bitstream frame format, supported boards, minimum host decoder) and a rollback record bound to retained identities let validation refuse every incompatible or unregistered rollback; 'previous version' is not a safe default.

*Model.* compatible(b, h, r) <=> format(b) in formats(h) and r in boards(b) and h >= min_host(b); a rollback is valid iff both identities are registered with matching digests, target < current, a reason is given, the matrix digest matches and compatible(target, host, board).

### T152 — Simulate packet loss, latency, and stale telemetry.

*Hypothesis.* Sequence numbers with modulo-2^32 serial arithmetic detect every loss, duplicate and reordering; timestamps with a declared clock offset never miss a stale frame, and their false alarms, like the misses of an offset estimated from minimum delay, occur at the rates the quantization, drift and bias predict.

*Model.* Gilbert-Elliott loss (p_gb=0.005, p_bg=0.2, loss 0.01/0.5), latency L = 2 ms + Gamma(2, 0.5 ms), duplicates p=0.002, period 1 ms, sampling phase U(0, 0.25 ms), timestamps quantized to 50 us, device/host drift 20 ppm, start sequence 2^32-1500, forced loss at 2^32-1. Truly stale iff L > 4 ms; receiver age = L + excess, excess = quantization + drift >= 0. P(L <= x) = 1 - exp(-g/s)(1 + g/s), g = x - 2 ms, s = 0.5 ms. Loss-count variance per frame p(1-p) + 2(l_b-l_g)^2 pi_g pi_b lambda/(1-lambda).

### T153 — Keep actuator writes disabled by default.

*Hypothesis.* A deny-by-default write policy refuses every write; enabling needs an externally issued, signed, in-scope, unexpired authorization verified against a trust anchor the lab does not hold, so no route opens it here.

*Model.* Gate(channel) = refuse unless enabled and verify(authorization, channel, now) passes; verify checks type, issuer, validity window, scope, signature, trust anchor in that order; the lab build has no trust anchor and no actuator transport.

### T154 — Treat control outputs as proposals until separately authorized.

*Hypothesis.* Control outputs can be computed and retained as proposals (frozen objects whose status is re-checked on every use), with content identities, while every conversion to a command is refused without separate authorization; the geometric proposal itself is sound to the order the Jacobi model predicts.

*Model.* Normal Jacobi field j(L) = j_lat(L) d + j_head(L) h; proposal h = -j_lat(L) d / j_head(L), undefined at a conjugate point (j_head(L) = 0). On the unit sphere h = -cot(L) d. With h applied, the residual separation is O(d^2).

## Results

| Task | Finding | Value | Evidence | Report |
| --- | --- | --- | --- | --- |
| T077 | Every exercised identity shows its predicted derivation, freshness and binding properties on the offline paths | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:be54599f083c` |
| T077 | Oscillator operation results carry no replay-stable numerical-result identity | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:be54599f083c` |
| T077 | The energy replay bundle identity excludes its replay receipt and verification | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:be54599f083c` |
| T077 | The retained log's device and kernel identities identify the producing GPU and code | (object of 4 entries; see the source report)  | `not_established` | `sha256:be54599f083c` |
| T077 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:be54599f083c` |
| T078 | Workbench sources retain the exact supplied bytes live, inside native bundles and after offline reopen | (object of 10 entries; see the source report)  | `numerically_verified` | `sha256:f479293ae058` |
| T078 | Non-canonical base64 transports are refused rather than normalized, and refused submissions retain no source | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:f479293ae058` |
| T078 | Evidence identity depends only on bytes: one byte string under two labels shares evidence_id while source_id differs | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:f479293ae058` |
| T078 | The oscillator recording path retains canonical content, not caller bytes: its file is a CIW re-serialization named by a content digest | {"recording_file_is_reserialization": true}  | `numerically_verified` | `sha256:f479293ae058` |
| T078 | The retained energy logs are real GPU energy measurements | {"declared_origins": ["synthetic_fixture"]}  | `not_established` | `sha256:f479293ae058` |
| T079 | Whitespace, key-order and float-spelling variants under one label retain distinct exact bytes and distinct evidence and source identities while their parsed content is canonically equal | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:f2bad39f4656` |
| T079 | Content-level identities coincide across the byte variants: one experiment_digest, log_digest and numerical_result_id | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:f2bad39f4656` |
| T079 | CIW canonical comparison is type-sensitive (1 != 1.0): a consistent int-for-float rewrite is refused by the stale log seal and, once resealed, retained as distinct evidence rather than aliased | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:f2bad39f4656` |
| T079 | A partial int-for-float rewrite of one copy of the solver settings is refused by CIW's cross-field solver check | {"observed": "Workload solver differs from plan"}  | `numerically_verified` | `sha256:f2bad39f4656` |
| T079 | A UTF-8 byte-order-mark variant with equal content is refused | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:f2bad39f4656` |
| T080 | Operation, execution, result, numerical-result and bundle identities never share a value, and CIW's step validator requires a new result identity, but no new numerical or operation identity, for a new occurrence | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:448b026f84b2` |
| T080 | Execution/result aliasing forgeries are refused on reopen with the pinned message; operation substitution is refused only by payload-shape validation | (list of 4 entries; see the source report)  | `numerically_verified` | `sha256:448b026f84b2` |
| T080 | Surviving mutant revision.gap: a workspace whose selection revision jumps to 1000, with records claiming revision 999, reopens | {"mutant": "revision.gap", "observed": "accepted"}  | `numerically_verified` | `sha256:448b026f84b2` |
| T080 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | {"surviving_mutants_in_this_task": ["revision.gap"]}  | `not_established` | `sha256:448b026f84b2` |
| T081 | The energy numerical_result_id is identical across original, sibling, replay, replay after offline reopen, replay of a replay and a separate session (same process and code) of one canonically identical log | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | A resealed metadata-only log edit changes the energy numerical_result_id while every analysed number stays equal: the identity binds the whole log through data.log_digest | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | The computed statistics.v1 result satisfies |mean| <= rms <= max(|min|, |max|) and agrees with a direct recomputation from the retained samples | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | CIW refuses to analyse a resealed variant of a log beside the original in one workbench when both keep one run_id | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Numerical edits that leave the retained source bytes unchanged and break a seal, a digest, the statistics bounds or the recomputed analysis are refused on reopen | (list of 4 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Surviving mutant energy-source.resealed: a resealed edit of the retained source log, with every derived record rebuilt, reopens with a different gross energy | {"mutant": "energy-source.resealed", "observed": "accepted"}  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Surviving mutant oscillator-stats.resealed: an in-bounds statistics edit with a recomputed seal reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Surviving mutant oscillator-stats.impossible-moments: resealed statistics that no sample set can have reopen | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Surviving mutant oscillator-stats.legacy: an edit to an unsealed legacy result reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:e773cf5095ee` |
| T081 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:e773cf5095ee` |
| T082 | Every execution, result, verification, bundle, session and replay occurrence has a fresh identity | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:43391e4d5601` |
| T082 | Reused or colliding occurrence identities are refused on reopen with the pinned message | (list of 5 entries; see the source report)  | `numerically_verified` | `sha256:43391e4d5601` |
| T082 | Surviving mutant fresh.created-at-shift: a backdated execution and result pair reopens | {"mutant": "fresh.created-at-shift", "observed": "accepted"}  | `numerically_verified` | `sha256:43391e4d5601` |
| T082 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:43391e4d5601` |
| T083 | Every retained replay receipt binds its source and replayed bundle digests, verification subject, fresh reproduction step, runtime digest and limited authority | {"binding_properties": 30, "held": 30, "receipts": 3}  | `numerically_verified` | `sha256:c075074308bf` |
| T083 | Transplanted receipts and false numerical-match claims are refused on reopen | (list of 3 entries; see the source report)  | `numerically_verified` | `sha256:c075074308bf` |
| T083 | Surviving mutant receipt.deleted: a replay bundle reopens without its replay receipt, indistinguishable from an original execution | {"mutant": "receipt.deleted", "observed": "accepted"}  | `numerically_verified` | `sha256:c075074308bf` |
| T083 | Replay agreement establishes verification by an independent party | (object of 2 entries; see the source report)  | `not_established` | `sha256:c075074308bf` |
| T083 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | {"surviving_mutants_in_this_task": ["receipt.deleted"]}  | `not_established` | `sha256:c075074308bf` |
| T084 | Replay receipt source digest forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 5 entries; see the source report)  | `numerically_verified` | `sha256:9a24c98e919a` |
| T084 | Surviving mutant receipt-source.sibling-execution: a receipt re-pointed, with its verification subject, at a sibling execution of the same bytes reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9a24c98e919a` |
| T084 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:9a24c98e919a` |
| T085 | Replay receipt replayed digest forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 4 entries; see the source report)  | `numerically_verified` | `sha256:1fd3e8370065` |
| T085 | Surviving mutant receipt-replayed.reidentified-bundle: a replay re-sessioned and dated before its source, with recomputed digests, reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:1fd3e8370065` |
| T085 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:1fd3e8370065` |
| T086 | Verification subject forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 6 entries; see the source report)  | `numerically_verified` | `sha256:14daf835e681` |
| T086 | The verification subject moves with the receipt source: T084's surviving mutant receipt-source.sibling-execution rebinds both to a sibling execution and reopens | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:14daf835e681` |
| T086 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | {"surviving_mutants_in_this_task": []}  | `not_established` | `sha256:14daf835e681` |
| T087 | Verification method forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 3 entries; see the source report)  | `numerically_verified` | `sha256:9c56709a369d` |
| T087 | Surviving mutant oscillator-method.injected: a sealed result carrying an injected verification_method reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9c56709a369d` |
| T087 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:9c56709a369d` |
| T088 | Independence flag forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 5 entries; see the source report)  | `numerically_verified` | `sha256:1ad1e1050e50` |
| T088 | Surviving mutant oscillator-independent.injected: sealed records carrying an injected independent: true reopen | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:1ad1e1050e50` |
| T088 | exchange._identity checks content only: a recomputed verification artifact claiming independent: true is accepted with status content_recomputed_not_authenticated | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1ad1e1050e50` |
| T088 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:1ad1e1050e50` |
| T089 | Admission status forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 7 entries; see the source report)  | `numerically_verified` | `sha256:288847889f82` |
| T089 | Surviving mutant oscillator-admission.injected: a sealed result carrying an injected state_admission reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:288847889f82` |
| T089 | A retained replay receipt or verification authorizes admission of the replayed result into canonical state | (object of 1 entries; see the source report)  | `not_established` | `sha256:288847889f82` |
| T089 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:288847889f82` |
| T090 | Provider runtime identity forgeries that leave a digest stale or contradict a recomputed, fixed or cross-referenced value are refused with the pinned message | (list of 6 entries; see the source report)  | `numerically_verified` | `sha256:bf59095ef3f6` |
| T090 | Surviving mutant oscillator-runtime.both: a provider runtime forged identically in execution and result reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:bf59095ef3f6` |
| T090 | Surviving mutant energy-runtime.all-bundles: a consistently forged analysis code digest reopens | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:bf59095ef3f6` |
| T090 | Surviving mutant energy-runtime.python-version: forged dependency versions reopen | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:bf59095ef3f6` |
| T090 | A consistently forged energy runtime identity is detected only when a replay recomputes the current analysis identity | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:bf59095ef3f6` |
| T090 | Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens | (object of 1 entries; see the source report)  | `not_established` | `sha256:bf59095ef3f6` |
| T091 | A saved workspace with oscillator results and an energy-accuracy original and replay reopens with no provider binding and reaches no execution path | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:32c92bd0d670` |
| T091 | The execution guard intercepts a replay attempted while it is active | "intercepted"  | `numerically_verified` | `sha256:32c92bd0d670` |
| T091 | Reopening recomputes the retained energy analysis to validate content | 7  | `numerically_verified` | `sha256:32c92bd0d670` |
| T091 | After reopen the binding-free energy-accuracy workflow still replays with a numerical match | true  | `numerically_verified` | `sha256:32c92bd0d670` |
| T092 | Requests to execute or replay unbound provider workflows, or to supply or bypass a binding, are refused with a named error and reach no provider process or adapter | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:b2710ad17a18` |
| T092 | A retained numerical-heat bundle whose values no provider computed passes reopen validation | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:b2710ad17a18` |
| T092 | The binding-free energy-accuracy replay succeeds in the same reopened session | true  | `numerically_verified` | `sha256:b2710ad17a18` |
| T092 | A content-consistent reopened bundle is acceptable as a verified production result | "not decided by the workbench"  | `not_established` | `sha256:b2710ad17a18` |
| T093 | Refused requests leave the in-memory session state and the session directory unchanged, and a re-save reproduces the saved workspace content | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:8f5748936b2a` |
| T093 | A refused reopen of a corrupted workspace creates no output directory | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:8f5748936b2a` |
| T093 | A refused recording operation is retained as a refused execution record | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:8f5748936b2a` |
| T094 | The golden retained workspaces reopen and validate with the current code without execution | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c6c04a854883` |
| T094 | Golden fixture SHA-256 digests | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c6c04a854883` |
| T094 | Reopening recomputes the golden energy analysis bit for bit on this platform | {"refused_as_platform_dependent": []}  | `numerically_verified` | `sha256:c6c04a854883` |
| T094 | The retained golden numerical-heat bundle names the pinned SCR revision, tree and recorded engine digest, and its values equal the integer Jacobi reference | [0, 16, 24, 16, 0]  | `numerically_verified` | `sha256:c6c04a854883` |
| T094 | The retained golden numerical-heat values were computed by the pinned SCR engine | "not established by reopen: a fabricated content-\u2026"  | `not_established` | `sha256:c6c04a854883` |
| T094 | Replay of the golden SCR bundle without a binding is refused | "operation_unavailable: No trusted repositories b\u2026"  | `numerically_verified` | `sha256:c6c04a854883` |
| T095 | Every malformed exchange fixture is refused by the CIW validator it targets, with the error text retained | {"fixtures": 19, "refused": 19}  | `numerically_verified` | `sha256:9f187ad8806e` |
| T095 | session.read_json accepts an overflowing number as infinity while the exchange and workbench parsers refuse it | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:9f187ad8806e` |
| T095 | session.read_json does not refuse a 20000-deep array with a ValueError | "not_refused_with_value_error"  | `numerically_verified` | `sha256:9f187ad8806e` |
| T095 | Session.from_workspace raises AttributeError, not a ValueError refusal, on a workspace holding only its version | "AttributeError"  | `numerically_verified` | `sha256:9f187ad8806e` |
| T095 | Session.from_workspace accepts an unknown top-level field and drops it on re-save | {"accepted": true, "dropped_on_resave": true}  | `numerically_verified` | `sha256:9f187ad8806e` |
| T095 | Workbench source.add reports malformed source JSON with text that names a bound runtime | "MALFORMED_RESPONSE: The bound runtime did not re\u2026"  | `numerically_verified` | `sha256:9f187ad8806e` |
| T096 | exchange._identity accepts every synthetic result and verification record sealed by an independently written canonical encoder, in any member order, and refuses every single-field mutation and a forged identity | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:91b2e9683aa2` |
| T096 | candidate_evidence.validate_response accepts valid synthetic ESM responses and refuses every boundary mutation | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:91b2e9683aa2` |
| T096 | Observation-batch identities are caller-declared: mutated batches pass exchange._identity | {"accepted": 24, "mutations": 24}  | `numerically_verified` | `sha256:91b2e9683aa2` |
| T096 | validate_response accepts unknown extra fields in an ESM response | {"accepted": 2, "cases": 2}  | `numerically_verified` | `sha256:91b2e9683aa2` |
| T096 | Passing provider-free conformance admits a candidate into canonical state | "not decided by the workbench"  | `not_established` | `sha256:91b2e9683aa2` |
| T097 | SCR executed through CIW's shared numerical-heat workflow and through its own Python API returns the declared integer heat fields | (object of 2 entries; see the source report)  | `provider_backed` | `sha256:1a2a57e2011d` |
| T097 | SCR heat outputs from the workbench and from SCR's Python API equal an independent integer reference | {"cases": 5, "mismatched_cells": 0}  | `independently_verified` | `sha256:1a2a57e2011d` |
| T097 | SCR replay reproduces the numerical identity with fresh execution occurrences and a non-independent verification | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:1a2a57e2011d` |
| T097 | The reopened SCR workspace carries no binding and refuses replay | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:1a2a57e2011d` |
| T097 | SET exchange conformance was not executed | (object of 3 entries; see the source report)  | `not_established` | `sha256:1a2a57e2011d` |
| T097 | The PPDA/SCR/SET exchange producer roundtrip was not executed | (object of 3 entries; see the source report)  | `not_established` | `sha256:1a2a57e2011d` |
| T097 | The integer heat field describes physical heat diffusion in a material | "not established: dimensionless integer arithmetic"  | `not_established` | `sha256:1a2a57e2011d` |
| T098 | The declared-workload and proved-heat workflows pin one SCR revision, while CIW declares more than one revision of SCR and of SET across its workflows | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:d754c46df8bd` |
| T098 | Every bound provider checkout is clean and at a CIW pin and, where CIW pins one, the pinned tree | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d754c46df8bd` |
| T098 | The working bytes of every clean bound checkout reproduce Git's HEAD tree id | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:d754c46df8bd` |
| T098 | Tracked-source and Cargo.lock digests of every readable bound checkout | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d754c46df8bd` |
| T098 | CIW's pinned subprocess adapter accepts each clean bound checkout for the module pins at its HEAD and refuses it for every other module pin | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:d754c46df8bd` |
| T098 | The SCR engine recorded for this run executes the SCR heat descriptor on the survey input | (object of 4 entries; see the source report)  | `provider_backed` | `sha256:d754c46df8bd` |
| T098 | A matching HEAD, tree and lock digest authenticates the upstream repository, toolchain and built engine | "not established: digests are not signatures"  | `not_established` | `sha256:d754c46df8bd` |
| T099 | cargo build --release --locked --offline of SCR execution-cli succeeds at the pinned revision | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:e298a463597d` |
| T099 | The locked build leaves Cargo.lock and the checkout unchanged and is bit-reproducible across fresh target directories | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:e298a463597d` |
| T099 | The freshly built engine's heat output equals an independent integer reference | [0, 219, 313, 219, 0]  | `independently_verified` | `sha256:e298a463597d` |
| T099 | The SP1 proved-heat locked build was not attempted | (object of 2 entries; see the source report)  | `not_established` | `sha256:e298a463597d` |
| T099 | A successful locked build makes the engine acceptable for production use | "not decided by the workbench"  | `not_established` | `sha256:e298a463597d` |
| T100 | Every earlier report present in the output directory when T100 runs keeps labels among the seven and physical and authority findings unestablished without acquisition | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | Every finding of those reports shows its label in the label column of its rendered Markdown row | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | A physical finding relabelled with a computational label is refused by the evidence validator | "Evidence label refused: basis supports not_estab\u2026"  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | render_markdown keeps the label column for a claim containing a pipe character | {"label_column_shifted": false, "row_cells": 3}  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | CIW keeps the synthetic energy fixture synthetic_only and refuses an unsealed relabel to physical | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | A resealed relabel of the synthetic energy fixture under a fresh occurrence is accepted by the workbench and classified as a physical-domain measurement | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | CIW refuses free-energy sources that relabel synthetic observations as physical or claim physical validation | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:581d024b2ef2` |
| T100 | The relabelled energy log is a physical GPU energy measurement | "not established: origin is operator-declared and\u2026"  | `not_established` | `sha256:581d024b2ef2` |
| T100 | The synthetic energy fixture characterizes real NVML counter accuracy | "not established: synthetic fixture"  | `not_established` | `sha256:581d024b2ef2` |
| T101 | PLSR decrease_resolution equals the documented bound transcribed in CIW at every evaluated matrix scale | {"cases": 288, "max_relative_difference": 0.0}  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | Inside the binary64 normal range the resolution scales exactly with 2^k and PLSR certifies exactly when eps > eps* | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | NUMERICAL_OVERFLOW first appears exactly where the CIW-recomputed decrease form overflows | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | A subnormal plant whose declared decrease form is exactly indefinite drives the PLSR resolution to zero; PLSR's code on it is recorded | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | Inconclusive threshold of the family A = s[[-eps, 1], [-1, -eps]], P = I: certified iff eps > 4 gamma_5 / (1 - 8u), independent of s | 2.2204460492503162e-15  | `analytic` | `sha256:db5496e9511a` |
| T101 | The documented decision order, transcribed in CIW, places the family's threshold at eps* in the normal range | {"mismatches": 0, "normal_range_cases": 112}  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | CIW's re-derived resolution scales exactly with a power-of-two matrix scale in the normal range | {"max_relative_deviation": 0.0, "scales": 14}  | `numerically_verified` | `sha256:db5496e9511a` |
| T101 | The subnormal witness's exact decrease form is indefinite while its binary64 evaluation is negative definite | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:db5496e9511a` |
| T102 | Power-of-two scaling of (A, P, x) inside LAPACK's window never changes the PLSR code or margin ratio | {"code_flips": 0, "evaluations": 550, "ratio_changes": 0}  | `numerically_verified` | `sha256:e6dab6845372` |
| T102 | Scaling P and x by powers of two never changes a discrete-time PLSR code, near the threshold included | {"code_flips": 0, "evaluations": 36}  | `numerically_verified` | `sha256:e6dab6845372` |
| T102 | Outside LAPACK's scaling window power-of-two rescaling of A and P never moves a PLSR code to a certificate that the exact class contradicts | {"evaluations": 250, "unsound": 0}  | `numerically_verified` | `sha256:e6dab6845372` |
| T102 | Scaling the witness by 2^-1074 drives its resolution to zero while its unit-scale code is DECREASE_NOT_DEFINITE; the scaled code is recorded | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:e6dab6845372` |
| T102 | No unscaled PLSR verdict certifies a form whose exact class is not negative definite | {"unsound": 0, "verdicts": 62}  | `independently_verified` | `sha256:e6dab6845372` |
| T102 | numpy.linalg.eigvalsh commutes exactly with power-of-two scaling while max|M| stays in [2^-485, 2^485] | {"inside_bitwise_equal": 72, "inside_total": 72}  | `numerically_verified` | `sha256:e6dab6845372` |
| T103 | PLSR decides the level gate exactly as the documented rule predicts across the near-limit scan | {"cases": 68, "disagreements": 0}  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | States from 2^-1074 to the largest binary64 number leave the PLSR code of a fixed negative definite form unchanged | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | Non-finite states are refused as input errors, not classified | {"inf": "raises ValueError", "nan": "raises ValueError"}  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | Matrices whose decrease form or scaled V leaves binary64 return NUMERICAL_OVERFLOW as predicted, and a control at the limit is still certified | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | The PLSR level gate misses exceedances when s^2 underflows: V > level is certified | {"cases": 68, "missed": 5}  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | The PLSR level gate reports OUTSIDE_LEVEL_SET for V below the level when s^2 overflows | {"cases": 68, "spurious": 9}  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | A finite in-box theta whose A(theta) overflows raises an input error instead of NUMERICAL_OVERFLOW | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | A subnormal plant whose declared decrease form is exactly indefinite drives the PLSR resolution to zero; PLSR's code on it is recorded | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T103 | The documented level rule, re-derived in CIW, misses exceedances when s^2 underflows and reports spurious ones when s^2 overflows | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:84a9bbcc4902` |
| T104 | PLSR never certifies a semidefinite or indefinite edge-case decrease form | {"cases": 28, "violations": 0}  | `independently_verified` | `sha256:33a816739795` |
| T104 | Every edge-case code is consistent with its exact class and exact distance from the resolution (resolved beyond two resolutions, never against the exact sign) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:33a816739795` |
| T104 | PLSR's Lyapunov solver and certificate constructor refuse a semidefinite Q and a singular P | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:33a816739795` |
| T104 | No verdict certifies with a P that quadratic() accepts although it is exactly indefinite | {"candidates": 16, "certifying_verdicts": 0}  | `numerically_verified` | `sha256:33a816739795` |
| T104 | Exact rational classes of the declared edge-case decrease forms | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:33a816739795` |
| T105 | Interior and boundary stiffness samples give the same PLSR code in all five unit systems | {"mismatches": 0, "samples": 10, "systems": 5}  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | check_vertices passes for the converted box in every unit system | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | Margin ratios are not invariant under non-uniform unit changes | {"interior0_ratio_spread": 7604466.762708691}  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | The declared box bounds 8 and 12 N/m and their binary64 neighbours keep their SI box decision in all five unit systems and under both conversion formulas | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | A parameter just above the SI bound is admitted after multiplying bound and sample by 1e-3 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | A parameter exactly on the SI bound is refused when bound and sample are converted by the two mathematically equal formulas k * 0.001 and k / 1000 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | The light-damping plant's verdict depends on the unit system although its exact decrease form is negative definite in all of them | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | P = [[6, 0.75], [0.75, 1]] is an exact common quadratic certificate for the SI stiffness box [8, 12] | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | Converting box bounds in binary64 collapses neighbours and depends on the formula used | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:4f21d6957a8d` |
| T105 | The declared stiffness box contains the stiffness of a real axis | null  | `not_established` | `sha256:4f21d6957a8d` |
| T106 | The eight rounding-free runtime-status-v1 codes are reached on one-parameter paths far from every threshold | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | Codes along each one-parameter path follow the documented decision order | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | The exactly indefinite P witnesses are never certified along their weak direction; whether they reach CERTIFICATE_NOT_POSITIVE is recorded | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | The runtime refuses to emit the five host-owned status codes | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | Pinned runtime constants: resolution factor, numerical policy, status vocabulary | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | The documented decision order, transcribed in CIW, assigns the eight rounding-free codes to the constructed path steps | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:28322d4c406c` |
| T106 | A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation | null  | `not_established` | `sha256:28322d4c406c` |
| T107 | No near-boundary case receives a certifying code unless its exact decrease form is negative definite | {"cases": 102, "violations": 0}  | `independently_verified` | `sha256:552d1db34038` |
| T107 | Beyond two resolutions from zero PLSR always resolves the sign | {"resolved_cases": 36, "unresolved": 0}  | `independently_verified` | `sha256:552d1db34038` |
| T107 | With a declared margin of three resolutions no case within two resolutions of zero is CERTIFIED_WITH_MARGIN | {"band_cases": 66, "certified": 0}  | `independently_verified` | `sha256:552d1db34038` |
| T107 | At required_margin 0 near-boundary spectra within two resolutions of zero receive CERTIFIED_WITH_MARGIN, and every such certificate is exactly sound | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:552d1db34038` |
| T107 | MARGIN_LOW appears exactly when the resolvable margin does not exceed the declared margin | {"margin_low_observed": true, "mismatches": 0}  | `numerically_verified` | `sha256:552d1db34038` |
| T107 | Share of exactly negative definite band cases answered NUMERICAL_INCONCLUSIVE without a declared margin | {"inconclusive_share": 0.4375}  | `provider_backed` | `sha256:552d1db34038` |
| T107 | The near-boundary family populates every exact resolution bin of max eig(M) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:552d1db34038` |
| T108 | Increasing required_margin never turns a failing PLSR verdict into a passing one | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:44934c28791e` |
| T108 | The switch from CERTIFIED_WITH_MARGIN to MARGIN_LOW happens exactly at required_margin = margin | {"mismatches": 0}  | `numerically_verified` | `sha256:44934c28791e` |
| T108 | Negative and non-finite required margins are refused | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:44934c28791e` |
| T108 | Monotonicity of the verdict in the declared margin follows from the decision order | "margin enters only as MARGIN_LOW iff margin <= r\u2026"  | `analytic` | `sha256:44934c28791e` |
| T108 | The documented rule re-derived in CIW is monotone on the same margin grids | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:44934c28791e` |
| T109 | Every certifying PLSR verdict on the adversarial plants uses an exactly valid certificate | {"certifying_verdicts": 30, "violations": 0}  | `independently_verified` | `sha256:cb3db4bd712d` |
| T109 | PLSR Lyapunov solutions agree with an independent solver to within 10 n^2 u cond(P) on every solved adversarial case | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:cb3db4bd712d` |
| T109 | Certified non-normal plants respect the Lyapunov transient bound ||exp(At)|| <= sqrt(cond P) | {"cases": 8, "max_ratio": 0.6180339887498949}  | `numerically_verified` | `sha256:cb3db4bd712d` |
| T109 | With P = I the non-normal plants are certified exactly when K < 2 sqrt 2, including K = 2.82 and 2.83 on either side | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:cb3db4bd712d` |
| T109 | PLSR's solve_lyapunov returns only exactly valid certificates on the exactly Hurwitz Jordan plants and otherwise raises ValueError at a documented gate | (object of 7 entries; see the source report)  | `independently_verified` | `sha256:cb3db4bd712d` |
| T109 | solve_lyapunov refuses an exactly Hurwitz plant for which an exactly valid quadratic certificate exists and PLSR's own verdict certifies it | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:cb3db4bd712d` |
| T109 | numpy.linalg.eigvals misplaces the exact eigenvalue -lambda of every defective test matrix by far more than machine precision; whether the sign flips is recorded | {"cases": 7, "misplaced_beyond_1e3_eps": 7, "wrong_sign": 5}  | `numerically_verified` | `sha256:cb3db4bd712d` |
| T110 | PLSR certifies each matrix with its own Lyapunov P exactly in the time convention where numpy finds it stable, and its solver refuses the other convention at a documented gate | {"matrices": 40, "mismatches": 0, "solver_refusals": 40}  | `independently_verified` | `sha256:08466cf6cf1b` |
| T110 | No Lyapunov P solved for one convention certifies a matrix in the other convention where numpy finds it unstable | {"cross_verdicts": 40, "unsound": 0}  | `independently_verified` | `sha256:08466cf6cf1b` |
| T110 | The two time interpretations give different PLSR outcomes exactly for the matrices whose stability differs between conventions | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:08466cf6cf1b` |
| T110 | Diagonal plants with P = I receive the code of each convention's exact decrease form | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:08466cf6cf1b` |
| T110 | A discrete plant refuses a theta_dot | {"code": "raises ValueError"}  | `numerically_verified` | `sha256:08466cf6cf1b` |
| T110 | A CIW Kronecker Lyapunov solution is positive definite exactly when numpy calls the matrix stable in that time convention | {"agreeing": 80, "solves": 80}  | `numerically_verified` | `sha256:08466cf6cf1b` |
| T111 | PLSR continuous-time Lyapunov solutions agree with the independent solver on the route family | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:2b611de00ecb` |
| T111 | PLSR discrete-time Lyapunov solutions agree with the independent solver on the route family | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:2b611de00ecb` |
| T111 | PLSR's solve_lyapunov returns a P exactly for the plants numpy's eigenvalues call stable and raises ValueError for the others | {"mismatches": 0, "plants": 50, "refused": 20}  | `independently_verified` | `sha256:2b611de00ecb` |
| T111 | PLSR's verdict certifies every numpy-stable plant with its own P and no numpy-unstable plant with P = I | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:2b611de00ecb` |
| T111 | The scalar route sees decrease at every sampled state of an indefinite form that PLSR reports DECREASE_NOT_DEFINITE | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:2b611de00ecb` |
| T111 | For n = 1 the PLSR verdict follows the sign of a throughout the normal range | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:2b611de00ecb` |
| T111 | The independent Lyapunov route (positive definite P) agrees with the numpy eigenvalue route on every route-family plant | {"agreeing": 50, "plants": 50}  | `numerically_verified` | `sha256:2b611de00ecb` |
| T112 | For the four simulated disturbance classes sup sqrt(V) stays below the sharp reachable-set supremum, and worst-case switching comes within 2 % of it | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:a9518b241b1c` |
| T112 | The sharp reachable-set supremum of sqrt(V) is converged in the quadrature step and lies below the quadratic ISS bound | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:a9518b241b1c` |
| T112 | Quadratic ISS-Lyapunov bound for the synthetic oscillator | (object of 4 entries; see the source report)  | `analytic` | `sha256:a9518b241b1c` |
| T112 | In one dimension the quadratic ISS bound w_bar / a is approached: sup |x| over 30 s is (1 - e^-30) of it | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:a9518b241b1c` |
| T112 | The series matrix exponential behind the exact ZOH simulation agrees with its closed form and, where available, an independent exponential | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:a9518b241b1c` |
| T112 | The ISS branch adds no runtime-status-v1 code, sample field or verdict to PLSR | "research branch only"  | `analytic` | `sha256:a9518b241b1c` |
| T112 | The disturbance bound w_bar = 0.5 holds for a physical plant | null  | `not_established` | `sha256:a9518b241b1c` |
| T112 | The ISS bound defines a safe operating envelope for a machine | null  | `not_established` | `sha256:a9518b241b1c` |
| T113 | Forwarded samples receive the runtime codes predicted from the declared model | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | The host accepts a window only when the kernel certifies theta_hat and both interval ends, so the near-bound estimate is rejected | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | The kernel refuses every host-owned code the adapter emitted | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | Kernel payloads built by the adapter carry only the declared model and numeric sample fields, and no envelope metadata | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | The adapter forwards theta_hat with both ends of its three-standard-error interval, and a near-bound estimate yields an interval end outside the box | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | Host statuses are decided in the adapter, are host-owned codes and are never forwarded | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | EKF theta estimates of forwarded windows lie within three standard errors of the synthetic truth | {"forwarded": 4, "misses": 0}  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | The adapter's common P is exactly valid at both box vertices of the declared discrete map | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:de5a6ddfe1ca` |
| T113 | The synthetic residual statistics describe a real encoder's performance | null  | `not_established` | `sha256:de5a6ddfe1ca` |
| T113 | The EKF standard error of theta covers the true parameter of a real axis at the stated rate | null  | `not_established` | `sha256:de5a6ddfe1ca` |
| T113 | The calibration referenced in the host envelope is valid | null  | `not_established` | `sha256:de5a6ddfe1ca` |
| T114 | The unit-balanced nominal P gives an exactly negative definite discrete decrease at every grid inertia | {"grid_points": 9, "not_negative_definite": 0}  | `numerically_verified` | `sha256:9591a17130b2` |
| T114 | With Q = I the nominal-model P does not cover the declared inertia interval | {"failing_grid_points": 4, "grid_points": 9}  | `numerically_verified` | `sha256:9591a17130b2` |
| T114 | The ZOH exponential agrees with its closed form and, where available, an independent exponential | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:9591a17130b2` |
| T114 | Without a level set the monitor's code is the same at every state; with the declared level it follows the exact V(x) > c decision | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9591a17130b2` |
| T114 | Every runtime code named as an abort trigger is produced by the declared monitor configuration, and the configuration produces no code outside the expected set | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9591a17130b2` |
| T114 | The servo-axis pilot is safe to operate | null  | `not_established` | `sha256:9591a17130b2` |
| T114 | The Lyapunov monitor may command, gate or release the axis | null  | `not_established` | `sha256:9591a17130b2` |
| T114 | The pilot configuration is acceptable for production use | null  | `not_established` | `sha256:9591a17130b2` |
| T114 | The monitor is ready for industrial deployment | null  | `not_established` | `sha256:9591a17130b2` |
| T114 | The placeholder inertia interval contains the real axis inertia | null  | `not_established` | `sha256:9591a17130b2` |
| T114 | Encoder and current-sensor calibrations of the bench are valid | null  | `not_established` | `sha256:9591a17130b2` |
| T142 | Exact floating-point operation counts of the scalar kernel restatements | (object of 5 entries; see the source report) operations per call | `numerically_verified` | `sha256:b29419cc2341` |
| T142 | Interpreter calls issued by ciw code per kernel call | (object of 6 entries; see the source report) calls | `numerically_verified` | `sha256:b29419cc2341` |
| T142 | Ranked Rust port recommendation | (list of 4 entries; see the source report)  | `numerically_verified` | `sha256:b29419cc2341` |
| T142 | Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer on the unit sphere | 4.440892098500626e-16  | `numerically_verified` | `sha256:b29419cc2341` |
| T142 | Rust ports of the ranked kernels are ready for industrial deployment | null  | `not_established` | `sha256:b29419cc2341` |
| T143 | Industrial interfaces that need C/C++ libraries behind a pinned subprocess boundary | (list of 5 entries; see the source report)  | `analytic` | `sha256:dcb8a151fb92` |
| T143 | Inventory validator refuses in-process bindings, write-capable directions or bus roles and unpinned entries | 10 refused mutations | `numerically_verified` | `sha256:dcb8a151fb92` |
| T143 | The listed interfaces are qualified for plant integration | null  | `not_established` | `sha256:dcb8a151fb92` |
| T143 | Vendor camera SDK acquisition meets its timing on real cameras | null  | `not_established` | `sha256:dcb8a151fb92` |
| T144 | Evidence and identity closure is standard-library Python with no native loading or process spawns | (list of 3 entries; see the source report)  | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Package-wide structural rules hold (no shell spawns, native loading only in declared hardware probes, no compiled extensions) | 0 violations | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Every process-spawning module names an identity in its code (heuristic, not a pin) | 0 violations | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Scanner flags forged modules that cross the boundary | 8 detected mutations | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Some process spawns run a PATH-resolved executable without comparing it to a pinned identity | {"witness": "ciw.lab.implementation_targets_serial"}  | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Numerical providers are invoked only through pinned executables | null  | `not_established` | `sha256:a91a942ff7a3` |
| T144 | Text search for 'subprocess' finds modules that spawn no process | {"witness": "ciw.acquired_dataset"}  | `numerically_verified` | `sha256:a91a942ff7a3` |
| T144 | Keeping native code behind subprocess boundaries makes machine interfaces safe | null  | `not_established` | `sha256:a91a942ff7a3` |
| T145 | Julia environment pinned and exercised through the CIW to SCR boundary | null  | `not_established` | `sha256:10d8dad92251` |
| T145 | Julia provider pin procedure | (list of 10 entries; see the source report)  | `analytic` | `sha256:10d8dad92251` |
| T145 | Symbolic torus Christoffel symbols and curvature agree with ciw.lab.surfaces.Torus | 4.440892098500626e-16  | `independently_verified` | `sha256:10d8dad92251` |
| T146 | Canonical JSON v1 test vectors (bytes and sha256) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:864796249c39` |
| T146 | ciw.telemetry.canonical reproduces the specification bytes on every accepted vector and tested float | (object of 4 entries; see the source report) mismatches | `numerically_verified` | `sha256:864796249c39` |
| T146 | ciw.core.identities.canonical_json reproduces the ASCII-escaped variant, not the specification bytes | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:864796249c39` |
| T146 | ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for non-ASCII text | (object of 1 entries; see the source report)  | `numerically_verified` | `sha256:864796249c39` |
| T146 | ciw.core.identities.canonical_json gives {1: 'x'} and {'1': 'x'} the same content identity | "50258d013c04133cf55f8dc9d9f096c7681c91045eaf870e\u2026"  | `numerically_verified` | `sha256:864796249c39` |
| T146 | Python canonicalizers accept values the specification refuses | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:864796249c39` |
| T146 | The exact shortest-digit rule agrees with CPython repr and NumPy Dragon4 on vector, random and decimal-tie binary64 values | (object of 4 entries; see the source report)  | `independently_verified` | `sha256:864796249c39` |
| T146 | CIW canonical JSON differs from RFC 8785 (JCS) numbers and key order | {"key_order_differs": true, "number_differences": 12}  | `numerically_verified` | `sha256:864796249c39` |
| T146 | Rust canonical JSON is byte-identical to the specification and refuses the same inputs | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:864796249c39` |
| T146 | Rust's own shortest float formatting breaks exact decimal ties away from the specification | {"decimal_ties": 96, "rust_shortest_differs": 42}  | `numerically_verified` | `sha256:864796249c39` |
| T146 | Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations | null  | `not_established` | `sha256:864796249c39` |
| T147 | CPU and GPU outputs agree under the tolerance policy on GPU hardware | null  | `not_established` | `sha256:a8039b2a05c8` |
| T147 | Bitwise policy detects reduction-order differences between float64 CPU orders | 123 rows of 128 | `numerically_verified` | `sha256:a8039b2a05c8` |
| T147 | Float64 reduction-order differences lie within the analytic error-bound policy | 0.0017336060475265365 fraction of the policy bound | `numerically_verified` | `sha256:a8039b2a05c8` |
| T147 | Float32 results violate the float64 policy and satisfy the float32 policy | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:a8039b2a05c8` |
| T147 | Dropped partial products larger than the policy bound plus the candidate's deviation from the reference are detected under both policies | (object of 2 entries; see the source report) faults | `numerically_verified` | `sha256:a8039b2a05c8` |
| T147 | The float32 policy misses dropped partial products up to about its bound | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:a8039b2a05c8` |
| T147 | GPU/CPU agreement establishes industrial readiness | null  | `not_established` | `sha256:a8039b2a05c8` |
| T148 | Correctly rounded exact accumulation is permutation-invariant | 1 distinct results | `independently_verified` | `sha256:3b95ad4aa494` |
| T148 | Fixed-order pairwise summation is reproducible for one order but not permutation-invariant | 4 distinct results | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | Kahan summation loses the sum [1, 1e100, 1, -1e100] that Neumaier summation keeps | {"exact": 2.0, "kahan": 0.0, "neumaier": 2.0}  | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | Observed errors of the sequential, pairwise, Neumaier and exact sums lie within their rigorous bounds | (object of 4 entries; see the source report) fraction of the bound | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | Observed Kahan errors lie within 2u sum|x| plus the declared second-order allowance | 0.34983344078055906 fraction of the allowance | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | The bound 2u|S| + 4n u^2 sum|x| does not bound Neumaier summation | (object of 3 entries; see the source report) error over bound | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | Distinct results under permutation for each algorithm and dataset | (object of 5 entries; see the source report) distinct results | `numerically_verified` | `sha256:3b95ad4aa494` |
| T148 | Deterministic reduction policy record | (object of 5 entries; see the source report)  | `analytic` | `sha256:3b95ad4aa494` |
| T149 | Telemetry frames round-trip every header field and channel bit-exactly | 0 failures | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | CIW table-driven CRC-32 agrees with zlib and the catalogue check value | (object of 3 entries; see the source report)  | `independently_verified` | `sha256:26dbf0a8c5a0` |
| T149 | Every burst of at most 32 bits and every single-bit error in a frame is refused by the decoder | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | A big-endian CRC trailer lets a 32-bit burst across the payload/CRC boundary escape | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | The burst guarantee holds only in the LSB-first bit order of the reflected CRC | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | Decoder, encoder and interface validator refuse every command, write or malformed path | 13 refusals | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | Host receiver exposes no sending or writing method | []  | `numerically_verified` | `sha256:26dbf0a8c5a0` |
| T149 | The frame format works on real FPGA links | null  | `not_established` | `sha256:26dbf0a8c5a0` |
| T149 | A telemetry-only interface guarantees the FPGA cannot actuate the machine | null  | `not_established` | `sha256:26dbf0a8c5a0` |
| T150 | Bitstream identity record binds bitstream, toolchain, constraints and source tree | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:053c6428e071` |
| T150 | Deployment of a synthetic placeholder bitstream is refused | "synthetic_bitstream"  | `numerically_verified` | `sha256:053c6428e071` |
| T150 | A real bitstream with this identity exists and is loaded on hardware | null  | `not_established` | `sha256:053c6428e071` |
| T150 | The bitstream is approved for production deployment | null  | `not_established` | `sha256:053c6428e071` |
| T151 | Compatibility rules and the set construction agree on every combination | {"combinations": 36, "compatible": 15, "disagreements": 0}  | `numerically_verified` | `sha256:387f7ebef3ef` |
| T151 | Rollback validation refuses incompatible, unregistered, no-op, forward and unexplained rollbacks | 8 refusals | `numerically_verified` | `sha256:387f7ebef3ef` |
| T151 | Rolling back to the previous bitstream version can be incompatible | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:387f7ebef3ef` |
| T151 | The rollback procedure is safe to execute on a production machine | null  | `not_established` | `sha256:387f7ebef3ef` |
| T151 | Rollback records are accepted for production change control | null  | `not_established` | `sha256:387f7ebef3ef` |
| T152 | Sequence-number gap detection recovers every lost frame across the 32-bit wrap | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | With the declared clock offset no stale frame is missed and false alarms match the timestamp quantization and drift prediction | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | An offset estimated from minimum delay misses stale frames at the rate its bias predicts | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | Loss rate and mean latency of a long run match the declared link model | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | Differencing sequence numbers in arrival order miscounts losses under reordering | {"arrival_order": 1258, "in_order": 119, "true": 120}  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | Non-modular differencing misses the loss at the 32-bit wrap | {"in_order": 119, "true": 120, "wrap_gap": [4294967295]}  | `numerically_verified` | `sha256:2f486c6d84d2` |
| T152 | Simulated loss, latency and staleness represent the real FPGA telemetry link | null  | `not_established` | `sha256:2f486c6d84d2` |
| T153 | Default policy refuses every write attempt | {"accepted": 0, "attempts": 1000}  | `numerically_verified` | `sha256:9d2a51b212ea` |
| T153 | Enabling writes is refused on every route, including a well-formed external record | 8 refused routes | `numerically_verified` | `sha256:9d2a51b212ea` |
| T153 | A frozen in-process policy object can be mutated | {"flag_mutated": true, "writes_still_refused": true}  | `numerically_verified` | `sha256:9d2a51b212ea` |
| T153 | The lab holds actuator write authority | null  | `not_established` | `sha256:9d2a51b212ea` |
| T153 | Disabled-by-default software writes make the machine safe | null  | `not_established` | `sha256:9d2a51b212ea` |
| T154 | Control outputs are constructed with status proposal and cannot be converted to a command here | {"refusals": 7, "status": "proposal"}  | `numerically_verified` | `sha256:0263e36ab3a8` |
| T154 | A frozen control proposal's status can be forced in memory, and the forced object is refused | {"refusals": 3, "status_forced": true}  | `numerically_verified` | `sha256:0263e36ab3a8` |
| T154 | Jacobi heading proposal cancels a lateral offset to second order | (object of 2 entries; see the source report) observed order | `numerically_verified` | `sha256:0263e36ab3a8` |
| T154 | Heading proposals are authorized for execution as actuator commands | null  | `not_established` | `sha256:0263e36ab3a8` |
| T154 | Applying the proposed heading corrections on a machine is safe | null  | `not_established` | `sha256:0263e36ab3a8` |

## Limitations

- T077: The retained log's device and kernel identities identify the producing GPU and code — not established.
- T077: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T078: The retained energy logs are real GPU energy measurements — not established.
- T080: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T081: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T082: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T083: Replay agreement establishes verification by an independent party — not established.
- T083: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T084: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T085: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T086: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T087: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T088: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T089: A retained replay receipt or verification authorizes admission of the replayed result into canonical state — not established.
- T089: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T090: Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens — not established.
- T092: A content-consistent reopened bundle is acceptable as a verified production result — not established.
- T094: The retained golden numerical-heat values were computed by the pinned SCR engine — not established.
- T096: Passing provider-free conformance admits a candidate into canonical state — not established.
- T097: SET exchange conformance was not executed — not established.
- T097: The PPDA/SCR/SET exchange producer roundtrip was not executed — not established.
- T097: The integer heat field describes physical heat diffusion in a material — not established.
- T097 is partial: Bind SCR and its engine in a fresh session, add the example source, execute, replay, save, reopen unbound under the guard and request replay; run four cases (including [0, 0, 1000, 0, 0] for 3 steps) 
- T098: A matching HEAD, tree and lock digest authenticates the upstream repository, toolchain and built engine — not established.
- T099: The SP1 proved-heat locked build was not attempted — not established.
- T099: A successful locked build makes the engine acceptable for production use — not established.
- T099 is partial: Bind --provider scr=<checkout> with cargo on PATH; command: CARGO_TARGET_DIR=<tmp> cargo build --release --locked --offline --manifest-path <scr>/crates/Cargo.toml -p execution-cli. The SP1 proved-hea
- T100: The relabelled energy log is a physical GPU energy measurement — not established.
- T100: The synthetic energy fixture characterizes real NVML counter accuracy — not established.
- T105: The declared stiffness box contains the stiffness of a real axis — not established.
- T106: A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation — not established.
- T112: The disturbance bound w_bar = 0.5 holds for a physical plant — not established.
- T112: The ISS bound defines a safe operating envelope for a machine — not established.
- T113: The synthetic residual statistics describe a real encoder's performance — not established.
- T113: The EKF standard error of theta covers the true parameter of a real axis at the stated rate — not established.
- T113: The calibration referenced in the host envelope is valid — not established.
- T114: The servo-axis pilot is safe to operate — not established.
- T114: The Lyapunov monitor may command, gate or release the axis — not established.
- T114: The pilot configuration is acceptable for production use — not established.
- T114: The monitor is ready for industrial deployment — not established.
- T114: The placeholder inertia interval contains the real axis inertia — not established.
- T114: Encoder and current-sensor calibrations of the bench are valid — not established.
- T142: Rust ports of the ranked kernels are ready for industrial deployment — not established.
- T143: The listed interfaces are qualified for plant integration — not established.
- T143: Vendor camera SDK acquisition meets its timing on real cameras — not established.
- T144: Numerical providers are invoked only through pinned executables — not established.
- T144: Keeping native code behind subprocess boundaries makes machine interfaces safe — not established.
- T145: Julia environment pinned and exercised through the CIW to SCR boundary — not established.
- T145 is partial: Probe for julia; record the pin procedure; derive torus geometry with SymPy and compare with the core.
- T146: Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations — not established.
- T147: CPU and GPU outputs agree under the tolerance policy on GPU hardware — not established.
- T147: GPU/CPU agreement establishes industrial readiness — not established.
- T147 is partial: Compute the dot products in four orders/precisions, compare with the harness under each policy, and drop each partial product in turn from the float64 and float32 candidates.
- T149: The frame format works on real FPGA links — not established.
- T149: A telemetry-only interface guarantees the FPGA cannot actuate the machine — not established.
- T150: A real bitstream with this identity exists and is loaded on hardware — not established.
- T150: The bitstream is approved for production deployment — not established.
- T151: The rollback procedure is safe to execute on a production machine — not established.
- T151: Rollback records are accepted for production change control — not established.
- T152: Simulated loss, latency and staleness represent the real FPGA telemetry link — not established.
- T153: The lab holds actuator write authority — not established.
- T153: Disabled-by-default software writes make the machine safe — not established.
- T154: Heading proposals are authorized for execution as actuator commands — not established.
- T154: Applying the proposed heading corrections on a machine is safe — not established.

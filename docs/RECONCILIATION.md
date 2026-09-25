# Implementation status

This document describes the repository at revision `617ca62`, whose executable
code is unchanged since `646aada`. For each contract area it records the
implemented behavior with the tests that assert it, the behavior the tests
cover only in part, and the measured limitations, stated as what is not
implemented or not supplied. Computational test success, matching replay
digests and content identities do not establish physical validity, calibration
traceability or verification.

## Records and identities

Implemented: `run.v1` structural validation with finite values, a strict
data-only embedded manifest, and content identities separate from execution
and verification identities:
`tests/test_adapter_seam.py::test_structural_validation_rejects_bad_record_anywhere`,
`::test_manifest_round_trip_is_detached_strict_and_data_only`,
`::test_evidence_is_content_bound_while_execution_and_verification_are_distinct`.
Raw bytes survive with distinct raw, calibrated and estimation identities:
`tests/test_investigation.py::test_raw_bytes_survive_with_distinct_corrected_and_estimation_identities`.
Exact geometric request bytes and covariance are retained:
`tests/test_geodesic.py::test_exact_raw_request_covariance_and_identity_are_retained`.
Every result stays `not_verified` with a null verification identity, and
resealing cannot lift that:
`tests/test_covariance_integration.py::test_jspt_coordinate_operation_uses_retained_joint_state`,
`tests/test_plsr.py::test_rehashing_cannot_detach_bindings_or_lift_claim_restrictions`.

Partial: the manifest `role` is required and accepts any nonempty string;
three values are in use (`instrument`, `measurement_adapter`,
`operation_provider`), and neither the validator nor a test restricts it to
them. The manifest declares no binding, entry point, time budget, parameters
or checks.

Not implemented: uncertainty columns beside channels (standard deviation,
interval, quantile or ensemble forms); covariance travels as result data. A run
carries no acquisition block; the acquisition time of a measurement-chain
record lives in its source metadata.

## Operations, executions and refusals

Implemented: versioned operation identities with explicit roles; a refused
invocation retained as `ciw.execution.v1` with no result, and a null runtime
when no provider was bound:
`tests/test_operation_runner.py::test_unknown_operation_retains_refusal_without_a_result`,
`::test_provider_cannot_publish_without_an_offline_payload_schema`,
`::test_generic_analysis_captures_effective_selection_and_preserves_global_selection`,
`tests/test_calibration_status.py::test_new_operation_version_reaches_admission_without_loading_a_provider`.
Capacity reserved for a pending operation and released on failure:
`tests/test_operation_runner.py::test_pending_operation_reserves_the_last_generic_capacity_slot`,
`::test_generic_publication_rechecks_capacity_after_legacy_analysis`,
`::test_failed_publication_releases_pending_reservation`. Domain refusals kept
distinct from errors and from held results:
`tests/test_subprocess_adapter.py::test_domain_refusal_has_no_result`,
`::test_refusal_reason_survives_session_save_and_reopen`,
`tests/test_plsr.py::test_numerical_refusal_is_distinct_from_a_certificate_violation`,
`tests/test_geodesic.py::test_domain_hold_and_refusals_remain_distinct`. A
calibration refusal at import writes no run, result or directory:
`tests/test_investigation.py::test_invalid_calibration_creates_neither_result_nor_output_directory`.

Partial: `operation.list` and `execution.list` are served, and no test sends
them. Exit `2` on a calibration refusal and exit `0` on a retained in-session
refusal are not asserted through the `ciw` process for the investigation
command; `ciw covariance` and `ciw covariance-replay` are exercised through the
Python API (`tests/test_covariance_integration.py`), not as processes.

Not implemented: a `result.created` broadcast (only `selection.changed` is
broadcast), refusal of a reused in-flight request id, cancellation, and an
instrument attach/detach protocol.

## Selection and clients

Implemented: one authoritative selection with a revision; a half-open interval
independent of the cursor; stale updates rejected without mutation:
`tests/test_protocol.py::SessionContractTests::test_cursor_and_interval_have_independent_semantics`,
`::test_stale_revision_is_rejected_without_mutation`,
`::test_explicit_analysis_interval_does_not_mutate_shared_selection`. Two
clients share the selection and reconnect:
`tests/test_integration.py::WorkbenchIntegrationTests::test_two_clients_share_selection_and_reconnect_after_disconnect`,
`::test_concurrent_edits_accept_one_revision_and_notify_both_clients`,
`::test_committed_selection_reaches_peers_if_sender_disconnects_before_reply`.
Terminal and service produce identical numbers:
`::test_headless_and_websocket_analysis_return_identical_numbers`. An
operation retains the selection it ran under:
`tests/test_operation_runner.py::test_generic_analysis_captures_effective_selection_and_preserves_global_selection`.
GTE refuses an interval that does not contain its whole batch:
`tests/test_geodesic.py::test_shared_session_executes_full_batch_and_rejects_observation_overrides`.

Partial: an unsupported protocol major is rejected
(`tests/test_protocol.py::SessionContractTests::test_bad_envelopes_and_unknown_operations_return_errors`);
the additive requests and fields are declared within major 1 without a minor
number, and the snapshot reports none. The 1 MiB incoming-message limit is
enforced by the transport and untested.

## Runtime binding and persistence

Implemented: source revision and tree, interpreter digest and dependency
versions verified before and after execution; bounded pipes and time; drift
refused:
`tests/test_subprocess_adapter.py::test_roundtrip_and_runtime_identity`,
`::test_source_drift_refuses_execution`,
`::test_pin_check_reads_bytes_even_when_git_assumes_unchanged`,
`::test_both_output_pipes_are_bounded`,
`::test_timeout_and_nonzero_exit_are_refused`,
`::test_expected_runtime_pins_are_enforced`,
`::test_arguments_do_not_allow_revision_or_command_injection`. Engines run in
separate processes at the pinned revisions:
`tests/test_investigation.py::test_domain_engines_execute_in_separate_pinned_processes`.
Reopening validates the whole workspace before any write and needs no domain
runtime:
`tests/test_replay.py::test_invalid_workspace_is_rejected_before_any_write`,
`::test_session_rejects_mismatched_scientific_evidence_before_writing`,
`tests/test_operation_runner.py::test_resealed_semantic_corruption_is_rejected_before_any_write`,
`::test_duplicate_json_keys_are_rejected_at_workspace_boundary`,
`tests/test_investigation.py::test_tampered_scientific_evidence_refuses_reopen_before_destination_writes`,
`::test_offline_inspection_needs_no_domain_runtime_and_leaves_source_unchanged`,
`tests/test_covariance_integration.py::test_saved_chain_inspects_without_importing_domain_engines`.
Replay keeps prior results and allocates new identities; an attempt with no
bound runtime is refused before any write:
`tests/test_investigation.py::test_offline_replay_retains_evidence_and_allocates_new_execution_and_result`,
`tests/test_covariance_integration.py::test_covariance_replay_preserves_old_result_and_allocates_new_identities`,
`::test_historical_pins_remain_replayable_after_runtime_upgrade`,
`tests/test_covariance_replay_refusal.py::test_unbound_latest_attempt_refuses_replay_before_runtime_resolution_or_writes`,
`tests/test_geodesic.py::test_replay_appends_events_without_replacing_evidence`.
Workspace versions 1 and 2, shutdown save and `serve --resume` reopen, in
process and across a signalled restart of the native service:
`tests/test_replay.py::test_saved_results_are_discoverable_and_identical_without_recomputation`,
`tests/test_operation_runner.py::test_unknown_operation_retains_refusal_without_a_result`
(version 2 with a retained execution),
`tests/test_deployment.py::test_graceful_event_shutdown_saves_selection_and_exact_result`,
`::test_resume_reopens_exact_evidence_and_results_without_computation`,
`::test_posix_signal_shutdown_and_restart_preserve_exact_saved_json`.

Implemented: the provider-free references (energy accuracy, thermal observer,
machine manifest, project graph, uncertainty validation) share one lifecycle in
`src/ciw/reference_workflow.py`; a committed retained workspace made from the
example inputs must reopen, reproduce its numbers (bit for bit for the exact
kinds, within the shared binary64 tolerance for the kernel-sensitive ones)
and replay freshly or refuse for a stated reason:
`tests/test_retained_compatibility.py` (eight tests). NumPy-backed references
record a numerical kernel probe in their algorithm identity; identities from
before the probe reopen and are refused replay naming the field:
`tests/test_reference_kernel.py`. `ciw workspace verify` reopens a workspace
offline and reports validity and per-bundle replayability with the differing
identity fields: `tests/test_workspace_verify.py`. The energy reopen check
tolerates roundoff-level differences but not changed conclusions:
`tests/test_energy_workflow.py::test_reopen_tolerates_kernel_level_rounding_of_error_metrics_but_not_changed_conclusions`.

The `runtime_mismatch` refusal of a saved revision outside the
current-or-historical allowlist is asserted without binding an adapter by
`tests/test_runtime_allowlist.py::test_saved_runtime_off_the_allowlist_is_refused_before_any_binding`;
`RUNTIME_UNAVAILABLE` and `RUNTIME_IO` by `tests/test_subprocess_adapter.py`;
refusal of a version-1 workspace carrying executions by
`tests/test_operation_runner.py::test_version_one_workspace_cannot_carry_executions`.

Partial: replay compares a recomputed data digest under the same pins; a
mismatch of the investigation and covariance verbs exits `2`, which needs the
pinned providers and is exercised only by the provider gates. Container
restart and resume are exercised only by `scripts/check_container.py`, which
needs a Docker daemon and did not run in this environment.

Not implemented: a multi-run journal, digest re-verification of the recording
and result files on disk, a persistence adapter, and in-process or remote
binding variants.

## Calibration applicability and expiry

Implemented: acquisition applicability persisted by the measurement-chain
adapter is re-checked and never rewritten; serving status is derived at one
explicit timezone-aware `evaluated_at` shared by all sources, over a half-open
validity interval; legacy records are labelled as derivations; saved bytes are
unchanged on every read surface: all eight tests of
`tests/test_calibration_status.py`, among them
`::test_current_interval_is_half_open_and_does_not_change_acquisition`,
`::test_session_read_surfaces_agree_and_leave_saved_result_bytes_unchanged`,
`::test_read_protocol_rejects_explicit_non_aware_or_null_time`;
`tests/test_rci_records.py::test_resealed_historical_applicability_cannot_be_rewritten`,
`::test_legacy_aware_timestamp_forms_are_retained_without_normalization`;
`tests/test_covariance_integration.py::test_v2_calibration_replays_after_serving_expiry`.
Missing, expired or mismatched calibration refuses the derived record:
`tests/test_investigation.py::test_invalid_calibration_creates_neither_result_nor_output_directory`.

Partial: the status is served in the `session.get` and `result.list` payloads,
as a sibling of the `result.get` payload in its response envelope, and on the
investigation summary; it is never written into a retained result. The
adapter's `calibration_not_yet_valid` reason is not asserted at the workbench
boundary.

Not implemented: a calibration profile bundle carrying serial number,
reference-standard designation, fit residuals and traceability; an in-session
correction operation; a calibration fixture corpus with expected outputs and
tolerances; a bench report; repeatability metrics; a physical reference
channel.

## Covariance artifacts and provenance

Implemented: `covariance-artifact.v1` with ordered quantities, per-axis units,
frame, reference values, full matrix, method, basis, provenance and
assumptions; a content identity over every field; symmetry and positive
semidefiniteness checked in correlation coordinates without repair; singular
matrices and mixed units accepted: all eleven tests of
`tests/test_covariance_artifacts.py`, among them
`::test_positive_semidefinite_including_singular_and_extreme_mixed_scales`,
`::test_invalid_matrix_refuses_without_repair`,
`::test_shared_sample_covariance_roundtrips_without_becoming_independent_sigmas`,
`::test_external_binding_rejects_reordered_quantities_mixed_unit_mismatch_and_frame`.
The native measurement-chain basis, the `[scale, zero_raw]` parameterization,
the coverage summary and the uncertainty scope are bound offline without
importing the provider: all eight tests of `tests/test_rci_records.py`.
Dependencies resolve to a retained result and its exact named covariance;
cycles and rehashed sources are refused:
`tests/test_covariance_records.py::test_dependencies_require_retained_result_and_exact_named_covariance`,
`::test_dependency_cycles_refuse_even_when_each_selected_artifact_matches`,
`::test_rehashed_fsrt_artifact_cannot_break_scientific_or_provenance_binding`,
`::test_rehashed_jspt_output_cannot_rewrite_map_or_source_semantics`. A shared
reference source and an undeclared model dependence are refused:
`tests/test_covariance_integration.py::test_common_reference_cannot_silently_enter_independent_assembly_composition`,
`::test_model_dependence_is_not_assumed`. Parameter covariance and every
contribution are retained, with the first-order term reproducible and a
diagonalized matrix distinguishable:
`tests/test_investigation.py::test_parameter_covariance_and_all_uncertainty_contributions_are_retained`.

Partial: the workbench admits one record per measurement chain, so no
cross-sample shared-parameter term exists in a workbench record although the
adapter batches records; the fixture showing a common offset surviving
averaging lives in the propagation provider's own suite and is run by neither
the workbench suite nor `scripts/check_adapters.py`. A relabelled
parameterization is refused
(`tests/test_rci_records.py::test_resealed_parameterization_stays_bound_to_native_assembly`),
and no other parameterization has an executable or an exercised conversion.
GTE's native covariances are retained exactly
(`tests/test_geodesic.py::test_exact_raw_request_covariance_and_identity_are_retained`)
and checked offline as symmetric positive semidefinite arrays: a resealed
indefinite tangent covariance and resealed negative ambient variances are
refused on reopen
(`::test_resealed_result_still_must_satisfy_source_and_scope[covariance]`,
`[mixed_covariance]`) and a negative input variance is refused by GTE itself
(`::test_domain_hold_and_refusals_remain_distinct[covariance]`); no test
exercises an asymmetric matrix. They are not `covariance-artifact.v1` and
cannot be selected by the propagation workflow.

Not implemented: a joint cross-assembly calibration model (an overlap is
refused as `unsupported_cross_assembly_dependence`); time-correlated
calibration error; a deterministic covariance corpus with expected outputs and
tolerances.

## Measurement-chain, estimation, propagation, geometric and verification integrations

Implemented, gated on the pinned checkouts and run by
`scripts/check_adapters.py`: the RCI to FSRT investigation through terminal
create, inspect and replay:
`tests/test_adapter_cli.py::test_real_cli_create_inspect_and_offline_replay`,
`tests/test_investigation.py::test_shared_session_executes_bound_provider_and_retains_unknown_provider_refusal`,
`::test_replay_preserves_last_model_override_and_physical_disagreement`,
`::test_unicode_domain_metadata_preserves_native_rci_digests`; the terminal
rendering keeps raw, calibrated, estimated, residual and refusal rows distinct
(`tests/test_adapter_cli.py::test_terminal_keeps_observation_estimate_and_refusal_distinct`,
ungated). FSRT v2 delivers six named artifacts; its `posterior` and
`reconciled` matrices equal the retained estimate covariances, and the
reconciled matrix carries a non-zero off-diagonal entry introduced by the
declared total (the posterior stage is diagonal for the independent-assembly
fixture):
`tests/test_covariance_integration.py::test_fsrt_stages_and_full_covariance_are_retained`,
`::test_calibration_provenance_and_native_parameterization_survive`. JSPT
propagates the retained joint state and its map is bound offline:
`::test_jspt_coordinate_operation_uses_retained_joint_state`,
`tests/test_covariance_records.py::test_jspt_schema_binds_declared_map_and_retains_nested_uncertainty_context`,
`::test_jspt_checks_cannot_promote_declarations_or_hide_numerical_changes`.
GTE executes the full batch, keeps held and refused outcomes distinct, replays
a policy override and rejects tampering, including the real terminal path: all
eleven tests of `tests/test_geodesic.py`, among them
`::test_policy_override_preserves_evidence_and_replays_held_result`,
`::test_real_terminal_create_inspect_replay`. The PLSR terminal bundle:
`tests/test_plsr.py::test_terminal_import_evaluate_inspect_and_replay_are_self_contained`,
`::test_domain_refusals_are_saved_evaluations_with_successful_exit`,
`::test_replay_mismatch_retains_new_evidence_and_returns_distinct_exit_code`,
`::test_tampered_bundle_is_rejected_before_replay_writes`,
`tests/test_plsr_engine.py::test_pin_rejects_changed_missing_and_extra_package_source`,
`::test_pin_rejects_different_distribution_version`.

Partial: FSRT's balance check is one-sided (`set_lcm.bridge.ciw`:
`score > chi2_quantile(rank, 0.999)`); its disagreement outcome survives replay
(`tests/test_investigation.py::test_replay_preserves_last_model_override_and_physical_disagreement`),
and `src/ciw/adapters/fsrt_records.py::validate_payload` re-derives the status
from the retained statistic on reopen, although no test reseals a
contradictory status and asserts the refusal; it is not an innovation or
estimation consistency test. JSPT does not derive
or verify a supplied Jacobian. PLSR results are not attached to shared
investigations.

Not implemented: information-block, reference-scale, solver and
validity-domain declarations by the estimation operation.

## Statistical validation of uncertainty

No statistical validation operation is implemented. No test asserts interval
coverage, innovation NIS or estimation NEES against a finite-sample band, and
no fixture of known noise separates an inflated covariance from an optimistic
one. The measurement-chain coverage summary describes which declared
components a covariance includes;
`tests/test_rci_records.py::test_resealed_coverage_summary_cannot_promote_or_drop_sources`
asserts that it cannot be promoted into coverage evidence. FSRT's one-sided
balance gate is a separate diagnostic and is never two-sided. Distinct source
identifiers are not evidence of independence, and synthetic uncertainty is
never promoted to a qualified estimate.

## Viewport

Implemented: the optional viewport renders the oscillator's phase and energy
views, shares the cursor, channel and interval with terminal clients, matches
responses by request id, refreshes on a revision conflict and marks a
disconnect stale. Its numeric cards are `sample.get` results. Headless checks
under `scripts/check_godot.py`: `godot/tests/protocol_smoke.gd` (two live
clients), `godot/tests/channel_generality.gd` (selector, cards and axes taken
from the record for two, three and five channels) and
`godot/tests/adapter_boundary.gd` (a generic record carrying `run_schema` or an
embedded manifest shows its identity and renders no values).

Partial: the viewport shows no results and no uncertainty; covariance,
calibration status and the geometric candidate are terminal and JSON only. No
check pairs one result between terminal and viewport.

Not implemented: a domain viewport for the measurement-chain, estimation,
propagation, geometric or PLSR integrations; a covariance-ellipse or other
uncertainty representation; in-terminal plots.

## Checks executed

- `python -m pytest -q` (Python 3.11, no external checkouts) on the
  `claude/sleepy-planck-mar1u3` branch after the retained-workspace compatibility
  gate, kernel probe, verification verb, CLI gap tests, audit hardening and
  the identity, record, validator and session check tables: **1705 passed,
  577 skipped, 38 subtests passed**; the same suite runs green in Prototype
  checks on Ubuntu and Windows for Python 3.11 and 3.12, while the PLSR
  terminal job and the provider gates stay red until the private providers are
  reachable (see [provider availability](PROVIDER_AVAILABILITY.md)).
- System audit on the same branch (Python 3.11, no external checkouts), each
  probe a script run against the live code: every session request kind with
  4,200 mutated payloads (0 escaped exceptions, every answer a well-formed
  envelope, catalog round-trips and no reservation left behind); 12
  transport-level frames against a served process (binary, malformed, deeply
  nested, non-finite, oversized, wrong version, spatial and bad-path
  connections: every one an error envelope or the documented close code);
  eight threads and, separately, six real WebSocket clients retaining,
  executing, replaying, inspecting and saving concurrently (no escaped
  exception, unique identities, saved workspace verifies); a saved workspace
  with one leaf changed at a time (1,400 reopens) and with keys removed, added
  or swapped (1,461 reopens): every content change refused with a ValueError
  and no reopen crashes; every provider-free reference under extreme numeric
  inputs (861 executions: refused or finite, none escaped); every example
  source of every kind with mutated leaves and, exhaustively, every numeric
  leaf replaced by text; `ciw workspace verify` on directories, missing,
  binary, truncated, deeply nested, oversized and sparse files (exit 2 with
  a message, never a traceback); a 150 s soak of execute, replay, save and
  reopen cycles (resident memory flat after warm-up); the whole suite in
  reverse file order (order independent) and the transport suites under
  `python -X dev` with resource warnings as errors (clean); all 140 `ciw`
  command lines quoted in the documentation parsed against the real parser;
  every markdown link resolved; every module imported with deprecation
  warnings as errors. Findings became the fixes and tests in
  `tests/test_audit_hardening.py`, the render binding in
  `tests/test_instruments.py`, the retention checks recorded above and the
  workspace reopen byte budget.
- Mutation gates on the same branch: `scripts/check_reference_mutants.py`
  (12 identity checks of the shared reference lifecycle and 10 record checks
  of the retained workbench, every one killed; before
  `tests/test_reference_identity_checks.py` and
  `tests/test_workbench_record_checks.py` existed, 7 and 8 of them survived)
  and `scripts/check_validator_mutants.py`, which derives a mutant from every
  guarded `raise` of the provider-free validators: before the
  `tests/test_*_checks.py` tables existed, 51 of 75 energy-log checks, 78 of
  79 thermal-contract checks, 24 of 30 uncertainty-source checks, 14 of 16
  consistency special-function checks, 15 of 15 machine-source checks and 6
  of 8 project-source checks survived; afterwards every mutant is killed
  except six clauses the script lists as logically redundant with a
  neighbouring clause. The same sweep over the session found 58 of 79
  request, payload and saved-workspace checks unguarded, because the suite
  asserted only that an error came back and never which one;
  `tests/test_session_checks.py` now names the code and message of each:
  73 of the 79 are killed and the six that survive are listed in the gate
  as repeated by a neighbouring check. Over the
  identified-design declaration validator (a provider-backed kind whose
  retention checks were added during this audit) 146 of 152 checks are
  still unguarded; that table is the recorded next step, not part of the
  gate.
- `python -m pytest` (Python 3.12 venv, no external checkouts) on 646aada: **404 passed, 46 skipped,
  38 subtests passed** in 92.09 s. Skips by reason: 1 × "Set CIW_RCI_REPO and CIW_FSRT_REPO to the pinned source checkouts" (test_adapter_cli.py); 12 × "Requires three clean pinned scientific checkouts" (test_covariance_integration.py); 19 × "Set CIW_GTE_REPO to exercise the real pinned GTE subprocess" (test_geodesic.py); 14 × "Set CIW_RCI_REPO and CIW_FSRT_REPO to exercise pinned domain subprocesses" (test_investigation.py) — 46 in total.
- `python scripts/check_adapters.py` on 646aada (clones the current pins rci f863bdd / fsrt 09a756d /
  jspt d910f5a / gte e55b8be and the historical pins rci 97fcc01 / fsrt d345888): **56 passed** in
  158.62 s, exit 0 (36 on 413fa4c before the GTE integration).
- On 413fa4c (before the GTE integration): `python -m pytest` 403 passed, 27 skipped, 38 subtests; the
  adapter gate 36 passed.
- `python scripts/check_godot.py --godot <Godot 4.5.2 headless>` on 617ca62 (code unchanged since
  646aada; no Godot file changed since 413fa4c): **PASS** — import, live protocol (protocol_smoke.gd),
  channel generality (channel_generality.gd: 11 named checks) and adapter boundary
  (adapter_boundary.gd: 16 named checks), 27 named checks in total, all PASS; generic adapters remain terminal-only without numerical reinterpretation.
- `python scripts/check_installed.py` on 413fa4c: **PASS** (installed distribution generates,
  analyzes and reopens retained evidence).
- Not executable in this environment: the container check (`scripts/check_container.py`, no
  Docker daemon), the Windows native controller, and the FSRT Windows/Python 3.13 NOAA
  reproduction, a failure in the pinned FSRT suite's own Windows workflow that predates the
  covariance adapter revision (see the recorded compatibility limitation in `COVARIANCE.md`).

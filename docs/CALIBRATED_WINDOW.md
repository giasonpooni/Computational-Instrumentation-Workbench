# Calibrated windows in the shared workbench

`ciw.calibrated-window.v1` connects TBRT → MCUR → STFE → GSIE, with SET
verification and fresh replay, through the existing session protocol. Add a
`calibrated-window` source, execute its operation, then inspect its retained
bundle, native instruments and GSIE context in the same result/fusion catalog.

The synthetic level example maps device times 10 and 12 to reference times 0
and 1. One affine profile maps raw values 2 and 4 to calibrated values 5 and 9.
The mean is 7, with variance 5.25; an independent scalar prior of mean 0 and
variance 1 gives posterior mean 1.12 and variance 0.84. These are computational
fixtures, not calibrated physical measurements.

## Bind and run

Use the five exact role-named checkouts in
`src/ciw/calibrated-window-runtimes.json`. This lane has its own SET revision;
existing telemetry and process provider pins remain unchanged.

```sh
ciw serve --calibrated-window-stack-root /trusted/window-stack --output-dir results
```

Send `source.add` with `kind: calibrated-window`, a label and base64 of the exact
bytes in `examples/calibrated-window/source.json`. Send `operation.execute` with
`operation_id: ciw.calibrated-window.v1` and `parameters: {source_id: ...}`.
The window, model and composition declaration are retained in those source
bytes. `bundle.replay`, `instrument.inspect`, `fusion.list`, `result.get` and
workspace save/restore use the existing interfaces. When also running the
process stack, `examples/shared-workbench/run.py --with-calibrated-window`
executes both in one session. Instrument inspection now also exposes retained
TBRT, MCUR and OIT steps from the existing process workflow.

## Uncertainty and compatibility

The source retains raw values and device timestamps unchanged. The clock map,
its synchronization evidence, calibration profile/evidence and their validity
periods remain separate from the derived reference times and calibrated values.
Source timestamps, the UTC epoch and calibration validity bounds must be exactly
representable at microsecond precision before provider binding. Extra trailing
fractional zeros are accepted; values requiring finer precision are refused
instead of truncated.
For N samples, one full joint covariance is ordered:

1. N device timestamps, clock skew, clock offset;
2. N indicated values, shared calibration gain, shared calibration offset.

Each native provider validates its marginal inputs and supplies its Jacobian.
CIW composes these into full time/time, value/value and time/value covariance
blocks using exact binary64-rational products with final floating-point
rounding. This preserves shared parameters, temporal correlation and declared
time/value correlation. The full joint matrix must be exactly symmetric and
positive semidefinite. `unknown` refuses; `declared_zero` means every
off-diagonal entry is zero. No covariance is imputed or repaired.

MCUR's native compatibility assessment must say that one affine profile
commutes with an arithmetic mean with full temporal uncertainty declared.
STFE consumes every mapped/calibrated sample and the complete propagated
value covariance. Its source reference binds the MCUR result as a derived
stream, without presenting it as newly acquired PPDA evidence. Nonlinear
calibration, profile switching, other window operations and silent sample
selection are refused in this version.

## Scope and authority

This is a scalar stationary-hold model on an explicitly declared nominal grid.
Clock uncertainty is retained, including cross-sample and time/value terms.
Window membership and calibration applicability are checked at the nominal
mapped acquisition times; this does not establish their validity over a
probabilistic time interval. Time uncertainty is not silently added as sensor
noise, and receipt time never substitutes for acquisition time.

OIT and FDIR do not run in this lane: observability remains `unresolved` and
fault assessment `not_run`. This context cannot enter the identified-design
workflow as a calibrated two-channel prior. The existing two-channel lane
continues to own the observable process-balance/fault-isolation demonstration.
ESM does not yet accept this new schema; candidate capture refuses it. Saved
workspaces restore historical content without restoring executable bindings,
fresh verification authority, eligibility or canonical admission.

Retained configuration, provider requests and numerical projections are compared
as canonical JSON bytes. Boolean, integer and floating-point substitutions such
as `true`, `1` and `1.0` are distinct, even after recomputing the outer digest.

`icrh.calibrated-window-to-state.v1` independently checks original/replay
bindings, native artifacts, shared covariance propagation and compatibility.
The installed gate executes actual providers and refuses skipped cases:

```sh
python scripts/check_calibrated_window.py --output-dir results/window-fixtures
```

Use `--stack-root /trusted/window-stack` to reuse existing exact pinned
checkouts. The output files are actual original/replay occurrences for ICRH.

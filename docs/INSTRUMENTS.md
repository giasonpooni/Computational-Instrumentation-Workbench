# Instruments: instructions and specifications

This catalogue records tools that can be used through the workbench and the
contracts needed to reproduce their results. An external tool is listed as
integrated only after its workbench entry point, saved evidence and replay path
have been exercised together.

| Instrument | Workbench status | Entry point |
| --- | --- | --- |
| `analytic-damped-oscillator.v1` | Integrated built-in synthetic instrument | `python -m ciw demo`, `analyze stats`, `analyze spectrum` |
| Parameterized Lyapunov Stability Runtime (PLSR) | External runtime ready for experimental computational use; workbench adapter pending | No PLSR workbench command yet |

## Synthetic damped oscillator

The built-in instrument produces a deterministic recording of an analytic
damped oscillator. Statistics, spectral analysis, saved results and reopening
work through the terminal. Source:
[`ciw.instruments`](../src/ciw/instruments.py); transport and record definitions:
[protocol v1](PROTOCOL.md).

### Install and run

Use Python 3.11 or newer in an activated virtual environment, from the repository
root. Installation uses the dependencies pinned in
[`pyproject.toml`](../pyproject.toml): NumPy 2.4.3 and websockets 16.0. The built-in
instrument ships with CIW; retain the CIW commit and Python version with an
investigation when reproducing it on another machine.

```text
python -m pip install -e .
python -m ciw demo --output recordings/demo.json
python -m ciw analyze stats --recording recordings/demo.json --channel q --start 2 --end 8 --output-dir results/stats
python -m ciw analyze spectrum --recording recordings/demo.json --channel q --start 0 --end 12 --output-dir results/spectrum
python -m ciw inspect results/spectrum/workspace.json
```

`demo` needs no external data. Analysis accepts the complete recorded-run JSON,
a channel (`q`, `v` or `energy`) and a half-open interval `[start, end)` in seconds.
The demo's model settings are fixed in v1. The `inspect` command reads a saved
file without executing an analysis. See the [quickstart](quickstart.md) for
virtual-environment setup, shared terminal use and the optional Godot viewport.

### Scientific specification

| Property | Delivered specification |
| --- | --- |
| Equation | `q'' + 2 gamma q' + omega_0^2 q = 0`, evaluated analytically |
| Parameters | Mass `1 kg`; natural frequency `0.8 Hz`; `omega_0 = 2 pi * 0.8 rad/s`; damping `gamma = 0.15 /s` |
| Initial state | Displacement `q(0) = 1 m`; velocity `v(0) = 0 m/s` |
| Energy | `0.5 * mass * (v^2 + omega_0^2 * q^2)` in joules |
| Samples and time | 768 uniformly spaced samples at 64 Hz on `[0, 12)` seconds; time since run start |
| Precision and frame | Retained `float64` scientific arrays; `oscillator-state` coordinate frame |
| Channels | `q`: metres; `v`: metres/second; `energy`: joules |
| `statistics.v1` | Sample count, mean, minimum, maximum and RMS of all retained samples in the selected interval |
| `spectrum.periodogram.v1` | One-sided periodogram PSD; constant detrend; periodic Hann window; density normalization |

Statistics require at least one selected sample; spectra require at least four.
For a spectrum of `N` samples, frequency spacing is `64 / N` Hz and density units
are the squared channel unit per Hz. Interior positive-frequency bins are
doubled; DC and an even-length Nyquist bin are not. The integral of the PSD equals
the window-weighted detrended mean square. Processing settings are included in
each result. This operation does not implement Welch averaging or a spectrogram.

### Outputs, identities and reopening

`demo` writes the recording to its requested path. Each analysis writes a source
recording copy, an individual result JSON and `workspace.json` to its output
directory, and prints a protocol-v1 response envelope whose `payload` is the
result record. Use separate output directories for
separate workspace snapshots: another analysis in the same directory replaces
`workspace.json` while retaining individual result files.

| Record | Fields and meaning |
| --- | --- |
| Recording | Instrument ID, `run_id`, scientific `evidence_id`, metadata, timestamps, channel values and derived render data |
| Result | `evidence_id`, `operation_id`, `execution_id`, `result_id`, source recording reference, selection revision, channel, interval and numerical `data` |
| Verification | `verification_id: null`, `verification_status: "not_verified"` |
| Workspace v1 | Complete recording, current selection, saved results and reserved view settings |

The evidence ID hashes the scientific content; the recording filename separately
binds the complete recording. These bindings detect inconsistent content. They
do not establish source authenticity or scientific verification. Analyses read
the full source arrays, and display geometry is a derived view.

Reopen the spectrum workspace in one terminal:

```text
python -m ciw serve --workspace results/spectrum/workspace.json --output-dir results/reopened
```

In another activated terminal, discover the restored results and save the session:

```text
python -m ciw send result.list
python -m ciw send workspace.save
```

Reopening validates and restores the recording, selection and existing results
without rerunning calculations. Existing evidence, result and execution IDs are
preserved. An explicit new analysis creates new execution and result IDs. The
[quickstart replay instructions](quickstart.md#reopen-an-investigation) explain
result inspection and selection persistence.

### Validation and limits

The checked-in [numerical tests](../tests/test_instruments.py) cover the analytic
fixture, energy decay, interval selection and PSD normalization, including DC and
Nyquist handling. [Integration tests](../tests/test_integration.py) compare terminal
and service calculations; [replay tests](../tests/test_replay.py) check saved
identities, content bindings and restoration without scientific recomputation.
Run them from a development installation:

```text
python -m pip install -e '.[dev]'
python -m pytest -q
```

The delivered input contract is one small uniformly sampled oscillator recording
per session. It is not a universal instrument format. Data are synthetic; device
acquisition, uncertainty estimates and physical validation are not supplied.

## PLSR: integration pending

PLSR evaluates quadratic Lyapunov certificates for declared linear and
affine-parameter models. Its standalone input boundary is implemented at commit
[`19ea6967060166ba09db6cd4563bd87bd6b3d196`](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/tree/19ea6967060166ba09db6cd4563bd87bd6b3d196).
This is the candidate runtime pin for the adapter; CIW does not yet import, invoke
or save PLSR evaluations through a workbench command.

The pinned [model-artifact guide](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/docs/MODEL-ARTIFACT-v1.md)
and [JSON Schema](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/src/lyapunov/schemas/model-artifact-v1.schema.json)
define `model-artifact-v1`: supplied matrices and certificate, parameter/rate
boxes, state ordering and units, time convention and sample period, margin
derivation, estimator identity and provenance. The standalone
[consumer instructions](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/consumer/README.md)
cover explicit sample inputs and emitted companion records.

The adapter must retain artifact/evidence digests, raw runtime status and all
three verdict booleans. `NUMERICAL_INCONCLUSIVE` is a numerical refusal;
`NOT_CERTIFIED` indicates a failed inequality. Exit zero from the standalone
consumer means a record was emitted, including refusals and violations. Current
claims remain computational only, with `may_authorize: false` and
`proof_status: NOT_CHECKED`; the kernel API remains marked `changing`.

## Documenting the next integration

Update this catalogue and the README when an integration is delivered. Each entry
must include its source and pinned version, supported runtime/environment, exact
setup and workbench commands, input ordering/units/time conventions, versioned
output and status semantics, saved-evidence and replay instructions, verification
state and limits, and links to validation evidence. A successful installation or
standalone example alone does not establish a completed workbench integration.

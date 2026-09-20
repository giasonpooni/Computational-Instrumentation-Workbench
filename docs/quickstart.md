# Run the first instrument

Computational Instrumentation Workbench is a terminal-first workbench that connects computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

This first slice is a local prototype with one **synthetic damped-oscillator instrument**. It does not yet acquire physical observations. Python owns the float64 scientific record and all calculations. A terminal can operate alone; the optional Godot window attaches to the same service.

For persistent native service control or the containerized backend, use the [deployment guide](../deploy/README.md).

## Install

Use Python 3.11 or newer. Commands below work from the repository root; activate the virtual environment before subsequent commands.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

On Linux/macOS: `python3 -m venv .venv`, then `source .venv/bin/activate`, then the same install command. If PowerShell activation is restricted, use `.venv\Scripts\python.exe -m ciw` in place of `ciw`; no policy change is necessary.

## Headless analysis

```text
ciw demo --output recordings/demo.json
ciw analyze stats --recording recordings/demo.json --channel q --start 2 --end 8 --output-dir results/stats
ciw analyze spectrum --recording recordings/demo.json --channel q --start 0 --end 12 --output-dir results/spectrum
ciw inspect results/spectrum/workspace.json
```

Both calculations emit structured JSON with independent evidence, operation, execution and result identities. The output directory contains a complete source recording, immutable result JSON and a portable `workspace.json`. Inspecting a file does not execute a calculation. Use separate output directories to retain distinct workspace snapshots; a subsequent analysis in the same directory updates its `workspace.json` while keeping individual result files.

The demo solves `q'' + 2 gamma q' + omega0^2 q = 0` analytically: mass 1 kg, natural frequency 0.8 Hz, damping gamma 0.15/s, initial displacement 1 m and initial velocity 0 m/s. It retains 768 samples at 64 Hz on `[0,12)` seconds. Energy is `0.5 m (v^2 + omega0^2 q^2)`. Numerical tests check energy decay, source selection and spectral normalization. These tests do not constitute physical validation of an instrument.

The spectrum is a one-sided, constant-detrended, periodic-Hann **periodogram PSD**, with explicit units and processing settings. It is not a Welch spectrum or spectrogram. Statistics and spectra always use the retained source arrays, never the reduced presentation or mesh.

## Shared terminal and viewport

Start the local session and keep this terminal running:

```text
ciw serve --recording recordings/demo.json --output-dir results/session
```

Open a second activated terminal:

```text
ciw send session.get
ciw watch
```

`watch` prints the current session and subsequent shared-selection events. Stop it with Ctrl+C to enter further commands. Plain requests also work without a viewport:

```powershell
ciw send sample.get --payload '{"time_s":3.0}'
ciw send selection.update --payload '{"expected_revision":0,"channel":"q","interval_s":[2,8]}'
ciw send analysis.stats
ciw send analysis.spectrum
ciw send workspace.save
```

Use the current revision from `session.get` in each selection update. A revision conflict means another client changed the selection: fetch the new state before retrying. On shells that alter JSON quoting, write the payload into a JSON file and pass `--payload-file request.json`.

Install or unzip [Godot 4.5.2 Standard](https://godotengine.org/download/archive/4.5.2-stable/), then import `godot/project.godot` and run it. Alternatively, use `godot --path godot` if the binary is on PATH. The client connects to `ws://127.0.0.1:8765` by default. It displays a phase portrait, backend-supplied energy surface/trajectory, shared cursor, channel/interval controls and numerical inspection. Source sample identity remains authoritative. Closing Godot leaves the Python session running.

The 3D axes represent state and energy; the surface is a function over state space, not physical terrain. Display scaling is declared separately and must not be read as a physical measurement.

## Reopen an investigation

```text
ciw serve --workspace results/session/workspace.json --output-dir results/reopened
ciw send result.list
```

Use a returned result ID with `result.get`. Reopening restores source data, selection and existing analysis records without rerunning the computation. New explicit analysis requests create new result and execution IDs. Save explicitly with `workspace.save` before shutting down when you want to preserve changed selection and the full result collection. There is no automatic acquisition, recomputation or restart after exit.

## Validation

```text
python -m pytest -q
godot --headless --path godot --editor --quit
python scripts/check_godot.py --godot godot
```

The scripted Godot check requires port 8765 to be free; it starts and stops its own temporary service. Pass your Godot executable's path if it is not on PATH. See `godot/README.md` for the live bridge smoke check and optional rendered capture. The protocol is specified in [PROTOCOL.md](PROTOCOL.md), with contributor requirements in [DEVELOPMENT.md](DEVELOPMENT.md).

## Current limits

- One small, uniformly sampled demo recording per session; q, v and energy channels. The oscillator demo has its own bounded record contract.
- The terminal client currently emits JSON and event lines. Rich/Textual panels and in-terminal plots are not implemented.
- Local native clients, text JSON, maximum 1 MiB incoming messages and 1,024 analysis results per session. No remote authentication, device acquisition or hard real-time control.
- Periodogram only; streaming telemetry, spectrograms, region occupancy, cancellation and binary arrays are not implemented.
- Local JSON is prototype persistence. An instrument catalog and a worker supervisor remain integration work; no replacement database is introduced.
- No uncertainty estimate or verification certificate is fabricated. Result records explicitly say `not_verified`.
- Performance at real instrument data volumes and a deployment bundle have not been established.

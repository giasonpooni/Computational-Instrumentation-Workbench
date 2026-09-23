# Oscillator operator card

This is the command-only walkthrough for the built-in synthetic oscillator.
The full contract is in [INSTRUMENTS.md](INSTRUMENTS.md); current coverage and
limits are in [INTEGRATION_COVERAGE.md](INTEGRATION_COVERAGE.md).

From the repository root:

    python -m pip install -e .
    python -m ciw demo --output recordings/demo.json
    python -m ciw analyze stats --recording recordings/demo.json --channel q --start 2 --end 8 --output-dir results/stats
    python -m ciw analyze spectrum --recording recordings/demo.json --channel q --start 0 --end 12 --output-dir results/spectrum
    python -m ciw inspect results/spectrum/workspace.json

Reopen the saved workspace in one terminal:

    python -m ciw serve --workspace results/spectrum/workspace.json --output-dir results/reopened

Inspect and save the restored session from another activated terminal:

    python -m ciw send result.list
    python -m ciw send workspace.save

Use the result ID returned by result.list with result.get when a single result
is needed. Use separate output directories for separate workspace snapshots.

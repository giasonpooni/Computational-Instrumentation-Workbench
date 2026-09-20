# Contributor instructions

This is the Computational Instrumentation Workbench: a terminal-first workbench connecting computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

- Preserve existing numerical engines. Python owns scientific records and calculations; Godot owns rendering and interaction.
- Terminal-only use must work. Closing the viewport must not stop the service.
- Use the versioned contract in `docs/PROTOCOL.md`; coordinate contract changes across clients and tests.
- Keep playback cursor separate from half-open analysis intervals. Reject stale selection updates.
- Retain distinct evidence, operation, execution, result, and verification identities. Never infer verification from a displayed scene.
- Compute from full-resolution scientific records, never render geometry or terminal summaries.
- Do not overwrite another contributor's branch or working tree. Use a separate branch and review integration against the latest remote refs.
- Claude is also working on this project and owns README, architecture documentation and ADRs. Codex leads executable prototype, protocol and verification. See `docs/coordination.md`.
- The user explicitly requested publishing validated increments directly to main. Fetch first, integrate other contributors' commits, and use ordinary fast-forward pushes; never force-push shared history.
- Run `python -m pytest` and the headless Godot import check when changing the corresponding component. Record unavailable checks honestly.
- For each successful tool integration, update the README's integrated-tools catalogue and `docs/INSTRUMENTS.md` in the same change. Include setup, runnable terminal examples, pinned tool/adapter versions, input/output specifications, status and error meanings, validation evidence, and limits. Keep standalone-ready or planned tools separate until a CIW end-to-end path has been exercised. See `docs/DEVELOPMENT.md#documenting-an-integrated-tool`.

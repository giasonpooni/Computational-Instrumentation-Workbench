# Build coordination

The user has assigned Codex to lead the build while Claude works in parallel.

## Current work split

| Contributor | Current scope |
| --- | --- |
| Claude | README, architecture document, architecture decision records and architectural review |
| Codex | Executable Python/NumPy instrument prototype, terminal CLI, shared session/protocol, optional Godot client, tests, continuous integration and integration checks |

This split is grounded in the user-provided report of Claude's current work. No direct message has been sent to Claude. Preserve Claude's documents and integrate changes through Git; do not claim that an unreviewed API is an agreed final platform contract.

The user explicitly requested validated work on `main`. Before publishing, fetch current refs, incorporate concurrent commits, run relevant checks and perform a normal fast-forward push. Never force-push over either contributor. Architecture changes should be reconciled against the executable protocol rather than silently changing one client.

## First milestone

One synthetic numerical instrument, one reusable analysis path, and two independent clients:

1. Run headless statistics and a periodogram; retain the source evidence and immutable results.
2. Attach the Godot phase/energy viewport to the same authoritative session.
3. Change shared selections in either client; inspect the same retained sample and preserve the analysis interval when moving the cursor.
4. Disconnect/reconnect the viewport without stopping the service.
5. Save/reopen the investigation without silently recomputing or replacing result identities.

The demo proves interface mechanics. It does not establish a production catalog, validated physical model or general-purpose scientific plotting system.

## Integration sequence

1. **Existing instrument adapter:** select one actual NumPy instrument and its reference fixtures. Preserve its headless calculation and numerical tolerances. Map quantities, units, timestamps, frames and missing-data conventions explicitly.
2. **Second and third instruments:** use a scalar/time-series instrument, state-estimation instrument and spatial result to test which contract elements are genuinely common. Capability metadata should determine available views; a scalar instrument does not need a mesh.
3. **Spectral and temporal expansion:** backend spectrograms, streaming with separate acquisition/publish/render rates, cancellation, stale-result handling under load and bounded history access.
4. **Deployment for a stated use:** package the selected environments and optional viewport; measure target workloads, precision, throughput, recovery and device access. Native/GPU/time-critical engines remain independent where appropriate.

Readiness is explicit: standalone-ready, workbench-ready and deployment-ready are distinct promotion stages. This first build is a prototype pending real-instrument integration. The workbench standardizes operations and result meaning; specialized instruments retain their scientific models.

# Circle geometry in the shared bench

These sources wrap the existing public synthetic
[`circle.json`](../adapters/circle.json) request as exact base64 bytes. They
contain no physical sensor recordings, surveyed frames or private BIM data.

Register `source.json` as kind `geometric-circle`, then execute
`ciw.geometric-circle.v1` with its `source_id`. The host must explicitly bind
the pinned GTE checkout. The workbench retains native projection, full joint
covariance, candidate policy and fresh replay through the same session used by
the other modules. `experiment.inspect` exposes a read-only geometry view.

| Source | Expected native outcome |
| --- | --- |
| `source.json` | Eligible under the declared geometric policy |
| `held.json` | Candidate held by the correction limit; original evidence retained |
| `stale.json` | `constraint_not_valid` refusal at acquisition time |
| `singular.json` | `numeric_geometry` refusal at the circle center |
| `covariance.json` | `input_covariance` refusal for negative variance |
| `unknown-geometry.json` | `unsupported_uncertainty` refusal |
| `frame-mismatch.json` | `input_frame` refusal |

The circle is declared fixed and exact. Input covariance includes cross-sample
terms. Retained successful results require exactly symmetric positive
semidefinite input covariance; the native operation's broader roundoff-tolerant
domain is not silently promoted into this shared profile. Entries must be
finite and exactly representable as binary64. Native output covariance retains
its numerical roundoff and original values. Tangent covariance remains attached to its bases and projected reference
points; ambient covariance is generally singular. No surveyed frame, physical
accuracy, trajectory, BIM acceptance or state admission follows from projection.

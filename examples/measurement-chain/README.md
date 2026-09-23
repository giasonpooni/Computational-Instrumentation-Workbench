# Retained measurement-chain example

`source.json` assembles the existing public synthetic inputs from
`examples/adapters/two-reservoir-covariance.json` and
`examples/adapters/tank-covariance-map.json`. Both raw records, assembly text,
calibration covariance and evidence are software fixtures, not device data or
physical calibration certificates. Original raw JSON bytes remain base64 encoded
and unchanged.

The workbench operation calls the existing RCI v2 calibration, FSRT v2 snapshot
investigation and JSPT covariance propagation. The complete native workspace is
retained, including separate FSRT/JSPT execution and result identities. A fresh
same-runtime reproduction has its own native occurrences. Native results keep
their `not_verified` status; the outer receipt only describes reproduction.

Declare independent assemblies and independent prior, observations and total.
Unknown dependence, shared uncertainty sources, stale calibration, inconsistent
record/assembly bindings and unsupported covariance mappings refuse. The JSPT
Jacobian and output reference are caller declarations, not verified derivatives
or new observations. A held FSRT reconciliation remains held.

The resulting typed workbench object is `measurement-chain-testbed`; it does not
create a GSIE fusion state or physical traceability.

# Acquired calibrated windows

`make_source.py` builds synthetic acquisition snapshots and explicit mappings
from native PPDA observation, record and document identities. It derives values
only from retained snapshot rows. The complete source snapshot bytes survive.

The mapping source selects each row in its first acquisition snapshot and
declares the clock map, affine calibration, quantity frame and full joint
covariance. `unknown` covariance or mismatched metadata refuses execution.
Raw device times and acquisition cursor order are never rewritten as event time.

Each mapping executes the existing TBRT → MCUR → STFE → GSIE workflow. Its
unchanged child bundle retains SET's numerical replay receipt; the outer receipt
checks mapping and child identities and does not claim independent verification.
Replay preserves the exact selected acquisition occurrence.

Repeated windows declare their own independent reference priors. They do not
silently consume the previous posterior or establish cross-window independence.

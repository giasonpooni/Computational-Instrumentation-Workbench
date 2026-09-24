"""Actuator write policy and control proposals (T153, T154).

Actuator writes are denied by default. Enabling them needs an authorization
record issued outside the workbench and verified against a trust anchor; the
lab build contains no trust anchor and no actuator transport, and it refuses
to issue authorization records itself, so every path ends in a refusal here.
Control outputs are immutable proposals: converting one into a command is
refused without separate authorization. These are software refusals inside
one Python process. They are defence in depth, not a security boundary, and
they establish nothing about machine safety or actuator authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .implementation_targets_serial import canonical_sha256

AUTHORIZATION_SCHEMA = "ciw.actuator-authorization.v1"
PROPOSAL_SCHEMA = "ciw.control-proposal.v1"


class AuthorityRefusal(PermissionError):
    """An actuator write or command conversion lacks the authority it needs."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AuthorizationRecord:
    """Shape of an externally issued authorization; the lab can describe it, not issue it."""

    issuer: str
    subject: str
    channels: tuple
    valid_from: str
    valid_until: str
    purpose: str
    signature: str | None = None
    schema: str = AUTHORIZATION_SCHEMA


def issue_authorization(*_, **__):
    raise AuthorityRefusal("lab_cannot_issue_authority", "The lab cannot create actuator authorization records")


def verify_authorization(record, *, channel: str, now: str) -> None:
    """Check an authorization in a fixed order; in the lab build the last check always refuses.

    ``now`` is an explicit ISO-8601 UTC instant so that results never depend on
    the wall clock.
    """
    if record is None:
        raise AuthorityRefusal("authorization_missing", "No actuator authorization record was supplied")
    if not isinstance(record, AuthorizationRecord) or record.schema != AUTHORIZATION_SCHEMA:
        raise AuthorityRefusal("authorization_wrong_type", "Authorization must be a ciw.actuator-authorization.v1 record")
    if record.issuer.strip().lower().startswith("ciw"):
        raise AuthorityRefusal("self_issued_authority", "The workbench cannot authorize its own actuator writes")
    if not record.valid_from <= now < record.valid_until:
        raise AuthorityRefusal("authorization_expired", "Authorization is not valid at the stated instant")
    if channel not in record.channels:
        raise AuthorityRefusal("out_of_scope", f"Authorization does not cover channel {channel}")
    if not record.signature:
        raise AuthorityRefusal("unsigned_authorization", "Authorization carries no signature")
    raise AuthorityRefusal("no_trust_anchor", "The lab build holds no actuator-authority trust anchor")


@dataclass(frozen=True)
class ActuatorWritePolicy:
    """Deny-by-default write gate. The ``enabled`` flag is never trusted on its own."""

    enabled: bool = False
    authorization: AuthorizationRecord | None = field(default=None)

    def check_write(self, channel: str, value: float, *, now: str) -> None:
        if not self.enabled or self.authorization is None:
            raise AuthorityRefusal("writes_disabled_by_default", "Actuator writes are disabled by default")
        # Re-verify on every write: a mutated flag alone must not open the gate.
        verify_authorization(self.authorization, channel=channel, now=now)
        raise AuthorityRefusal("no_actuator_transport", "The lab build has no actuator transport")

    def enable(self, authorization, *, channel: str, now: str) -> "ActuatorWritePolicy":
        verify_authorization(authorization, channel=channel, now=now)
        raise AuthorityRefusal("no_actuator_transport", "The lab build has no actuator transport")


class ProposalRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ControlProposal:
    """A control output. Its status is fixed at ``proposal``; it is not a command."""

    channel: str
    value: float
    unit: str
    produced_by: str
    inputs_sha256: str
    rationale: str
    status: str = "proposal"

    def __post_init__(self):
        if self.status != "proposal":
            raise AuthorityRefusal("proposal_status_fixed", "Control outputs are proposals; their status cannot be set")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ProposalRefusal("nonfinite_proposal", "A control proposal value must be a finite float")

    def record(self) -> dict:
        # Frozen dataclasses can be altered with object.__setattr__; re-check before anything is derived.
        _require_proposal(self)
        body = {"schema": PROPOSAL_SCHEMA, "channel": self.channel, "value": self.value, "unit": self.unit,
                "produced_by": self.produced_by, "inputs_sha256": self.inputs_sha256, "rationale": self.rationale,
                "status": self.status}
        return dict(body, proposal_sha256=canonical_sha256(body))


def _require_proposal(proposal) -> None:
    if proposal.status != "proposal":
        raise AuthorityRefusal("proposal_status_tampered", "A control output whose status was altered is refused")


def to_command(proposal, authorization=None, *, now: str):
    """Refused unless separately authorized; the lab build cannot complete the conversion."""
    if not isinstance(proposal, ControlProposal):
        raise AuthorityRefusal("not_a_proposal", "Only control proposals can be considered for conversion")
    _require_proposal(proposal)
    if authorization is None:
        raise AuthorityRefusal("proposal_not_authorized", "Control outputs are proposals until separately authorized")
    verify_authorization(authorization, channel=proposal.channel, now=now)
    raise AuthorityRefusal("no_actuator_transport", "The lab build has no actuator transport")


def heading_correction(surface, u0, heading: float, length: float, lateral: float, steps: int = 200,
                       singular_ratio: float = 1e-6) -> ControlProposal:
    """Heading change that nulls a lateral offset at arclength ``length`` to first order.

    The normal Jacobi field satisfies j(L) = j_lat(L) d + j_head(L) h, so
    h = -j_lat(L) d / j_head(L). At a conjugate point j_head(L) = 0 and no
    heading correction exists; the proposal is refused there.
    """
    from .jacobi import transfer

    matrix = transfer(surface, u0, heading, length, steps).matrix()
    j_lat, j_head = float(matrix[0, 0]), float(matrix[0, 1])
    if abs(j_head) <= singular_ratio * max(1.0, abs(j_lat)):
        raise ProposalRefusal("conjugate_point", "No heading correction exists at a conjugate point (j_head(L) = 0)")
    value = float(-j_lat * lateral / j_head)
    inputs = {"surface": surface.describe(), "u0": [float(x) for x in np.asarray(u0)], "heading": float(heading),
              "length": float(length), "lateral": float(lateral), "steps": int(steps),
              "transfer": [[float(x) for x in row] for row in matrix]}
    return ControlProposal(channel="heading_offset", value=value, unit="rad",
                           produced_by="ciw.lab.implementation_targets T154 (Jacobi transfer, RK4)",
                           inputs_sha256=canonical_sha256(inputs),
                           rationale="First-order Jacobi cancellation of a lateral offset at the stated arclength")

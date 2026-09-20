# Proposed licensing policy — pending review

This is a proposal for future original first-party workbench development.
It is not an adopted license transition or a statement about the repository's
current visibility. The accompanying `LICENSE` is a review draft; the
`LicenseRef-Notation-Systems-Proprietary` package expression names that custom
draft rather than an open-source license.

## Scope and existing rights

The proposed proprietary scope covers original workbench execution,
persistence, synchronization, and presentation code only to the extent the
relevant rights holders authorize those terms. No repository-wide license
was found in the audited base revision
`84e1cae46efc28b8dac0a8111f78a756ba696b7d`. Absence of a license is not evidence
that one person owns every contribution or that no separate agreement exists.

Earlier permissions and licenses remain available for the versions and
material to which they apply. A future notice must not be applied retroactively
to erase those grants, remove attribution, or override file-specific licenses.
Public visibility and GitHub's platform permissions are separate from a
software license. Restricting future repository access does not retrieve
existing copies or make previously public material confidential.

## Independently licensed components

The workbench does not own or relicense the scientific domain runtimes. The
following license inventory is tied to the current pinned revisions; a pin
change requires another review of its own license and notices.

| Runtime | Pinned revision | Existing license |
| --- | --- | --- |
| RCI | `f863bdd69d49224e0cdc871943bbb052e5b0a975` | MIT |
| FSRT | `09a756dd9cdd3a9bb6cb14b5cd498f6259937ac2` | MIT |
| JSPT | `d910f5a1d7f6dd5f2dd87dfca66990f714f97b18` | MIT |
| GTE | `e55b8be2b3ba05f7e6c6a31807c77b3e42700f07` | GNU Affero General Public License v3 |
| PLSR | `19ea6967060166ba09db6cd4563bd87bd6b3d196` | MIT |

Preserve upstream licenses and copyright notices when redistributing any of
these components. GTE's AGPL terms require specific review of the intended
combination, modifications, distribution, and network use before any such
deployment is described as wholly proprietary. A subprocess boundary alone
does not establish a legal exception or resolve that review.

NumPy, websockets, setuptools, pytest, Python and container base components,
and the optional Godot host retain their own licenses. This inventory is not
a complete bill of materials for a deployed image or workstation. Any
distribution must inventory the actual included components and preserve their
required notices and other obligations.

## Adoption requirements

Before replacing this proposal with effective terms, obtain legal review and
confirmation of the relevant ownership, assignments, contributor agreements,
and previously granted rights. Commit authorship or an automated contributor
name does not establish legal title.

The adopting change must state the covered revision and effective date,
identify the authorized rights holders, retain historical grants and
third-party exceptions, and confirm the proposed use of copyleft components.
Package metadata must match the terms actually adopted. Access controls,
publication settings, and any confidential-data handling are separate actions;
this policy does not assert they have been changed.

# Architecture Decision Records

This directory holds the Architecture Decision Records (ADRs) for the Computational Instrumentation Workbench.

Each ADR records one decision: the context that made it necessary, the decision itself, its consequences, and the alternatives that were considered. ADRs are numbered in the order they were accepted and are never edited after acceptance; a later ADR supersedes an earlier one when a decision changes.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-project-name-and-definition.md) | Project name and definition | Accepted |
| [0002](0002-foundational-runtime-positioning.md) | Position the workbench as a foundational instrumentation runtime | Accepted |
| [0003](0003-reference-implementation-arrangement.md) | Reference implementation arrangement | Accepted |

## Adding an ADR

1. Copy the section structure of the most recent ADR (Status, Date, Context, Decision, Consequences, Alternatives considered).
2. Number it with the next free four-digit number and give it a short kebab-case filename.
3. Add a row to the index above.
4. If it replaces an earlier decision, set the earlier ADR's status to `Superseded by NNNN`.

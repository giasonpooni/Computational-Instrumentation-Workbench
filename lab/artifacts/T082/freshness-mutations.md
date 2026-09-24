| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T082 | fresh.energy-replay-reuse | workspace | energy replay bundle | full | killed | Declared workload bundles must have distinct execution and reproduction occurrences | Declared workload bundles must have distinct execution and reproduction occurrences | killed |
| T082 | fresh.energy-reproduction-reuse | workspace | energy replay bundle | full | killed | Declared workload bundles must have distinct execution and reproduction occurrences | Declared workload bundles must have distinct execution and reproduction occurrences | killed |
| T082 | fresh.cross-namespace | workspace | energy replay bundle | full | killed | Identity collision between recording operations and retained workflows | Identity collision between recording operations and retained workflows | killed |
| T082 | fresh.oscillator-duplicate | workspace | oscillator execution and result | local | killed | Saved result identity mismatch or duplication | Saved result identity mismatch or duplication | killed |
| T082 | fresh.created-at-one | workspace | oscillator execution | local | killed | Execution/result created_at binding mismatch | Execution/result created_at binding mismatch | killed |
| T082 | fresh.created-at-shift | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |

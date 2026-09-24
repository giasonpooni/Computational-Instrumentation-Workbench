| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T090 | oscillator-runtime.naive | workspace | oscillator execution | none | killed | Operation record integrity mismatch | Operation record integrity mismatch | killed |
| T090 | oscillator-runtime.execution-only | workspace | oscillator execution | local | killed | Execution/result runtime binding mismatch | Execution/result runtime binding mismatch | killed |
| T090 | oscillator-runtime.empty | workspace | oscillator execution and result | local | killed | Invalid execution runtime identity | Invalid execution runtime identity | killed |
| T090 | oscillator-runtime.both | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |
| T090 | energy-runtime.naive | workspace | energy replay bundle runtime | none | killed | Energy analysis bundle identity, schema or size differs | Energy analysis bundle identity, schema or size differs | killed |
| T090 | energy-runtime.replay-only | workspace | energy replay bundle runtime | full | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T090 | energy-runtime.malformed | workspace | energy bundle runtimes | full | killed | Invalid retained analysis implementation identity | Invalid retained analysis implementation identity | killed |
| T090 | energy-runtime.all-bundles | workspace | energy bundle runtimes | full | accepted | accepted | accepted | SURVIVED |
| T090 | energy-runtime.python-version | workspace | energy bundle runtimes | full | accepted | accepted | accepted | SURVIVED |

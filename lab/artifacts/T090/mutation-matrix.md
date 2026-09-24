| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T080 | alias.result-execution | workspace | oscillator result | local | killed | Saved result identity mismatch or duplication | Saved result identity mismatch or duplication | killed |
| T080 | alias.execution-result | workspace | oscillator execution | local | killed | Execution/result execution_id binding mismatch | Execution/result execution_id binding mismatch | killed |
| T080 | alias.result-prefix | workspace | oscillator execution and result | local | killed | Invalid saved result identity | Invalid saved result identity | killed |
| T080 | alias.operation | workspace | oscillator execution and result | local | killed | Invalid saved spectrum data fields or sample count | Invalid saved spectrum data fields or sample count | killed |
| T080 | alias.swap-pairing | workspace | oscillator executions and results | local | accepted | accepted | accepted | SURVIVED |
| T080 | revision.gap | workspace | oscillator selection history | local | accepted | accepted | accepted | SURVIVED |
| T081 | energy-data.naive | workspace | energy replay result data | none | killed | Energy analysis bundle identity, schema or size differs | Energy analysis bundle identity, schema or size differs | killed |
| T081 | energy-data.reforged | workspace | energy replay result data | full | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T081 | energy-source.resealed | workspace | retained energy source log | full | accepted | accepted | accepted | SURVIVED |
| T081 | oscillator-stats.naive | workspace | oscillator result data | none | killed | Operation record integrity mismatch | Operation record integrity mismatch | killed |
| T081 | oscillator-stats.out-of-bounds | workspace | oscillator result data | local | killed | Saved statistics mean is outside its bounds | Saved statistics mean is outside its bounds | killed |
| T081 | oscillator-stats.resealed | workspace | oscillator result data | local | accepted | accepted | accepted | SURVIVED |
| T081 | oscillator-stats.impossible-moments | workspace | oscillator result data | local | accepted | accepted | accepted | SURVIVED |
| T081 | oscillator-stats.legacy | workspace | legacy oscillator result data | none | accepted | accepted | accepted | SURVIVED |
| T082 | fresh.energy-replay-reuse | workspace | energy replay bundle | full | killed | Declared workload bundles must have distinct execution and reproduction occurrences | Declared workload bundles must have distinct execution and reproduction occurrences | killed |
| T082 | fresh.energy-reproduction-reuse | workspace | energy replay bundle | full | killed | Declared workload bundles must have distinct execution and reproduction occurrences | Declared workload bundles must have distinct execution and reproduction occurrences | killed |
| T082 | fresh.cross-namespace | workspace | energy replay bundle | full | killed | Identity collision between recording operations and retained workflows | Identity collision between recording operations and retained workflows | killed |
| T082 | fresh.oscillator-duplicate | workspace | oscillator execution and result | local | killed | Saved result identity mismatch or duplication | Saved result identity mismatch or duplication | killed |
| T082 | fresh.created-at-one | workspace | oscillator execution | local | killed | Execution/result created_at binding mismatch | Execution/result created_at binding mismatch | killed |
| T082 | fresh.created-at-shift | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |
| T083 | receipt.numerical-match-false | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T083 | receipt.transplanted | workspace | energy bundle | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T083 | receipt.transplanted-resealed | workspace | energy bundle | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T083 | receipt.transplanted-full | workspace | energy bundle | full | accepted | accepted | accepted | SURVIVED |
| T083 | receipt.fabricated | workspace | energy bundle | full | accepted | accepted | accepted | SURVIVED |
| T083 | receipt.deleted | workspace | energy replay bundle | none | accepted | accepted | accepted | SURVIVED |
| T084 | receipt-source.naive | workspace | energy replay receipt | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T084 | receipt-source.replay-id | workspace | energy replay receipt | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T084 | receipt-source.subject-rebound | workspace | energy replay receipt | local | killed | Replay source must already belong to this workbench | Replay source must already belong to this workbench | killed |
| T084 | receipt-source.self | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T084 | receipt-source.other-source | workspace | energy replay receipt | local | killed | Replay source must already belong to this workbench | Replay source must already belong to this workbench | killed |
| T084 | receipt-source.sibling-execution | workspace | energy replay receipt | local | accepted | accepted | accepted | SURVIVED |
| T085 | receipt-replayed.naive | workspace | energy replay receipt | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T085 | receipt-replayed.replay-id | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T085 | receipt-replayed.source | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T085 | receipt-replayed.sibling | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T085 | receipt-replayed.reidentified-bundle | workspace | energy replay bundle | full | accepted | accepted | accepted | SURVIVED |
| T086 | receipt-subject.naive | workspace | energy replay receipt verification | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T086 | receipt-subject.resealed | workspace | energy replay receipt verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T086 | bundle-subject.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T086 | oscillator-verification.resealed | workspace | oscillator result | local | killed | Protocol v1 saved results must remain not_verified with verification_id null | Protocol v1 saved results must remain not_verified with verification_id null | killed |
| T086 | oscillator-subject.injected | workspace | oscillator result | local | accepted | accepted | accepted | SURVIVED |
| T087 | receipt-method.naive | workspace | energy replay receipt verification | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T087 | receipt-method.resealed | workspace | energy replay receipt verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T087 | bundle-method.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T087 | oscillator-method.injected | workspace | oscillator result | local | accepted | accepted | accepted | SURVIVED |
| T088 | receipt-independent.naive | workspace | energy replay receipt verification | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T088 | receipt-independent.resealed | workspace | energy replay receipt verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T088 | bundle-independent.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T088 | oscillator-independent.injected | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |
| T089 | receipt-admission.naive | workspace | energy replay receipt | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T089 | receipt-admission.resealed | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T089 | bundle-authority.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T089 | result-authority.reforged | workspace | energy result authority | full | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T089 | oscillator-status.resealed | workspace | oscillator result | local | killed | Protocol v1 saved results must remain not_verified with verification_id null | Protocol v1 saved results must remain not_verified with verification_id null | killed |
| T089 | oscillator-admission.injected | workspace | oscillator result | local | accepted | accepted | accepted | SURVIVED |
| T090 | oscillator-runtime.naive | workspace | oscillator execution | none | killed | Operation record integrity mismatch | Operation record integrity mismatch | killed |
| T090 | oscillator-runtime.execution-only | workspace | oscillator execution | local | killed | Execution/result runtime binding mismatch | Execution/result runtime binding mismatch | killed |
| T090 | oscillator-runtime.empty | workspace | oscillator execution and result | local | killed | Invalid execution runtime identity | Invalid execution runtime identity | killed |
| T090 | oscillator-runtime.both | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |
| T090 | energy-runtime.naive | workspace | energy replay bundle runtime | none | killed | Energy analysis bundle identity, schema or size differs | Energy analysis bundle identity, schema or size differs | killed |
| T090 | energy-runtime.replay-only | workspace | energy replay bundle runtime | full | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T090 | energy-runtime.malformed | workspace | energy bundle runtimes | full | killed | Invalid retained analysis implementation identity | Invalid retained analysis implementation identity | killed |
| T090 | energy-runtime.all-bundles | workspace | energy bundle runtimes | full | accepted | accepted | accepted | SURVIVED |
| T090 | energy-runtime.python-version | workspace | energy bundle runtimes | full | accepted | accepted | accepted | SURVIVED |
| T086 | esm.bundle-subject | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |
| T086 | exchange.verification-subject | validator | exchange verification identity (pure validator) | none | killed | verification_id does not match the artifact content | verification_id does not match the artifact content | killed |
| T088 | esm.inspection-independent | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | Native ESM inspection binding or scope mismatch | Native ESM inspection binding or scope mismatch | killed |
| T088 | esm.candidate-independent | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |
| T088 | exchange.verification-independent | validator | exchange verification identity (pure validator) | local | accepted | accepted:content_recomputed_not_authenticated | accepted:content_recomputed_not_authenticated | accepted (by design) |
| T089 | esm.canonical-admission | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM may retain candidate evidence only | ESM may retain candidate evidence only | killed |
| T089 | esm.candidate-admitted | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |

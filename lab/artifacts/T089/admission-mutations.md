| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T089 | receipt-admission.naive | workspace | energy replay receipt | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T089 | receipt-admission.resealed | workspace | energy replay receipt | local | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T089 | bundle-authority.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T089 | result-authority.reforged | workspace | energy result authority | full | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T089 | oscillator-status.resealed | workspace | oscillator result | local | killed | Protocol v1 saved results must remain not_verified with verification_id null | Protocol v1 saved results must remain not_verified with verification_id null | killed |
| T089 | oscillator-admission.injected | workspace | oscillator result | local | accepted | accepted | accepted | SURVIVED |
| T089 | esm.canonical-admission | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM may retain candidate evidence only | ESM may retain candidate evidence only | killed |
| T089 | esm.candidate-admitted | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |

| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T088 | receipt-independent.naive | workspace | energy replay receipt verification | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T088 | receipt-independent.resealed | workspace | energy replay receipt verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T088 | bundle-independent.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T088 | oscillator-independent.injected | workspace | oscillator execution and result | local | accepted | accepted | accepted | SURVIVED |
| T088 | esm.inspection-independent | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | Native ESM inspection binding or scope mismatch | Native ESM inspection binding or scope mismatch | killed |
| T088 | esm.candidate-independent | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |
| T088 | exchange.verification-independent | validator | exchange verification identity (pure validator) | local | accepted | accepted:content_recomputed_not_authenticated | accepted:content_recomputed_not_authenticated | accepted (by design) |

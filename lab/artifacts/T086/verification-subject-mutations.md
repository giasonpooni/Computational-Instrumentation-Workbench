| Task | Mutant | Kind | Target | Recompute | Predicted | Pinned message | Observed | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T086 | receipt-subject.naive | workspace | energy replay receipt verification | none | killed | Invalid retained energy replay receipt | Invalid retained energy replay receipt | killed |
| T086 | receipt-subject.resealed | workspace | energy replay receipt verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T086 | bundle-subject.resealed | workspace | energy bundle verification | local | killed | Retained energy analysis binding differs | Retained energy analysis binding differs | killed |
| T086 | oscillator-verification.resealed | workspace | oscillator result | local | killed | Protocol v1 saved results must remain not_verified with verification_id null | Protocol v1 saved results must remain not_verified with verification_id null | killed |
| T086 | esm.bundle-subject | validator | ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator) | none | killed | ESM candidate does not bind the selected native bundle | ESM candidate does not bind the selected native bundle | killed |
| T086 | exchange.verification-subject | validator | exchange verification identity (pure validator) | none | killed | verification_id does not match the artifact content | verification_id does not match the artifact content | killed |

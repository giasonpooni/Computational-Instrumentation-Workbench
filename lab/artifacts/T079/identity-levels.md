| Identity | Level | Distinct values over 8 variants |
| --- | --- | --- |
| workbench evidence_id (bundle artifact_ref and sha256 are copies of it) | byte | 8 |
| workbench source_id | byte + label (one shared label here) | 8 |
| bundle experiment_digest | canonical content | 1 |
| log_digest (sealed inside the log) | canonical content | 1 |
| numerical_result_id | canonical content of analysed data, which includes log_digest | 1 |
| recording evidence_id (oscillator) | canonical content of scientific fields | n/a |
| bundle_digest / result_id / verification_id / replay_id | canonical content including fresh occurrences | n/a |

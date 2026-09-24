# Uncertainty validation: finite-sample consistency of declared covariances

`ciw.uncertainty-validation.v1` is a provider-free shared-session operation. It
takes retained samples of a declared reference value, an estimate and the
estimate's covariance, optionally with innovations and their covariances, and
reports whether the declared covariances are consistent with the observed
errors at a declared confidence. It separates a covariance that is too small
from one that is too large instead of merging both into one verdict, and it
names every statistic that falls outside its band.

The operation is a statistical check of declared uncertainty, not a physical
validation. Reference values are declared, either a synthetic fixture or a
held-out reference the operator supplies, and their own uncertainty is not
modelled. No state is admitted and no equipment is authorized.

## Source

Kind `uncertainty-validation`, schema `ciw.uncertainty-validation-source.v1`:

| Field | Meaning |
| --- | --- |
| `configuration` | The fixed read-only policy; any other value is refused |
| `request.confidence` | Two-sided band probability between 0.5 and 0.999 |
| `request.state_names`, `state_units`, `frame`, `time_basis` | Ordered state declaration, at most sixteen states |
| `request.reference_origin` | `synthetic_fixture` or `declared_reference` |
| `request.cross_sample_dependence` | `declared_independent` or `unknown` |
| `request.samples[]` | `time`, `reference`, `estimate`, `covariance`; two to 4096 entries with strictly increasing times |
| `request.innovations[]` | Optional `time`, `innovation`, `covariance`, with `measurement_names` and `measurement_units` |

Every covariance must pass the workbench's admission rule in correlation
coordinates and additionally be positive definite with a bounded condition
number, because each sample is normalized by its own inverse covariance. A
singular, asymmetric or nonfinite matrix is refused before retention; nothing
is repaired.

## Statistics and bands

| Statistic | Definition | Band |
| --- | --- | --- |
| NEES | mean over samples of `e_k^T P_k^-1 e_k` with `e_k = estimate - reference` | two-sided chi-square interval with `K * n` degrees of freedom, divided by `K` |
| NIS | the same over innovations and their covariances | chi-square with `K * m` degrees of freedom, divided by `K` |
| Coverage | per component, the fraction of errors inside `z * sigma_k` where `z` is the two-sided normal bound at the confidence | Clopper-Pearson interval for the observed fraction; consistent when it contains the nominal probability |
| Bias | per component, the mean error over its standard error under the declared covariances | `|z| <= z_confidence` |

The chi-square and binomial quantiles come from `src/ciw/consistency_math.py`,
pure-Python regularized incomplete gamma and beta functions checked against
reference values in `tests/test_consistency_math.py`. No SciPy is required.

Statuses are explicit: a NEES or NIS mean above its band is
`covariance_too_small`, below it `covariance_too_large`; coverage is
`under_covering` or `over_covering`; bias is `biased_at_declared_uncertainty`.
The verdict is `consistent` only when every applicable statistic is, and lists
each failure otherwise.

## Statistical authority

The bands assume independent samples. With `cross_sample_dependence: unknown`
the same numbers are computed and reported, but the verdict's authority is
`diagnostic_only` and the statistical scope names the unknown dependence, in
the same way the residual monitor withholds alarm authority. Declaring
independence is the operator's assertion; the operation cannot establish it.

Coverage is weak for small sample counts: with 64 samples a complete coverage
of 64 out of 64 still contains the nominal 0.95 in its interval, so an inflated
covariance is detected by NEES and NIS long before coverage flags it. The
shipped `covariance-too-large` fixture shows exactly that.

## Fixtures and commands

`examples/uncertainty-validation/generate.py` writes four sources from one
seeded draw over a constant-velocity trajectory with a shared scalar position
innovation; `tests/test_uncertainty_validation.py` checks that the committed
files are the generator's output.

| Fixture | Declared covariance | Expected outcome |
| --- | --- | --- |
| `consistent.json` | the true covariance | consistent, statistical authority |
| `covariance-too-small.json` | one quarter of the truth | NEES, NIS and coverage flag it |
| `covariance-too-large.json` | four times the truth | NEES and NIS flag it; coverage cannot at 64 samples |
| `unknown-dependence.json` | the true covariance, dependence unknown | consistent numbers, diagnostic authority |

Register a fixture as a `uncertainty-validation` source through `source.add`
and execute `ciw.uncertainty-validation.v1` with its `source_id`; no repository
binding is needed. The retained bundle, `result.get`, `execution.list` and the
`experiment.inspect` projection expose the statistics, and `bundle.replay`
records a fresh occurrence whose numerical identity must match. Reopening a
workspace validates the retained statistics against a fresh reference
computation with a binary64 tolerance on the numbers and exact equality on
every status, name and scope.

## Limitations

- Reference values are treated as exact; a reference with its own uncertainty
  must be composed into the error covariance before it enters this operation.
- Independence between samples is declared, never inferred.
- The chi-square bands are exact for Gaussian errors with correct covariances;
  they are diagnostics, not proofs, for other error laws.
- No physical model, calibration or plant claim follows from a consistent
  verdict. The operation does not admit state or authorize any action.

# Variational Free-Energy Sensor Fusion on Curved Surfaces

The `variational-free-energy` workflow, operation
`ciw.variational-free-energy.v1`, estimates two initial path errors from two
synthetic measurement sources on a declared curved surface. It compares an
iterated Gaussian approximation with the analytical Gaussian posterior, then
checks estimates and held-out predictions against retained simulated truth.
Its purpose is to make optimization, uncertainty, model assumptions and
prediction errors inspectable in one workbench session.

This is a bounded linear-Gaussian numerical experiment. The truth labels are
synthetic; reported performance is **simulation performance**, not measured
physical performance. Sensor calibration, surveyed geometry, physical state
admission and physical stability remain unestablished. The workbench does not
infer that lowering free energy makes an assumed model physically correct.

The variational objective has the evidence-bound interpretation described by
[Friston (2010)](https://www.nature.com/articles/nrn2787). The connection between
Gaussian filtering and free-energy optimization motivates the comparison with
an exact posterior; see [Baltieri and Isomura
(2021)](https://arxiv.org/abs/2111.10530). This bench implements the finite
declared estimation problem below; it does not implement the broader biological
or active-inference claims discussed in those papers.

## Coordinates, observations and retained evidence

The physical latent vector is

\[
x=(\text{initial lateral error in m},\;
   \text{initial heading error in radian})^T.
\]

Positive declared scales form \(S=\operatorname{diag}(s_1,s_2)\), with
\(x=Sz\). Inference and optimization use the dimensionless coordinates \(z\).
Means transform as \(m_x=Sm_z\), and covariances as
\(\Sigma_x=S\Sigma_zS^T\). These scales are part of the experiment identity;
changing them changes the numerical meaning of a fixed gradient step.

Native CSG supplies a Jacobi transfer \(\Phi_i\) at each declared arclength.
Each source has a declared scalar observation row \(H_i\), so its physical
linear observation row is \(G_i^{\rm phys}=H_i\Phi_i\). If observation scales
form the positive diagonal matrix \(D\), the normalized training problem uses

\[
y=D^{-1}y_{\rm phys},\qquad G=D^{-1}G^{\rm phys}S,\qquad
R=D^{-1}R_{\rm phys}D^{-T}.
\]

The implemented profile fixes the first source to lateral displacement and the
second to heading, so their rows select the corresponding transfer component.

The two sources each provide one training and one held-out scalar observation.
Each observation group has its own full \(2\times2\) noise covariance. Noise
cross-covariance between the training and held-out groups is explicitly zero
in this profile; noise is also declared independent of the latent draw.
Within-group correlation must be retained, even when the rows have different
source names. Held-out observations still share uncertainty through their
common latent state.

The source retains the assumed model separately from the synthetic generator:
curvature, prior, observation definitions, noise covariances, bias settings,
solver settings, actual latent draws and actual noise realizations. A random
seed alone is not the evidence. Up to 128 prior-predictive trials are retained.
Truth and held-out observations are used for evaluation; neither enters the
training posterior or selects the solver's stopping point.

The generator's observation equations are checked against the retained truth,
bias and noise arrays. The statistical claim that those arrays are independent
Gaussian draws remains an operator declaration; neither the seed nor an
algebraic consistency check authenticates that sampling law.

The Gaussian reference is exact for the **retained linear observation
matrices**. Those matrices come from CSG's numerical transfer calculation;
Gaussian exactness does not certify the continuous-surface approximation or
the declared curvature. Gaussian draws also do not establish that a physical
trajectory satisfies CSG's separately retained linearization assumptions.

## Exact posterior and normalized free energy

All quantities in this section are normalized. Let

\[
z\sim\mathcal N(m_0,P_0),\qquad
y\mid z\sim\mathcal N(Gz+b,R),\qquad
q(z)=\mathcal N(m,\Sigma),
\]

with positive-definite \(P_0,R,\Sigma\), latent dimension \(d=2\), and training
observation count \(p=2\). The likelihood uses the **assumed** bias \(b\);
unmodelled generator bias remains an evaluation mismatch.

The posterior precision, mean and covariance are

\[
\Lambda=P_0^{-1}+G^TR^{-1}G,\qquad
\eta=P_0^{-1}m_0+G^TR^{-1}(y-b),\qquad
m_\star=\Lambda^{-1}\eta,\quad \Sigma_\star=\Lambda^{-1}.
\]

The reference calculation instead conditions in observation space, using
\(C=GP_0G^T+R\), \(K=P_0G^TC^{-1}\), and

\[
m_\star=m_0+K(y-b-Gm_0),\qquad
\Sigma_\star=(I-KG)P_0(I-KG)^T+KRK^T.
\]

The second expression is the Joseph covariance form. Implementations use
linear solves and positive-definite factorizations; the inverse notation
defines the mathematics. A proper prior can make the posterior well-defined
even when the two observation rows alone do not identify both latent
coordinates. Posterior existence therefore does not establish observability.

With natural logarithms, variational free energy is the negative ELBO,
\(F=E_q[\log q(z)-\log p(z,y)]\). Its complete normalization is

\[
\begin{aligned}
2F={}&(m-m_0)^TP_0^{-1}(m-m_0)
 +(y-b-Gm)^TR^{-1}(y-b-Gm)\\
 &+\operatorname{tr}(\Lambda\Sigma)
 +\log\det P_0+\log\det R-\log\det\Sigma
 +p\log(2\pi)-d.
\end{aligned}
\]

The independently calculated evidence is

\[
\log p(y)=-\tfrac12\{(y-b-Gm_0)^TC^{-1}(y-b-Gm_0)
 +\log\det C+p\log(2\pi)\}.
\]

Thus \(F+\log p(y)\) equals
\(\mathrm{KL}(q\Vert p(z\mid y))\), with the explicit Gaussian check

\[
\mathrm{KL}=\tfrac12\{
\operatorname{tr}(\Lambda\Sigma)
 +(m-m_\star)^T\Lambda(m-m_\star)-d
 -\log\det\Lambda-\log\det\Sigma\}.
\]

The evidence/ELBO identity is the standard variational-inference decomposition;
see [Blei, Kucukelbir and McAuliffe
(2017)](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf).
A small KL gap means agreement with this assumed posterior. It can coexist
with a wrong curvature, biased sensors or incorrect noise assumptions.

Density values depend on coordinates. Returning to physical observations gives
\(\log p(y_{\rm phys})=\log p(y)-\log\det D\) and
\(F_{\rm phys}=F+\log\det D\). The latent-coordinate Jacobians cancel between
\(q\) and the prior. The KL gap is unchanged. Reports use the declared
normalized coordinates and nats; absolute free energies from different units
or different observations are not interchangeable accuracy scores.

## Iteration and its discrete stability check

The exact derivatives are

\[
\nabla_m F=\Lambda m-\eta,\qquad
\nabla_\Sigma F=\tfrac12(\Lambda-\Sigma^{-1}).
\]

Mean iteration is ordinary gradient descent:

\[
m_{k+1}=m_k-\alpha(\Lambda m_k-\eta).
\]

Writing \(e_k=m_k-m_\star\) gives the actual discrete error map
\(e_{k+1}=Ae_k\), where \(A=I-\alpha\Lambda\). The mean-only quadratic
contribution is \(V(e)=\tfrac12e^T\Lambda e\). PLSR checks this matrix with
\(P=\Lambda/2\) in the declared normalized coordinates, including the
discrete expression \(A^TPA-P\). This is a numerical matrix check for the
optimizer; it is not a formal proof or a certificate of physical path stability.
For this positive-definite quadratic, \(0<\alpha<2/\lambda_{\max}(\Lambda)\)
is the exact-arithmetic mean convergence condition.

Covariance is also iterated, through its precision \(Q_k=\Sigma_k^{-1}\):

\[
Q_{k+1}=(1-\beta)Q_k+\beta\Lambda,\qquad \Sigma_{k+1}=Q_{k+1}^{-1},
\qquad 0<\beta<1.
\]

This is precision relaxation, not an additive Euclidean covariance-gradient
step. It preserves positive definiteness in exact arithmetic and approaches
the posterior covariance. The PLSR mean check does not certify this separate
update. A converged mean with the wrong covariance still has a nonzero full
Gaussian KL gap. Retain both matrix evolution and the full objective rather
than reporting mean convergence as distributional convergence.

The status `converged` means the declared mean-gradient and relative-precision
residual tolerances were met. It is not a certificate that KL is below a
particular bound. With an anisotropic precision matrix, a small normwise
relative residual can still hide appreciable error in a weak direction.
Inspect the separately retained full KL gap and covariance comparison.

## Held-out predictions and empirical coverage

For the held-out group, let \(G_h,b_h,R_h\) be the assumed normalized model.
Given a Gaussian estimate \((m,\Sigma)\), the noisy-observation prediction is

\[
E[y_h\mid y]=G_hm+b_h,\qquad
\operatorname{Cov}(y_h\mid y)=G_h\Sigma G_h^T+R_h.
\]

The term \(G_h\Sigma G_h^T\) alone describes the latent signal; it is too
narrow for a noisy-observation interval. These formulas rely on the declared
zero cross-group noise covariance. A profile with correlated training and
held-out noise would require joint Gaussian conditioning, including its
additional mean correction.

A single trial reports error and interval-containment outcomes against its
retained truth. Empirical coverage summarizes the retained prior-predictive
ensemble using the exact reference posterior for each trial; the iterative
trajectory is evaluated for the selected representative trial. Matched prior
and likelihood draws provide a useful computational
calibration check; they do not establish sensor calibration. This distinction
is consistent with the simulation-based checking framework of [Talts et al.
(2018)](https://arxiv.org/abs/1804.06788).

Marginal intervals and a joint two-coordinate ellipse are different sets.
For a nominal 95% marginal interval use the corresponding normal quantile;
for a two-dimensional Gaussian ellipse, the squared Mahalanobis threshold is
\(-2\log(0.05)\). State and held-out coverage have separate denominators.
The two coordinates in one trial are not two independent experiments.

Report the actual covered count and trial count. At 128 independent trials,
nominal 95% coverage has a binomial standard error of about 1.93 percentage
points; an observed fraction is not an exact probability. A confidence interval
for that fraction describes Monte Carlo uncertainty, not systematic model
error. Fixed-truth repeated-noise trials have a different coverage meaning
from prior-predictive trials. The mismatch fixtures deliberately violate the
matched-model premise and are interpreted as stress experiments.

## Fixtures and their interpretation

The live client is `examples/variational-free-energy/run.py`; the declared
source files are in the same directory.

| Source file | Purpose | Interpretation |
| --- | --- | --- |
| `baseline.json` | Matched geometry, bias and noise model | Compare exact conditioning, native inference, iterative mean/covariance and held-out prediction. |
| `correlated-noise.json` | Retain nonzero within-group source correlation in the assumed model | Condition on the complete covariance jointly; source labels do not imply independence. |
| `ignored-correlation.json` | Generate correlated noise while fitting the declared simplified covariance | Check this fixture's uncertainty and held-out effects; omitted correlation does not distort every direction in the same way. |
| `sensor-bias.json` | Retain a generator bias absent from the assumed likelihood | Optimization can converge while truth error and predictive error remain large. |
| `wrong-curvature.json` | Generate observations with a different curvature from the fitted transfer | Posterior agreement validates inference for the fitted matrix, not that matrix's geometry. |
| `unstable-step.json` | Use a mean step beyond its discrete convergence bound | Retain the failed numerical condition and finite iteration evidence; do not relabel it as successful convergence. |

The unstable fixture must excite an unstable eigenmode. A step outside the
global bound need not visibly diverge from a specially chosen initial condition
that has no component in that mode. No fixture supports a universal claim that
one mismatch always increases every error measure.

The retained fixtures give these native results (128 trials, nominal 95% joint
sets). Coverage uses exact-reference posteriors in every row; the KL gap and
optimizer status describe the representative iterative run. Consequently the
unstable-step row retains the baseline reference coverage while its own
iterative estimate fails:

| Case | Iteration status | Final KL gap, approximately | Latent joint count | Held-out joint count |
| --- | --- | --- | --- | --- |
| Baseline | Converged | `3.5e-17` | 122/128 | 121/128 |
| Correlation retained | Converged | `4.3e-19` | 121/128 | 124/128 |
| Correlation ignored | Converged | `3.5e-17` | 118/128 | 116/128 |
| Sensor bias | Converged | `3.5e-17` | 3/128 | 35/128 |
| Wrong curvature | Converged | `9.9e-19` | 41/128 | 29/128 |
| Excessive step | Iteration limit | `8.1e5` | 122/128 | 121/128 |

Baseline PLSR matrix margin is about `3.319`; the excessive-step margin is
about `-14.877`. These refer to the fixed normalized mean iteration. The saved
records include full traces, actual interval counts and Wilson intervals;
near-zero KL values are subject to the declared floating-point error budget.

## Execution, persistence and acceptance

Use Python 3.12 or newer, NumPy 2.4.3 and the `bench-models` extra
(`jsonschema==4.26.0`). The native roles are CSG at `bbc535a`, GSIE at `5241eee`
and PLSR at `19ea696`; retained runtime records bind their full immutable
revisions, source trees, interpreter and dependencies. CSG owns the transfer,
GSIE provides the native Gaussian comparison, PLSR checks the actual discrete
mean map, and CIW owns this small experimental free-energy kernel and its
composition.

Provision the exact clean native checkouts in `csg/`, `gsie/` and `plsr/`
under a trusted stack directory, then start the workbench:

```sh
python -m pip install -e '.[bench-models]'
ciw serve --free-energy-stack-root /trusted/free-energy
```

In a second terminal, run one case or all six with fresh replay in the shared
session:

```sh
python examples/variational-free-energy/run.py --url ws://127.0.0.1:8765 --case baseline
python examples/variational-free-energy/run.py --url ws://127.0.0.1:8765 --case all
```

The installed-wheel gate provisions all three pinned providers and requires
the native tests to finish without skips:

```sh
python scripts/check_free_energy.py --output-dir results/free-energy-gate
```

Repository paths are host bindings, not instructions recovered from a saved
artifact. Preserve source bytes, normalization, model/generator separation,
native inputs and responses, iteration history, analytical comparisons and
evaluation realizations. Fresh replay creates new occurrence identities while
preserving matching numerical identities. Offline inspection checks retained
structure and bindings without executing providers; it does not authenticate
the provenance of a fabricated artifact. Replay from the same implementation
is a reproduction check, not an independent scientific validation.

Acceptance checks cover the following distinct claims:

- Information-space posterior parameters agree with observation-space
  conditioning and the native Gaussian result, including correlated noise.
- Directional finite differences agree with the mean and covariance
  derivatives; symmetric off-diagonal covariance perturbations include both
  matrix entries in the Frobenius inner product.
- Full free energy, log evidence and analytic Gaussian KL agree within a
  declared floating-point budget, including their normalization constants.
  Near-zero cancellation is reported within that budget, not hidden as an
  arbitrary positive objective.
- Stable mean iteration and positive-definite precision relaxation approach
  the exact posterior; the unstable fixture retains a failed stability check.
- Changing only held-out observations or truth labels cannot change the fitted
  posterior, iterative path or PLSR matrix. Changing training data can.
- Held-out predictions include observation noise; empirical counts and
  denominators reproduce from retained trials, without an automatic physical
  calibration claim.
- Invalid shapes, nonfinite values, non-positive scales, invalid covariance,
  unsupported noise cross-covariance and out-of-budget work are refused.
  Resealing a changed assumption cannot bypass input/result bindings.
- Installed-wheel execution, shared-session inspection, offline restore and
  fresh replay exercise the same retained operation and native runtime pins.

Future measurement selection needs a separate expected information objective,
such as expected posterior entropy reduction over explicitly costed candidate
measurements. It must respect shared noise and acquisition dependencies. It is
not implemented by minimizing the current observation's variational free
energy, and this workflow does not automatically choose or acquire a next
measurement.

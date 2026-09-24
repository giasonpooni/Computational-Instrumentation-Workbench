# Disturbance-aware (ISS) research branch for PLSR-style certificates

Status: research branch; not a runtime feature and not part of runtime-status-v1.

## Definition

x' = f(x, w) is input-to-state stable if |x(t)| <= beta(|x(0)|, t) + gamma(sup |w|) for a class-KL beta and a class-K gamma.

## ISS-Lyapunov function

V with a1(|x|) <= V(x) <= a2(|x|) and V' <= -a3(|x|) + sigma(|w|); for linear x' = A x + B w and V = x^T P x: V' <= -c V + 2 sqrt(V) ||P^(1/2) B|| |w| with c = min eig(P^-1 Q).

## Data a host must supply

- the disturbance input matrix B (or a bound on how w enters), in the declared state units
- a disturbance bound w_bar with units, provenance and the procedure that established it
- whether w is sampled, held or continuous, and the sampling period if the model is discrete
- the declared model uncertainty that w is meant to cover, separately from exogenous disturbance
- the certificate P and Q used, with the Lyapunov-equation residual

## What PLSR must not claim

- an ISS gain, ultimate bound or invariant set: runtime-status-v1 evaluates the undisturbed decrease only
- that a CERTIFIED_WITH_MARGIN sample is robust to disturbances or unmodelled dynamics
- that a declared required_margin corresponds to any physical disturbance level
- that a disturbance bound supplied by a host is true of a physical plant
- any new status code for disturbance robustness (a new code is a new runtime-status version)

## Synthetic numerical illustration

synthetic: A = [[0, 1], [-4, -1.2]], B = [0, 1]^T, w_bar = 0.5, Q = I; exact ZOH simulation with h = 0.005 s over 30 s from x(0) = 0 for four bounded disturbance classes, compared with the quadratic ISS bound and with the sharp reachable-set supremum; plus the scalar system x' = -x + w, where the bound is approached as t grows.

Analytic bound: sqrt(V) <= 1.61832 (c = 0.445949, beta = 0.360844); |x| <= 2.26221.

| Disturbance | sup sqrt(V) | sup sqrt(V) / bound | sup abs(x) |
| --- | --- | --- | --- |
| constant | 0.256983 | 0.1588 | 0.189173 |
| worst-case switching | 0.409331 | 0.2529 | 0.530671 |
| resonant sinusoid | 0.324048 | 0.2002 | 0.416665 |
| random held | 0.186461 | 0.1152 | 0.207311 |

Sharp reference: the largest sqrt(V) reachable from x(0) = 0 under any |w| <= w_bar is 0.411754 (0.2544 of the ISS bound); the quadratic ISS bound is conservative by the Cauchy-Schwarz step.

Scalar system: exact sup |x| = 0.5, bound = 0.5.

## Open research questions

- a resolution-aware float64 evaluation of the ISS inequality analogous to decrease_resolution
- sampled-data ISS: inter-sample behaviour of a discrete certificate under held disturbances
- how a host would evidence w_bar from acquired data (hardware-gated; not available here)

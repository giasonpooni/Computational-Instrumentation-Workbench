# Curvature signal versus integrator error (L = 2)

Ratio = signal / abs(computed - true deviation); resolved when >= 10 for every finer step; (r) marks a roundoff-limited error (below 64 ulp(L), or below 1000 ulp(L) and not decreasing at the next step).

| surface | K | signal abs(j_head(L) - L) | euler: ratio at N = 4 / 16 / 128; resolved from N | midpoint: ratio at N = 4 / 16 / 128; resolved from N | rk4: ratio at N = 4 / 16 / 128; resolved from N |
| --- | --- | --- | --- | --- | --- |
| plane | 0 | 0 | - / - / -; no signal | - / - / -; no signal | - / - / -; no signal |
| cylinder | 0 | 0 | - / - / -; no signal | - / - / -; no signal | - / - / -; no signal |
| sphere R=1 | 1 | 1.091 | 1.85 / 8.73 / 75.8; 32 | 157 / 632 / 3.31e+04; 4 | 8.67e+04 / 8.4e+05 / 2.72e+09; 4 |
| sphere R=10 | 0.01 | 0.01331 | 1.6 / 5.59 / 43.1; 32 | 16.2 / 260 / 1.67e+04; 4 | 1.29e+05 / 3.33e+07 / 1.36e+11; 4 |
| sphere R=100 | 0.0001 | 0.0001333 | 1.6 / 5.57 / 42.9; 32 | 16 / 256 / 1.64e+04; 4 | 1.28e+07 / 3.26e+09 / 3e+11 (r); 4 |
| sphere R=1e4 | 1e-08 | 1.333e-08 | 1.6 / 5.57 / 42.9; 32 | 16 / 256 / 1.64e+04; 4 | 6.79e+08 (r) / 6.59e+07 (r) / 1.54e+07 (r); 4 |
| sphere R=1e7 | 1e-14 | 1.333e-14 | 1.62 (r) / 5.98 (r) / 8.52 (r); never | 14.8 (r) / 63.1 (r) / 14.8 (r); 4 | 63.1 (r) / 63.1 (r) / 9.93 (r); never |
| sphere R=1e8 | 1e-16 | 1.333e-16 | 1 (r) / 1 (r) / 1 (r); never | 1 (r) / 1 (r) / 1 (r); never | 1 (r) / 1 (r) / 1 (r); never |
| torus R=2 | varies | 0.25 | 1.76 / 6.28 / 48.1; 32 | 40.5 / 750 / 5.03e+04; 4 | 2.64e+04 / 2.15e+06 / 7.27e+09; 4 |
| torus R=64 equator | 0.0154 | 0.02045 | 1.6 / 5.6 / 43.2; 32 | 16.3 / 263 / 1.68e+04; 4 | 8.46e+04 / 2.18e+07 / 8.98e+10; 4 |
| saddle c=1 | varies | 0.5139 | 1.88 / 7.1 / 55.6; 32 | 102 / 1.25e+03 / 7.85e+04; 4 | 4.94e+03 / 4.13e+06 / 1.38e+10; 4 |
| bump h=0.5 | varies | 0.1039 | 1.54 / 5.55 / 43.2; 32 | 23.7 / 517 / 3.77e+04; 4 | 1.24e+04 / 2.83e+06 / 1.11e+10; 4 |
| bump h=0.05 | varies | 0.001144 | 1.61 / 6.19 / 49.3; 32 | 47.7 / 3.15e+03 / 9.89e+05; 4 | 1.48e+03 / 4.46e+05 / 1.83e+09; 4 |
| hyperbolic k=1 | -1 | 1.627 | 1.44 / 4.13 / 29.2; 64 | 7.51 / 91 / 5.37e+03; 8 | 614 / 1.17e+05 / 4.41e+08; 4 |

# Mathematical model exploration

CIW now exposes a small educational layer around the retained Julia oscillator
without changing what the operation claims. An `experiment.inspect` view
contains an `educational_model` card with:

- the state and energy equations in LaTeX;
- symbols, meanings, units and declared values;
- the damping ratio and regime classification;
- the observed energy behavior of the retained trajectory;
- assumptions, suggested exercises and bounded augmentation controls; and
- source, evidence, result and execution provenance.

The card is a read-only projection. It does not re-run Julia, add measurement
uncertainty, establish physical validity or admit a state.

## Offline what-if preview

The terminal can evaluate a bounded analytic preview while an operator is
learning the model:

```text
ciw julia-oscillator preview \
  --source examples/julia/oscillator.json \
  --set model.gamma_s_inv=0 \
  --set initial_state.q0_m=0.5 \
  --output results/julia-oscillator/free-preview.json
```

Allowed paths are `model.omega_0_rad_s`, `model.gamma_s_inv`,
`model.mass_kg`, `initial_state.q0_m` and `initial_state.v0_m_s`. The source
validator keeps the same finite bounds and underdamped profile restrictions as
the registered Julia operation.

A preview is deliberately marked `hypothetical_offline_preview` and has no
`execution_id` or `result_id`. Its `preview_id` identifies the preview file,
not a retained scientific result. To retain the changed trajectory, submit the
augmented source through `ciw.julia-oscillator.v1`; that creates a new
execution and result with the normal raw-byte, runtime and replay bindings.

The preview uses the independent analytic reference, so it is useful for
teaching equations, units, damping and energy. It does not substitute for a
Julia provider execution or for physical calibration.

Each preview also reports a `comparison` object. It gives `max_abs_delta` and
`final_delta` for position, velocity and energy, plus baseline and preview
energy endpoints. This makes the effect of a bounded parameter change visible
without confusing an offline what-if with a retained experiment.


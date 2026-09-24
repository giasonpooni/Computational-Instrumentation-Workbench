# Evidence-bound encoder position

This example retains an evidence bundle, a candidate machine manifest, its
deterministic challenge report and one position request in the shared
workbench, then executes the provider-free `ciw.encoder-position.v1` operation
over them. Ten declared documentation, observation, firmware and validation
sources back the manifest of one leadscrew carriage axis; every value is a
synthetic fixture and nothing is read from a device.

```sh
ciw source add --kind machine-manifest --file examples/machine-manifest/source.json
ciw operation execute ciw.encoder-position.v1 --source SOURCE_ID
```

No repository binding is needed; the operation is served by the independent
Python reference in `src/ciw/machine_manifest.py`. A source whose challenge
report is stale for its candidate and evidence, or whose sealed digests are
broken, is refused before retention. The retained result is the position and
its local linearized uncertainty under the declared kinematic model; physical
validation, state admission and hardware actuation remain `not_performed`.

`make_source.py` is the authoring script for the committed `source.json`.

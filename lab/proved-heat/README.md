# Retained proved-heat gate runs

Each directory here, `lab/proved-heat/<run-id>/`, is one passing run of the SP1
proved-heat gate, the gate steps of
[`.github/workflows/proved-heat.yml`](../../.github/workflows/proved-heat.yml):
the guest ELF rebuilt from SCR's committed recipe, the native engine and SP1
prover built, and `scripts/check_proved_heat.py` producing, verifying,
replaying and re-verifying a real SP1 proof from an installed CIW wheel and
rejecting a tampered one. A run is made on a provisioned host with
`scripts/run_proved_heat_locally.py` and retained with

```sh
ciw lab proved-heat retain results/proved-heat --retained lab --run-id <run-id> --host "<host description>"
```

as described in [docs/LAB.md, Proved-heat gate records](../../docs/LAB.md#proved-heat-gate-records).
No run here is produced by CI or by `scripts/refresh_lab.py`, which never
touches this directory.

| Path | Contents |
| --- | --- |
| `<run-id>/run.json` | `ciw.lab-proved-heat-run.v1`: the operator-declared host, the date, how the run was made (which workflow steps ran, for how long, and how the provisioning steps were satisfied), host facts, toolchain versions, SCR, SP1 and CIW source identities, the compiler archive digest, operator observations, what was left out and the record's limitations; never a host path |
| `<run-id>/manifest.json` | `ciw.lab-proved-heat-manifest.v1`: SHA-256 and size of every other file, and of the gate's own bytes inside each `.gz` file |
| `<run-id>/build.json` | The guest rebuild from the committed recipe: recipe identity, guest ELF SHA-256, compiler archive and SP1 revision |
| `<run-id>/source-checks.json.gz` | The SP1 build-source check: tracked sources unchanged by the build, the clean reference and the generated files left out of it |
| `<run-id>/gate/gate.json` | `ciw.proved-heat-gate.v1`: status, SCR and SP1 pins, native artifact digests, tests passed, installed wheel and the gate's measurements |
| `<run-id>/gate/tests.xml` | The gate's JUnit record |
| `<run-id>/gate/source.json`, `gate/original.json.gz`, `gate/replay.json.gz` | The heat source and the original and replayed proved-heat bundles with their proofs, so a retained proof can be re-verified later with `ciw proof verify` after decompressing it (`gunzip -k gate/original.json.gz`) |
| `<run-id>/gate/reverification.json` | The gate's re-verification of the original bundle's retained proof |

The gate outputs are retained as the gate wrote them, so they name the
recording host's temporary and checkout locations as identity metadata:
`build.json`'s `elf_path`, and the SCR runtime identity (`repository_root` and
`python_executable`) in both bundles and in `gate/reverification.json`'s
verifier runtimes. The bundle digests and the verification id seal those
identities, so they cannot be redacted without invalidating the record;
`run.json` and `manifest.json` hold no host path. The session workspace the
gate also writes is left out (its two bundles are the retained ones) and so are
the driver's console logs, both named with their reason in `run.json`.

`ciw lab verify`, `ciw lab proved-heat verify --retained lab` and
`scripts/check_lab.py` check each record for integrity only: every file
matches the manifest and no unrecorded file is present, the gate passed with
its required native tests, its SCR and SP1 revisions and trees, guest recipe,
guest ELF and compiler archive are the pins CIW declares, and the retained
bundles, replay receipt and re-verification report pass CIW's offline
proved-heat validation. No proof is re-verified and nothing is re-executed.
T099 reads the latest record, which `scripts/check_lab.py` binds as its
`proved-heat-record` provider, and labels what it establishes
`provider_backed`: the gate's outcome as recorded at CIW's pins, not a
verification by T099. The gate's engine and prover are
`operator_asserted_not_attested`, its times, memory and proof sizes are
measurements of its recording host, and the digests are unkeyed: a deliberate
edit that recomputes them is caught only by review of `git diff lab/proved-heat`.

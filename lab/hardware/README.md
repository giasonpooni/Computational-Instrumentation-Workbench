# Retained hardware runs

Each directory here, `lab/hardware/<run-id>/`, is one run of hardware tasks
made by an operator on the capture host (for example the RTX 2080 host, where
`ciw energy probe` succeeds), or a run that retained operator captures, and
is retained with

```sh
ciw lab hardware retain <run output> --retained lab --run-id <run-id> --host "<host description>"
```

as described in [docs/LAB.md, Hardware evidence](../../docs/LAB.md#hardware-evidence).
None is present until an operator retains one; no run here is produced by CI
or by `scripts/refresh_lab.py`, which never touches this directory.

| Path | Contents |
| --- | --- |
| `<run-id>/capture.json` | `ciw.lab-hardware-run.v1`: run identity, the operator-declared host, run date, CIW version and package digest, tasks and report identities, provider and capture bindings by role and digest (never host paths), hardware-measured count |
| `<run-id>/reports/T*.json` | The run's `ciw.lab-task-report.v1` reports, as the capture host wrote them |
| `<run-id>/artifacts/T*/` | Their artifacts, including the raw bytes that hardware-measured findings cite and any operator captures (`capture-<role>.*`) |
| `<run-id>/run-log.json` | Elapsed times of the run; not findings |

Each run's hardware-measured findings rest on a hardware probe that succeeded
in their own task on the capture host; such a run cannot be recomputed in CI
or on any other host. `ciw lab verify`, `ciw lab hardware verify --retained
lab` and `scripts/check_lab.py` check each run for integrity only: its reports
validate and hold no host path, its artifacts match their recorded digests and
no unrecorded file is present, every hardware-measured finding cites raw bytes
its own task retained after a hardware probe succeeded in that task itself
(for an operator capture, a probe of that capture's instrument), and
`capture.json` agrees with the run. Operator captures are retained and not
authenticated: they are bound to no device and never support an established
physical label by themselves. A run of captures alone involves no host device
and can be re-analysed on any host from its retained capture artifacts
(docs/LAB.md, Operator captures). The declared host is the operator's
statement. Digests and report identities are unkeyed: they detect accidental
edits, and a deliberate edit is caught by review of `git diff lab/hardware`.

`ciw lab queue --retained lab` and `ciw lab next --retained lab` show the
latest valid hardware run of a task beside its main-run state and label,
marked with the run identity; neither replaces the main run's labels.
`ciw lab unmeasured --retained lab` lists each run's hardware-measured counts
beside the main run's and T167's, never added to them.

# Retained second-platform figure records

Each directory here, `lab/figure-platforms/<record-id>/`, is one run of
[`.github/workflows/figures.yml`](../../.github/workflows/figures.yml):
`scripts/check_figures.py --retained lab` on `windows-latest`, which
re-executes every retained figure task whose providers it can bind and
compares each figure with the retained copy. The workflow uploads its result
as the `figure-check-windows` artifact; a record is retained from that
artifact's zip with

```sh
python scripts/retain_figure_check.py figure-check-windows.zip --retained lab \
    --repository OWNER/REPO --workflow .github/workflows/figures.yml --run-id RUN --run-attempt 1 \
    --head-sha SHA --artifact-id ID --artifact-name figure-check-windows --artifact-digest sha256:HEX
```

as described in [docs/LAB.md, Second-platform figure records](../../docs/LAB.md#second-platform-figure-records).
The script refuses a zip whose SHA-256 is not the declared artifact digest.
`scripts/refresh_lab.py` never touches this directory.

| Path | Contents |
| --- | --- |
| `<record-id>/figure-check.json` | `ciw.lab-figure-check.v1`: every compared figure's outcome, its retained and fresh digests and, for a declared figure, what identifies its retained copy on any kernel; the tasks not re-executed or not comparable there with their reasons; the platform (OS, Python, NumPy, BLAS, OpenBLAS kernel, CIW package digest) |
| `<record-id>/figure-check.md` | The run's summary of the same comparison |
| `<record-id>/fresh/artifacts/T*/*.svg` | The fresh SVG of each figure the run reports as a mismatch, and no other |
| `<record-id>/record.json` | `ciw.lab-figure-platform-record.v1`: repository, workflow, run id and attempt, head commit, artifact id, name and digest, both text files' digest and line endings as the zip held them, and the platform block of `figure-check.json`; never a host path |
| `<record-id>/manifest.json` | `ciw.lab-figure-platform-manifest.v1`: SHA-256 and size of every other file |

A Windows run writes the two text files with CRLF line endings; the record
stores them with LF, as the repository stores text, and `record.json` records
their bytes in the zip so verification can give them back. The fresh reports
and the SVGs of matching figures are left out.

`ciw lab verify`, `ciw lab figure-platform verify --retained lab` and
`scripts/check_lab.py` check each record for integrity only: every file
matches the manifest and no unrecorded file is present, both text files give
back the zip's bytes, both schemas hold, every figure entry agrees with its
declaration and digests, the summary counts follow from the figure list and
`record.json` names the check's platform. Nothing is re-executed. T158 reads
the latest record, which `scripts/check_lab.py` binds as its
`figure-platform-record` provider: it refuses a record made on its own
operating system, counts the outcomes only of entries whose retained figure is
still its run's, and labels the regeneration on Windows `provider_backed`,
the CI run's outcome as recorded, not a comparison by T158. The provenance is
as declared and the digests are unkeyed: a deliberate edit that recomputes
them is caught only by review of `git diff lab/figure-platforms`.

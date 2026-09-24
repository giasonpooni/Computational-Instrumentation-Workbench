# Lab release report of T001-T164

Scope: the reports of T001-T164, retained before this task ran. T165 reads only earlier reports, so T165-T168 are not in it; queue-state.json and the dashboard cover the whole run.

- CIW version: 0.1.0
- Queue: 168 tasks; 164 reported; not reported: T165, T166, T167, T168
- Release digest: `sha256:0f3127c76ae72fedf33f4d9f1a99857b634ba2f934a7919587e08f1c6524b828` (unsigned; covers task states, headline labels, finding claims and finding labels; sha256 over the UTF-8 bytes of ciw.core.identities.canonical_json (sorted keys, no whitespace, ASCII escapes, NaN refused; the encoding of report identities) of the list [[task, state, headline label, [[claim, label], ...]], ...] in queue order)
- Physical validation: `not_established`

| Section | Tasks | Reported |
| --- | --- | --- |
| Geodesic/Jacobi experiments | 18 | 18 |
| Flat torus and topology | 14 | 14 |
| Arbitrary surfaces and discrete geometry | 12 | 12 |
| Instrument observation experiments | 15 | 15 |
| Sensor-fusion experiments | 17 | 17 |
| CIW exchange and provenance | 24 | 24 |
| Lyapunov runtime experiments | 14 | 14 |
| Energy and GPU experiments | 11 | 11 |
| Manufacturing and robotic use cases | 16 | 16 |
| Rust, Python, Julia, C++, GPU, and FPGA | 13 | 13 |
| Research and portfolio work | 14 | 10 |

| State | Tasks |
| --- | --- |
| blocked | 2 |
| completed | 147 |
| partial | 15 |

| Evidence label | Findings |
| --- | --- |
| `analytic` | 11 |
| `synthetic` | 0 |
| `numerically_verified` | 824 |
| `provider_backed` | 10 |
| `hardware_measured` | 0 |
| `independently_verified` | 69 |
| `not_established` | 187 |

A passing check outranks provenance in the label rules, so the basis components each finding declares are counted beside the labels (a finding counts once per component it declares):

| Basis component | Findings |
| --- | --- |
| `acquisition` | 0 |
| `derivation` | 145 |
| `independent_check` | 69 |
| `provider` | 96 |
| `reference_checks` | 868 |
| `synthetic_inputs` | 366 |
| `none` | 165 |

| Runtime | Revision | Tree or digest | Tasks |
| --- | --- | --- | --- |
| cbsr | `b543969cb80a69a5df1274eee10d22549bcea773` | `eef36f7fe740b4deb562e9dda9383e94294c6503` | T077 |
| csg | `bbc535af29c30997e56fd120320c570830676462` | `181b6eb73288d001f45c39bb149b1a80a431f34b` | T005, T008, T098 |
| ftr | `dc918562cd9e351a65475d29f46963c9f2fd7db8` | `1f082c6b443f2e301ac5889122912f26968ae756` | T019, T020, T026, T098 |
| gsie | `de38873db1ba98744d395dfa7df1d3b5b7bb3059` | `dc364802cf846329e48eaaf0832fa4d47c2abd35` | T077 |
| plsr | `19ea6967060166ba09db6cd4563bd87bd6b3d196` | `a6fee6b3b4eb7852b31f5a285fe8c50b535f25762e1daf82e36720ef8dd1b7c7` | T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T113, T114 |
| ppda | `209985a8c7482aba037021035c2f0b550c9b1ddc`<br>`a29845e13e55de30b24ae752b896058041d653e6` | `9d7150ca433e78e637210c5b2849edba171677d7`<br>`7140bf71289161245e73801e0cd934dc93156509` | T077<br>T097, T098 |
| rust | `-`<br>`-`<br>`-` | `014343936914720e0dab50e5e34a1396f59c0ca5d4ac262cc553b5fbd869ce1c`<br>`4073fb9da1955d54a26a7d6da4f71aef0e2310af667cdd245b3cbed9fca4fcb4`<br>`bdf545c7e9050cc557a68e36f825914ec65fd4e91b25a56a842da477412dccb7` | T142, T146<br>T117<br>T117 |
| scr | `a59aba283b0304faeeb3e5d305087e7709e171ca` | `4068a711534932e8d89bb0d87d373376dafdf6cd` | T097, T098, T099 |
| scr-exchange | `5f0409743e0098a0691a88302a9b3dcdcbcf25fd` | `4b9ba4bf1f95562d299714d4ed4bf7b9d0c6e913` | T097, T098 |
| set | `1467ec5058b3e7ebd6ba4a45f2d9b49148a2560d`<br>`542e672be512bf43b61253f2b2a43cd967cb3062` | `029ee11bcb68d55ac58afae773b875326ccfc9c0`<br>`25c667fc507eb8404e1f077d46e87ed2fc6abc3d` | T077<br>T097, T098 |
| sp1 | `b38b61209e45e969289e70d5cf79dc763460bc41` | `7deca3aced8d8eb84dfcede98285a192c862ea4c` | T099 |
| stfe | `54dd43b657b92d87c2fc263dfd99437c664be353` | `152816aca45290c81d770570851541edae69a706` | T077 |

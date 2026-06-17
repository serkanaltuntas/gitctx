# Open Source Standard

gitctx should be more than an open-weight model. The target is a release that
lets outside developers inspect, modify, retrain, fine-tune, redistribute, and
independently verify the system.

## Required Public Artifacts

| Area | Must publish |
|---|---|
| CLI/source | Source, build scripts, tests, release workflow, install docs |
| Training code | Model config, tokenizer config, loader config, loss/objective code, trainer invocation |
| Data pipeline | Repo discovery rules, license filters, extraction scripts, diff normalization, label schema |
| Teacher pipeline | Teacher model/version/license records, prompts, decoding configs, verifier/ranker code |
| Data documentation | Source manifests, data card, filtering stats, license summary, excluded/private-data policy |
| Dataset artifacts | Derived examples when legally safe; otherwise reproducible manifests and scripts |
| Model artifacts | Weights, tokenizer, config, quantized variants, checkpoints when practical |
| Evals | Tasks, scorers, metrics, baseline results, eval card, held-out policy |
| Reproducibility | Commit hash, environment lock, commands, random seeds, hardware notes, known gaps |
| Governance | Contributing guide, code of conduct, security policy, release checklist |

## License Defaults

- Code: Apache-2.0.
- Documentation: Apache-2.0 unless a release states otherwise.
- Model weights: Apache-2.0 when all upstream obligations allow it.
- Dataset manifests/cards: CC-BY-4.0 or CC0 where appropriate.
- Generated labels: release only under terms compatible with source repository
  licenses and teacher model licenses.

## Claim Language

- `open weights`: only model parameters are public.
- `source available`: code is visible but not under an OSI-compatible license.
- `reproducible open recipe`: code, manifests, and commands are public, but
  some raw data cannot be redistributed.
- `full open-source release`: code, data information, parameters, and
  modification/redistribution rights are public under compatible terms.

Do not call a release fully open source unless it passes the release checklist.

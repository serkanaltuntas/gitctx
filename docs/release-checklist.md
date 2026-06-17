# Release Checklist

Use this checklist before any public model, dataset, or CLI release.

## Legal and Data

- [ ] Every source repository has a license decision.
- [ ] Every teacher model has a recorded license and output-use decision.
- [ ] No closed-model outputs are present.
- [ ] No private customer data is present.
- [ ] No private held-out eval content is present.
- [ ] Generated labels are redistributable under compatible terms.
- [ ] Third-party notices are complete.

## Model

- [ ] Model weights are included.
- [ ] Tokenizer and config are included.
- [ ] Quantized artifacts are documented.
- [ ] Model card states intended use, limitations, risks, and eval results.
- [ ] Checksums are published.

## Data and Evaluation

- [ ] Source manifests are published.
- [ ] Data-generation scripts are published.
- [ ] Teacher prompts and decoding configs are published.
- [ ] Data card is published.
- [ ] Eval card is published.
- [ ] Public eval commands reproduce reported results.
- [ ] Held-out policy is documented.

## Software

- [ ] Source code is under Apache-2.0 unless a file states otherwise.
- [ ] Build and test commands are documented.
- [ ] Release tag exists.
- [ ] Install instructions are tested.
- [ ] Security contact and contribution rules are documented.


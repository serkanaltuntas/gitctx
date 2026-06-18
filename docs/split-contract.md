# Dataset Split Contract

Status: binding for the next artifact after `pilot-v0`.
Date: 2026-06-19.

gitctx must not train and evaluate on adjacent examples that leak repository
style, repeated refactors, or teacher prompt artifacts across splits. The split
contract is therefore stricter than a random row-level split.

## Split Names

| Split | Purpose | Visibility | Allowed use |
|---|---|---|---|
| `DEV` | Build and debug the data, prompt, verifier, and training loop | Inspectable | Training, ablations, error analysis |
| `REPORT` | Measure progress that can be reported publicly | Public aggregate, examples only after release review | Model selection and progress reporting |
| `HELD_OUT` | Final release gate | Private aggregate until release | Final pass/fail only |

`pilot-v0` is `DEV` only. It is useful for pipeline proof, not model-quality
claims.

## Assignment Unit

The default split unit is repository plus time window, not individual rows.

Reason: adjacent commits in the same repository often share file paths, style,
ongoing refactors, release context, and commit-message conventions. A random
row split can make a model look better by leaking repository-local patterns.

For the next artifact:

- a repository may contribute to `DEV` and `REPORT` only when the time windows
  are non-overlapping and recorded before teacher generation;
- `HELD_OUT` should prefer repository-level separation;
- no exact source commit may appear in more than one split;
- no generated label, review record, or training example may be moved into a
  different split after teacher generation;
- split changes require a new artifact version.

## Lock Order

Splits are locked in this order:

1. Select source repositories and pinned revisions.
2. Assign repository/time-window split eligibility.
3. Extract candidate source diffs.
4. Review source diffs for teacher labeling.
5. Generate teacher labels.
6. Review generated labels.
7. Promote reviewed labels into SFT artifacts.
8. Run baseline reports.

Do not assign `REPORT` or `HELD_OUT` after seeing generated model outputs.

## Split Plan File

For any artifact that contains more than DEV-only smoke data, split assignment
must be recorded in a split-plan JSON document before source-diff extraction.

The format is defined in
[`schemas/split-plan.schema.json`](../schemas/split-plan.schema.json). A format
example is available at
[`examples/split-plan.example.json`](../examples/split-plan.example.json).

Each window records:

- repository URL;
- assigned split;
- inclusive `start` timestamp;
- exclusive `end` timestamp;
- reason for the split decision.

Windows for the same repository must not overlap. During extraction, a commit is
included only when its source commit timestamp falls into one recorded window.
The worker writes the `split_plan_path` into the source artifact report so the
private data repository can prove which split plan produced an artifact.

Example worker shape:

```bash
make pilot-source \
  GITCTX_DATA_DIR="$HOME/LAB/gitctx-data" \
  SPLIT_PLAN="$HOME/LAB/gitctx-data/manifests/split-plan.next.json"
```

## Training Rules

- `DEV` may be used for training.
- `REPORT` targets must not be used for training.
- `HELD_OUT` targets must not be used for training, prompt tuning, verifier
  tuning, model selection, or release copy.
- Historical commit subjects are weak comparison context, not ground truth.
- Teacher-generated labels are not training records until human review promotes
  them with `accept` or `edit`.

## Minimum Dataset Conditions

These are minimum gates for claims, not hard limits on experimentation.

| Stage | Minimum artifact | What may be claimed |
|---|---:|---|
| Local smoke training | 100-1,000 `DEV` records | Loader/training code runs; no quality claim |
| GCTX-1 proof run | 10,000 `DEV` train records, 1,000 `REPORT` records, 1,000 reserved `HELD_OUT` candidates | A 60M-100M model can be compared against deterministic and teacher baselines on `REPORT` |
| GCTX-2 public beta candidate | 50,000+ `DEV` train records, 5,000+ `REPORT`, 5,000+ `HELD_OUT` | Public model-quality discussion if model/data/eval cards are signed |

Repository diversity minimum for GCTX-1:

- at least 25 training repositories;
- at least 5 `REPORT` repositories or non-overlapping report windows;
- at least 5 `HELD_OUT` repositories;
- at least two programming-language ecosystems;
- no single repository may contribute more than 25% of training records.

These numbers can be raised after the next pilot. Lowering them requires a new
written decision.

## Baselines Required Before Training Claims

Every model-quality claim must compare against:

- deterministic template/scorer baseline;
- raw teacher label baseline;
- historical subject baseline;
- reviewed SFT target baseline.

The `pilot-v0` baseline showed why historical subjects are insufficient:
historical subjects passed Conventional Commit format on only 5 of 114 records.

## Release Rule

A release may describe full examples as public only when the data card says
which source diffs, teacher outputs, and review targets are redistributable.
When full examples cannot be redistributed, publish source manifests, scripts,
prompts, schemas, aggregate stats, and reproduction instructions instead.

# Teacher Input Artifacts

Teacher input artifacts are private JSONL files that contain the prompt payloads
sent to the approved local teacher model.

The first smoke artifact is:

```text
artifacts/teacher/teacher-inputs.smoke.jsonl
```

Named artifacts use the same layout:

```text
artifacts/teacher/teacher-inputs.<artifact>.jsonl
artifacts/teacher/generated-labels.<artifact>.jsonl
artifacts/teacher/generated-labels.<artifact>.report.json
reviews/generated-labels.<artifact>.review.jsonl
```

It is generated only from source-diff records whose review decision is
`accepted_for_teacher_labeling`.

Each record includes:

- source repository, license, commit, parent commit, split, changed paths, and
  historical subject;
- the pinned teacher model id, revision, license, prompt version, and decoding
  config;
- the teacher runtime, runtime model id, local tag id, size, and context length
  when the teacher runs through Ollama;
- system and user messages;
- full diff text and `diff_sha256`;
- `input_status`.

Full diff text is source-derived data. Keep teacher input artifacts private
until a data-card and redistribution review decides what can be published.

Create the smoke teacher inputs:

```bash
make teacher-inputs
```

Create the pilot teacher inputs after the pilot source-diff review is complete:

```bash
make pilot-teacher-inputs
make pilot-teacher-input-check
```

By default, smoke teacher inputs target the local Ollama teacher approved in
`docs/teacher-decisions/ollama-qwen2.5-coder-7b-smoke.md`:

```text
teacher_model_id: ollama/qwen2.5-coder:7b
teacher_runtime: ollama
teacher_revision: dae161e27b0e
```

Validate and refresh checksums:

```bash
make teacher-input-check
```

Generate labels with local Ollama:

```bash
make teacher-generate
make teacher-generate-check
```

Generate pilot labels with local Ollama:

```bash
make pilot-teacher-generate
make pilot-teacher-generate-check
```

Generation sends one teacher input record per Ollama call. Do not batch many
diffs into one prompt; per-record calls keep provenance, validation, retries,
and resume behavior clean.

Prepare generated-label human review:

```bash
make generated-review-template REVIEWER="reviewer@example.com"
make generated-review-check
```

Prepare pilot generated-label human review:

```bash
make pilot-generated-review-template REVIEWER="reviewer@example.com"
make pilot-generated-review-check
```

The review artifact is private:

```text
reviews/generated-labels.smoke.review.jsonl
```

Each generated-label review records `accept`, `edit`, or `reject`, optional
issue tags, optional edited text, reviewer identity, timestamp, and notes.
Generated labels are not approved as training data until this review is
complete and the training-artifact step promotes only accepted or edited labels.

Create the pilot supervised fine-tuning artifact:

```bash
make pilot-train-artifact
make pilot-train-artifact-check
```

See [`training-artifacts.md`](training-artifacts.md) for the promotion contract.

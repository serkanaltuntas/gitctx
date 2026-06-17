# Teacher Input Artifacts

Teacher input artifacts are private JSONL files that contain the prompt payloads
sent to the approved local teacher model.

The first smoke artifact is:

```text
artifacts/teacher/teacher-inputs.smoke.jsonl
```

It is generated only from source-diff records whose review decision is
`accepted_for_teacher_labeling`.

Each record includes:

- source repository, license, commit, parent commit, split, changed paths, and
  historical subject;
- the pinned teacher model id, revision, license, prompt version, and decoding
  config;
- system and user messages;
- full diff text and `diff_sha256`;
- `input_status`.

Full diff text is source-derived data. Keep teacher input artifacts private
until a data-card and redistribution review decides what can be published.

Create the smoke teacher inputs:

```bash
make teacher-inputs
```

Validate and refresh checksums:

```bash
make teacher-input-check
```

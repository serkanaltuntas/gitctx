# Verified teacher reference overlays

`delta_targets.generate` accepts an explicit `output_format="text"` option for
unconstrained, single-header teacher generation. The default remains `"json"`.
Both formats require a pinned licensed teacher, complete DEV-only input, a valid
Conventional Commit, normal generation completion, matching prompt token counts,
and the fixed student answer budget. Neither format approves a candidate.

Plain responses retain the exact header, including case, spaces and a breaking
marker. Only terminal CR/LF characters may be removed. A response containing a
body, explanation, code fence or invalid header is rejected; the decoder does not
search for a usable substring. JSON responses retain the original fields and
construct the target only from `type`, `scope`, and `subject`. Optional teacher
analysis remains outside the target.

`reference_overlay.build_override` independently decodes the stored raw response
and checks field and target equality before accepting an explicit full-source
semantic attestation. Changing the target, or editing derived fields without
matching raw output, invalidates the overlay. Old candidates without an
`output_format` field continue to use the JSON contract. Mechanical validation
does not establish factual accuracy or independent human approval.

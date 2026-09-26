"""Decode teacher output without inventing or editing a replacement message."""
import json
import re

from gitctx.conventional import DEFAULT_TYPES, HEADER_RE, parse_commit_message


def decode_response(raw, output_format="json"):
    """Return fields and exact target; text may only lose terminal CR/LF bytes.

    JSON preserves all fields (including optional analysis) but only uses the
    three declared message fields. Text does not remove fences, explanations,
    spaces, or a body to manufacture a valid header.
    """
    if not isinstance(raw, str):
        raise ValueError("teacher response must be text")
    if output_format == "json":
        fields = json.loads(raw)
        if not isinstance(fields, dict) or any(
            not isinstance(fields.get(k), str) for k in ("type", "scope", "subject")
        ):
            raise ValueError("invalid fields")
        scope = f"({fields['scope']})" if fields["scope"] else ""
        target = fields["type"] + scope + ": " + fields["subject"]
    elif output_format == "text":
        target = raw.rstrip("\r\n")
        match = HEADER_RE.fullmatch(target)
        if match is None:
            raise ValueError("invalid plain header")
        fields = {k: match[k] or "" for k in ("type", "scope", "subject")}
        if match["breaking"]:
            fields["breaking"] = True
    else:
        raise ValueError("unknown teacher output format")
    if fields["type"] not in DEFAULT_TYPES:
        raise ValueError("invalid type")
    if not fields["subject"].strip():
        raise ValueError("empty subject")
    if not re.fullmatch(r"([a-zA-Z0-9_.-]{1,24})?", fields["scope"]):
        raise ValueError("invalid scope")
    if "\n" in target or "\r" in target or len(target.splitlines()) != 1:
        raise ValueError("subject only")
    parse_commit_message(target)
    return fields, target

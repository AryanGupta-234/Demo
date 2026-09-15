"""Field-preserving truncation for long Work Notes / Comments journals.

Why this exists
----------------
The user turn in each training example is a JSON blob containing the fixed
CR field taxonomy *plus* the raw Work Notes / Comments journal text for that
CR. Journals are the one field group with unbounded length in the real
ServiceNow export. Two failure modes must both be avoided:

1. Naive whole-message truncation (e.g. "just cut the JSON string at N
   chars") can sever the JSON mid-token and can just as easily eat into the
   fixed descriptive/signoff/context fields as into the journal text --
   destroying exactly the fields this project has been careful to keep in
   the schema even when historically sparse (e.g. Change plan).

2. Relying on the tokenizer's/trainer's default `max_length` truncation is
   worse: HF/TRL truncate from the right by default, and the assistant
   target (the actual supervised completion) sits at the *end* of the
   formatted chat sequence. A long journal can silently push the assistant
   target partially or entirely out of the training window, so the model
   is "trained" on a truncated or missing target without any error.

Policy implemented here
------------------------
* CR descriptive/signoff/context fields are NEVER touched.
* Only the journal fields (work_notes, comments, legacy_comments_and_work_notes,
  other_notes) are ever shortened, and only when they exceed a budget.
* ServiceNow journal entries are chronologically headed
  ("MM-DD-YYYY HH:MM:SS - actor (role)"). We segment on that header and keep
  the earliest entry (initial context) plus as many of the most recent
  entries as fit the budget, dropping only the middle -- and we say so
  explicitly in the text, rather than silently disappearing content.
* If no headers are found (free-text journal), we fall back to head+tail
  character truncation with an explicit "omitted" marker.
* The assistant message is never modified by this module.
"""
from __future__ import annotations

import json
import re
from typing import Any

JOURNAL_KEYS = ("work_notes", "comments", "legacy_comments_and_work_notes", "other_notes")

_ENTRY_HEADER_RE = re.compile(r"(?=\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - )")


def _segment_entries(text: str) -> list[str]:
    if not text:
        return []
    parts = _ENTRY_HEADER_RE.split(text)
    entries = [p for p in parts if p.strip()]
    return entries if len(entries) > 1 else [text]


def shrink_field(text: str, char_budget: int) -> str:
    """Shrink one journal string to roughly char_budget characters, keeping
    the earliest entry and the most recent entries, dropping the middle."""
    if not text or len(text) <= char_budget:
        return text
    entries = _segment_entries(text)
    if len(entries) <= 2:
        half = max(1, char_budget // 2)
        head, tail = text[:half], text[-half:]
        return f"{head}\n...[middle omitted for length]...\n{tail}"

    first = entries[0]
    kept_tail: list[str] = []
    budget = char_budget - len(first)
    for entry in reversed(entries[1:]):
        if budget - len(entry) <= 0:
            break
        kept_tail.insert(0, entry)
        budget -= len(entry)
    omitted = len(entries) - 1 - len(kept_tail)
    marker = f"\n...[{omitted} earlier journal entries omitted for length]...\n" if omitted > 0 else ""
    return first + marker + "".join(kept_tail)


def shrink_user_payload(payload: dict[str, Any], *, char_budget_per_field: int) -> tuple[dict[str, Any], bool]:
    changed = False
    out = dict(payload)
    for key in JOURNAL_KEYS:
        value = out.get(key)
        if isinstance(value, str) and len(value) > char_budget_per_field:
            out[key] = shrink_field(value, char_budget_per_field)
            changed = True
    return out, changed


def shrink_user_message_json(content: str, *, char_budget_per_field: int) -> tuple[str, bool]:
    """Apply shrink_user_payload to a JSON-encoded user message string."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content, False
    if not isinstance(payload, dict):
        return content, False
    shrunk, changed = shrink_user_payload(payload, char_budget_per_field=char_budget_per_field)
    if not changed:
        return content, False
    return json.dumps(shrunk, ensure_ascii=False, default=str), True

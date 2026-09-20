"""Context-window planning, strict prompt contracts, and target assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable


TARGET_PREFIXES = {
    "self",
    "participant",
    "person_external",
    "org",
    "project",
    "idea",
    "place",
    "thing",
    "other",
    "ambiguous",
}


@dataclass(frozen=True)
class TagBatch:
    rows: tuple[dict, ...]
    target_gidxs: tuple[int, ...]


Tagger = Callable[[str, str], str]


SYSTEM_PROMPT = """You extract who or what each target message is primarily about.
Return one strict JSON array and no prose. Each object must have this shape:
{"gidx": 1, "targets": ["project:Example"]}
Allowed labels are self, participant:<name>, person_external:<name>, org:<name>,
project:<label>, idea:<label>, place:<label>, thing:<label>, other, and ambiguous.
Return exactly one object per requested gidx, in the same order. Use an empty targets
list when no target is clear. Return at most three concise targets per message."""


def plan_batches(rows: list[dict], *, window: int = 6, context: int = 3) -> list[TagBatch]:
    if window < 1 or window > 100:
        raise ValueError("window must be between 1 and 100")
    if context < 0 or context > 100:
        raise ValueError("context must be between 0 and 100")
    batches: list[TagBatch] = []
    for start in range(0, len(rows), window):
        end = min(len(rows), start + window)
        context_start = max(0, start - context)
        batch_rows = tuple(rows[context_start:end])
        batches.append(TagBatch(batch_rows, tuple(int(row["gidx"]) for row in rows[start:end])))
    return batches


def build_user_prompt(batch: TagBatch, *, participants: Iterable[str] = ("User", "Assistant")) -> str:
    participant_list = [str(value)[:80] for value in participants]
    targets = set(batch.target_gidxs)
    lines = [f"Participants: {json.dumps(participant_list, ensure_ascii=False)}"]
    lines.append("[CTX] lines are context only. [MSG] lines require output.")
    for row in batch.rows:
        gidx = int(row["gidx"])
        tag = "[MSG]" if gidx in targets else "[CTX]"
        sender = str(row.get("sender") or "Unknown")[:80]
        content = str(row.get("content_text") or "(empty)")
        if len(content) > 12_000:
            content = content[:12_000] + "\n[truncated]"
        lines.append(f"{tag} [{gidx}] {sender}: {content}")
    lines.append("TARGET_GIDXS: " + json.dumps(list(batch.target_gidxs)))
    return "\n".join(lines)


def parse_and_validate(response: str, expected_gidxs: tuple[int, ...]) -> list[dict]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Tagger response is not strict JSON.") from exc
    if not isinstance(payload, list) or len(payload) != len(expected_gidxs):
        raise ValueError("Tagger response count does not match the requested messages.")
    validated: list[dict] = []
    for expected, item in zip(expected_gidxs, payload, strict=True):
        if not isinstance(item, dict) or type(item.get("gidx")) is not int or item.get("gidx") != expected:
            raise ValueError(f"Tagger response is not aligned at gidx {expected}.")
        targets = item.get("targets")
        if not isinstance(targets, list) or len(targets) > 3:
            raise ValueError(f"targets for gidx {expected} must be a list of at most three labels.")
        clean: list[str] = []
        for value in targets:
            if not isinstance(value, str) or not value or len(value) > 80:
                raise ValueError(f"Invalid target label for gidx {expected}.")
            prefix = value.split(":", 1)[0]
            if prefix not in TARGET_PREFIXES:
                raise ValueError(f"Unsupported target prefix for gidx {expected}: {prefix}")
            if prefix in {"participant", "person_external", "org", "project", "idea", "place", "thing"} and (":" not in value or not value.split(":", 1)[1].strip()):
                raise ValueError(f"Target label requires a value for gidx {expected}: {value}")
            if prefix in {"self", "other", "ambiguous"} and value != prefix:
                raise ValueError(f"Target label takes no suffix for gidx {expected}: {value}")
            clean.append(value)
        if len(clean) != len(set(clean)):
            raise ValueError(f"Duplicate target labels for gidx {expected}.")
        validated.append({"gidx": expected, "targets": clean})
    return validated


def tag_rows(
    rows: list[dict],
    tagger: Tagger,
    *,
    window: int = 6,
    context: int = 3,
    participants: Iterable[str] = ("User", "Assistant"),
) -> list[dict]:
    tagged = [dict(row) for row in rows]
    by_gidx = {int(row["gidx"]): row for row in tagged}
    for batch in plan_batches(tagged, window=window, context=context):
        prompt = build_user_prompt(batch, participants=participants)
        assignments = parse_and_validate(tagger(SYSTEM_PROMPT, prompt), batch.target_gidxs)
        for assignment in assignments:
            by_gidx[assignment["gidx"]]["targets"] = assignment["targets"]
    return tagged

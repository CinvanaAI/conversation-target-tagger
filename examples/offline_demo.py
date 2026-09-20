"""Authored fixture answers exercise batching; they do not perform inference."""
import json
from pathlib import Path
from conversation_target_tagger import normalize_export, tag_rows

# These are reviewed answers for the three supplied synthetic messages.
FIXTURE_TARGETS = {1: ["place:Workshop"], 2: ["project:GardenWatch"], 3: ["project:GardenWatch"]}


def deterministic_tagger(_system: str, prompt: str) -> str:
    target_ids = json.loads(prompt.rsplit("TARGET_GIDXS: ", 1)[1].splitlines()[0])
    return json.dumps([{"gidx": value, "targets": FIXTURE_TARGETS[value]} for value in target_ids])


if __name__ == "__main__":
    source = Path(__file__).with_name("synthetic_conversations.json")
    rows = normalize_export(json.loads(source.read_text(encoding="utf-8")))
    print(json.dumps(tag_rows(rows, deterministic_tagger, window=2, context=1), indent=2))

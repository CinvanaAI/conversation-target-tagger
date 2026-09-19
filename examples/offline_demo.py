import json
from pathlib import Path

from conversation_target_tagger import normalize_export, tag_rows


def deterministic_tagger(_system: str, prompt: str) -> str:
    marker = "TARGET_GIDXS: "
    target_ids = json.loads(prompt.split(marker, 1)[1].splitlines()[0])
    return json.dumps([{"gidx": value, "targets": ["project:GardenWatch"]} for value in target_ids])


source = Path(__file__).with_name("synthetic_conversations.json")
rows = normalize_export(json.loads(source.read_text(encoding="utf-8")))
print(json.dumps(tag_rows(rows, deterministic_tagger, window=2, context=1), indent=2))

"""Print both complete synthetic requests and the exact requested assignments."""
import json
from pathlib import Path
from conversation_target_tagger import normalize_export
from conversation_target_tagger.tagging import build_user_prompt, plan_batches
from examples.offline_demo import deterministic_tagger

rows = normalize_export(json.loads(Path(__file__).with_name("synthetic_conversations.json").read_text(encoding="utf-8")))
trace = []
for batch in plan_batches(rows, window=2, context=1):
    prompt = build_user_prompt(batch)
    trace.append({"target_indices": batch.target_gidxs, "prompt": prompt,
                  "authored_response": json.loads(deterministic_tagger("", prompt))})
print(json.dumps(trace, indent=2))

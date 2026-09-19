# Conversation Target Tagger

This package turns both common ChatGPT `conversations.json` shapes into one deterministic chronological message stream, then asks a local Ollama model who or what each message is about.

It solves the hard parts that made the original one-file experiment useful:

- supports both `messages` lists and `mapping` graphs;
- retains empty, hidden, system, and non-mainline nodes instead of silently dropping them;
- assigns a stable global chronological index;
- supplies bounded prior-message context for pronoun resolution;
- requires an exact one-to-one JSON response for every requested message;
- rejects misaligned, duplicated, unsupported, or oversized target labels;
- keeps Ollama on loopback unless remote access is explicitly enabled;
- omits message text and raw message objects from output by default.

## Privacy model

The model request necessarily contains the text being classified. The default transport accepts only a loopback Ollama server. Generated JSONL contains message metadata and target labels, but not message text or raw export objects unless the operator explicitly enables those fields.

This is a local processing tool, not an anonymizer. Titles, timestamps, node identifiers, and target labels can still be sensitive.

## Try the offline path

```powershell
python -m unittest discover -s tests -v
python -m examples.offline_demo
python -m conversation_target_tagger.cli examples/synthetic_conversations.json output --plan
```

The first two commands do not call a model. A real local run looks like:

```powershell
conversation-target-tagger conversations.json output --model qwen2.5:7b-instruct
```

Use `--include-content` only when the generated evidence directory is approved to contain conversation text. `--include-raw` carries substantially more privacy risk.

## Limits

- Target labels are model judgments, not ground truth.
- Context windows help with references but cannot reconstruct missing conversation state.
- The tool processes every exported node; it does not decide which branch is canonical.
- Output files contain source hashes and metadata but are not encrypted.
- No live Ollama call is made by the test suite.

See [ORIGIN.md](../ORIGIN.md) and [SECURITY.md](../SECURITY.md).

# Conversation Target Tagger

Put exported messages in order, then attach a checked target label to each one.

Label who or what exported messages concern while preserving chronological identity and enough context to resolve references.

## See it work

**Input:** The supplied synthetic GardenWatch conversations.

**Result:** Three chronologically indexed messages with project:GardenWatch labels from the declared demo callback.

[Read the captured output](examples/result.txt) | [Inspect the example](examples/offline_demo.py)

Python 3.11 or newer. From the repository root:

```sh
python -m pip install -e .
python -m examples.offline_demo
```

The example uses synthetic material and runs offline. The captured output comes from executing this example, not a hand-written mockup.

## How it works

The normalizer handles message lists and mapping graphs without silently dropping hidden or non-mainline nodes. Tagging assigns stable global indices, supplies bounded context, and checks that a response labels each requested message exactly once. The offline example uses a deterministic callback so the data path is inspectable without a model.

Implementation: [conversation_target_tagger/normalize.py](conversation_target_tagger/normalize.py), [conversation_target_tagger/tagging.py](conversation_target_tagger/tagging.py), [conversation_target_tagger/ollama.py](conversation_target_tagger/ollama.py).

## Limits

The callback always returns the same target and is not a classification benchmark. Live labels are model judgments. The library example prints synthetic text; the production CLI omits content by default. A live Ollama call was not tested.

[Reference and CLI details](docs/REFERENCE.md) | [Origin](ORIGIN.md) | [MIT license](LICENSE.md)

# Conversation Target Tagger

Put exported messages in order, then attach a checked target label to each one.

Label who or what exported messages concern while preserving chronological identity and enough context to resolve references.

## See it work

**Input:** The supplied synthetic GardenWatch conversations.

**Result:** Three chronologically indexed messages: one Workshop label and two GardenWatch labels, supplied by the explicitly authored demo callback.

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

The callback looks up authored fixture answers and is not a classification benchmark. Live labels are model judgments. The library example prints synthetic text; the production CLI omits content by default. A live Ollama call was not tested.

[Reference and CLI details](docs/REFERENCE.md) | [Origin](ORIGIN.md) | [MIT license](LICENSE.md)

## Follow a context window

Run `python -m examples.context_trace` or read [the complete requests and responses](examples/context-trace.json). The first batch targets indices 1 and 2. The second includes index 2 as `[CTX]` and requests only index 3, so the reply “Yes, the sensor…” can be read alongside the previous GardenWatch question. Global order also places the separate workshop message before both garden messages. The fixture callback supplies known answers by index; a model must infer those answers itself.

To integrate another model, pass a `tagger(system_prompt, user_prompt) -> str` callback to `tag_rows`. It must return strict JSON with integer indices in requested order and at most three labels per message. Missing/reordered indices, booleans masquerading as index 1, duplicate labels, empty named labels and unsupported prefixes fail. Errors stop the call; retry/backoff policy belongs to your integration.

Context is the preceding **global** stream, which can include a different conversation. It is not a reconstruction of one graph branch. Stable indices are stable for the same normalized input, not permanent IDs across changing exports; keep conversation/node identities for joins. New event schemas and evaluated classifier quality would require additional fixtures and measured runs.

For preserving same-ID conversation variants before classification, see [ChatGPT Export Archive](https://github.com/CinvanaAI/chatgpt-export-to-conversation-engine). [Origin](ORIGIN.md) separates the historical idea from the public continuation.

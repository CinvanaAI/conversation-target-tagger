# Origin

This is a public continuation of a September 2025 script named `chronological_targets_ollama.py` found beside a private ChatGPT export experiment.

The historical script established the central idea: preserve every message from both export shapes, sort them globally, pass bounded prior context to a local model, and require an exact response for each chronological index.

The public package separates normalization, batching, validation, transport, and CLI concerns; adds injectable offline tests; defaults Ollama to loopback; bounds prompts and target labels; writes atomically; refuses accidental output replacement; and removes the original person/project-specific label examples. No conversation exports, generated classifications, images, or account records are included.

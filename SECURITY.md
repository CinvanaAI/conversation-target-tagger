# Security and Privacy

- Treat every conversation export as private input.
- The default Ollama origin must resolve to loopback. Remote model servers require `--allow-remote` and should be trusted explicitly. Prefer HTTPS for remote hosts; plain HTTP exposes conversation text to the network path.
- Ollama endpoints must be bare origins without embedded credentials, paths, queries, or fragments.
- Prompts contain message text even when public output omits it.
- Output content and complete raw messages are separate opt-ins.
- Existing generated files are not replaced without `--force`; the flag only replaces this tool's two known files.
- Writes use temporary files followed by atomic replacement.
- Model output is untrusted and must satisfy the exact JSON, index, count, prefix, size, and duplication contract before it is attached.
- Conversation content can contain prompt-injection text that influences classification. The strict output contract limits the result to bounded labels, but those labels remain untrusted model judgments.
- The tool does not execute model output, open attachments, or follow URLs from conversations.

Before publishing generated output, inspect it independently. Target labels, titles, timestamps, and IDs may identify people or private projects even when content is omitted.

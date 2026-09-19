"""Command-line interface for local, strict conversation target tagging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .normalize import normalize_export, public_row
from .ollama import OllamaTagger
from .tagging import plan_batches, tag_rows


OUTPUT_NAMES = ("tagged_messages.jsonl", "run_summary.json")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tag a ChatGPT export chronologically using a local Ollama model.")
    parser.add_argument("input", type=Path, help="ChatGPT conversations JSON file")
    parser.add_argument("output", type=Path, help="Directory for generated evidence")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--participants", nargs="*", default=["User", "Assistant"])
    parser.add_argument("--include-content", action="store_true", help="Retain message text in generated JSONL")
    parser.add_argument("--include-raw", action="store_true", help="Retain complete raw message objects (high privacy risk)")
    parser.add_argument("--allow-remote", action="store_true", help="Permit a non-loopback Ollama origin")
    parser.add_argument("--force", action="store_true", help="Overwrite this tool's two known output files")
    parser.add_argument("--plan", action="store_true", help="Normalize and report the batch plan without model calls or writes")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    data, source_hash = _load(source)
    rows = normalize_export(data, retain_raw=args.include_raw)
    batches = plan_batches(rows, window=args.window, context=args.context)
    plan = {
        "source_sha256": source_hash,
        "conversation_count": len(data),
        "message_count": len(rows),
        "batch_count": len(batches),
        "window": args.window,
        "context": args.context,
        "includes_content_in_output": bool(args.include_content),
        "includes_raw_messages_in_output": bool(args.include_raw),
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in OUTPUT_NAMES]
    if not args.force and any(path.exists() for path in targets):
        raise FileExistsError("Output exists; use --force to replace this tool's known output files.")
    tagger = OllamaTagger(base_url=args.host, model=args.model, allow_remote=args.allow_remote)
    tagged = tag_rows(
        rows,
        tagger,
        window=args.window,
        context=args.context,
        participants=args.participants,
    )
    jsonl = "".join(
        json.dumps(
            public_row(row, include_content=args.include_content, include_raw=args.include_raw),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for row in tagged
    )
    _atomic_text(output / "tagged_messages.jsonl", jsonl)
    _atomic_text(output / "run_summary.json", json.dumps({**plan, "status": "complete"}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), **plan}, indent=2))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

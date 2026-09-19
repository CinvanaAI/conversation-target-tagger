"""Lossless-enough normalization for the two common ChatGPT export shapes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _timestamp(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _content(content: Any) -> tuple[str, list[str], str]:
    if isinstance(content, str):
        return "text", [content], content
    if not isinstance(content, dict):
        return "", [], ""
    content_type = str(content.get("content_type") or "")
    parts = content.get("parts")
    text_parts = [part for part in parts if isinstance(part, str)] if isinstance(parts, list) else []
    return content_type, text_parts, "\n".join(text_parts)


def _role(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    author = message.get("author")
    return str(author.get("role") or "") if isinstance(author, dict) else ""


def _sender(role: str) -> str:
    lowered = role.casefold()
    if "user" in lowered:
        return "User"
    if "assistant" in lowered:
        return "Assistant"
    if "system" in lowered:
        return "System"
    if "tool" in lowered:
        return "Tool"
    return role or "Unknown"


def _row(
    conversation: dict[str, Any],
    message: dict[str, Any] | None,
    *,
    conversation_index: int,
    sequence: int,
    node_id: Any,
    parent_id: Any,
    children: Any,
    retain_raw: bool,
) -> dict[str, Any]:
    message = message if isinstance(message, dict) else None
    role = _role(message)
    content_type, content_parts, content_text = _content(message.get("content") if message else None)
    metadata = message.get("metadata") if message and isinstance(message.get("metadata"), dict) else {}
    message_time = message.get("create_time") if message else None
    conversation_time = conversation.get("create_time")
    row: dict[str, Any] = {
        "conversation_id": str(conversation.get("id") or f"conversation-{conversation_index}"),
        "conversation_title": str(conversation.get("title") or ""),
        "conversation_create_time": conversation_time,
        "conversation_create_time_iso": _iso(conversation_time),
        "node_id": str(node_id) if node_id is not None else None,
        "parent_id": str(parent_id) if parent_id is not None else None,
        "children_ids": [str(value) for value in children] if isinstance(children, list) else [],
        "role": role,
        "sender": _sender(role),
        "create_time": message_time,
        "create_time_iso": _iso(message_time),
        "content_type": content_type,
        "content_parts": content_parts,
        "content_text": content_text,
        "hidden": bool(metadata.get("is_visually_hidden_from_conversation", False)),
        "model_slug": metadata.get("model_slug"),
        "status": message.get("status") if message else None,
        "end_turn": message.get("end_turn") if message else None,
        "sequence_in_conversation": sequence,
        "conversation_index": conversation_index,
        "sort_timestamp": _timestamp(message_time, _timestamp(conversation_time)),
    }
    if retain_raw:
        row["raw_message"] = message
    return row


def normalize_export(data: Any, *, retain_raw: bool = False) -> list[dict[str, Any]]:
    """Normalize every message/node and assign a deterministic global chronological index."""
    if not isinstance(data, list):
        raise ValueError("Export root must be a list of conversations.")
    rows: list[dict[str, Any]] = []
    for conversation_index, conversation in enumerate(data, start=1):
        if not isinstance(conversation, dict):
            raise ValueError(f"Conversation {conversation_index} is not an object.")
        messages = conversation.get("messages")
        if isinstance(messages, list):
            for sequence, message in enumerate(messages, start=1):
                if not isinstance(message, dict):
                    message = None
                rows.append(
                    _row(
                        conversation,
                        message,
                        conversation_index=conversation_index,
                        sequence=sequence,
                        node_id=message.get("id") if message else None,
                        parent_id=message.get("parent_id") if message else None,
                        children=message.get("children") if message else [],
                        retain_raw=retain_raw,
                    )
                )
            continue
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            continue
        for sequence, (node_id, node) in enumerate(mapping.items(), start=1):
            node = node if isinstance(node, dict) else {}
            rows.append(
                _row(
                    conversation,
                    node.get("message"),
                    conversation_index=conversation_index,
                    sequence=sequence,
                    node_id=node_id,
                    parent_id=node.get("parent"),
                    children=node.get("children"),
                    retain_raw=retain_raw,
                )
            )
    rows.sort(
        key=lambda item: (
            item["sort_timestamp"],
            _timestamp(item.get("conversation_create_time")),
            item["conversation_index"],
            item["sequence_in_conversation"],
        )
    )
    for gidx, row in enumerate(rows, start=1):
        row["gidx"] = gidx
    return rows


def public_row(row: dict[str, Any], *, include_content: bool = False, include_raw: bool = False) -> dict[str, Any]:
    """Remove processing-only and privacy-sensitive fields unless explicitly retained."""
    result = {key: value for key, value in row.items() if key not in {"sort_timestamp", "conversation_index"}}
    if not include_content:
        result.pop("content_text", None)
        result.pop("content_parts", None)
    if not include_raw:
        result.pop("raw_message", None)
    return result

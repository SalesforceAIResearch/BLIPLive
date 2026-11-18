"""Causal ZSS7: three recent frames, latest response, fresh prefill per step."""

from __future__ import annotations

import json
import re

from .memory import ResponseMemory, format_frame_time
from .prompts import SYSTEM_PROMPT

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def parse_json_predictions(text: str) -> list[dict]:
    """Best-effort parse of the current-timestamp JSON protocol."""
    if not text:
        return []
    candidate = re.sub(r"^```(?:json)?\s*", "", text.strip())
    candidate = re.sub(r"\s*```\s*$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        match = _JSON_ARRAY_RE.search(text)
        try:
            parsed = json.loads(match.group(0)) if match else None
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
    if not isinstance(parsed, list):
        return []
    output = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            timestamp = float(item.get("time"))
        except (TypeError, ValueError):
            continue
        output.append({"time": timestamp, "content": content.strip()})
    return sorted(output, key=lambda item: item["time"])


def json_array_parse_status(text: str) -> str:
    """Classify whether the protocol parser received a valid or recoverable array."""
    if not text or not text.strip():
        return "empty"
    candidate = re.sub(r"^```(?:json)?\s*", "", text.strip())
    candidate = re.sub(r"\s*```\s*$", "", candidate)
    try:
        return "valid" if isinstance(json.loads(candidate), list) else "invalid"
    except (TypeError, ValueError, json.JSONDecodeError):
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return "invalid"
        try:
            return "recovered" if isinstance(json.loads(match.group(0)), list) else "invalid"
        except (TypeError, ValueError, json.JSONDecodeError):
            return "invalid"


def ensure_unique_times(predictions: list[dict], *, step: float = 0.001) -> list[dict]:
    """Make PAUC custom IDs unique by shifting duplicate times by one millisecond."""
    output = []
    previous: float | None = None
    for prediction in sorted(predictions, key=lambda item: float(item.get("time", 0.0))):
        timestamp = float(prediction.get("time", 0.0))
        if previous is not None and timestamp <= previous:
            timestamp = previous + step
        output.append({**prediction, "time": timestamp})
        previous = timestamp
    return output


def stream(
    runtime,
    observations,
    *,
    question,
    frame_times,
    window_frames=3,
    max_new_tokens=192,
    tolerance=0.5,
):
    if len(observations) != len(frame_times):
        raise ValueError("One timestamp is required per frame")
    memory = ResponseMemory()
    predictions, trace = [], []
    for index, current in enumerate(frame_times):
        start = max(0, index - window_frames + 1)
        question_prompt = f"User question (asked at time 0): {question}"
        selected_times = frame_times[start : index + 1]
        selected = observations[start : index + 1]
        if hasattr(runtime, "generate_messages"):
            content = [
                {
                    "type": "text",
                    "text": f"{question_prompt}\n{memory.render()}\n\nCurrent observation:",
                }
            ]
            for image, time in zip(selected, selected_times):
                content.extend(
                    [
                        {"type": "text", "text": f"\n{format_frame_time(time)}\n"},
                        {"type": "image", "image": image},
                    ]
                )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            raw = runtime.generate_messages(messages, max_new_tokens=max_new_tokens)
        else:
            bos = runtime.tokenizer.bos_token or ""
            prompt = (
                f"{bos}<|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
                f"<|turn>user\n{question_prompt}<turn|>\n"
                f"<|turn>user\n{memory.render()}\n\nCurrent observation:"
            )
            block = [runtime.embed_text(prompt)]
            for frame, time in zip(selected, selected_times):
                block.append(runtime.embed_text(f"\n{format_frame_time(time)}\n"))
                block.extend(runtime.visual_block(frame))
            block.append(runtime.embed_text("<turn|>\n<|turn>model\n"))
            raw = runtime.greedy_from_embeds(block, max_new_tokens=max_new_tokens)
        parsed = parse_json_predictions(raw)
        # This is the reference protocol's current-timestamp acceptance rule,
        # not a post-hoc deduplication or cooldown.
        accepted = [
            {"time": float(current), "content": p["content"]}
            for p in parsed
            if abs(p["time"] - current) <= tolerance
        ][:1]
        predictions.extend(accepted)
        trace.append({"time": current, "raw": raw, "parse_status": json_array_parse_status(raw)})
        memory.update(accepted)
    return ensure_unique_times(predictions), trace

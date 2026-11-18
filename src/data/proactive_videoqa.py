"""Strict WEB loading: missing videos or questions must not silently shrink a run."""

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(annotation, video_root, *, expected_examples=500):
    source = json.loads(Path(annotation).read_text())
    if expected_examples is not None and len(source) != expected_examples:
        raise ValueError(f"Expected {expected_examples} WEB examples, found {len(source)}")
    root = Path(video_root)
    index, stems = {}, {}
    for path in sorted(root.rglob("*.mp4")):
        index[str(path.relative_to(root).with_suffix(""))] = path
        stems.setdefault(path.stem, []).append(path)
    examples, seen = [], set()
    for item in source:
        question_id = str(item["question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen.add(question_id)
        key = str(Path(question_id).with_suffix(""))
        path = index.get(key)
        if path is None:
            candidates = stems.get(Path(key).name, [])
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous video: {question_id}")
            path = candidates[0]
        question = next(
            (
                t["content"]
                for t in item.get("conversation", [])
                if t.get("role") == "user" and t.get("content")
            ),
            None,
        )
        if not question:
            raise ValueError(f"Missing question: {question_id}")
        # Gold answers/windows are deliberately excluded from model input.
        examples.append({"question_id": question_id, "question": question, "path": str(path)})
    return examples

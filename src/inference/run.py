"""Generate resumable WEB prediction shards (one model replica per CUDA GPU)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from data.proactive_videoqa import file_hash, load_examples, read_jsonl
from harness.prompts import SYSTEM_PROMPT
from harness.protocol import stream


def code_hash():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = [root / "metrics.py"]
    for package in ("data", "harness", "inference", "evaluation"):
        paths.extend((root / package).rglob("*.py"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke test only; full WEB uses all 500 examples")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["protocol"] != "zss7" or config["runtime"] not in ("gemma", "qwen"):
        raise ValueError("This release supports Gemma/Qwen ZSS7 only")
    if config["window_frames"] < 1 or config["max_new_tokens"] < 1:
        raise ValueError("Frame window and token budget must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    examples = load_examples(
        args.annotations, args.videos, expected_examples=config["expected_examples"]
    )
    if args.limit:
        examples = examples[: args.limit]
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config,
        "annotation_sha256": file_hash(args.annotations),
        "video_root": str(args.videos.resolve()),
        "num_examples": len(examples),
        "question_ids": [e["question_id"] for e in examples],
        "world_size": world,
        "video_inventory_sha256": hashlib.sha256(
            json.dumps(
                [
                    (
                        e["question_id"],
                        Path(e["path"]).stat().st_size,
                        Path(e["path"]).stat().st_mtime_ns,
                    )
                    for e in examples
                ]
            ).encode()
        ).hexdigest(),
        "code_sha256": code_hash(),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    meta = args.output / f"run-rank-{rank}.json"
    shard = args.output / f"predictions-rank-{rank}.jsonl"
    if meta.exists() and json.loads(meta.read_text())["identity"] != identity:
        raise ValueError("Run settings changed; use a new output directory")
    if not meta.exists() and shard.exists():
        raise ValueError("Prediction shard has no run metadata; use a new output directory")
    records = read_jsonl(shard) if shard.exists() else []
    done = [str(r["question_id"]) for r in records]
    assigned = examples[rank::world]
    if len(done) != len(set(done)) or not set(done) <= {e["question_id"] for e in assigned}:
        raise ValueError("Invalid or duplicate IDs in existing prediction shard")
    from data.video import load_video

    if config["runtime"] == "gemma":
        from inference.gemma import GemmaRuntime as Runtime
    else:
        from inference.qwen import QwenRuntime as Runtime
    import torch

    versions = {}
    for name in ("torch", "transformers", "opencv-python", "Pillow", "accelerate"):
        versions[name] = importlib.metadata.version(name)
    if meta.exists() and json.loads(meta.read_text())["versions"] != versions:
        raise ValueError("Dependency versions changed; use a new output directory")
    metadata = {
        "identity": identity,
        "rank": rank,
        "versions": versions,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else "CPU",
        "complete": False,
    }
    meta.write_text(json.dumps(metadata, indent=2) + "\n")
    runtime = None
    with shard.open("a", encoding="utf-8") as handle:
        for example in assigned:
            if example["question_id"] in done:
                continue
            if runtime is None:
                runtime = Runtime.load(
                    model_name=config["model_name"], model_revision=config["model_revision"]
                )
            images, times = load_video(example["path"], source_fps=config["source_fps"])
            observations = (
                runtime.encode_pils(images) if hasattr(runtime, "encode_pils") else images
            )
            predictions, trace = stream(
                runtime,
                observations,
                question=example["question"],
                frame_times=times,
                window_frames=config["window_frames"],
                max_new_tokens=config["max_new_tokens"],
                tolerance=config["tolerance"],
            )
            record = {
                "question_id": example["question_id"],
                "model_response_list": predictions,
                "trace": trace,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(
                f"rank={rank} question={example['question_id']} responses={len(predictions)}",
                flush=True,
            )
    metadata["complete"] = True
    meta.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

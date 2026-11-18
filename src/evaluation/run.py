"""Merge complete prediction shards, optionally judge, then report PAUC/PAUC-F."""

import argparse
import json
from pathlib import Path

from data.proactive_videoqa import file_hash, read_jsonl
from evaluation.report import report


def merge_predictions(run, annotation):
    meta = json.loads((run / "run-rank-0.json").read_text())
    identity = meta["identity"]
    if file_hash(annotation) != identity["annotation_sha256"]:
        raise ValueError("Annotations differ from generation")
    records = {}
    for rank in range(identity["world_size"]):
        shard_meta = json.loads((run / f"run-rank-{rank}.json").read_text())
        if shard_meta["identity"] != identity or not shard_meta["complete"]:
            raise ValueError(f"Rank {rank} is incomplete or has different settings")
        for record in read_jsonl(run / f"predictions-rank-{rank}.jsonl"):
            key = str(record["question_id"])
            if key in records:
                raise ValueError(f"Duplicate prediction: {key}")
            records[key] = record
    if set(records) != set(identity["question_ids"]):
        raise ValueError("Prediction IDs do not match the selected benchmark examples")
    path = run / "predictions.jsonl"
    path.write_text(
        "".join(
            json.dumps(records[key], ensure_ascii=False) + "\n" for key in identity["question_ids"]
        )
    )
    return path, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--scores", type=Path, help="Use existing official judged answers; no API calls"
    )
    parser.add_argument("--judge-model", help="Override the configuration's judge model")
    parser.add_argument("--judge-workers", type=int, default=8)
    args = parser.parse_args()
    predictions, identity = merge_predictions(args.run, args.annotations)
    config = identity["config"]
    judge_model = args.judge_model or config["judge_model"]
    if args.scores:
        scores = args.scores
    else:
        from evaluation.judge import judge_predictions

        scores = judge_predictions(
            predictions, args.annotations, args.run, model=judge_model, workers=args.judge_workers
        )
    report(
        predictions,
        scores,
        args.run / "summary.json",
        judge_model=judge_model,
        omega=config["omega"],
    )


if __name__ == "__main__":
    main()

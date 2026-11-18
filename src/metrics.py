"""Standalone PAUC and PAUC-F (pauc-f-v1), using only the standard library.

Inputs may be in-memory lists or paths: predictions are JSONL, judged answers
are the official PAUC JSON schema. Public summary scores are percentages.
Judgments use scores 0/1/2; API judging is intentionally separate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCORER_VERSION = "pauc-f-v1"
DEFAULT_OMEGA = 0.5


def _records(value, *, jsonl=False):
    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8") as handle:
            return (
                [json.loads(line) for line in handle if line.strip()]
                if jsonl
                else json.load(handle)
            )
    return list(value)


def _by_id(records):
    result = {}
    for record in records:
        key = str(record["question_id"])
        if key in result:
            raise ValueError(f"Duplicate question_id: {key}")
        result[key] = record
    return result


def _validate_scores(scored, omega):
    if not math.isfinite(omega) or not 0 <= omega <= 1:
        raise ValueError("omega must be between 0 and 1")
    if not scored:
        raise ValueError("No scored examples")
    _by_id(scored)
    for record in scored:
        if not record["answer"]:
            raise ValueError("Each example must contain gold turns")
        for turn in sorted(record["answer"], key=lambda t: t["reply_timespan"][0]):
            start, end = map(float, turn["reply_timespan"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("Gold windows must have finite, positive duration")
            for time, quality in turn["judge_scores"].items():
                if not start <= float(time) <= end or quality not in (0, 1, 2):
                    raise ValueError("Judgments must be in-window with quality 0, 1, or 2")


def area_under_line_ratio(points, max_x, max_y, omega=0.5, start_score=0.5):
    """Integrate the official piecewise-constant quality curve, preserving point order."""
    if not points:
        return 0.0
    points = sorted(points, key=lambda point: point[0])
    points = [(x * (1 - omega), y) for x, y in points]
    points.append((max_x, points[-1][1]))
    previous_x, previous_y, area = 0, start_score, 0
    for x, y in points:
        area += (x - previous_x) * previous_y
        previous_x, previous_y = x, y
    return area / (max_x * max_y)


def fire_times(record: dict) -> list[float]:
    """Return every physical assistant fire, preserving duplicates."""
    return [
        float(turn["time"])
        for turn in record.get("model_response_list", [])
        if str(turn.get("role", "assistant")).lower() == "assistant"
    ]


def assign_value_adding_fires(
    fires: list[float],
    turns: list[dict],
    *,
    timestamp_tolerance: float = 1e-6,
) -> set[int]:
    """Assign the earliest positive judged fire to each gold turn.

    Process windows in earliest-start order, as in the reference. Values are indices
    into ``fires``, rather than timestamps, so two physical fires with the same
    timestamp remain distinct denominator entries.
    """
    assigned: set[int] = set()
    ordered_turns = sorted(
        enumerate(turns),
        key=lambda item: (
            float(item[1]["reply_timespan"][0]),
            float(item[1]["reply_timespan"][1]),
            item[0],
        ),
    )
    for _, turn in ordered_turns:
        judged = sorted(
            (
                (float(timestamp), float(score))
                for timestamp, score in (turn.get("judge_scores") or {}).items()
            ),
            key=lambda item: item[0],
        )
        for timestamp, score in judged:
            if score <= 0:
                continue
            fire_index = next(
                (
                    index
                    for index, fire_time in enumerate(fires)
                    if index not in assigned
                    and math.isclose(
                        fire_time,
                        timestamp,
                        rel_tol=0.0,
                        abs_tol=timestamp_tolerance,
                    )
                ),
                None,
            )
            if fire_index is not None:
                assigned.add(fire_index)
                break
    return assigned


def precision_counts(predictions: list[dict], scored: list[dict]) -> dict[str, int]:
    """Count physical fires and at most one value-adding fire per gold turn."""
    scores_by_id = {str(item["question_id"]): item for item in scored}
    prediction_ids = {str(item["question_id"]) for item in predictions}
    missing = sorted(prediction_ids - scores_by_id.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} prediction IDs are absent from scored artifacts; first={missing[0]}"
        )

    total = 0
    useful = 0
    for prediction in predictions:
        question_id = str(prediction["question_id"])
        fires = fire_times(prediction)
        total += len(fires)
        useful += len(assign_value_adding_fires(fires, scores_by_id[question_id]["answer"]))
    return {
        "num_fires": total,
        "num_value_adding_fires": useful,
        "num_wasted_fires": total - useful,
    }


def score_pauc(scored, *, omega=DEFAULT_OMEGA):
    """Return official gold-turn-mean PAUC on a 0–100 scale.

    No judged response in a gold window contributes 25, as in official PAUC.
    Zero-duration windows are rejected, rather than silently changing the mean.
    """
    scored = _records(scored)
    _validate_scores(scored, omega)
    values = []
    for record in scored:
        for turn in record["answer"]:
            start, end = turn["reply_timespan"]
            points = [(float(time) - start, score) for time, score in turn["judge_scores"].items()]
            values.append(area_under_line_ratio(points, end - start, 2, omega) if points else 0.25)
    return {
        "metric": "PAUC",
        "omega": omega,
        "pauc": 100 * (sum(values) / len(values)),
        "num_videos": len(scored),
        "num_gold_turns": len(values),
    }


def score_pauc_f(predictions, scored, *, omega=DEFAULT_OMEGA, judge_model="gpt-4.1"):
    """Score every physical assistant fire, including repeats/out-of-window fires."""
    predictions = _records(predictions, jsonl=True)
    scored = _records(scored)
    pred_by_id, scored_by_id = _by_id(predictions), _by_id(scored)
    if pred_by_id.keys() != scored_by_id.keys():
        raise ValueError("Prediction and scored question IDs must match exactly")
    for record in predictions:
        if any(not math.isfinite(t) or t < 0 for t in fire_times(record)):
            raise ValueError("Fire timestamps must be finite and nonnegative")
    raw = score_pauc(scored, omega=omega)
    counts = precision_counts(predictions, scored)
    recall = raw["pauc"] / 100
    precision = (
        counts["num_value_adding_fires"] / counts["num_fires"] if counts["num_fires"] else 0.0
    )
    harmonic = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric": "PAUC-F",
        "scorer_version": SCORER_VERSION,
        "omega": omega,
        "judge_model": judge_model,
        "headline_eligible": judge_model != "timing",
        "fire_assignment": "earliest-positive-per-turn; earliest-window-first; one-to-one",
        "pauc_f": 100 * harmonic,
        "pauc": raw["pauc"],
        "pauc_recall_component": raw["pauc"],
        "fire_precision": 100 * precision,
        "num_videos": raw["num_videos"],
        "num_gold_turns": raw["num_gold_turns"],
        **counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Include to calculate PAUC-F as well")
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        score_pauc_f(args.predictions, args.scores, omega=args.omega, judge_model=args.judge_model)
        if args.predictions
        else score_pauc(args.scores, omega=args.omega)
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

"""Official cumulative-answer PAUC judging with a resumable request cache."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.proactive_videoqa import read_jsonl

INSTRUCTION = "You are an evaluator for a video question answering system. Your task is to rate the whether the predicted answer covers the key points of the ground truth answer. Use the following scale to assign a score:\n- 3: Mostly covered; the predicted answer covers all key information in the ground truth answer, though it may have minor inaccuracies or rephrases.\n- 2: Partially covered; the predicted answer has some correct information, but also contains significant inaccuracies or missing key points.\n- 1: Incorrect; the predicted answer may be related to the ground truth answer, but most of the information is missing, or the predicted answer is not relevant to the question or in very poor quality.Output the score only, do not add more explanations."


def build_requests(predictions, gold, *, model="gpt-4.1"):
    """Match official PAUC's cumulative, text-deduplicated judge requests."""
    gold_by_id = {str(item["question_id"]): item for item in gold}
    requests = []
    for prediction in predictions:
        key = str(prediction["question_id"])
        example = gold_by_id[key]
        fires = [
            p
            for p in prediction["model_response_list"]
            if p.get("role", "assistant") == "assistant"
        ]
        for turn_index, turn in enumerate(example["answer"]):
            start, end = turn["reply_timespan"]
            sentences = []
            for fire in fires:
                if not start <= fire["time"] <= end or fire["content"] in sentences:
                    continue
                sentences.append(fire["content"])
                text = (
                    f"Question: {example['conversation'][0]['content']}\n"
                    f"Ground Truth Answer: {turn['content']}\n"
                    f"Predicted Answer: {' '.join(sentences)}"
                )
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": INSTRUCTION},
                        {"role": "user", "content": text},
                    ],
                }
                requests.append(
                    {
                        "custom_id": f"{key}={turn_index}={fire['time']}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
    return requests


def process_scores(predictions, gold, responses):
    """Apply the official 1/2/3 conversion and related-window tail filter."""
    examples = {str(e["question_id"]): copy.deepcopy(e) for e in gold}
    streams = {
        str(p["question_id"]): [
            f for f in p["model_response_list"] if f.get("role", "assistant") == "assistant"
        ]
        for p in predictions
    }
    for key, fires in streams.items():
        for turn in examples[key]["answer"]:
            turn["judge_scores"] = {}
            turn["preds"] = [
                (f["time"], f["content"])
                for f in fires
                if turn["reply_timespan"][0] <= f["time"] <= turn["reply_timespan"][1]
            ]
    requested = set()
    for response in responses:
        key, turn_index, timestamp = response["custom_id"].rsplit("=", 2)
        requested.add(key)
        turn = examples[key]["answer"][int(turn_index)]
        content = response["response"]["body"]["choices"][0]["message"]["content"]
        if not content or content[0] not in "123":
            raise ValueError(f"Invalid judge response: {content!r}")
        turn["judge_scores"][float(timestamp)] = int(content[0]) - 1
    for key in streams:
        for turn in examples[key]["answer"]:
            if key not in requested:
                turn["preds"] = []
            if len(turn["judge_scores"]) <= 1:
                continue
            last_score = 0
            for timestamp, score in sorted(turn["judge_scores"].items()):
                if timestamp <= turn["related_timespan"][1]:
                    last_score = score
                elif score < last_score:
                    del turn["judge_scores"][timestamp]
    return [examples[str(p["question_id"])] for p in predictions]


def judge_predictions(predictions, gold, output, *, model="gpt-4.1", workers=8):
    """Use OPENAI_API_KEY and optional OPENAI_BASE_URL; reuse exact cached requests."""
    from openai import OpenAI

    output = Path(output)
    cache = output / "judge-cache"
    cache.mkdir(exist_ok=True)
    predictions = read_jsonl(predictions)
    gold = json.loads(Path(gold).read_text())
    requests = build_requests(predictions, gold, model=model)
    if workers < 1:
        raise ValueError("Judge worker count must be positive")
    (output / "judge_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))

    # Deduplicate identical bodies before parallel execution to avoid concurrent cache writes.
    def cache_key(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    bodies = {cache_key(r["body"]): r["body"] for r in requests}

    def execute(item):
        key, body = item
        path = cache / f"{key}.json"
        if path.exists():
            response = json.loads(path.read_text())
        else:
            response_obj = OpenAI().chat.completions.create(**body, timeout=120)
            content = response_obj.choices[0].message.content or ""
            if not content or content[0] not in "123":
                raise ValueError(f"Invalid judge response: {content!r}")
            response = {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": content}}]},
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(response))
            temporary.replace(path)
        return key, response

    # No client is constructed for an all-silent or fully cached evaluation.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cached = dict(pool.map(execute, bodies.items()))
    responses = [
        {"custom_id": r["custom_id"], "response": cached[cache_key(r["body"])], "error": None}
        for r in requests
    ]
    (output / "judge_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses))
    scores_path = output / "judged_answers.json"
    scores_path.write_text(
        json.dumps(process_scores(predictions, gold, responses), indent=2) + "\n"
    )
    return scores_path

# BLIP-Live

[Kanchana Ranasinghe](https://scholar.google.com/citations?user=K2WBZTwAAAAJ), [Honglu Zhou](https://scholar.google.com/citations?user=U-Uzcs8AAAAJ), [Dominick Reilly](https://scholar.google.com/citations?user=YlFKOTkAAAAJ), [Le Xue](https://scholar.google.com/citations?user=2_SZkwMAAAAJ&hl=en), [Shusheng Yang](https://scholar.google.com/citations?user=v6dmW5cntoMC), [Juan Carlos Niebles](https://scholar.google.com/citations?user=hqNhUCYAAAAJ), [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ), [Ran Xu](https://scholar.google.com/citations?user=sgBB2sUAAAAJ)

[Project Page](https://bliplive.salesforceresearch.ai/) | [Paper](https://arxiv.org/abs/2611.21553v1)

PAUC and PAUC-F metrics, plus a minimal causal streaming harness for Gemma and Qwen.
This first release supports the 500-video **ProactiveVideoQA WEB** evaluation.
Training and inference with our trained checkpoints will follow in a future release.

## Installation

Use Python 3.12. Metric calculation has no third-party dependencies:

```bash
pip install -e .
python -m unittest discover -s tests -v
```

For model generation and API judging, install the harness dependencies in a CUDA
environment. Install the appropriate CUDA build of PyTorch for your machine first
if necessary; the reference dependency versions are declared in `pyproject.toml`.

```bash
pip install -e ".[harness]"
```

The example configurations use BF16 full-size models and SDPA. Each GPU holds a
complete model replica; multiple GPUs shard videos, not model weights. The recorded
experiments used eight H200 GPUs. Obtain access to the configured Hugging Face models and authenticate with `hf auth login`
or `HF_TOKEN` before generation. Model weights are downloaded on first use.

## WEB quick start

Obtain the [ProactiveVideoQA dataset](https://huggingface.co/datasets/wangyueqian/ProactiveVideoQA)
and point the launchers to the released WEB directory:

```text
/path/to/WEB/
├── anno.json
└── videos/
    └── …500 MP4 videos
```

Use the released 2-FPS WEB videos. The reference pipeline samples end-aligned at
1 FPS using `source_fps=2`, a three-frame causal window, latest-response memory,
greedy decoding, and the unchanged ZSS7 prompt. Re-encoding videos or substituting
a different source FPS changes observations and timestamps.

Set `OPENAI_API_KEY` for GPT-4.1 judging. `OPENAI_BASE_URL` may optionally point to
an OpenAI-compatible endpoint. **The examples make paid judge API calls after
generation.** Exact cached requests are reused on reruns.

```bash
# One GPU by default. Set NUM_GPUS=8 to shard across eight model replicas.
bash examples/run_web_qwen.sh /path/to/WEB outputs/qwen_web
bash examples/run_web_gemma.sh /path/to/WEB outputs/gemma_web
```

Both examples generate predictions, judge answers, and report raw PAUC and PAUC-F
at `omega=0.5`. They require all 500 annotations and all corresponding videos;
missing videos or mismatched IDs produce an error rather than a partial score.
Run only one launcher per output directory at a time.

To generate a small smoke-test subset without making judge calls:

```bash
python -m inference.run --config configs/qwen_web.json \
  --annotations /path/to/WEB/anno.json --videos /path/to/WEB/videos \
  --output outputs/qwen_smoke --limit 2
```

To run evaluation separately, including against existing official judgments:

```bash
python -m evaluation.run --run outputs/qwen_web \
  --annotations /path/to/WEB/anno.json

# Offline scoring with saved judgments; no API calls.
python -m evaluation.run --run outputs/qwen_web \
  --annotations /path/to/WEB/anno.json --scores /path/to/judged_answers.json
```

Generation resumes completed videos with the same command and output directory.
Code, config, annotation, video inventory, dependency, subset, or worker-count changes
require a new directory.
Each run stores per-rank predictions and provenance, merged `predictions.jsonl`,
judge requests/responses, `judge-cache/`, `judged_answers.json`, and `summary.json`.
Predictions include raw per-step outputs to help inspect parsing and timing.

## Standalone metrics

Copy `src/metrics.py` into another project, or install this repository:

```python
from metrics import score_pauc, score_pauc_f

raw = score_pauc("judged_answers.json", omega=0.5)
result = score_pauc_f("predictions.jsonl", "judged_answers.json", omega=0.5)
print(raw["pauc"], result["pauc_f"])
```

Both functions also accept lists of records. All public summary scores are
percentages on a 0–100 scale. To score from the command line:

```bash
python src/metrics.py --predictions predictions.jsonl \
  --scores judged_answers.json --omega 0.5
```

Predictions use the official JSONL schema:

```json
{"question_id":"video.mp4","model_response_list":[{"time":2.5,"content":"The person opens the door."}]}
```

Judged answers are a JSON list. Each record contains `question_id` and `answer`;
each answer contains `reply_timespan: [start, end]` and `judge_scores`, mapping
timestamp strings to quality scores 0, 1, or 2. The standalone scorer does not
infer semantic quality from answer text; the evaluation pipeline supplies it.

PAUC averages quality/timing area over gold turns, including its official 25-point
no-reply score. PAUC-F harmonically combines that recall with pooled physical-fire
precision: each turn can credit at most one earliest positive judged fire, and
each physical fire can be credited once. Repeats and out-of-window fires remain
in the denominator. No fires produce PAUC-F zero. Zero-duration windows are
rejected explicitly. The scorer version is `pauc-f-v1`.

## Recorded WEB reference results

These are historical zero-shot ZSS7 results, not fresh runs of this release.
Both use GPT-4.1 judging at `omega=0.5` on 500 videos / 1,328 gold turns.

| Base model + harness | PAUC | PAUC-F |
| --- | ---: | ---: |
| Gemma-4-26B-A4B-it | 57.30 | 43.39 |
| Qwen3.5-35B-A3B | 39.53 | 54.24 |

The configs pin model revisions and protocol settings. Original Gemma WEB
generation came from a separate checkout; the original Qwen WEB revision was
recorded as a hub-cache revision rather than an exact hash. The Qwen config
therefore pins the revision used elsewhere in the reference project. Fresh
inference and judge outputs may differ; this release does not yet claim exact
end-to-end reproduction of either historical generation run.

## Layout

- `src/metrics.py`: standalone PAUC and PAUC-F.
- `src/harness/`: streaming protocol, fixed prompt, latest-response memory.
- `src/inference/`: Gemma and Qwen model loading and generation.
- `src/evaluation/`: shard validation, API judging, and reporting.
- `src/data/`: WEB annotation loading and reference frame sampling.
- `configs/`: user-editable reproduction settings.
- `examples/`: two complete WEB launchers.
- `tests/test_metrics.py`: small offline correctness tests.
- `src/train/` and `src/inference/trained/`: future release notes.

## Contributing and responsible use

See [contribution guidelines](CONTRIBUTING.md), our
[Code of Conduct](CODE_OF_CONDUCT.md), and [AI ethics disclaimer](AI_ETHICS.md).
Report security concerns using [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE.txt).
Dataset and model terms are provided by their respective publishers.

Copyright (c) 2026 Salesforce, Inc.

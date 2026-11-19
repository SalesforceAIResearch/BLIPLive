# BLIP-Live

[Project Page](https://bliplive.salesforceresearch.ai/) | [Arxiv](https://arxiv.org/abs/2611.21553v1)

Authors: 
[Kanchana Ranasinghe](https://scholar.google.com/citations?user=K2WBZTwAAAAJ), [Honglu Zhou](https://scholar.google.com/citations?user=U-Uzcs8AAAAJ), [Dominick Reilly](https://scholar.google.com/citations?user=YlFKOTkAAAAJ), [Le Xue](https://scholar.google.com/citations?user=2_SZkwMAAAAJ&hl=en), [Shusheng Yang](https://scholar.google.com/citations?user=v6dmW5cntoMC), [Juan Carlos Niebles](https://scholar.google.com/citations?user=hqNhUCYAAAAJ), [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ), [Ran Xu](https://scholar.google.com/citations?user=sgBB2sUAAAAJ)


## Installation

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

## Examples 

See [examples/README.md](examples/README.md) for more details.


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

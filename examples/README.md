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
uv run --extra harness python -m inference.run --config configs/qwen_web.json \
  --annotations /path/to/WEB/anno.json --videos /path/to/WEB/videos \
  --output outputs/qwen_smoke --limit 2
```

To run evaluation separately, including against existing official judgments:

```bash
uv run --extra harness python -m evaluation.run --run outputs/qwen_web \
  --annotations /path/to/WEB/anno.json

# Offline scoring with saved judgments; no API calls.
uv run --extra harness python -m evaluation.run --run outputs/qwen_web \
  --annotations /path/to/WEB/anno.json --scores /path/to/judged_answers.json
```

Generation resumes completed videos with the same command and output directory.
Code, config, annotation, video inventory, dependency, subset, or worker-count changes
require a new directory.
Each run stores per-rank predictions and provenance, merged `predictions.jsonl`,
judge requests/responses, `judge-cache/`, `judged_answers.json`, and `summary.json`.
Predictions include raw per-step outputs to help inspect parsing and timing.


## Recorded WEB reference results

These are historical zero-shot ZSS7 results, not fresh runs of this release.
Both use GPT-4.1 judging at `omega=0.5` on 500 videos / 1,328 gold turns.

| Base model + harness | PAUC | PAUC-F |
| --- | ---: | ---: |
| Gemma-4-26B-A4B-it | 57.30 | 43.39 |
| Qwen3.5-35B-A3B | 39.53 | 54.24 |

# Source attribution

This release extracts the PAUC-F scorer and Gemma/Qwen ZSS7 WEB evaluation path
from the BLIP-Live development repository (`stream-gemma`), reference commit
`34a771e52e11fa3d339d216762db79485fd684a8`.

PAUC integration, cumulative-answer judging, the original judge instruction, and
the related-window postprocessing rule are adapted from
[ProactiveVideoQA](https://github.com/yellow-binary-tree/ProactiveVideoQA),
`pauc/judge_llm_scores_openai.py`, reference commit
`6fe9a3338b3dd02d15a657d293c02986afb91bf1`.
Please cite the benchmark when using its dataset or metric:

Yueqian Wang et al. *ProactiveVideoQA: A Comprehensive Benchmark Evaluating
Proactive Interactions in Video Large Language Models*. arXiv:2507.09313, 2025.

No dataset, model weights, private credentials, or original experiment artifacts
are included in this repository.

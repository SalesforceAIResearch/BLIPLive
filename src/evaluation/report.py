"""Report versioned PAUC-F alongside raw PAUC."""

import json
from pathlib import Path

from metrics import score_pauc_f


def report(predictions, scores, output, *, judge_model="gpt-4.1", omega=0.5):
    summary = score_pauc_f(predictions, scores, omega=omega, judge_model=judge_model)
    Path(output).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    return summary

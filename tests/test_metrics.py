"""Small, offline, hand-calculated checks; no model or judge dependencies."""

import copy
import unittest

from metrics import score_pauc, score_pauc_f


def case(fires=(), judgments=None):
    predictions = [
        {
            "question_id": "video",
            "model_response_list": [{"time": time, "content": "answer"} for time in fires],
        }
    ]
    scored = [
        {
            "question_id": "video",
            "answer": [{"reply_timespan": [0, 10], "judge_scores": judgments or {}}],
        }
    ]
    return predictions, scored


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        predictions, scored = case([0], {"0": 2})
        self.assertEqual(score_pauc(scored)["pauc"], 100)
        self.assertEqual(score_pauc_f(predictions, scored)["pauc_f"], 100)

    def test_silence(self):
        predictions, scored = case()
        self.assertEqual(score_pauc(scored)["pauc"], 25)
        self.assertEqual(score_pauc_f(predictions, scored)["pauc_f"], 0)

    def test_delayed_response(self):
        predictions, scored = case([8], {"8": 2})
        # omega=.5 moves t=8 to t=4: (4*.5 + 6*2) / (10*2) = .7.
        self.assertAlmostEqual(score_pauc(scored)["pauc"], 70)
        self.assertAlmostEqual(score_pauc_f(predictions, scored)["pauc_f"], 200 * 0.7 / 1.7)
        self.assertAlmostEqual(score_pauc(scored, omega=0)["pauc"], 40)
        self.assertAlmostEqual(score_pauc(scored, omega=1)["pauc"], 100)

    def test_repeats_and_out_of_window_fires_count(self):
        predictions, scored = case([0, 0, 20], {"0": 2})
        result = score_pauc_f(predictions, scored)
        self.assertEqual(result["num_fires"], 3)
        self.assertEqual(result["num_value_adding_fires"], 1)
        self.assertAlmostEqual(result["pauc_f"], 50)

    def test_wrong_response_can_be_worse_than_silence(self):
        predictions, scored = case([0], {"0": 0})
        self.assertEqual(score_pauc(scored)["pauc"], 0)
        self.assertEqual(score_pauc_f(predictions, scored)["pauc_f"], 0)

    def test_pool_precision_and_average_gold_turns(self):
        predictions, scored = case([0], {"0": 2})
        other_pred, other_scored = case([20, 21])
        other_pred[0]["question_id"] = other_scored[0]["question_id"] = "second"
        result = score_pauc_f(predictions + other_pred, scored + other_scored)
        # R=(1+.25)/2=.625; P=1/3; F=10/23.
        self.assertAlmostEqual(result["pauc"], 62.5)
        self.assertAlmostEqual(result["pauc_f"], 1000 / 23)

    def test_overlapping_windows_credit_each_physical_fire_once(self):
        predictions, scored = case([2], {"2": 2})
        scored[0]["answer"].append({"reply_timespan": [1, 8], "judge_scores": {"2": 2}})
        result = score_pauc_f(predictions, scored)
        self.assertEqual(result["num_gold_turns"], 2)
        self.assertEqual(result["num_value_adding_fires"], 1)

    def test_invalid_inputs_fail(self):
        predictions, scored = case([0], {"0": 2})
        for omega in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                score_pauc(scored, omega=omega)
        with self.assertRaises(ValueError):
            score_pauc_f(predictions + predictions, scored)
        with self.assertRaises(ValueError):
            score_pauc_f([], scored)
        broken = copy.deepcopy(scored)
        broken[0]["answer"][0]["reply_timespan"] = [1, 1]
        with self.assertRaises(ValueError):
            score_pauc(broken)


if __name__ == "__main__":
    unittest.main()

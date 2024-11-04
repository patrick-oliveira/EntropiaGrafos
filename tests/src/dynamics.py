import unittest as ut
import numpy as np

from opdynamics.model.dynamics import mutate, distort
from opdynamics.model.model import Model
from opdynamics.utils.types import TransitionProbabilities


class TestMutation(ut.TestCase):
    model_params = {
        "graph_type": "barabasi",
        "network_size": 500,
        "memory_size": 256,
        "code_length": 5,
        "kappa": 0,
        "lambd": 0,
        "alpha": 0,
        "omega": 0,
        "gamma": 0,
        "preferential_attachment": 2,
        "polarization_grouping_type": 0
    }

    def test_mutate(self):
        def f(i_bit: int, tp: TransitionProbabilities) -> float:
            i_bit = 0
            o_bits = []
            for _ in range(10**5):
                o_bits.append(mutate(i_bit, tp))
            mutate_mean = np.mean(o_bits)
            mutate_mean = round(mutate_mean, 1)
            return mutate_mean

        tp = {
            0: 0.5,
            1: 0.5
        }
        mutate_mean = f(0, tp)
        self.assertEqual(
            mutate_mean,
            0.5,
            "The mean of the output bits should be 0.5."
        )
        mutate_mean = f(1, tp)
        self.assertEqual(
            mutate_mean,
            0.5,
            "The mean of the output bits should be 0.5."
        )

        tp = {
            0: 0.1,
            1: 0.9
        }
        mutate_mean = f(0, tp)
        self.assertEqual(
            mutate_mean,
            0.9,
            "The mean of the output bits should be 0.9."
        )
        mutate_mean = f(1, tp)
        self.assertEqual(
            mutate_mean,
            0.1,
            "The mean of the output bits should be 0.1."
        )

        tp = {
            0: 0.9,
            1: 0.1
        }
        mutate_mean = f(0, tp)
        self.assertEqual(
            mutate_mean,
            0.1,
            "The mean of the output bits should be 0.1."
        )
        mutate_mean = f(1, tp)
        self.assertEqual(
            mutate_mean,
            0.9,
            "The mean of the output bits should be 0.9."
        )

    def test_distort(self):
        def f(i_code: np.ndarray, tp: TransitionProbabilities) -> float:
            dist_sum = []
            for _ in range(10**5):
                dist_sum.append(distort(i_code, tp).sum())
            mutate_mean = np.mean(dist_sum)
            mutate_mean = round(mutate_mean, 1)
            return mutate_mean

        code = np.array([0, 1, 0, 1, 0])

        tp = {
            0: 0.5,
            1: 0.5
        }
        mean = f(code, tp)
        self.assertEqual(
            mean,
            2.5,
            "The mean of the output bits should be 2.5."
        )

        tp = {
            0: 0.1,
            1: 0.9
        }
        mean = f(code, tp)
        self.assertEqual(
            mean,
            0.5,
            "The mean of the output bits should be 0.5."
        )

        tp = {
            0: 0.9,
            1: 0.1
        }
        mean = f(code, tp)
        self.assertEqual(
            mean,
            4.5,
            "The mean of the output bits should be 4.5."
        )

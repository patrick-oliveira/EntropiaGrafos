import unittest as ut

from opdynamics.utils.reading_tools import param_to_hash
from opdynamics.simulation.utils import get_param_tuple


class TestReadingTools(ut.TestCase):
    test_params = [
        {
            "simulation_parameters": {
                'graph_type': 'barabasi',
                'network_size': 500,
                'memory_size': 256,
                'code_length': 5,
                'kappa': 0,
                'lambd': 0,
                'alpha': 0.6,
                'omega': 0,
                'gamma': 0,
                'preferential_attachment': 2,
                'polarization_type': 0,
            },
            "general_parameters": {
                'T': 1000,
                'num_repetitions': 100,
                'early_stop': True,
                'epsilon': 1e-08,
                'results_path': 'results/single_polarized_group/'
            }
        },
        {
            "simulation_parameters": {
                'graph_type': 'barabasi',
                'network_size': 500,
                'memory_size': 256,
                'code_length': 5,
                'kappa': 0,
                'lambd': 1,
                'alpha': 0.6,
                'omega': 0,
                'gamma': 0,
                'preferential_attachment': 2,
                'polarization_type': 0,
            },
            "general_parameters": {
                'T': 1000,
                'num_repetitions': 100,
                'early_stop': True,
                'epsilon': 1e-08,
                'results_path': 'results/single_polarized_group/'
            }
        },
        {
            "simulation_parameters": {
                'graph_type': 'barabasi',
                'network_size': 500,
                'memory_size': 256,
                'code_length': 5,
                'kappa': 0,
                'lambd': 1,
                'alpha': 0.6,
                'omega': 0,
                'gamma': 0,
                'preferential_attachment': 2,
                'polarization_type': 0,
                "distribution": "from_list",
                "base_list": [0, 5, 10]
            },
            "general_parameters": {
                'T': 1000,
                'num_repetitions': 100,
                'early_stop': True,
                'epsilon': 1e-08,
                'results_path': 'results/single_polarized_group/'
            }
        }
    ]

    expected_hash = [
        'dde7c6e3f95fd0b5ea2a9c1bcd9bd6616a71042601155dae7ac7c86ec93e8c46',
        'ff0178775322fe26d404ab8c345019736fef275b2a082534217eb9e427ff0d83',
        'b3eb5e8ac447e35091defa2857ec823d6f809deb9d87442cf9fcdbedda939612'
    ]

    def test_hashing(self):
        for k, param in enumerate(self.test_params):
            self.assertEqual(
                param_to_hash(get_param_tuple(param)),
                self.expected_hash[k]
            )

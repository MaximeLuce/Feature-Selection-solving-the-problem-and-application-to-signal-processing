import unittest
import numpy as np
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import DecodedFeatureSelectionProblem
from app.decoders import SigmaDecoder


class DummyMaskProblem:
    def __init__(self, num_features):
        self.num_features = num_features
        self.evaluations_count = 0

    def reset_counter(self):
        self.evaluations_count = 0

    def evaluate_batch(self, masks):
        masks = np.asarray(masks, dtype=int)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        self.evaluations_count += masks.shape[0]
        return masks.sum(axis=1).astype(float)

    def evaluate(self, mask):
        return float(self.evaluate_batch([mask])[0])


class DeterministicDecoder:
    def bounds(self, num_features):
        return np.tile(np.array([[0.0, 1.0]], dtype=float), (num_features, 1))

    def decode_batch(self, vectors, num_features, rng):
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return (vectors[:, :num_features] >= 0.5).astype(int)


class FixedPopulationDE(DifferentialEvolution):
    def _generate_population(self):
        self.population = np.array([
            [0.9, 0.1, 0.2],
            [0.8, 0.8, 0.1],
            [0.3, 0.7, 0.9],
            [0.6, 0.4, 0.4],
        ], dtype=float)

    def _mutate(self, fitness):
        return self.population.copy()

    def _crossover(self, mutants):
        return mutants


class TestSinglePassPopulationDecode(unittest.TestCase):
    def test_feature_selection_problem_returns_masks_used_for_fitness(self):
        problem = DecodedFeatureSelectionProblem(
            DummyMaskProblem(num_features=3), DeterministicDecoder()
        )
        vectors = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.4]], dtype=float)
        result = problem.evaluate_population(vectors)
        expected_masks = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        np.testing.assert_array_equal(result.masks, expected_masks)
        np.testing.assert_array_equal(result.fitness, problem.problem.evaluate_batch(expected_masks))

    def test_de_monitoring_uses_evaluated_masks(self):
        problem = DecodedFeatureSelectionProblem(
            DummyMaskProblem(num_features=3), DeterministicDecoder()
        )
        algorithm = FixedPopulationDE(problem, config={
            "popsize": 4, "max_generations": 1, "target_fitness": -1.0,
            "monitoring": {
                "metrics": ["best_fitness", "evaluation_count", "fitness_mean",
                            "fitness_worst", "best_mask", "worst_mask", "population_diversity"],
            },
        })
        result = algorithm.run()
        expected_masks = result.history[0]["best_mask"]
        self.assertIsNotNone(result.history[0]["best_mask"])
        self.assertIsNotNone(result.history[0]["worst_mask"])
        self.assertIn("population_diversity", result.history[0])
        self.assertEqual(result.history[0]["evaluations_count"], 4)
        self.assertEqual(result.history[1]["evaluations_count"], 8)


if __name__ == "__main__":
    unittest.main()

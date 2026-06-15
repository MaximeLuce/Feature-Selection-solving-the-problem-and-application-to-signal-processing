import numpy as np

from .results import PopulationEvaluation


class BenchmarkProblem:
    def __init__(self, name, bounds, minimum=None, minimum_value=None):
        self.name = name
        self.bounds = np.asarray(bounds, dtype=float)
        self.minimum = None if minimum is None else np.asarray(minimum, dtype=float)
        self.minimum_value = minimum_value
        self.evaluations_count = 0

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def num_features(self):
        return len(self.bounds)

    def __call__(self, x):
        return self.objective(x)

    def reset(self):
        self.evaluations_count = 0

    def reset_counter(self):
        self.evaluations_count = 0

    def objective(self, x):
        raise NotImplementedError

    def evaluate(self, x):
        """Single evaluation — accepts any array-like input."""
        self.evaluations_count += 1
        return float(self.objective(np.asarray(x)))

    def evaluate_batch(self, masks):
        """Batch evaluation for binary masks (SA/NBPSO contract)."""
        masks = np.asarray(masks, dtype=float)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        self.evaluations_count += masks.shape[0]
        return np.array([self.objective(m) for m in masks])

    def evaluate_population(self, vectors):
        """Batch evaluation for continuous vectors (DE contract)."""
        vectors = np.asarray(vectors, dtype=float)
        fitness = np.array([self.objective(x) for x in vectors])
        self.evaluations_count += vectors.shape[0]
        return PopulationEvaluation(fitness=fitness, masks=None)


class SphereProblem(BenchmarkProblem):
    def __init__(self):
        super().__init__(
            name="Sphere",
            bounds=[[-5.12, 5.12], [-5.12, 5.12]],
            minimum=[0, 0],
            minimum_value=0,
        )

    def objective(self, x):
        return np.sum(x**2)


class RastriginProblem(BenchmarkProblem):
    def __init__(self):
        super().__init__(
            name="Rastrigin",
            bounds=[[-5.12, 5.12], [-5.12, 5.12]],
            minimum=[0, 0],
            minimum_value=0,
        )

    def objective(self, x):
        return 10 * len(x) + np.sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])


class AckleyProblem(BenchmarkProblem):
    def __init__(self):
        super().__init__(
            name="Ackley",
            bounds=[[-5.12, 5.12], [-5.12, 5.12]],
            minimum=[0, 0],
            minimum_value=0,
        )

    def objective(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * np.array(x)))
        return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)


class OneMaxProblem(BenchmarkProblem):
    """Maximize count of 1s in a binary vector. Fitness = n - sum(mask), so 0 = perfect."""

    def __init__(self, num_features=100):
        super().__init__(
            name="OneMax",
            bounds=[[0, 1]] * num_features,
            minimum=[1] * num_features,
            minimum_value=0,
        )

    def objective(self, x):
        arr = np.asarray(x, dtype=float)
        # Round continuous values to 0/1 for consistent behavior
        bits = (arr > 0.5).astype(int)
        return self.dim - np.sum(bits)

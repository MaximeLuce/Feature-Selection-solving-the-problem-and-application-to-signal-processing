import numpy as np


class Problem:
    def __init__(self, name, bounds, minimum=None, minimum_value=None):
        self.name = name
        self.bounds = np.asarray(bounds, dtype=float)
        self.minimum = None if minimum is None else np.asarray(minimum, dtype=float)
        self.minimum_value = minimum_value
        self.evaluations_count = 0

    @property
    def dim(self):
        return len(self.bounds)

    def __call__(self, x):
        return self.objective(x)

    def reset(self):
        self.evaluations_count = 0

    def objective(self, x):
        raise NotImplementedError

    def evaluate(self, x):
        self.evaluations_count += 1
        return self.objective(x)


class SphereProblem(Problem):
    def __init__(self):
        super().__init__(
            name="Sphere",
            bounds=[[-5.12, 5.12], [-5.12, 5.12]],
            minimum=[0, 0],
            minimum_value=0,
        )

    def objective(self, x):
        return np.sum(x**2)


class RastriginProblem(Problem):
    def __init__(self):
        super().__init__(
            name="Rastrigin",
            bounds=[[-5.12, 5.12], [-5.12, 5.12]],
            minimum=[0, 0],
            minimum_value=0,
        )

    def objective(self, x):
        return 10 * len(x) + np.sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])


class AckleyProblem(Problem):
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

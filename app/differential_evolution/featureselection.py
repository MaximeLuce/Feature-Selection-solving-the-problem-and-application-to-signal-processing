import numpy as np
from app.Utilities.masks import repair_zero_masks
from .results import PopulationEvaluation


class DecodedFeatureSelectionProblem:
    def __init__(self, problem, decoder):
        self.problem = problem
        self.decoder = decoder
        self.bounds = decoder.bounds(problem.num_features)
        self.rng = np.random.default_rng()

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def evaluations_count(self):
        return self.problem.evaluations_count

    def reset(self):
        self.problem.reset_counter()
        self.rng = np.random.default_rng()

    def evaluate_population(self, population):
        vectors = np.asarray(population, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        masks = self.decoder.decode_batch(vectors, self.problem.num_features, self.rng)
        masks = repair_zero_masks(masks, self.rng)
        fitness = self.problem.evaluate_batch(masks)
        return PopulationEvaluation(fitness=fitness, masks=masks)
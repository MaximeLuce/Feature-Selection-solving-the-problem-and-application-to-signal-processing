# app/differential_evolution/algorithm.py

import numpy as np
from app.monitoring import Monitor
from app.differential_evolution.results import DEResult, PopulationEvaluation


STRATEGY_METHODS = {
    "rand/1": ("_mutate_rand_1", 3),
    "best/1": ("_mutate_best_1", 2),
    "rand/2": ("_mutate_rand_2", 5),
    "best/2": ("_mutate_best_2", 4),
    "current-to-best/1": ("_mutate_current_to_best_1", 2),
    "current-to-rand/1": ("_mutate_current_to_rand_1", 3),
}

DEFAULT_CONFIG = {
    "popsize": 100,
    "F1": 0.5,
    "F2": 0.5,
    "CR": 0.7,
    "strategy": "rand/1",
    "max_generations": 1000,
    "seed": None,
    "target_fitness": 0.01,
    "monitoring": {
        "metrics": [],
    },
}


class DifferentialEvolution:
    def __init__(self, problem, config=None):
        config = config or {}
        self.problem = problem

        self.popsize = config.get("popsize", DEFAULT_CONFIG["popsize"])
        self.F1 = config.get("F1", DEFAULT_CONFIG["F1"])
        self.F2 = config.get("F2", 1-self.F1)
        self.CR = config.get("CR", DEFAULT_CONFIG["CR"])
        self.strategy = config.get("strategy", DEFAULT_CONFIG["strategy"])
        self.max_generations = config.get("max_generations", DEFAULT_CONFIG["max_generations"])
        self.seed = config.get("seed", DEFAULT_CONFIG["seed"])
        self.target_fitness = config.get("target_fitness", DEFAULT_CONFIG["target_fitness"])

        if self.popsize < 3:
            raise ValueError(f"popsize must be >= 3, got {self.popsize}.")
        if self.max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {self.max_generations}.")
        if not (0 <= self.CR <= 1):
            raise ValueError(f"CR must be in [0, 1], got {self.CR}.")

        # Resolve strategy
        if self.strategy not in STRATEGY_METHODS:
            valid = ", ".join(STRATEGY_METHODS)
            raise ValueError(f"Unknown strategy: {self.strategy}. Use one of: {valid}.")
        method_name, self.strategy_samples = STRATEGY_METHODS[self.strategy]
        self._mutate_strategy = getattr(self, method_name)

        if self.popsize <= max(3, self.strategy_samples):
            raise ValueError(
                f"Population size {self.popsize} is too small"
                f"{self.strategy}; need more than {max(3, self.strategy_samples):} individuals."
            )

        # State
        self.bounds = self.problem.bounds
        self.dim = self.problem.dim
        
        self.reset()

        self.monitor = Monitor(config.get("monitoring", DEFAULT_CONFIG["monitoring"]))

    def reset(self):
        self.generation = 0
        self.best_vector = np.empty(0)
        self.best_mask = np.empty(0, dtype=int)
        self.best_fitness = np.inf
        self.population = np.empty((self.popsize, self.dim))
        self.masks = None
        self.generator = np.random.default_rng(self.seed)
        self.problem.reset()

    def _generate_population(self):
        self.population = self.generator.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.popsize, self.dim),
        )

    def _evaluate_population(self, population):
        return self.problem.evaluate_population(population)

    def _select(self, trials, trial_masks, fitness, masks, trial_fitness):
        improved = trial_fitness < fitness
        self.population[improved] = trials[improved]
        if masks is not None and trial_masks is not None:
            masks[improved] = trial_masks[improved]
        fitness[improved] = trial_fitness[improved]
        return fitness, masks

    def _update_best(self, fitness, masks):
        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(fitness[best_idx])
            self.best_vector = self.population[best_idx].copy()
            if masks is not None:
                self.best_mask = masks[best_idx].copy()

    # --- Strategy methods ---

    def _mutate_rand_1(self, ids, best_idx):
        return self.population[ids[:, 0]] + self.F1 * (
            self.population[ids[:, 1]] - self.population[ids[:, 2]]
        )

    def _mutate_best_1(self, ids, best_idx):
        return self.population[best_idx] + self.F1 * (
            self.population[ids[:, 0]] - self.population[ids[:, 1]]
        )

    def _mutate_rand_2(self, ids, best_idx):
        return (
            self.population[ids[:, 0]]
            + self.F1 * (self.population[ids[:, 1]] - self.population[ids[:, 2]])
            + self.F2 * (self.population[ids[:, 3]] - self.population[ids[:, 4]])
        )

    def _mutate_best_2(self, ids, best_idx):
        return (
            self.population[best_idx]
            + self.F1 * (self.population[ids[:, 0]] - self.population[ids[:, 1]])
            + self.F2 * (self.population[ids[:, 2]] - self.population[ids[:, 3]])
        )

    def _mutate_current_to_best_1(self, ids, best_idx):
        return (
            self.population
            + self.F1 * (self.population[best_idx] - self.population)
            + self.F2 * (self.population[ids[:, 0]] - self.population[ids[:, 1]])
        )

    def _mutate_current_to_rand_1(self, ids, best_idx):
        return (
            self.population
            + self.F1 * (self.population[ids[:, 0]] - self.population)
            + self.F2 * (self.population[ids[:, 1]] - self.population[ids[:, 2]])
        )

    def _mutate(self, fitness):
        if self.popsize <= self.strategy_samples:
            raise ValueError(
                f"Population size {self.popsize} is too small for strategy "
                f"{self.strategy}; need more than {self.strategy_samples} individuals."
            )
        # pop x pop scores
        scores = self.generator.random((self.popsize, self.popsize))
        # guarantee uniqueness - diag to inf so the individual can't sample itself
        scores[np.arange(self.popsize), np.arange(self.popsize)] = np.inf
        # take k best samples
        ids = np.argpartition(
            scores,
            self.strategy_samples - 1,
            axis=1,
        )[:, :self.strategy_samples]
        best_idx = np.argmin(fitness)
        return self._mutate_strategy(ids, best_idx)

    def _crossover(self, mutants):
        N, D = self.population.shape
        mask = self.generator.random((N, D)) < self.CR
        j_rand = self.generator.integers(0, D, size=N)
        mask[np.arange(N), j_rand] = True
        return np.where(mask, mutants, self.population)

    def _stop(self):
        return (
            self.generation >= self.max_generations
            or self.best_fitness < self.target_fitness
        )

    def run(self):
        self.reset()
        self.monitor.start()

        # Initialize and evaluate
        self._generate_population()
        eval_result = self._evaluate_population(self.population)
        self._update_best(eval_result.fitness, eval_result.masks)

        self.monitor.record_population(eval_result.fitness, eval_result.masks)

        # Main loop
        while not self._stop():
            self.generation += 1
            mutants = self._mutate(eval_result.fitness)
            trials = self._crossover(mutants)
            trials = np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])

            eval_result = self._evaluate_population(trials)
            fitness, masks = self._select(
                trials, eval_result.masks,
                eval_result.fitness, eval_result.masks,
                eval_result.fitness,
            )
            eval_result = PopulationEvaluation(fitness=fitness, masks=masks)
            self._update_best(fitness, masks)

            self.monitor.record_population(fitness, masks)

        self.monitor.finish(self.generation, self.best_fitness, self.problem.evaluations_count)
        monitor_result = self.monitor.build_result_fields()

        return DEResult(
            best_vector=self.best_vector,
            best_mask=self.best_mask,
            best_fitness=self.best_fitness,
            generations=self.generation,
            evaluations=self.problem.evaluations_count,
            wall_ns=monitor_result["wall_ns"],
            cpu_ns=monitor_result["cpu_ns"],
            history=monitor_result["history"],
        )

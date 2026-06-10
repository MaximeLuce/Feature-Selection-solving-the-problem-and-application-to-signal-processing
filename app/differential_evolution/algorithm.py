# algorithm.py

import numpy as np
from .monitoring import Monitor


            
class Result:
    def __init__(self, best: np.ndarray, best_fitness: float):
        self.best = best
        self.best_fitness = best_fitness
        
class Algorithm:
    def __init__(self, problem):
        self.problem = problem
        self.best = None
        self.best_fitness = np.inf

    def reset(self):
        self.best = None
        self.best_fitness = np.inf
        
    def run(self) -> Result:
        raise NotImplementedError

class DifferentialEvolution(Algorithm):
    STRATEGY_METHODS = {
        'rand/1': ('_mutate_rand_1', 3),
        'best/1': ('_mutate_best_1', 2),
        'rand/2': ('_mutate_rand_2', 5),
        'best/2': ('_mutate_best_2', 4),
        'current-to-best/1': ('_mutate_current_to_best_1', 2),
        'current-to-rand/1': ('_mutate_current_to_rand_1', 3),
    }
    
    DEFAULT_CONFIG = {
        'popsize': 100,
        'F1': 0.5,
        'F2': 0.5,
        'CR': 0.7,
        'strategy': 'rand/1',
        'max_generations': 1000,
        'seed': None,
    }

    def __init__(self, problem, config=None):
        super().__init__(problem)
        config = config or {}
        self.popsize = config.get('popsize', self.DEFAULT_CONFIG['popsize'])
        self.F1 = config.get('F1', self.DEFAULT_CONFIG['F1'])
        self.F2 = config.get('F2', self.DEFAULT_CONFIG['F2'])
        self.CR = config.get('CR', self.DEFAULT_CONFIG['CR'])
        self.strategy = config.get('strategy', self.DEFAULT_CONFIG['strategy'])
        self.max_generations = config.get('max_generations', self.DEFAULT_CONFIG['max_generations'])
        self.seed = config.get('seed', self.DEFAULT_CONFIG['seed'])
        self.generation = 0
        self.generator = np.random.default_rng(self.seed)
        self.bounds = self.problem.bounds
        self.dim = self.problem.dim
        self.population = np.empty((self.popsize, self.dim))
        self.strategy_method, self.strategy_samples = self._get_strategy(self.strategy)
        self.monitor = Monitor()
        
    def reset(self):
        super().reset()
        self.generation = 0
        self.generator = np.random.default_rng(self.seed)
        self.population = np.empty((self.popsize, self.dim))
        self.problem.reset()

    def _get_strategy(self, name):
        if name not in self.STRATEGY_METHODS:
            valid_strategies = ', '.join(self.STRATEGY_METHODS)
            raise ValueError(f'Unknown strategy: {name}. Use one of: {valid_strategies}')

        method_name, samples_count = self.STRATEGY_METHODS[name]
        return getattr(self, method_name), samples_count
    
    def _stop(self):
        return self.generation >= self.max_generations or self.best_fitness < 0.01
    
    def _generate_population(self):
        self.population = self.generator.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.popsize, self.dim))
    
    def _evaluate_population(self, population):
        return np.array([self.problem.evaluate(x) for x in population])

    def _mutate_rand_1(self, ids, best_idx):
        return self.population[ids[:, 0]] + self.F1 * (self.population[ids[:, 1]] - self.population[ids[:, 2]])

    def _mutate_best_1(self, ids, best_idx):
        return self.population[best_idx] + self.F1 * (self.population[ids[:, 0]] - self.population[ids[:, 1]])

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
        ids = self.generator.integers(0, self.popsize, size=(self.popsize, self.strategy_samples))
        best_idx = np.argmin(fitness)
        return self.strategy_method(ids, best_idx)

    def _crossover(self, mutants):
        N, D = self.population.shape
        mask = self.generator.random((N, D)) < self.CR
        j_rand = self.generator.integers(0, D, size=N)
        mask[np.arange(N), j_rand] = True
        return np.where(mask, mutants, self.population)

    def _select(self, trials, fitness, trial_fitness):
        improved = trial_fitness < fitness
        self.population[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]
        return fitness

    def _update_best(self, fitness):
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best = self.population[best_idx].copy()
            self.best_fitness = float(fitness[best_idx])

    def run(self):
        self.reset()
        self._generate_population()
        fitness = self._evaluate_population(self.population)
        self._update_best(fitness)

        while not self._stop():
            self.generation += 1
            mutants = self._mutate(fitness)
            trials = self._crossover(mutants)
            trials = np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])
            trial_fitness = self._evaluate_population(trials)
            fitness = self._select(trials, fitness, trial_fitness)
            self._update_best(fitness)
        
        return Result(self.best, self.best_fitness)
    

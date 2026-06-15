import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.monitoring import Monitor, MonitoringMetric
from app.OptimizationAlgorithm.OptimizationAlgorithm import OptimizationAlgorithm
from app.Problem.Individual import Individual
from app.Utilities.masks import repair_zero_mask


@dataclass
class SAResult:
    best_mask: list[int]
    best_fitness: float
    evaluations: int
    expected_final_temperature: float
    final_temperature: float
    wall_ns: int | None = None
    cpu_ns: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class SimulatedAnnealing(OptimizationAlgorithm):
    DEFAULT_MONITORING = {
        "metrics": [],
        "record_every": 1,
    }

    def __init__(
        self,
        problem,
        max_evaluations=10000,
        initial_temp=1000.0,
        cooling_rate=0.99,
        seed=None,
        min_temperature=1e-10,
        monitoring=None,
    ):
        super().__init__(problem)
        self.max_evaluations = max_evaluations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.seed = seed
        self.min_temperature = min_temperature
        monitoring_config = dict(self.DEFAULT_MONITORING)
        monitoring_config.update(monitoring or {})
        self.monitor = Monitor(monitoring_config)
        self.generator = np.random.default_rng(self.seed)

    def _expected_final_temperature(self):
        cooling_steps = max(0, self.max_evaluations - 1)
        return max(
            self.min_temperature,
            float(self.initial_temp * (self.cooling_rate ** cooling_steps)),
        )

    def _random_mask(self):
        mask = self.generator.integers(0, 2, size=self.problem.num_features)
        return repair_zero_mask(mask, self.generator)

    def generate_neighbor(self, current_individual):
        neighbor_mask = np.asarray(current_individual.features_mask, dtype=int).copy()
        idx_to_flip = int(self.generator.integers(0, len(neighbor_mask)))
        neighbor_mask[idx_to_flip] = 1 - neighbor_mask[idx_to_flip]
        neighbor_mask = repair_zero_mask(neighbor_mask, self.generator)
        return Individual(neighbor_mask.tolist())

    def run(self):
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive.")
        if self.initial_temp <= 0:
            raise ValueError("initial_temp must be positive.")
        if not 0 < self.cooling_rate < 1:
            raise ValueError("cooling_rate must be between 0 and 1.")
        if self.min_temperature <= 0:
            raise ValueError("min_temperature must be positive.")

        self.problem.reset_counter()
        self.generator = np.random.default_rng(self.seed)
        self.monitor.record_every = max(1, self.max_evaluations // 100)

        expected_final_temperature = self._expected_final_temperature()
        self.monitor.start()

        current_sol = Individual(self._random_mask().tolist())
        current_sol.evaluate(self.problem)
        best_sol = copy.deepcopy(current_sol)
        temperature = float(self.initial_temp)
        accepted_moves = 0

        self.monitor.record_annealing(
            generation=0,
            best_fitness=float(best_sol.fitness),
            current_fitness=float(current_sol.fitness),
            best_mask=list(best_sol.features_mask),
            current_mask=list(current_sol.features_mask),
            temperature=temperature,
            evaluations_count=self.problem.evaluations_count,
            acceptance_rate=0.0,
        )

        for generation in range(1, self.max_evaluations):
            neighbor = self.generate_neighbor(current_sol)
            neighbor.evaluate(self.problem)

            delta_f = neighbor.fitness - current_sol.fitness
            accepted = False

            if delta_f < 0:
                current_sol = neighbor
                accepted = True
                if neighbor.fitness < best_sol.fitness:
                    best_sol = copy.deepcopy(neighbor)
            else:
                probability = math.exp(-delta_f / temperature)
                if float(self.generator.random()) < probability:
                    current_sol = neighbor
                    accepted = True

            if accepted:
                accepted_moves += 1

            temperature = max(self.min_temperature, temperature * self.cooling_rate)
            self.monitor.record_annealing(
                generation=generation,
                best_fitness=float(best_sol.fitness),
                current_fitness=float(current_sol.fitness),
                best_mask=list(best_sol.features_mask),
                current_mask=list(current_sol.features_mask),
                temperature=float(temperature),
                evaluations_count=self.problem.evaluations_count,
                acceptance_rate=float(accepted_moves / generation),
            )

        self.monitor.finish(
            generation=max(0, self.max_evaluations - 1),
            best_fitness=float(best_sol.fitness),
            evaluations_count=self.problem.evaluations_count,
        )
        monitor_result = self.monitor.build_result_fields()

        return SAResult(
            best_mask=list(best_sol.features_mask),
            best_fitness=float(best_sol.fitness),
            evaluations=self.problem.evaluations_count,
            expected_final_temperature=expected_final_temperature,
            final_temperature=float(temperature),
            wall_ns=monitor_result["wall_ns"],
            cpu_ns=monitor_result["cpu_ns"],
            history=monitor_result["history"],
        )

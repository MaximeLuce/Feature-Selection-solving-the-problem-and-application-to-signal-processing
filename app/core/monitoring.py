import time
from enum import StrEnum
from typing import Any

import numpy as np


class MonitoringMetric(StrEnum):
    ELAPSED_TIME_NS = "elapsed_time_ns"
    CPU_TIME_NS = "cpu_time_ns"
    EVALUATION_COUNT = "evaluation_count"
    BEST_FITNESS = "best_fitness"
    FITNESS_MEAN = "fitness_mean"
    FITNESS_WORST = "fitness_worst"
    FITNESS_STD = "fitness_std"
    BEST_MASK = "best_mask"
    WORST_MASK = "worst_mask"
    HAMMING_DISTANCE = "hamming_distance"
    POPULATION_DIVERSITY = "population_diversity"
    TEMPERATURE = "temperature"
    CURRENT_FITNESS = "current_fitness"
    CURRENT_MASK = "current_mask"
    ACCEPTANCE_RATE = "acceptance_rate"
    EXPECTED_FINAL_TEMPERATURE = "expected_final_temperature"
    FINAL_TEMPERATURE = "final_temperature"


class Monitor:
    def __init__(self, config=None):
        config = config or {}
        self.metrics = {
            MonitoringMetric(m) for m in config.get("metrics", [])
        }
        self.record_every = max(1, int(config.get("record_every", 1)))
        self.reset()

    @staticmethod
    def _hamming_distance(masks: np.ndarray) -> float:
        """Average Hamming distance from the centroid (O(n*d))."""
        masks = np.asarray(masks, dtype=float)
        centroid = (masks.mean(axis=0) > 0.5).astype(int)
        return float(np.mean(np.sum(masks != centroid, axis=1)))

    def reset(self):
        self.wall_start_ns: int | None = None
        self.cpu_start_ns: int | None = None
        self.wall_ns: int | None = None
        self.cpu_ns: int | None = None
        self.generation = 0
        self.population_evaluations = 0
        self.records: list[dict[str, Any]] = []
        self.seen_best_fitness: float = float("inf")
        self.seen_best_mask: np.ndarray | None = None

    def start(self):
        self.reset()
        if MonitoringMetric.ELAPSED_TIME_NS in self.metrics:
            self.wall_start_ns = time.perf_counter_ns()
        if MonitoringMetric.CPU_TIME_NS in self.metrics:
            self.cpu_start_ns = time.thread_time_ns()

    def record_population(self, fitness: np.ndarray, masks: np.ndarray | None = None):
        fitness = np.asarray(fitness, dtype=float)
        self.population_evaluations += fitness.shape[0]
        self.generation += 1

        if self.generation % self.record_every != 0:
            return

        record: dict[str, Any] = {"event": "generation_completed", "generation": self.generation}

        if MonitoringMetric.EVALUATION_COUNT in self.metrics:
            record["evaluations_count"] = self.population_evaluations
        if MonitoringMetric.FITNESS_MEAN in self.metrics:
            record["fitness_avg"] = float(fitness.mean())
        if MonitoringMetric.FITNESS_WORST in self.metrics:
            record["fitness_worst"] = float(fitness.max())
        if MonitoringMetric.FITNESS_STD in self.metrics:
            record["fitness_std"] = float(fitness.std())

        best_idx = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))

        if masks is not None:
            masks = np.asarray(masks)
            if MonitoringMetric.WORST_MASK in self.metrics:
                record["worst_mask"] = masks[worst_idx].tolist()
            gen_best_fitness = float(fitness[best_idx])
            if gen_best_fitness < self.seen_best_fitness:
                self.seen_best_fitness = gen_best_fitness
                self.seen_best_mask = masks[best_idx].tolist()

        if MonitoringMetric.BEST_FITNESS in self.metrics:
            record["best_fitness"] = self.seen_best_fitness
        if MonitoringMetric.BEST_MASK in self.metrics:
            record["best_mask"] = list(self.seen_best_mask) if self.seen_best_mask is not None else None
        if MonitoringMetric.HAMMING_DISTANCE in self.metrics and masks is not None:
            record["population_diversity"] = self._hamming_distance(masks)

        self.records.append(record)

    def record_annealing(self, generation, best_fitness, current_fitness,
                         best_mask, current_mask, temperature,
                         evaluations_count, acceptance_rate=0.0):
        record: dict[str, Any] = {"event": "generation_completed", "generation": generation}

        if MonitoringMetric.BEST_FITNESS in self.metrics:
            record["best_fitness"] = best_fitness
        if MonitoringMetric.CURRENT_FITNESS in self.metrics:
            record["current_fitness"] = current_fitness
        if MonitoringMetric.BEST_MASK in self.metrics:
            record["best_mask"] = best_mask
        if MonitoringMetric.CURRENT_MASK in self.metrics:
            record["current_mask"] = current_mask
        if MonitoringMetric.TEMPERATURE in self.metrics:
            record["temperature"] = temperature
        if MonitoringMetric.ACCEPTANCE_RATE in self.metrics:
            record["acceptance_rate"] = acceptance_rate
        if MonitoringMetric.EVALUATION_COUNT in self.metrics:
            record["evaluations_count"] = evaluations_count

        self.records.append(record)

    def finish(self, generation, best_fitness, evaluations_count):
        if MonitoringMetric.ELAPSED_TIME_NS in self.metrics and self.wall_start_ns is not None:
            self.wall_ns = time.perf_counter_ns() - self.wall_start_ns
        if MonitoringMetric.CPU_TIME_NS in self.metrics and self.cpu_start_ns is not None:
            self.cpu_ns = time.thread_time_ns() - self.cpu_start_ns

        record: dict[str, Any] = {"event": "run_end", "generation": generation}
        if MonitoringMetric.BEST_FITNESS in self.metrics:
            record["best_fitness"] = best_fitness
        if MonitoringMetric.EVALUATION_COUNT in self.metrics:
            record["evaluations_count"] = evaluations_count
        if MonitoringMetric.ELAPSED_TIME_NS in self.metrics:
            record["elapsed_time_ns"] = self.wall_ns
        if MonitoringMetric.CPU_TIME_NS in self.metrics:
            record["cpu_time_ns"] = self.cpu_ns
        self.records.append(record)

    def build_result_fields(self):
        return {"wall_ns": self.wall_ns, "cpu_ns": self.cpu_ns, "history": self.records}

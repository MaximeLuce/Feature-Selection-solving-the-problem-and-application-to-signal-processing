import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MonitoringEvent(StrEnum):
    RUN_START = "run_start"
    GENERATION_COMPLETED = "generation_completed"
    FITNESS_EVALUATED = "fitness_evaluated"
    RUN_END = "run_end"


class MonitoringMetric(StrEnum):
    ELAPSED_TIME_NS = "elapsed_time_ns"
    CPU_TIME_NS = "cpu_time_ns"
    EVALUATION_COUNT = "evaluation_count"
    BEST_FITNESS = "best_fitness"
    FITNESS_MEAN = "fitness_mean"
    FITNESS_WORST = "fitness_worst"
    FITNESS_STD = "fitness_std"


@dataclass
class Event:
    name: MonitoringEvent
    data: dict[str, Any]


class Monitor:
    def __init__(self, config=None):
        config = config or {}
        self.metrics = {
            MonitoringMetric(metric)
            for metric in config.get("metrics", [])
        }
        self.generation_metrics = self.metrics & {
            MonitoringMetric.BEST_FITNESS,
            MonitoringMetric.EVALUATION_COUNT,
            MonitoringMetric.FITNESS_MEAN,
            MonitoringMetric.FITNESS_WORST,
            MonitoringMetric.FITNESS_STD,
        }
        self.run_end_metrics = self.metrics & {
            MonitoringMetric.ELAPSED_TIME_NS,
            MonitoringMetric.CPU_TIME_NS,
            MonitoringMetric.EVALUATION_COUNT,
            MonitoringMetric.BEST_FITNESS,
        }
        self.wall_start_ns: int | None = None
        self.cpu_start_ns: int | None = None
        self.wall_ns: int | None = None
        self.cpu_ns: int | None = None
        self.fitness_evaluations: int = 0
        self.records: list[dict[str, Any]] = []

    def emit(self, event):
        if event.name == MonitoringEvent.RUN_START:
            if MonitoringMetric.ELAPSED_TIME_NS in self.metrics:
                self.wall_start_ns = time.perf_counter_ns()
                self.wall_ns = None
            if MonitoringMetric.CPU_TIME_NS in self.metrics:
                self.cpu_start_ns = time.process_time_ns()
                self.cpu_ns = None
            return

        if event.name == MonitoringEvent.GENERATION_COMPLETED:
            if not self.generation_metrics:
                return
            record = {
                "event": event.name.value,
                "generation": event.data.get("generation"),
            }
            if MonitoringMetric.BEST_FITNESS in self.generation_metrics:
                record["best_fitness"] = event.data.get("best_fitness")
            if MonitoringMetric.EVALUATION_COUNT in self.generation_metrics:
                record["evaluations_count"] = event.data.get("evaluations_count")
            fitness = event.data.get("fitness")
            if fitness is not None:
                if MonitoringMetric.FITNESS_MEAN in self.generation_metrics:
                    record["fitness_avg"] = float(fitness.mean())
                if MonitoringMetric.FITNESS_WORST in self.generation_metrics:
                    record["fitness_worst"] = float(fitness.max())
                if MonitoringMetric.FITNESS_STD in self.generation_metrics:
                    record["fitness_std"] = float(fitness.std())
            self.records.append(record)
            return

        if event.name == MonitoringEvent.RUN_END:
            if MonitoringMetric.ELAPSED_TIME_NS in self.metrics and self.wall_start_ns is not None:
                self.wall_ns = time.perf_counter_ns() - self.wall_start_ns
            if MonitoringMetric.CPU_TIME_NS in self.metrics and self.cpu_start_ns is not None:
                self.cpu_ns = time.process_time_ns() - self.cpu_start_ns
            if not self.run_end_metrics:
                return
            record : dict[str, Any] = {
                "event": event.name.value,
            }
            generation = event.data.get("generation")
            if generation is not None:
                record["generation"] = generation
            if MonitoringMetric.BEST_FITNESS in self.run_end_metrics:
                record["best_fitness"] = event.data.get("best_fitness")
            if MonitoringMetric.EVALUATION_COUNT in self.run_end_metrics:
                record["evaluations_count"] = event.data.get("evaluations_count")
            if MonitoringMetric.ELAPSED_TIME_NS in self.run_end_metrics:
                record["elapsed_time_ns"] = self.wall_ns
            if MonitoringMetric.CPU_TIME_NS in self.run_end_metrics:
                record["cpu_time_ns"] = self.cpu_ns
            self.records.append(record)
            
        if event.name == MonitoringEvent.FITNESS_EVALUATED:
            self.fitness_evaluations += 1

    def build_result_fields(self):
        return {
            "wall_ns": self.wall_ns,
            "cpu_ns": self.cpu_ns,
            "history": self.records.copy(),
        }

    def get_monitoring_records(self):
        return self.build_result_fields()

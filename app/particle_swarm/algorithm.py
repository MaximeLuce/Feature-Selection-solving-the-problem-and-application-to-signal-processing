

import numpy as np

from app.differential_evolution.monitoring import Event, Monitor, MonitoringEvent

from .results import NBPSOResult


class NewBinaryParticleSwarmOptimization:
    DEFAULT_CONFIG = {
        "swarm_size": 50,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "max_generations": 100,
        "seed": None,
        "target_fitness": 0.01,
        "monitoring": {
            "metrics": [],
        },
    }

    def __init__(self, problem, config=None):
        config = config or {}
        self.problem = problem
        self.swarm_size = config.get("swarm_size", self.DEFAULT_CONFIG["swarm_size"])
        self.w = config.get("w", self.DEFAULT_CONFIG["w"])
        self.c1 = config.get("c1", self.DEFAULT_CONFIG["c1"])
        self.c2 = config.get("c2", self.DEFAULT_CONFIG["c2"])
        self.max_generations = config.get(
            "max_generations",
            self.DEFAULT_CONFIG["max_generations"],
        )
        self.seed = config.get("seed", self.DEFAULT_CONFIG["seed"])
        self.target_fitness = config.get(
            "target_fitness",
            self.DEFAULT_CONFIG["target_fitness"],
        )
        monitoring_config = dict(self.DEFAULT_CONFIG["monitoring"])
        monitoring_config.update(config.get("monitoring", {}))
        self.monitor = Monitor(monitoring_config)
        self.generator = np.random.default_rng(self.seed)
        self.num_features = self.problem.num_features
        self.generation = 0
        self.positions = np.zeros((self.swarm_size, self.num_features), dtype=int)
        self.velocities = np.zeros((self.swarm_size, self.num_features), dtype=float)
        self.personal_best_positions = np.zeros_like(self.positions)
        self.personal_best_fitness = np.full(self.swarm_size, np.inf, dtype=float)
        self.best = np.zeros(self.num_features, dtype=int)
        self.best_fitness = np.inf

    def reset(self):
        self.problem.reset_counter()
        self.generator = np.random.default_rng(self.seed)
        self.generation = 0
        self.positions = np.zeros((self.swarm_size, self.num_features), dtype=int)
        self.velocities = np.zeros((self.swarm_size, self.num_features), dtype=float)
        self.personal_best_positions = np.zeros_like(self.positions)
        self.personal_best_fitness = np.full(self.swarm_size, np.inf, dtype=float)
        self.best = np.zeros(self.num_features, dtype=int)
        self.best_fitness = np.inf

    def _stop(self):
        return (
            self.generation >= self.max_generations
            or self.best_fitness < self.target_fitness
        )

    def _initialize_swarm(self):
        self.positions = self.generator.integers(
            0,
            2,
            size=(self.swarm_size, self.num_features),
        )
        self.velocities = self.generator.uniform(
            -1.0,
            1.0,
            size=(self.swarm_size, self.num_features),
        )

    def _evaluate_swarm(self):
        return np.array([self.problem.evaluate(position) for position in self.positions])

    def _update_personal_best(self, fitness):
        improved = fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = fitness[improved]

    def _update_global_best(self):
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.best_fitness:
            self.best = self.personal_best_positions[best_idx].copy()
            self.best_fitness = float(self.personal_best_fitness[best_idx])

    def _update_velocities(self):
        r1 = self.generator.random((self.swarm_size, self.num_features))
        r2 = self.generator.random((self.swarm_size, self.num_features))
        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.personal_best_positions - self.positions)
            + self.c2 * r2 * (self.best - self.positions)
        )

    def _update_positions(self):
        clipped = np.clip(self.velocities, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped))
        random_values = self.generator.random((self.swarm_size, self.num_features))
        self.positions = (random_values < probabilities).astype(int)

    def run(self):
        self.reset()
        self.monitor.emit(
            Event(
                MonitoringEvent.RUN_START,
                {
                    "seed": self.seed,
                    "swarm_size": self.swarm_size,
                    "max_generations": self.max_generations,
                    "algorithm": "nbpso",
                },
            )
        )

        self._initialize_swarm()
        fitness = self._evaluate_swarm()
        self._update_personal_best(fitness)
        self._update_global_best()
        self.monitor.emit(
            Event(
                MonitoringEvent.GENERATION_COMPLETED,
                {
                    "generation": self.generation,
                    "fitness": fitness,
                    "best_fitness": self.best_fitness,
                    "evaluations_count": self.problem.evaluations_count,
                },
            )
        )

        while not self._stop():
            self.generation += 1
            self._update_velocities()
            self._update_positions()
            fitness = self._evaluate_swarm()
            self._update_personal_best(fitness)
            self._update_global_best()
            self.monitor.emit(
                Event(
                    MonitoringEvent.GENERATION_COMPLETED,
                    {
                        "generation": self.generation,
                        "fitness": fitness,
                        "best_fitness": self.best_fitness,
                        "evaluations_count": self.problem.evaluations_count,
                    },
                )
            )

        self.monitor.emit(
            Event(
                MonitoringEvent.RUN_END,
                {
                    "generation": self.generation,
                    "best_fitness": self.best_fitness,
                    "evaluations_count": self.problem.evaluations_count,
                },
            )
        )
        monitor_result = self.monitor.build_result_fields()

        return NBPSOResult(
            best=self.best.copy(),
            best_fitness=self.best_fitness,
            generations=self.generation,
            evaluations=self.problem.evaluations_count,
            wall_ns=monitor_result["wall_ns"],
            cpu_ns=monitor_result["cpu_ns"],
            history=monitor_result["history"],
        )

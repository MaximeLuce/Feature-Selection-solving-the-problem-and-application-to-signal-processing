import numpy as np

from app.monitoring import Monitor
from app.Utilities.masks import repair_zero_masks

from .results import NBPSOResult


class NewBinaryParticleSwarmOptimization:
    DEFAULT_CONFIG = {
        "swarm_size": 50,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "vmax": 4.0,
        "max_generations": 100,
        "seed": None,
        "target_fitness": 0.01,
        "monitoring": {
            "metrics": [],
            "record_every": 1,
        },
    }

    def __init__(self, problem, config=None):
        config = config or {}
        self.problem = problem
        self.swarm_size = config.get("swarm_size", self.DEFAULT_CONFIG["swarm_size"])
        self.w = config.get("w", self.DEFAULT_CONFIG["w"])
        self.c1 = config.get("c1", self.DEFAULT_CONFIG["c1"])
        self.c2 = config.get("c2", self.DEFAULT_CONFIG["c2"])
        self.vmax = config.get("vmax", self.DEFAULT_CONFIG["vmax"])
        self.max_generations = config.get("max_generations", self.DEFAULT_CONFIG["max_generations"])
        self.seed = config.get("seed", self.DEFAULT_CONFIG["seed"])
        self.target_fitness = config.get("target_fitness", self.DEFAULT_CONFIG["target_fitness"])

        # Validate
        if self.swarm_size < 1:
            raise ValueError(f"swarm_size must be >= 1, got {self.swarm_size}.")
        if self.max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {self.max_generations}.")
        if self.vmax <= 0:
            raise ValueError(f"vmax must be > 0, got {self.vmax}.")

        self.monitor = Monitor(config.get("monitoring", self.DEFAULT_CONFIG["monitoring"]))
        self.generator = np.random.default_rng(self.seed)
        self.num_features = self.problem.num_features
        self._init_state()

    def _init_state(self):
        self.generation = 0
        self.positions = np.zeros((self.swarm_size, self.num_features), dtype=int)
        self.velocities_to_zero = np.zeros((self.swarm_size, self.num_features), dtype=float)
        self.velocities_to_one = np.zeros((self.swarm_size, self.num_features), dtype=float)
        self.personal_best_positions = np.zeros_like(self.positions)
        self.personal_best_fitness = np.full(self.swarm_size, np.inf, dtype=float)
        self.best = np.zeros(self.num_features, dtype=int)
        self.best_fitness = np.inf

    def reset(self):
        self.problem.reset_counter()
        self.generator = np.random.default_rng(self.seed)
        self._init_state()

    def _stop(self):
        return (
            self.generation >= self.max_generations
            or self.best_fitness < self.target_fitness
        )

    def _initialize_swarm(self):
        self.positions = self.generator.integers(0, 2, size=(self.swarm_size, self.num_features))
        self.positions = repair_zero_masks(self.positions, self.generator)
        self.velocities_to_zero.fill(0.0)
        self.velocities_to_one.fill(0.0)

    def _update_personal_best(self, fitness):
        improved = fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = fitness[improved]

    def _update_global_best(self):
        best_idx = int(np.argmin(self.personal_best_fitness))
        if self.personal_best_fitness[best_idx] < self.best_fitness:
            self.best = self.personal_best_positions[best_idx].copy()
            self.best_fitness = float(self.personal_best_fitness[best_idx])

    def _update_velocities(self):
        r1 = self.generator.random((self.swarm_size, self.num_features))
        r2 = self.generator.random((self.swarm_size, self.num_features))

        personal_to_one = np.where(
            self.personal_best_positions == 1,
            r1 * self.c1,
            -(r1 * self.c1),
        )
        personal_to_zero = -personal_to_one

        global_to_one = np.where(
            self.best == 1,
            r2 * self.c2,
            -(r2 * self.c2),
        )
        global_to_zero = -global_to_one

        self.velocities_to_one = np.clip(
            self.w * self.velocities_to_one + personal_to_one + global_to_one,
            -self.vmax,
            self.vmax,
        )
        self.velocities_to_zero = np.clip(
            self.w * self.velocities_to_zero + personal_to_zero + global_to_zero,
            -self.vmax,
            self.vmax,
        )

    def _update_positions(self):
        selected_velocity = np.where(
            self.positions == 0,
            self.velocities_to_one,
            self.velocities_to_zero,
        )
        probabilities = 1.0 / (1.0 + np.exp(-selected_velocity))
        flips = self.generator.random((self.swarm_size, self.num_features)) < probabilities
        self.positions = np.where(flips, 1 - self.positions, self.positions)
        self.positions = repair_zero_masks(self.positions, self.generator)

    def run(self):
        self.reset()
        self.monitor.start()

        self._initialize_swarm()
        fitness = self.problem.evaluate_batch(self.positions)
        self._update_personal_best(fitness)
        self._update_global_best()
        self.monitor.record_population(fitness, self.positions.copy())

        while not self._stop():
            self.generation += 1
            self._update_velocities()
            self._update_positions()
            fitness = self.problem.evaluate_batch(self.positions)
            self._update_personal_best(fitness)
            self._update_global_best()
            self.monitor.record_population(fitness, self.positions.copy())

        self.monitor.finish(self.generation, self.best_fitness, self.problem.evaluations_count)
        monitor_result = self.monitor.build_result_fields()

        return NBPSOResult(
            best_mask=self.best.copy(),
            best_fitness=self.best_fitness,
            generations=self.generation,
            evaluations=self.problem.evaluations_count,
            wall_ns=monitor_result["wall_ns"],
            cpu_ns=monitor_result["cpu_ns"],
            history=monitor_result["history"],
        )

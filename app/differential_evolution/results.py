from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PopulationEvaluation:
    fitness: np.ndarray
    masks: np.ndarray | None = None


@dataclass
class DEResult:
    best_vector: np.ndarray
    best_mask: np.ndarray
    best_fitness: float
    generations: int
    evaluations: int
    wall_ns: int | None
    cpu_ns: int | None
    history: list[dict[str, Any]] = field(default_factory=list)

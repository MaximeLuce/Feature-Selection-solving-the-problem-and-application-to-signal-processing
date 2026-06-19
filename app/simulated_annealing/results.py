from dataclasses import dataclass, field
from typing import Any


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

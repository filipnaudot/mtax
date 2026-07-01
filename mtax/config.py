from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QBAFSemantics:
    BASIC: str = "basic_model"
    QUADRATIC_ENERGY: str = "QuadraticEnergy_model"
    SQUARED_DFQUAD: str = "SquaredDFQuAD_model"
    EULER_BASED_TOP: str = "EulerBasedTop_model"
    EULER_BASED: str = "EulerBased_model"
    DFQUAD: str = "DFQuAD_model"


@dataclass(frozen=True)
class ExchangeConfig:
    max_rounds: int = 100
    max_retries: int = 3
    stop_when_resolved: bool = True
    resolution: Literal["stance", "top_r"] = "stance"
    top_r: int = 2
    semantics: str = QBAFSemantics.DFQUAD
    resolution_threshold: float = 0.5

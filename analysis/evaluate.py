import argparse
from dataclasses import dataclass
from typing import Sequence

from mtax.config import QBAFSemantics


SUPPORTED_SEMANTICS = (
    QBAFSemantics.BASIC,
    QBAFSemantics.QUADRATIC_ENERGY,
    QBAFSemantics.SQUARED_DFQUAD,
    QBAFSemantics.EULER_BASED_TOP,
    QBAFSemantics.EULER_BASED,
    QBAFSemantics.DFQUAD,
)


@dataclass(frozen=True)
class EvaluationConfig:
    graph_size: int
    qbaf_size: int
    semantics: tuple[str, ...]
    seed: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1: raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Run the MTAX evaluation.")
    parser.add_argument("--graph-size", type=positive_int, default=30)
    parser.add_argument("--qbaf-size", type=positive_int, default=15)
    parser.add_argument(
        "--semantics",
        nargs="+",
        choices=SUPPORTED_SEMANTICS,
        default=SUPPORTED_SEMANTICS,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.qbaf_size > args.graph_size:
        parser.error("--qbaf-size must not exceed --graph-size")
    return EvaluationConfig(
        graph_size=args.graph_size,
        qbaf_size=args.qbaf_size,
        semantics=tuple(args.semantics),
        seed=args.seed,
    )


if __name__ == "__main__":
    parse_args()

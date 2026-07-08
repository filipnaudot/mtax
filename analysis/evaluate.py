import argparse
import random
from dataclasses import dataclass
from typing import Sequence

from mtax.bm import BipolarMultitree
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
    num_topics: int
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
    parser.add_argument("--num-topics", type=positive_int, default=3)
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
    if args.num_topics >= args.graph_size:
        parser.error("--num-topics must be smaller than --graph-size")
    return EvaluationConfig(
        graph_size=args.graph_size,
        num_topics=args.num_topics,
        qbaf_size=args.qbaf_size,
        semantics=tuple(args.semantics),
        seed=args.seed,
    )


def generate_universal_bm(graph_size: int, num_topics: int, seed: int) -> BipolarMultitree:
    labels = [f"argument_{index}" for index in range(graph_size)]
    bm = BipolarMultitree(topics=set(labels[:num_topics]))
    random_generator = random.Random(seed)
    for source in labels[num_topics:]:
        target = random_generator.choice(sorted(bm.arguments))
        kind = random_generator.choice(("attack", "support"))
        bm.add_relation(source, target, kind)
    return bm


if __name__ == "__main__":
    config = parse_args()
    generate_universal_bm(config.graph_size, config.num_topics, config.seed)

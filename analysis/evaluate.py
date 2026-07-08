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
    extra_edge_probability: float
    semantics: tuple[str, ...]
    seed: int
    visualize: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1: raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0: raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Run the MTAX evaluation.")
    parser.add_argument("--graph-size", type=positive_int, default=30)
    parser.add_argument("--num-topics", type=positive_int, default=3)
    parser.add_argument("--qbaf-size", type=positive_int, default=15)
    parser.add_argument("--extra-edge-probability", type=probability, default=0.5)
    parser.add_argument("--semantics", nargs="+", choices=SUPPORTED_SEMANTICS, default=SUPPORTED_SEMANTICS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args(argv)
    if args.qbaf_size > args.graph_size:
        parser.error("--qbaf-size must not exceed --graph-size")
    if args.num_topics >= args.graph_size:
        parser.error("--num-topics must be smaller than --graph-size")
    return EvaluationConfig(
        graph_size=args.graph_size,
        num_topics=args.num_topics,
        qbaf_size=args.qbaf_size,
        extra_edge_probability=args.extra_edge_probability,
        semantics=tuple(args.semantics),
        seed=args.seed,
        visualize=args.visualize,
    )


def generate_universal_bm(graph_size: int, num_topics: int, seed: int, extra_edge_probability: float = 0.5) -> BipolarMultitree:
    labels = [f"argument_{index}" for index in range(graph_size)]
    bm = BipolarMultitree(topics=set(labels[:num_topics]))
    random_generator = random.Random(seed)
    for source in labels[num_topics:]:
        targets = sorted(bm.arguments)
        target = random_generator.choice(targets)
        kind = random_generator.choice(("attack", "support"))
        bm.add_relation(source, target, kind)
        if random_generator.random() < extra_edge_probability:
            random_generator.shuffle(targets)
            for extra_target in targets:
                if extra_target == target: continue
                try:
                    kind = random_generator.choice(("attack", "support"))
                    bm.add_relation(source, extra_target, kind)
                    break
                except ValueError:
                    continue
    return bm


def visualize_bm(bm: BipolarMultitree, output_path: str = "universal_bm") -> str:
    from graphviz import Digraph
    graph = Digraph("universal_bm", format="png")
    graph.attr(rankdir="BT")
    for argument in sorted(bm.arguments):
        graph.node(argument, shape="box" if argument in bm.topics else "ellipse")
    for source, target, kind in sorted(bm.relations):
        graph.edge(source, target, label=kind, color="red" if kind == "attack" else "green")
    return graph.render(output_path, cleanup=True)


if __name__ == "__main__":
    config = parse_args()
    universal_bm = generate_universal_bm(
        config.graph_size,
        config.num_topics,
        config.seed,
        config.extra_edge_probability,
    )
    if config.visualize:
        print(visualize_bm(universal_bm))

import argparse
import random
from dataclasses import dataclass
from typing import Sequence

from mtax.agent import MTAXAgent
from mtax.bm import BipolarMultitree
from mtax.config import QBAFSemantics
from mtax.schema import Argument, Disclosure, Relation
from qbaf import QBAFramework


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
    num_agents: int
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
    parser.add_argument("--num-agents", type=positive_int, default=2)
    parser.add_argument("--extra-edge-probability", type=probability, default=0.5)
    parser.add_argument("--semantics", nargs="+", choices=SUPPORTED_SEMANTICS, default=SUPPORTED_SEMANTICS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args(argv)
    if args.qbaf_size > args.graph_size:
        parser.error("--qbaf-size must not exceed --graph-size")
    if args.qbaf_size < args.num_topics:
        parser.error("--qbaf-size must be at least --num-topics")
    if args.num_topics >= args.graph_size:
        parser.error("--num-topics must be smaller than --graph-size")
    return EvaluationConfig(
        graph_size=args.graph_size,
        num_topics=args.num_topics,
        qbaf_size=args.qbaf_size,
        num_agents=args.num_agents,
        extra_edge_probability=args.extra_edge_probability,
        semantics=tuple(args.semantics),
        seed=args.seed,
        visualize=args.visualize,
    )


def generate_bm(graph_size: int, num_topics: int, seed: int, extra_edge_probability: float = 0.5) -> BipolarMultitree:
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
    assert len(bm.arguments) == graph_size, "generated BM has wrong number of arguments"
    assert bm.topics == set(labels[:num_topics]), "generated BM has wrong topics"
    return bm


def derive_private_agent(agent: MTAXAgent, universal_bm: BipolarMultitree, qbaf_size: int, seed: int, default_semantics: str) -> MTAXAgent:
    random_generator = random.Random(seed)
    selected = set(universal_bm.topics)
    selected_relations: list[tuple[str, str, str]] = []
    while len(selected) < qbaf_size:
        frontier = [relation for relation in universal_bm.relations
                    if relation[0] not in selected and relation[1] in selected]
        assert frontier, "universal BM has no frontier relation for private extraction"
        source, target, kind = random_generator.choice(sorted(frontier))
        selected_relations.append((source, target, kind))
        selected.add(source)
    for source, target, kind in sorted(universal_bm.relations):
        if source in selected and target in selected and (source, target, kind) not in selected_relations:
            selected_relations.append((source, target, kind))
    agent.initialize(sorted(universal_bm.topics), default_semantics)
    relations = tuple(Relation(source=source, target=target, kind=kind) # type: ignore
                      for source, target, kind in selected_relations)
    if relations:
        agent.ingest(Disclosure(arguments=tuple(Argument(label=argument, text=argument)
                                                for argument in sorted(selected - universal_bm.topics)), relations=relations))
    private_relations = {(relation.source, relation.target, relation.kind) for relation in agent.private_relations}
    assert len(agent.private_qbaf.arguments) == qbaf_size, "private QBAF has wrong number of arguments"
    assert set(agent.topics) == universal_bm.topics, "private agent has wrong topics"
    assert set(agent.private_qbaf.arguments) <= universal_bm.arguments, "private QBAF uses unknown arguments"
    assert private_relations <= universal_bm.relations, "private QBAF uses unknown relations"
    return agent


def visualize_bm(bm: BipolarMultitree, output_path: str = "public_exchange") -> str:
    from graphviz import Digraph
    graph = Digraph("public_exchange", format="png")
    graph.attr(rankdir="BT")
    for argument in sorted(bm.arguments):
        graph.node(argument, shape="box" if argument in bm.topics else "ellipse")
    for source, target, kind in sorted(bm.relations):
        graph.edge(source, target, color="red" if kind == "attack" else "green")
    return graph.render(output_path, cleanup=True)


def visualize_qbaf(qbaf: QBAFramework, topics: set[str], output_path: str = "private_qbaf") -> str:
    from graphviz import Digraph
    graph = Digraph("private_qbaf", format="png")
    graph.attr(rankdir="BT")
    for argument in sorted(qbaf.arguments):
        label = f"{argument}\ninitial={qbaf.initial_strength(argument):.2f}\nfinal={qbaf.final_strength(argument):.2f}"
        graph.node(argument, label=label, shape="box" if argument in topics else "ellipse")
    for source, target in sorted(qbaf.attack_relations.relations):
        graph.edge(source, target, color="red")
    for source, target in sorted(qbaf.support_relations.relations):
        graph.edge(source, target, color="green")
    return graph.render(output_path, cleanup=True)


if __name__ == "__main__":
    config = parse_args()
    public_bm = generate_bm(config.graph_size, config.num_topics, config.seed, config.extra_edge_probability)
    agents = tuple(
        derive_private_agent(MTAXAgent(f"agent_{index}"), public_bm, config.qbaf_size, config.seed+(index+1), config.semantics[index % len(config.semantics)])
        for index in range(config.num_agents)
    )
    if config.visualize:
        print(visualize_bm(public_bm))
        for agent in agents:
            print(visualize_qbaf(agent.private_qbaf, set(agent.topics), agent.name))

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence

from eval_agents import CounterfactualAgent, GreedyAgent, ShallowAgent
from mtax.agent import MTAXAgent
from mtax.bm import BipolarMultitree
from mtax.config import ExchangeConfig, QBAFSemantics
from mtax.disclosure_measure import kendall_tau_b, topic_ranking
from mtax.mtax import MTAX
from mtax.schema import Argument, Disclosure, Relation
from qbaf import QBAFramework


result_csv_headers = [
    "experiment",
    "parameter",
    "value",
    "strategy",
    "runs",
    "resolved",
    "resolution_rate",
    "avg_rounds",
    "avg_contributions",
    "avg_final_ranking_agreement",
]


EVALUATION_SEMANTICS = QBAFSemantics.DFQUAD
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
TOPIC_VALUES = (2, 4, 6, 8, 10, 12, 14)
AGENT_VALUES = (2, 4, 6, 8, 10)
DENSITY_VALUES = (0.0, 0.25, 0.5, 0.75)
SHALLOW_MAX_CONTRIBUTIONS = 3
STRATEGIES = ("shallow", "greedy", "counterfactual")


@dataclass(frozen=True)
class EvaluationConfig:
    graph_size: int
    num_topics: int
    qbaf_size: int
    num_agents: int
    max_rounds: int
    max_attempts: int
    runs: int
    extra_edge_probability: float
    seed: int
    visualize: bool
    experiment: str


@dataclass(frozen=True)
class ExperimentCase:
    experiment: str
    parameter: str
    value: int | float
    config: EvaluationConfig


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
    parser.add_argument("--max-rounds", type=positive_int, default=100)
    parser.add_argument("--max-attempts", type=positive_int, default=100)
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--extra-edge-probability", type=probability, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--experiment", choices=("base", "topics", "agents", "density", "all"), default="base")
    args = parser.parse_args(argv)
    max_num_topics = max(args.num_topics, max(TOPIC_VALUES)) if args.experiment in ("topics", "all") else args.num_topics
    if args.qbaf_size > args.graph_size:
        parser.error("--qbaf-size must not exceed --graph-size")
    if args.qbaf_size < max_num_topics:
        parser.error("--qbaf-size must be at least the maximum number of topics in the selected experiment")
    if max_num_topics >= args.graph_size:
        parser.error("--graph-size must be larger than the maximum number of topics in the selected experiment")
    if args.num_topics < 2:
        parser.error("--num-topics must be at least 2")
    if args.num_agents < 2:
        parser.error("--num-agents must be at least 2")
    return EvaluationConfig(
        graph_size=args.graph_size,
        num_topics=args.num_topics,
        qbaf_size=args.qbaf_size,
        num_agents=args.num_agents,
        max_rounds=args.max_rounds,
        max_attempts=args.max_attempts,
        runs=args.runs,
        extra_edge_probability=args.extra_edge_probability,
        seed=args.seed,
        visualize=args.visualize,
        experiment=args.experiment,
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
    for topic in sorted(universal_bm.topics):
        argument = Argument(label=topic, text=topic)
        agent.private_strengths[topic] = agent.rate(argument)
        agent.private_qbaf.modify_initial_strength(topic, agent.private_strengths[topic])
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


def create_exchange(config: EvaluationConfig, strategy: str, seed: int) -> tuple[BipolarMultitree, MTAX]:
    public_bm = generate_bm(config.graph_size, config.num_topics, seed, config.extra_edge_probability)
    agents = []
    for index in range(config.num_agents):
        agent_seed = seed + index + 1
        if strategy == "shallow":
            agent = ShallowAgent(f"agent_{index}", seed=agent_seed, max_contributions=SHALLOW_MAX_CONTRIBUTIONS)
        elif strategy == "greedy":
            agent = GreedyAgent(f"agent_{index}", seed=agent_seed)
        elif strategy == "counterfactual":
            agent = CounterfactualAgent(f"agent_{index}", seed=agent_seed)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        agents.append(derive_private_agent(agent, public_bm, config.qbaf_size, agent_seed, EVALUATION_SEMANTICS))
    agents = tuple(agents)
    exchange = MTAX(list(agents),
                    sorted(public_bm.topics),
                    ExchangeConfig(max_rounds=config.max_rounds, stop_when_resolved=True, resolution="top_r", semantics=EVALUATION_SEMANTICS))
    return public_bm, exchange


def run_exchange(exchange: MTAX) -> None:
    states = []
    for state in exchange:
        states.append(state)
    assert states, "exchange did not run"
    assert states[-1].round_index <= exchange.config.max_rounds, "exchange exceeded max rounds"


def experiment_cases(config: EvaluationConfig) -> list[ExperimentCase]:
    cases = []
    if config.experiment in ("base", "all"):
        cases.append(ExperimentCase("base", "base", 0, config))
    if config.experiment in ("topics", "all"):
        cases.extend(
            ExperimentCase("topics", "num_topics", value, replace(config, num_topics=value))
            for value in TOPIC_VALUES
        )
    if config.experiment in ("agents", "all"):
        cases.extend(
            ExperimentCase("agents", "num_agents", value, replace(config, num_agents=value))
            for value in AGENT_VALUES
        )
    if config.experiment in ("density", "all"):
        cases.extend(
            ExperimentCase("density", "extra_edge_probability", value, replace(config, extra_edge_probability=value))
            for value in DENSITY_VALUES
        )
    return cases


def find_unresolved_seed(config: EvaluationConfig, seed: int) -> int:
    for attempt in range(config.max_attempts):
        candidate_seed = seed + attempt
        _, trial_exchange = create_exchange(config, "counterfactual", candidate_seed)
        if not trial_exchange.is_resolved():
            return candidate_seed
    raise RuntimeError("could not generate an initially unresolved exchange")


def average_ranking_agreement(agents: Sequence[MTAXAgent], topics: Sequence[str]) -> float:
    if len(agents) < 2:
        return 1.0
    rankings = [topic_ranking(agent.private_qbaf, topics) for agent in agents]
    agreements = [
        kendall_tau_b(left, right)
        for left, right in combinations(rankings, 2)
    ]
    return sum(agreements) / len(agreements)


if __name__ == "__main__":
    config = parse_args()
    cases = experiment_cases(config)
    total = len(cases) * config.runs * len(STRATEGIES)
    done = 0
    last_public_bm = None
    last_exchange = None

    with open(RESULT_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result_csv_headers)
        for case in cases:
            results: dict[str, list] = {name: [] for name in STRATEGIES}
            agreements: dict[str, list[float]] = {name: [] for name in STRATEGIES}
            for run_index in range(case.config.runs):
                run_seed = case.config.seed + run_index * case.config.max_attempts
                seed = find_unresolved_seed(case.config, run_seed)
                for name in STRATEGIES:
                    public_bm, exchange = create_exchange(case.config, name, seed)
                    assert not exchange.is_resolved(), "exchange is initially resolved"
                    if config.visualize:
                        last_public_bm = public_bm
                        last_exchange = exchange
                    run_exchange(exchange)
                    results[name].append(exchange.result())
                    agreements[name].append(average_ranking_agreement(exchange.agents, exchange.topics))
                    done += 1
                    print(f"\rprogress {done}/{total} ({done / total:.0%})", end="", file=sys.stderr, flush=True)

            for name in STRATEGIES:
                strategy_results = results[name]
                resolved = sum(result.resolved for result in strategy_results)
                avg_rounds = sum(result.rounds for result in strategy_results) / len(strategy_results)
                avg_contributions = sum(len(result.trace) for result in strategy_results) / len(strategy_results)
                avg_agreement = sum(agreements[name]) / len(agreements[name])
                writer.writerow([
                    case.experiment,
                    case.parameter,
                    case.value,
                    name,
                    len(strategy_results),
                    resolved,
                    f"{resolved / len(strategy_results):.3f}",
                    f"{avg_rounds:.2f}",
                    f"{avg_contributions:.2f}",
                    f"{avg_agreement:.3f}",
                ])
    print(file=sys.stderr)
    if config.visualize and last_public_bm is not None and last_exchange is not None:
        print(visualize_bm(last_public_bm))
        for agent in last_exchange.agents:
            print(visualize_qbaf(agent.private_qbaf, set(last_exchange.topics), agent.name))

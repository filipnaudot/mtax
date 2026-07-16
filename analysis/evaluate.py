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
    "rating_mode",
    "strategies",
    "runs",
    "resolved",
    "resolution_rate",
    "avg_rounds",
    "avg_final_ranking_agreement",
]

influence_csv_headers = [
    "experiment",
    "parameter",
    "value",
    "rating_mode",
    "strategy",
    "runs",
    "avg_influence",
    "avg_contributions",
]


EVALUATION_SEMANTICS = QBAFSemantics.DFQUAD
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
INFLUENCE_PATH = os.path.join(os.path.dirname(__file__), "behavior_influence.csv")
TOPIC_VALUES = (2, 4, 6, 8, 10, 12, 14)
AGENT_VALUES = (3, 5, 7, 9, 11)
DENSITY_VALUES = (0.0, 0.25, 0.5, 0.75)
SHALLOW_MAX_CONTRIBUTIONS = 3
STRATEGIES = ("shallow", "greedy", "counterfactual")
STRATEGY_CHOICES = frozenset(STRATEGIES)


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
    rating_mode: str
    strategies: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentCase:
    experiment: str
    parameter: str
    value: int | float | str
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
    parser.add_argument("--oracle-graph-size", dest="graph_size", type=positive_int, default=30)
    parser.add_argument("--num-topics", type=positive_int, default=3)
    parser.add_argument("--qbaf-size", type=positive_int, default=15)
    parser.add_argument("--num-agents", type=positive_int, default=3)
    parser.add_argument("--max-rounds", type=positive_int, default=100)
    parser.add_argument("--max-attempts", type=positive_int, default=100)
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--extra-edge-probability", type=probability, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--experiment", choices=("base", "rating", "topics", "agents", "density", "all"), default="base")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES, help="Strategies assigned to agents in rotation.")
    args = parser.parse_args(argv)
    max_num_topics = max(args.num_topics, max(TOPIC_VALUES)) if args.experiment in ("topics", "all") else args.num_topics
    if args.qbaf_size > args.graph_size:
        parser.error("--qbaf-size must not exceed --oracle-graph-size")
    if args.qbaf_size < max_num_topics:
        parser.error("--qbaf-size must be at least the maximum number of topics in the selected experiment")
    if max_num_topics >= args.graph_size:
        parser.error("--oracle-graph-size must be larger than the maximum number of topics in the selected experiment")
    if args.num_topics < 2:
        parser.error("--num-topics must be at least 2")
    if args.num_agents < 2:
        parser.error("--num-agents must be at least 2")
    if args.num_agents < len(args.strategies):
        parser.error("--num-agents must be at least the number of strategies")
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
        rating_mode="random",
        strategies=tuple(args.strategies),
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


def create_agent(strategy: str, name: str, seed: int, rating_mode: str) -> MTAXAgent:
    if strategy == "shallow":
        return ShallowAgent(name, seed=seed, rating_mode=rating_mode, max_contributions=SHALLOW_MAX_CONTRIBUTIONS)
    if strategy == "greedy":
        return GreedyAgent(name, seed=seed, rating_mode=rating_mode)
    if strategy == "counterfactual":
        return CounterfactualAgent(name, seed=seed, rating_mode=rating_mode)
    raise ValueError(f"unknown strategy: {strategy}")


def agent_strategy(config: EvaluationConfig, agent_index: int) -> str:
    return config.strategies[agent_index % len(config.strategies)]


def create_exchange(config: EvaluationConfig, seed: int) -> tuple[BipolarMultitree, MTAX]:
    universal_bm = generate_bm(config.graph_size, config.num_topics, seed, config.extra_edge_probability)
    agents = []
    for index in range(config.num_agents):
        agent_seed = seed + index + 1
        agent = create_agent(strategy=agent_strategy(config, index), name=f"agent_{index}", seed=agent_seed, rating_mode=config.rating_mode)
        agents.append(derive_private_agent(agent, universal_bm, config.qbaf_size, agent_seed, EVALUATION_SEMANTICS))
    agents = tuple(agents)
    exchange = MTAX(list(agents),
                    sorted(universal_bm.topics),
                    ExchangeConfig(max_rounds=config.max_rounds, stop_when_resolved=True, resolution="top_r", semantics=EVALUATION_SEMANTICS))
    return universal_bm, exchange


def run_exchange(exchange: MTAX) -> None:
    states = []
    for state in exchange:
        states.append(state)
    assert states, "exchange did not run"
    assert states[-1].round_index <= exchange.config.max_rounds, "exchange exceeded max rounds"


def experiment_cases(config: EvaluationConfig) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []

    def add_cases(experiment: str, parameter: str, values: Sequence[int | float], config_parameter: str | None = None) -> None:
        for value in values:
            for rating_mode in ("random", "stable"):
                changes: dict[str, int | float | str] = {"rating_mode": rating_mode}
                if config_parameter is not None:
                    changes[config_parameter] = value
                cases.append(ExperimentCase(
                    experiment,
                    parameter,
                    value,
                    replace(config, **changes),
                ))

    if config.experiment == "base":
        add_cases("base", "base", (0,))
    if config.experiment in ("rating", "all"):
        for rating_mode in ("random", "stable"):
            cases.append(ExperimentCase(
                "rating",
                "rating_mode",
                rating_mode,
                replace(config, rating_mode=rating_mode),
            ))
    if config.experiment in ("topics", "all"):
        add_cases("topics", "num_topics", TOPIC_VALUES, "num_topics")
    if config.experiment in ("agents", "all"):
        add_cases("agents", "num_agents", tuple(value for value in AGENT_VALUES if value >= len(config.strategies)), "num_agents")
    if config.experiment in ("density", "all"):
        add_cases("density", "extra_edge_probability", DENSITY_VALUES, "extra_edge_probability")
    return cases


def find_unresolved_seed(config: EvaluationConfig, seed: int) -> int:
    for attempt in range(config.max_attempts):
        candidate_seed = seed + attempt
        _, trial_exchange = create_exchange(config, candidate_seed)
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


def behavior_influence(config: EvaluationConfig, exchange: MTAX, initial_rankings: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    influence = {strategy: [] for strategy in config.strategies}
    final_rankings = {
        agent.name: topic_ranking(agent.private_qbaf, exchange.topics)
        for agent in exchange.agents
    }
    for source_index, source_agent in enumerate(exchange.agents):
        strategy = agent_strategy(config, source_index)
        source_initial = initial_rankings[source_agent.name]
        for target_agent in exchange.agents:
            if target_agent.name == source_agent.name:
                continue
            target_initial = initial_rankings[target_agent.name]
            target_final = final_rankings[target_agent.name]
            before = kendall_tau_b(target_initial, source_initial)
            after = kendall_tau_b(target_final, source_initial)
            influence[strategy].append(after - before)
    return influence


def behavior_contributions(config: EvaluationConfig, exchange: MTAX) -> dict[str, int]:
    agent_strategies = {
        agent.name: agent_strategy(config, index)
        for index, agent in enumerate(exchange.agents)
    }
    counts = {strategy: 0 for strategy in config.strategies}
    for contribution in exchange.state.trace:
        counts[agent_strategies[contribution.agent]] += 1
    return counts


if __name__ == "__main__":
    config = parse_args()
    cases = experiment_cases(config)
    strategy_label = " ".join(config.strategies)
    total = len(cases) * config.runs
    done = 0
    with open(RESULT_PATH, "w", newline="") as result_file, open(INFLUENCE_PATH, "w", newline="") as influence_file:
        result_writer = csv.writer(result_file)
        influence_writer = csv.writer(influence_file)
        result_writer.writerow(result_csv_headers)
        influence_writer.writerow(influence_csv_headers)
        for case in cases:
            results = []
            agreements: list[float] = []
            influence_by_strategy = {strategy: [] for strategy in config.strategies}
            contributions_by_strategy = {strategy: 0 for strategy in config.strategies}
            for run_index in range(case.config.runs):
                run_seed = case.config.seed + run_index * case.config.max_attempts
                seed = find_unresolved_seed(case.config, run_seed)
                universal_bm, exchange = create_exchange(case.config, seed)
                assert not exchange.is_resolved(), "exchange is initially resolved"
                initial_rankings = {
                    agent.name: topic_ranking(agent.private_qbaf, exchange.topics)
                    for agent in exchange.agents
                }
                if config.visualize:
                    print(visualize_bm(universal_bm))
                    for agent in exchange.agents:
                        print(visualize_qbaf(agent.private_qbaf, set(exchange.topics), agent.name))
                run_exchange(exchange)
                results.append(exchange.result())
                agreements.append(average_ranking_agreement(exchange.agents, exchange.topics))
                for strategy, values in behavior_influence(case.config, exchange, initial_rankings).items():
                    influence_by_strategy[strategy].extend(values)
                for strategy, count in behavior_contributions(case.config, exchange).items():
                    contributions_by_strategy[strategy] += count
                done += 1
                print(f"\rprogress {done}/{total} ({done / total:.0%})", end="", file=sys.stderr, flush=True)

            resolved = sum(result.resolved for result in results)
            avg_rounds = sum(result.rounds for result in results) / len(results)
            avg_agreement = sum(agreements) / len(agreements)
            result_writer.writerow([
                case.experiment,
                case.parameter,
                case.value,
                case.config.rating_mode,
                strategy_label,
                len(results),
                resolved,
                f"{resolved / len(results):.3f}",
                f"{avg_rounds:.2f}",
                f"{avg_agreement:.3f}",
            ])
            for strategy in config.strategies:
                influence_values = influence_by_strategy[strategy]
                influence_writer.writerow([
                    case.experiment,
                    case.parameter,
                    case.value,
                    case.config.rating_mode,
                    strategy,
                    len(results),
                    f"{sum(influence_values) / len(influence_values):.3f}",
                    f"{contributions_by_strategy[strategy] / len(results):.2f}",
                ])
    print(file=sys.stderr)

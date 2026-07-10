from collections.abc import Mapping, Sequence
from itertools import combinations
from math import sqrt

from qbaf import QBAFramework

from mtax.schema import Relation




###################################
# SINGLE TOPIC
###################################
def single_topic_disclosure_effects(public_qbaf: QBAFramework, private_qbaf: QBAFramework, topic: str) -> list[tuple[Relation, float]]:
    public_attacks = public_qbaf.attack_relations.relations
    public_supports = public_qbaf.support_relations.relations
    private_relations = [
        Relation(source=source, target=target, kind="attack")
        for source, target in private_qbaf.attack_relations.relations - public_attacks
    ] + [
        Relation(source=source, target=target, kind="support")
        for source, target in private_qbaf.support_relations.relations - public_supports
    ]
    before = public_qbaf.final_strength(topic)
    effects: list[tuple[Relation, float]] = []
    for relation in private_relations:
        if relation.target not in public_qbaf.arguments:
            continue

        hypothetical_qbaf = public_qbaf.copy()
        if relation.source not in hypothetical_qbaf.arguments:
            hypothetical_qbaf.add_argument(relation.source, private_qbaf.initial_strength(relation.source))
        if relation.kind == "attack":
            hypothetical_qbaf.add_attack_relation(relation.source, relation.target)
        else:
            hypothetical_qbaf.add_support_relation(relation.source, relation.target)

        effect = hypothetical_qbaf.final_strength(topic) - before
        effects.append((relation, effect))
    return sorted(effects, key=lambda item: (-item[1], item[0].source, item[0].target, item[0].kind))




###################################
# MULTI TOPIC (RANKING)
###################################
def topic_ranking(qbaf: QBAFramework, topics: Sequence[str]) -> dict[str, float]:
    return {topic: qbaf.final_strength(topic) for topic in topics}


def kendall_tau_b(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    if first.keys() != second.keys():
        raise ValueError("Rankings must contain the same topics")
    concordant = 0
    discordant = 0
    ties_first = 0
    ties_second = 0
    for left, right in combinations(first, 2):
        first_order = (first[left] > first[right]) - (first[left] < first[right])
        second_order = (second[left] > second[right]) - (second[left] < second[right])
        if first_order == 0 and second_order != 0:
            ties_first += 1
        elif second_order == 0 and first_order != 0:
            ties_second += 1
        elif first_order == second_order != 0:
            concordant += 1
        elif first_order != 0 and second_order != 0:
            discordant += 1
    denominator = sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    if denominator == 0:
        pairs = combinations(first, 2)
        return 1.0 if all(first[left] == first[right] and second[left] == second[right] for left, right in pairs) else 0.0
    return (concordant - discordant) / denominator


def ranking_disclosure_effects(public_qbaf: QBAFramework,
                               private_qbaf: QBAFramework,
                               topics: Sequence[str],
                               relations: Sequence[Relation] | None = None) -> list[tuple[Relation, float]]:
    if relations is None:
        public_attacks = public_qbaf.attack_relations.relations
        public_supports = public_qbaf.support_relations.relations
        relations = [
            Relation(source=source, target=target, kind="attack")
            for source, target in private_qbaf.attack_relations.relations - public_attacks
        ] + [
            Relation(source=source, target=target, kind="support")
            for source, target in private_qbaf.support_relations.relations - public_supports
        ]
    private_ranking = topic_ranking(private_qbaf, topics)
    before = kendall_tau_b(topic_ranking(public_qbaf, topics), private_ranking)
    effects: list[tuple[Relation, float]] = []
    for relation in relations:
        if relation.target not in public_qbaf.arguments:
            continue
        hypothetical_qbaf = public_qbaf.copy()
        if relation.source not in hypothetical_qbaf.arguments:
            hypothetical_qbaf.add_argument(relation.source, private_qbaf.initial_strength(relation.source))
        if relation.kind == "attack":
            hypothetical_qbaf.add_attack_relation(relation.source, relation.target)
        else:
            hypothetical_qbaf.add_support_relation(relation.source, relation.target)
        after = kendall_tau_b(topic_ranking(hypothetical_qbaf, topics), private_ranking)
        effects.append((relation, after - before))
    return sorted(effects, key=lambda item: (-item[1], item[0].source, item[0].target, item[0].kind))

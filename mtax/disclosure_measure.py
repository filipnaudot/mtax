from qbaf import QBAFramework

from mtax.schema import Relation


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

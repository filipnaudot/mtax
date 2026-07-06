from __future__ import annotations

from copy import deepcopy

from qbaf import QBAFramework

from mtax.bm import BipolarMultitree
from mtax.schema import Argument, Disclosure, Pass, Relation


NEGATIVE = -1
NEUTRAL = 0
POSITIVE = 1



class MTAXAgent:
    def __init__(self,
                 name: str,
                 private_arguments: dict[str, Argument] | None = None,
                 private_strengths: dict[str, float] | None = None,
                 private_relations: list[Relation] | None = None,
                 semantics: str | None = None,
                 negative_below: float = 0.5,
                 positive_above: float = 0.5) -> None:
        if negative_below > positive_above:
            raise ValueError("negative_below must not exceed positive_above")
        self.name = name
        self._semantics_override = semantics
        self.semantics = semantics
        self.negative_below = negative_below
        self.positive_above = positive_above
        self.topics: tuple[str, ...] = ()
        self.private_arguments: dict[str, Argument] = private_arguments if private_arguments is not None else {}
        self.private_strengths: dict[str, float] = private_strengths if private_strengths is not None else {}
        self.private_relations: list[Relation] = private_relations if private_relations is not None else []
        self._private_qbaf: QBAFramework | None = None


    @property
    def private_qbaf(self) -> QBAFramework:
        if self._private_qbaf is None:
            raise RuntimeError("Agent has not been initialized by an MTAX exchange.")
        return self._private_qbaf


    def initialize(self, topics: list[str], default_semantics: str) -> None:
        self.topics = tuple(topics)
        self.semantics = self._semantics_override or default_semantics
        arguments = list(dict.fromkeys((*self.topics, *self.private_arguments)))
        known_arguments = set(arguments)
        for relation in self.private_relations:
            unknown = [f"{source} '{target}'"
                       for source, target in (("source", relation.source), ("target", relation.target))
                       if target not in known_arguments]
            if unknown: raise ValueError(f"Private relation has unknown {' and '.join(unknown)}")
        for argument in arguments:
            self.private_strengths.setdefault(argument, 0.5)
        attacks = [
            (relation.source, relation.target)
            for relation in self.private_relations
            if relation.kind == "attack"
        ]
        supports = [
            (relation.source, relation.target)
            for relation in self.private_relations
            if relation.kind == "support"
        ]
        self._private_qbaf = QBAFramework(
            arguments,
            [self.private_strengths[argument] for argument in arguments],
            attacks,
            supports,
            semantics=self.semantics,
        )


    def contribute(self, violation_feedback: str | None = None) -> Disclosure | Pass | None:
        raise NotImplementedError


    def rate(self, argument: Argument) -> float:
        return 0.5


    def stance(self, topic: str) -> int:
        strength = self.private_qbaf.final_strength(topic)
        if strength < self.negative_below:
            return NEGATIVE
        if strength > self.positive_above:
            return POSITIVE
        return NEUTRAL


    def build_qbaf_from_bm(self, public_bm: BipolarMultitree) -> QBAFramework:
        arguments = [
            argument
            for argument in self.private_qbaf.arguments
            if argument in public_bm.arguments
        ]
        initial_strengths = [
            self.private_qbaf.initial_strength(argument)
            for argument in arguments
        ]
        attacks = [
            (source, target)
            for source, target in self.private_qbaf.attack_relations.relations
            if (source, target, "attack") in public_bm.relations
        ]
        supports = [
            (source, target)
            for source, target in self.private_qbaf.support_relations.relations
            if (source, target, "support") in public_bm.relations
        ]
        return QBAFramework(arguments, initial_strengths, attacks, supports, semantics=self.semantics)


    def available_relations(self, public_bm: BipolarMultitree) -> list[tuple[Relation, float]]:
        private_relations = [
            Relation(source=source, target=target, kind="attack")
            for source, target in self.private_qbaf.attack_relations.relations
        ] + [
            Relation(source=source, target=target, kind="support")
            for source, target in self.private_qbaf.support_relations.relations
        ]
        available = []
        source_strengths: dict[str, float] = {}
        for relation in private_relations:
            if (relation.source, relation.target, relation.kind) in public_bm.relations:
                continue
            candidate = deepcopy(public_bm)
            try:
                candidate.add_relation(relation.source, relation.target, relation.kind)
            except ValueError:
                continue
            if relation.source not in source_strengths:
                source_strengths[relation.source] = self.private_qbaf.final_strength(relation.source)
            available.append((relation, source_strengths[relation.source]))
        return sorted(available, key=lambda item: (item[0].source, item[0].target, item[0].kind))


    def ingest(self, disclosure: Disclosure) -> None:
        for argument in disclosure.arguments:
            if argument.label not in self.private_arguments:
                self.private_arguments[argument.label] = argument
                self.private_strengths[argument.label] = self.rate(argument)
                self.private_qbaf.add_argument(argument.label, self.private_strengths[argument.label])
        for relation in disclosure.relations:
            if relation not in self.private_relations:
                self.private_relations.append(relation)
                if relation.kind == "attack":
                    self.private_qbaf.add_attack_relation(relation.source, relation.target)
                else:
                    self.private_qbaf.add_support_relation(relation.source, relation.target)

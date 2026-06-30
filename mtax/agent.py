from __future__ import annotations

from qbaf import QBAFramework

from mtax.schema import Argument, Disclosure, Pass, Relation



class MTAXAgent:
    def __init__(self,
                 name: str,
                 private_arguments: dict[str, Argument] | None = None,
                 private_strengths: dict[str, float] | None = None,
                 private_relations: list[Relation] | None = None,
                 semantics: str | None = None) -> None:
        self.name = name
        self._semantics_override = semantics
        self.semantics = semantics
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

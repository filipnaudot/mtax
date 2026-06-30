from __future__ import annotations

from mtax.schema import Argument, Disclosure, Relation



class MTAXAgent:
    def __init__(self,
                 name: str,
                 private_arguments: dict[str, Argument] | None = None,
                 private_strengths: dict[str, float] | None = None,
                 private_relations: list[Relation] | None = None) -> None:
        self.name = name
        self.topics: tuple[str, ...] = ()
        self.private_arguments: dict[str, Argument] = private_arguments if private_arguments is not None else {}
        self.private_strengths: dict[str, float] = private_strengths if private_strengths is not None else {}
        self.private_relations: list[Relation] = private_relations if private_relations is not None else []

    def initialize(self, topics: list[str]) -> None:
        self.topics = tuple(topics)

    def contribute(self, violation_feedback: str | None = None) -> Disclosure | None:
        raise NotImplementedError

    def rate(self, argument: Argument) -> float:
        return 0.5

    def ingest(self, disclosure: Disclosure) -> None:
        for argument in disclosure.arguments:
            if argument.label not in self.private_arguments:
                self.private_arguments[argument.label] = argument
                self.private_strengths[argument.label] = self.rate(argument)
        for relation in disclosure.relations:
            if relation not in self.private_relations:
                self.private_relations.append(relation)

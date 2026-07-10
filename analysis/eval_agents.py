import random

from mtax.agent import MTAXAgent
from mtax.bm import BipolarMultitree
from mtax.disclosure_measure import ranking_disclosure_effects
from mtax.schema import Argument, Disclosure, Pass


class CounterfactualAgent(MTAXAgent):
    def __init__(self, name: str, seed: int, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.random_generator = random.Random(seed)

    def rate(self, argument: Argument) -> float:
        return self.random_generator.random()

    def contribute(self, public_bm: BipolarMultitree, violation_feedback: str | None = None) -> Disclosure | Pass:
        available = [relation for relation, _ in self.available_relations(public_bm)]
        if not available:
            return Pass(action="pass")
        effects = ranking_disclosure_effects(self.build_qbaf_from_bm(public_bm), self.private_qbaf, self.topics, relations=available)
        if not effects or effects[0][1] <= 0:
            return Pass(action="pass")
        relation = effects[0][0]
        arguments = ()
        if relation.source not in public_bm.arguments:
            arguments = (self.private_arguments.get(relation.source, Argument(label=relation.source, text=relation.source)),)
        return Disclosure(arguments=arguments, relations=(relation,))

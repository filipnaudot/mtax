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



class GreedyAgent(MTAXAgent):
    def __init__(self, name: str, seed: int, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.random_generator = random.Random(seed)


    def rate(self, argument: Argument) -> float:
        return self.random_generator.random()


    def contribute(self, public_bm: BipolarMultitree, violation_feedback: str | None = None) -> Disclosure | Pass:
        available = self.available_relations(public_bm)
        if not available:
            return Pass(action="pass")
        effects = ranking_disclosure_effects(self.build_qbaf_from_bm(public_bm),
                                             self.private_qbaf,
                                             self.topics,
                                             relations=[relation for relation, _ in available])
        preserving_relations = {relation for relation, effect in effects if effect >= 0} # type: ignore
        candidates = [(relation, strength) for relation, strength in available if relation in preserving_relations]
        if not candidates:
            return Pass(action="pass")
        distances = {topic: 0 for topic in public_bm.topics}
        pending = set(public_bm.arguments) - set(public_bm.topics)
        while pending:
            progressed = False
            for argument in list(pending):
                target_distances = [distances[target] for source, target, _ in public_bm.relations 
                                    if source == argument and target in distances]
                if target_distances:
                    distances[argument] = min(target_distances) + 1
                    pending.remove(argument)
                    progressed = True
            if not progressed:
                break

        relation = max(candidates, key=lambda item: (item[1], -distances.get(item[0].target, len(public_bm.arguments))))[0]
        arguments = ()
        if relation.source not in public_bm.arguments:
            arguments = (self.private_arguments.get(relation.source, Argument(label=relation.source, text=relation.source)),)
        return Disclosure(arguments=arguments, relations=(relation,))



class ShallowAgent(MTAXAgent):
    def __init__(self, name: str, seed: int, max_contributions: int = 1, **kwargs) -> None:
        super().__init__(name, **kwargs)
        if max_contributions < 1:
            raise ValueError("max_contributions must be at least 1")
        self.random_generator = random.Random(seed)
        self.max_contributions = max_contributions


    def rate(self, argument: Argument) -> float:
        return self.random_generator.random()


    def contribute(self, public_bm: BipolarMultitree, violation_feedback: str | None = None) -> Disclosure | Pass:
        available = [(relation, strength)
                     for relation, strength in self.available_relations(public_bm)
                     if relation.target in public_bm.topics]
        if not available:
            return Pass(action="pass")
        effects = ranking_disclosure_effects(self.build_qbaf_from_bm(public_bm),
                                             self.private_qbaf,
                                             self.topics,
                                             relations=[relation for relation, _ in available])
        preserving_relations = {relation for relation, effect in effects if effect >= 0} # type: ignore
        candidates = [(relation, strength) for relation, strength in available if relation in preserving_relations]
        selected = [relation for relation, _ in sorted(candidates, key=lambda item: (-item[1], item[0].source, item[0].target, item[0].kind))[:self.max_contributions]]
        if not selected:
            return Pass(action="pass")
        argument_labels = tuple(dict.fromkeys(relation.source for relation in selected
                                              if relation.source not in public_bm.arguments))
        arguments = tuple(self.private_arguments.get(label, Argument(label=label, text=label)) for label in argument_labels)
        return Disclosure(arguments=arguments, relations=tuple(selected))

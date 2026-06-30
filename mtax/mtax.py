from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mtax.bm import BipolarMultitree
from mtax.config import ExchangeConfig
from mtax.schema import Argument, Disclosure, Relation

if TYPE_CHECKING:
    from mtax.agent import MTAXAgent






#######################################
# DATA CLASSES
#######################################
@dataclass(frozen=True)
class Contribution:
    disclosure: Disclosure
    agent: str
    round_index: int


@dataclass(frozen=True)
class PublishError:
    agent: str
    contribution: Contribution
    relation: Relation | None
    reason: str


@dataclass
class DialogueState:
    topics: list[str]
    public_arguments: dict[str, Argument] = field(default_factory=dict)
    public_bm: BipolarMultitree = field(init=False)
    trace: list[Contribution] = field(default_factory=list)
    publish_errors: list[PublishError] = field(default_factory=list)
    round_index: int = 0

    def __post_init__(self) -> None:
        self.public_bm = BipolarMultitree(topics=set(self.topics))


@dataclass(frozen=True)
class ExchangeResult:
    topics: list[str]
    resolved: bool
    rounds: int
    final_state: DialogueState
    final_strengths: dict[str, float]
    final_stances: dict[str, bool]
    trace: list[Contribution]
    metrics: dict[str, float | int | bool | object]

#######################################






class MTAX:
    def __init__(self, agents: list[MTAXAgent], topics: list[str], config: ExchangeConfig | None = None) -> None:
        self.agents = agents
        self.topics = topics
        self.config = config or ExchangeConfig()
        self._state = DialogueState(topics=topics)
        for agent in self.agents:
            agent.initialize(topics)


    def __iter__(self):
        while self._state.round_index < self.config.max_rounds:
            yield self.step()


    @property
    def state(self) -> DialogueState:
        return self._state


    def step(self) -> DialogueState:
        if self._state.round_index >= self.config.max_rounds:
            return self._state
        self._state.publish_errors = []
        for agent in self.agents:
            feedback = None
            errors: list[PublishError] = []
            for _ in range(self.config.max_retries + 1):
                disclosure = agent.contribute(violation_feedback=feedback)
                if disclosure is None:
                    break
                contribution = Contribution(disclosure=disclosure, agent=agent.name, round_index=self._state.round_index,)
                errors = self._publish(contribution)
                if not errors:
                    self._state.trace.append(contribution)
                    for current_agent in self.agents:
                        current_agent.ingest(disclosure)
                    break
                feedback = errors[0].reason
            else:
                self._state.publish_errors.extend(errors)
        self._state.round_index += 1
        return self._state


    def _publish(self, contribution: Contribution) -> list[PublishError]:
        disclosure = contribution.disclosure
        new_arguments = {argument.label: argument for argument in disclosure.arguments}
        if len(new_arguments) != len(disclosure.arguments):
            return [self._publish_error(contribution, "Argument labels must be unique within a disclosure.")]

        existing_labels = self._state.public_bm.arguments
        redefined = existing_labels & new_arguments.keys()
        if redefined:
            label = sorted(redefined)[0]
            return [self._publish_error(contribution, f"Argument '{label}' is already in the exchange.")]

        known_labels = existing_labels | new_arguments.keys()
        for relation in disclosure.relations:
            if relation.source not in known_labels:
                return [self._publish_error(contribution,
                                            f"Relation source '{relation.source}' is not in the exchange or disclosure.",
                                            relation)]

        related_labels = set()
        for relation in disclosure.relations:
            related_labels.add(relation.source)
            related_labels.add(relation.target)
        unused_arguments = new_arguments.keys() - related_labels
        if unused_arguments:
            label = sorted(unused_arguments)[0]
            return [self._publish_error(contribution, f"Argument '{label}' has no relation.")]

        try:
            self._state.public_bm.add_relations(disclosure.relations)
        except ValueError as error:
            return [self._publish_error(contribution, str(error))]

        self._state.public_arguments.update(new_arguments)
        return []


    @staticmethod
    def _publish_error(contribution: Contribution, reason: str, relation: Relation | None = None) -> PublishError:
        return PublishError(
            agent=contribution.agent,
            contribution=contribution,
            relation=relation,
            reason=reason,
        )



    def contributor_mapping(self, relation: Relation) -> tuple[str, int] | None:
        for contribution in self._state.trace:
            if relation in contribution.disclosure.relations:
                return contribution.agent, contribution.round_index
        return None


    def result(self) -> ExchangeResult:
        resolved = False
        return ExchangeResult(
            topics=list(self.topics),
            resolved=resolved,
            rounds=self._state.round_index,
            final_state=self._state,
            final_strengths={},
            final_stances={},
            trace=list(self._state.trace),
            metrics={
                "rounds": self._state.round_index,
                "num_contributions": len(self._state.trace),
                "resolved": resolved,
            },
        )

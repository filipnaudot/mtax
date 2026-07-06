from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mtax.bm import BipolarMultitree
from mtax.config import ExchangeConfig
from mtax.resolution import Resolution
from mtax.schema import Argument, Disclosure, Pass, Relation
from mtax.turn_taking import BasicTurnTaking, TurnTaking

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


@dataclass(frozen=True)
class AgentStatus:
    agent: str
    outcome: Literal["published", "rejected", "passed", "no_response"]
    detail: str | None = None
    attempts: int = 1


class InvalidAgentResponse(ValueError):
    pass


@dataclass
class DialogueState:
    topics: list[str]
    public_arguments: dict[str, Argument] = field(default_factory=dict)
    public_bm: BipolarMultitree = field(init=False)
    trace: list[Contribution] = field(default_factory=list)
    publish_errors: list[PublishError] = field(default_factory=list)
    agent_statuses: list[AgentStatus] = field(default_factory=list)
    round_index: int = 0

    def __post_init__(self) -> None:
        self.public_bm = BipolarMultitree(topics=set(self.topics))


@dataclass(frozen=True)
class ExchangeResult:
    topics: list[str]
    resolved: bool
    termination_reason: Literal["resolved", "max_rounds"] | None
    rounds: int
    final_state: DialogueState
    final_strengths: dict[str, float]
    final_stances: dict[str, bool]
    trace: list[Contribution]
    metrics: dict[str, float | int | bool | object]

#######################################






class MTAX:
    def __init__(self,
                 agents: list[MTAXAgent],
                 topics: list[str],
                 config: ExchangeConfig | None = None,
                 turn_taking: TurnTaking | None = None) -> None:
        self.agents = agents
        self.topics = topics
        self.config = config or ExchangeConfig()
        self.turn_taking = turn_taking or BasicTurnTaking()
        if self.config.resolution == "top_r" and (len(self.topics) < 2 or not 1 <= self.config.top_r <= len(self.topics)):
            raise ValueError("top_r requires at least 2 topics and must not exceed the number of topics")
        self._state = DialogueState(topics=topics)
        for agent in self.agents:
            agent.initialize(topics, self.config.semantics)
        self.resolution = Resolution(self.agents, self.topics)


    def __iter__(self):
        while self._state.round_index < self.config.max_rounds:
            if (self._state.round_index > 0) and self.config.stop_when_resolved and self.is_resolved():
                break
            yield self.step()


    @property
    def state(self) -> DialogueState:
        return self._state


    def step(self) -> DialogueState:
        if self._state.round_index >= self.config.max_rounds:
            return self._state
        self._state.publish_errors = []
        self._state.agent_statuses = []
        for agent in self.turn_taking(self.agents):
            feedback = None
            errors: list[PublishError] = []
            invalid_response = None
            for attempt in range(1, self.config.max_retries + 2):
                try:
                    disclosure = agent.contribute(violation_feedback=feedback)
                except InvalidAgentResponse as error:
                    invalid_response = str(error)
                    feedback = invalid_response
                    continue
                if disclosure is None:
                    invalid_response = ("No structured response was returned. Return a Disclosure or an explicit Pass.")
                    feedback = invalid_response
                    continue
                if isinstance(disclosure, Pass):
                    self._state.agent_statuses.append(AgentStatus(agent.name, "passed", disclosure.reason, attempt))
                    break
                contribution = Contribution(disclosure=disclosure, agent=agent.name, round_index=self._state.round_index,)
                errors = self._publish(contribution)
                if not errors:
                    self._state.trace.append(contribution)
                    labels = ", ".join(argument.label for argument in disclosure.arguments)
                    self._state.agent_statuses.append(AgentStatus(agent.name, "published", labels or "relations only", attempt))
                    for current_agent in self.agents:
                        current_agent.ingest(disclosure)
                    break
                feedback = errors[0].reason
            else:
                if errors:
                    self._record_rejection(agent.name, errors, self.config.max_retries + 1)
                else:
                    self._state.agent_statuses.append(AgentStatus(agent.name, "no_response", invalid_response, self.config.max_retries + 1))
        self._state.round_index += 1
        return self._state


    def _record_rejection(self, agent: str, errors: list[PublishError], attempts: int) -> None:
        self._state.publish_errors.extend(errors)
        reason = errors[0].reason if errors else "Disclosure was rejected."
        self._state.agent_statuses.append(AgentStatus(agent, "rejected", reason, attempts))


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


    def is_resolved(self) -> bool:
        if self.config.resolution == "top_r":
            return self.resolution.top_r(self.config.top_r)
        return self.resolution.stance()


    def result(self) -> ExchangeResult:
        resolved = False
        if (self._state.round_index > 0):
            resolved = self.is_resolved()
        termination_reason = None
        if resolved and self.config.stop_when_resolved:
            termination_reason = "resolved"
        elif self._state.round_index >= self.config.max_rounds:
            termination_reason = "max_rounds"
        return ExchangeResult(
            topics=list(self.topics),
            resolved=resolved,
            termination_reason=termination_reason,
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

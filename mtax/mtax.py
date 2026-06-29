from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from mtax.config import ExchangeConfig

if TYPE_CHECKING:
    from mtax.agent import Agent


#####################################################
# DATA CLASSES
#####################################################
@dataclass(frozen=True)
class Argument:
    label: str
    text: str


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class Contribution:
    label: str
    argument: str
    relations: tuple[Relation, ...] = ()
    agent: str = ""
    round_index: int = -1


@dataclass
class DialogueState:
    topics: list[str]
    public_arguments: dict[str, Argument] = field(default_factory=dict)
    public_relations: set[Relation] = field(default_factory=set)
    trace: list[Contribution] = field(default_factory=list)
    round_index: int = 0


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
#####################################################


class MTAX:
    def __init__(self, agents: list[Agent], topics: list[str], config: ExchangeConfig | None = None) -> None:
        self.agents = agents
        self.topics = topics
        self.config = config or ExchangeConfig()
        self._state = DialogueState(topics=topics)

    @property
    def state(self) -> DialogueState:
        return self._state

    def step(self) -> DialogueState:
        if self._state.round_index >= self.config.max_rounds:
            return self._state
        for agent in self.agents:
            agent.ingest_public_state(self._state)
            contribution = agent.step(self._state)
            if contribution is not None:
                recorded = replace(
                    contribution,
                    agent=agent.name,
                    round_index=self._state.round_index,
                )
                self._state.trace.append(recorded)
                self._publish(recorded)
                for current_agent in self.agents:
                    current_agent.ingest_public_state(self._state)
        self._state.round_index += 1
        return self._state

    def _publish(self, contribution: Contribution) -> None:
        self._state.public_arguments[contribution.label] = Argument(
            label=contribution.label,
            text=contribution.argument,
        )
        self._state.public_relations.update(contribution.relations)

    def contributor_mapping(self, relation: Relation) -> tuple[str, int] | None:
        for contribution in self._state.trace:
            for current_relation in contribution.relations:
                if current_relation == relation:
                    return contribution.agent, contribution.round_index
        return None

    def run(self) -> ExchangeResult:
        while self._state.round_index < self.config.max_rounds:
            self.step()
        return self.result()

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

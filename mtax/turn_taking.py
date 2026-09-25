from abc import ABC, abstractmethod

from mtax.agent import MTAXAgent


class TurnTaking(ABC):
    def __init__(self, agents: list[MTAXAgent]) -> None:
        self.agents = agents

    @abstractmethod
    def __call__(self, time_step: int) -> MTAXAgent:
        """Return the agent selected to act."""


class BasicTurnTaking(TurnTaking):
    def __call__(self, time_step: int) -> MTAXAgent:
        return self.agents[time_step % len(self.agents)]

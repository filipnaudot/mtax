from abc import ABC, abstractmethod

from mtax.agent import MTAXAgent


class TurnTaking(ABC):
    @abstractmethod
    def __call__(self, agents: list[MTAXAgent]) -> list[MTAXAgent]:
        """Return the agents selected to act, in turn order."""


class BasicTurnTaking(TurnTaking):
    def __call__(self, agents: list[MTAXAgent]) -> list[MTAXAgent]:
        return list(agents)

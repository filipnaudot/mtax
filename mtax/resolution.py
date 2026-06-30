from collections.abc import Sequence

from mtax.agent import MTAXAgent


class Resolution:
    def __init__(self, agents: Sequence[MTAXAgent], topics: Sequence[str]) -> None:
        self.agents = agents
        self.topics = topics


    def top_r(self, r: int) -> bool:
        if not 1 <= r <= len(self.topics):
            raise ValueError("r must be between 1 and the number of topics")
        if not self.agents:
            return False
        rankings = [self._top_r(agent, r) for agent in self.agents]
        return None not in rankings and len(set(rankings)) == 1


    def stance(self, negative_below: float = 0.5, positive_above: float = 0.5) -> bool:
        if negative_below > positive_above:
            raise ValueError("negative_below must not exceed positive_above")
        if not self.agents:
            return False
        return all(
            len({
                self._stance(
                    agent.private_qbaf.final_strength(topic),
                    negative_below,
                    positive_above,
                )
                for agent in self.agents
            }) == 1
            for topic in self.topics
        )
    

    def _top_r(self, agent: MTAXAgent, r: int) -> tuple[str, ...] | None:
        strengths = {
            topic: agent.private_qbaf.final_strength(topic)
            for topic in self.topics
        }
        ranking = tuple(sorted(self.topics, key=strengths.__getitem__, reverse=True))
        relevant = ranking[: min(r + 1, len(ranking))]
        if any(strengths[first] == strengths[second] for first, second in zip(relevant, relevant[1:])):
            return None
        return ranking[:r]


    @staticmethod
    def _stance(strength: float, negative_below: float, positive_above: float) -> str:
        if strength < negative_below:
            return "negative"
        if strength > positive_above:
            return "positive"
        return "neutral"

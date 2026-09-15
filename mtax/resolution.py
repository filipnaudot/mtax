from collections.abc import Sequence
from itertools import groupby

from mtax.agent import MTAXAgent


class Resolution:
    def __init__(self, agents: Sequence[MTAXAgent], topics: Sequence[str]) -> None:
        self.agents = agents
        self.topics = topics


    def top_r(self, r: int) -> bool:
        if len(self.topics) < 2 or not 1 <= r <= len(self.topics):
            raise ValueError("top_r requires at least 2 topics and must not exceed the number of topics")
        if not self.agents:
            return False
        rankings = [self.top_r_ranking(agent, r) for agent in self.agents]
        return len(set(rankings)) == 1


    def stance(self) -> bool:
        if not self.agents:
            return False
        return all(
            len({agent.stance(topic) for agent in self.agents}) == 1
            for topic in self.topics
        )
    

    def top_r_ranking(self, agent: MTAXAgent, r: int) -> tuple[frozenset[str], ...]:
        strengths = {
            topic: agent.private_qbaf.final_strength(topic)
            for topic in self.topics
        }
        ranking = sorted(self.topics, key=strengths.__getitem__, reverse=True)
        groups = []
        covered_topics = 0
        for _, topics in groupby(ranking, key=strengths.__getitem__):
            group = frozenset(topics)
            groups.append(group)
            covered_topics += len(group)
            if covered_topics >= r: break
        return tuple(groups)

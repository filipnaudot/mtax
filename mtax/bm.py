from __future__ import annotations

from copy import deepcopy
from collections import deque
from collections.abc import Iterable

from mtax.schema import Relation




class BipolarMultitree:
    def __init__(self, topics: set[str]) -> None:
        self.topics: frozenset[str] = frozenset(topics)
        self._outgoing: dict[str, set[str]] = {topic: set() for topic in topics}
        self._incoming: dict[str, set[str]] = {topic: set() for topic in topics}
        self._relations: set[tuple[str, str, str]] = set()


    @property
    def arguments(self) -> set[str]:
        return set(self._outgoing)


    @property
    def relations(self) -> set[tuple[str, str, str]]:
        return set(self._relations)


    def add_relation(self, source: str, target: str, kind: str) -> None:
        if source in self.topics:
            raise ValueError(f"topic '{source}' cannot be a relation source. Outgoing relations from a topic argument is not allowed.")
        if target not in self._outgoing:
            raise ValueError(f"'{target}' is not in the exchange and can therefore not be the relation target.")
        if self._reaches(start=target, destination=source):
            raise ValueError(f"'{source}' --> '{target}' would create a cycle in the public exchange graph which is not allowed.")

        conflicting_node = self._find_single_path_conflict(source, target)
        if conflicting_node is not None:
            raise ValueError(f"'{source}' --> '{target}' creates two paths to '{conflicting_node}'")
        
        self._outgoing.setdefault(source, set()).add(target)
        self._incoming.setdefault(source, set())
        self._incoming.setdefault(target, set()).add(source)
        self._relations.add((source, target, kind))


    def add_relations(self, relations: Iterable[Relation]) -> None:
        pending = list(relations)
        candidate = deepcopy(self)
        while pending:
            next_relation = None
            for relation in pending:
                if relation.target in candidate.arguments:
                    next_relation = relation
                    candidate.add_relation(relation.source, relation.target, relation.kind)
                    pending.remove(relation)
                    break
            if next_relation is None:
                unresolved = ", ".join(
                    f"'{relation.source}' --> '{relation.target}'"
                    for relation in pending
                )
                available_targets = ", ".join(
                    f"'{argument}'"
                    for argument in sorted(candidate.arguments)
                )
                raise ValueError(
                    f"Cannot connect unresolved relations: {unresolved}. "
                    f"Each target must be an available argument or topic. "
                    f"Available targets: {available_targets}."
                )
        self._outgoing = candidate._outgoing
        self._incoming = candidate._incoming
        self._relations = candidate._relations


    def _reaches(self, start: str, destination: str) -> bool:
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == destination:
                return True
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self._outgoing.get(node, ()))
        return False


    def _bfs(self, sources: set[str], edges: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(edges.get(node, ()))
        return visited


    def _find_single_path_conflict(self, source: str, target: str) -> str | None:
        if source not in self._outgoing:
            return None  # new argument, no ancestors —-> trivially safe

        ancestors_of_source = self._bfs({source}, self._incoming)
        descendants_of_target = self._bfs({target}, self._outgoing)
        reachable_from_source_ancestors = self._bfs(ancestors_of_source, self._outgoing)

        overlap = reachable_from_source_ancestors & descendants_of_target
        if not overlap:
            return None
        topic_overlap = overlap & self.topics
        return next(iter(topic_overlap if topic_overlap else overlap))

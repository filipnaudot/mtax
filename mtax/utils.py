from __future__ import annotations

from mtax.mtax import MTAX, Contribution, Relation


class MTAXTerminalUI:
    def __init__(self, exchange: MTAX) -> None:
        self.exchange = exchange

    def render(self) -> None:
        state = self.exchange.state
        result = self.exchange.result()
        print("\n" + "=" * 72)
        print(f"MTAX | round {state.round_index}/{self.exchange.config.max_rounds}")
        print(f"topics   : {', '.join(state.topics)}")
        print(f"resolved : {result.resolved}")
        print("-" * 72)
        self._render_arguments()
        self._render_relations()
        self._render_trace_tail()
        print("=" * 72)

    def _render_arguments(self) -> None:
        print("Public arguments")
        if not self.exchange.state.public_arguments:
            print("  none")
            return
        for argument in self.exchange.state.public_arguments.values():
            print(f"  {argument.label}: {argument.text}")

    def _render_relations(self) -> None:
        print("\nPublic relations")
        if not self.exchange.state.public_relations:
            print("  none")
            return
        for relation in sorted(self.exchange.state.public_relations, key=self._relation_key):
            contributor = self.exchange.contributor_mapping(relation)
            source = f"{relation.source} {relation.kind} {relation.target}"
            if contributor is None:
                print(f"  {source}")
            else:
                agent, round_index = contributor
                print(f"  {source}  [{agent}, round {round_index}]")

    def _render_trace_tail(self, size: int = 4) -> None:
        print("\nLatest contributions")
        trace = self.exchange.state.trace[-size:]
        if not trace:
            print("  none")
            return
        for contribution in trace:
            print(self._format_contribution(contribution))


    def _relation_key(self, relation: Relation) -> tuple[str, str, str]:
        return relation.target, relation.kind, relation.source


    def _format_contribution(self, contribution: Contribution) -> str:
        return (f"  r{contribution.round_index} {contribution.agent}: "
                f"{contribution.label} - {contribution.argument}")

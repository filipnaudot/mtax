from __future__ import annotations

from collections import defaultdict
from shutil import get_terminal_size
from textwrap import shorten

from mtax.mtax import MTAX, Contribution, Relation


class MTAXTerminalUI:
    def __init__(self, exchange: MTAX) -> None:
        self.exchange = exchange

    def render(self) -> None:
        state = self.exchange.state
        result = self.exchange.result()
        width = max(64, min(get_terminal_size((88, 24)).columns, 100))
        status = "RESOLVED" if result.resolved else "ACTIVE"
        if state.round_index >= self.exchange.config.max_rounds and not result.resolved:
            status = "FINISHED"

        print()
        print(f"╭─ MTAX {'─' * (width - 8)}╮")
        summary = (
            f"Round {state.round_index}/{self.exchange.config.max_rounds}"
            f"  ·  {len(state.public_arguments)} arguments"
            f"  ·  {len(state.public_bm.relations)} relations"
            f"  ·  {status}"
        )
        print(f"│ {summary:<{width - 2}} │")
        print(f"╰{'─' * width}╯")

        self._render_topic_strengths()
        self._render_agent_statuses(width)
        self._render_graph(width)
        self._render_trace_tail()

    def _render_topic_strengths(self) -> None:
        self._print_section("TOPIC STRENGTHS")
        for topic in self.exchange.topics:
            print(f"  ◎ {topic}")
            for agent in self.exchange.agents:
                try:
                    strength = agent.private_qbaf.final_strength(topic)
                except (RuntimeError, ValueError):
                    print(f"      {agent.name:<12} unavailable")
                    continue
                print(f"      {agent.name:<12} {self._strength_bar(strength)}  {strength:.3f}")

    def _render_agent_statuses(self, width: int) -> None:
        state = self.exchange.state
        round_index = max(0, state.round_index - 1)
        self._print_section(f"AGENT STATUS · ROUND {round_index}")
        if not state.agent_statuses:
            print("  Waiting for the first round.")
            return

        markers = {
            "published": "✓",
            "rejected": "✗",
            "passed": "○",
            "no_response": "?",
        }
        for status in state.agent_statuses:
            if status.outcome == "published":
                message = f"published {status.detail}"
            elif status.outcome == "rejected":
                attempt_label = "attempt" if status.attempts == 1 else "attempts"
                message = f"rejected after {status.attempts} {attempt_label}: {status.detail}"
            elif status.outcome == "passed":
                message = "passed" + (f": {status.detail}" if status.detail else "")
            else:
                message = "invalid response" + (f": {status.detail}" if status.detail else "")
            available = max(20, width - len(status.agent) - 10)
            message = shorten(message, width=available, placeholder="…")
            print(f"  {markers[status.outcome]} {status.agent:<12} {message}")

    def _render_graph(self, width: int) -> None:
        self._print_section("PUBLIC ARGUMENT GRAPH")
        state = self.exchange.state
        if not state.public_arguments:
            print("  No arguments disclosed yet.")
            return

        relations_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for source, target, kind in state.public_bm.relations:
            relations_by_source[source].append((target, kind))

        for source in self._ordered_sources(relations_by_source):
            argument = state.public_arguments[source]
            available = max(20, width - len(source) - 8)
            description = shorten(argument.text, width=available, placeholder="…")
            print(f"  [{source}]  {description}")
            relations = sorted(relations_by_source[source], key=lambda item: (item[0], item[1]))
            for index, (target, kind) in enumerate(relations):
                branch = "└─" if index == len(relations) - 1 else "├─"
                marker = "+ SUPPORT" if kind == "support" else "− ATTACK "
                destination = f"◎ {target}" if target in state.public_bm.topics else f"[{target}]"
                relation = Relation(source=source, target=target, kind=kind)
                contributor = self.exchange.contributor_mapping(relation)
                attribution = ""
                if contributor is not None:
                    agent, round_index = contributor
                    attribution = f"  ·  {agent}, r{round_index}"
                print(f"      {branch} {marker} ──▶ {destination}{attribution}")

    def _ordered_sources(self, relations_by_source: dict[str, list[tuple[str, str]]]) -> list[str]:
        depths = {topic: 0 for topic in self.exchange.topics}
        pending = set(relations_by_source)
        while pending:
            progressed = False
            for source in list(pending):
                target_depths = [
                    depths[target]
                    for target, _ in relations_by_source[source]
                    if target in depths
                ]
                if target_depths:
                    depths[source] = max(target_depths) + 1
                    pending.remove(source)
                    progressed = True
            if not progressed:
                break
        return sorted(relations_by_source, key=lambda source: (-depths.get(source, 0), source))

    def _render_trace_tail(self, size: int = 5) -> None:
        self._print_section("RECENT ACTIVITY")
        trace = self.exchange.state.trace[-size:]
        if not trace:
            print("  No contributions yet.")
            return
        for contribution in trace:
            print(self._format_contribution(contribution))

    @staticmethod
    def _format_contribution(contribution: Contribution) -> str:
        labels = ", ".join(argument.label for argument in contribution.disclosure.arguments)
        arguments = f"arguments: {labels}" if labels else "relations only"
        relation_count = len(contribution.disclosure.relations)
        return (
            f"  r{contribution.round_index:<3} {contribution.agent:<12} "
            f"{arguments}  ·  {relation_count} relation{'s' if relation_count != 1 else ''}"
        )

    @staticmethod
    def _strength_bar(strength: float, width: int = 14) -> str:
        bounded = max(0.0, min(1.0, strength))
        filled = round(bounded * width)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    @staticmethod
    def _print_section(title: str) -> None:
        print(f"\n  {title}")
        print(f"  {'─' * len(title)}")

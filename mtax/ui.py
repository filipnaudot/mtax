from __future__ import annotations

from collections import defaultdict
from textwrap import shorten

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mtax.mtax import MTAX, Contribution, Relation


class MTAXTerminalUI:
    def __init__(self, exchange: MTAX, console: Console | None = None) -> None:
        self.exchange = exchange
        self.console = console or Console()

    def render(self) -> None:
        self.console.print(Group(
            self._summary(),
            self._topic_strengths(),
            self._agent_statuses(),
            self._graph(),
            self._recent_activity(),
        ))

    def _summary(self) -> Panel:
        state = self.exchange.state
        result = self.exchange.result()
        status = "RESOLVED" if result.resolved else "ACTIVE"
        if state.round_index >= self.exchange.config.max_rounds and not result.resolved:
            status = "FINISHED"
        style = {"RESOLVED": "green", "ACTIVE": "yellow", "FINISHED": "red"}[status]
        summary = Text.assemble(
            f"Round {state.round_index}/{self.exchange.config.max_rounds}",
            "  •  ",
            f"{len(state.public_arguments)} arguments",
            "  •  ",
            f"{len(state.public_bm.relations)} relations",
            "  •  ",
            (status, f"bold {style}"),
        )
        return Panel(summary, title="[bold]MTAX[/bold]", border_style=style)

    def _topic_strengths(self) -> Table:
        table = Table(title="TOPIC STRENGTHS", box=box.SIMPLE, expand=True)
        table.add_column("Topic", style="bold cyan")
        table.add_column("Agent")
        table.add_column("Strength", justify="right")
        table.add_column("Value", justify="right")

        for topic in self.exchange.topics:
            for index, agent in enumerate(self.exchange.agents):
                try:
                    strength = agent.private_qbaf.final_strength(topic)
                    bar = self._strength_bar(strength)
                    value = f"{strength:.3f}"
                except (RuntimeError, ValueError):
                    bar = Text("unavailable", style="dim")
                    value = "—"
                table.add_row(topic if index == 0 else "", agent.name, bar, value)
        return table

    def _agent_statuses(self) -> Panel | Table:
        state = self.exchange.state
        round_index = max(0, state.round_index - 1)
        if not state.agent_statuses:
            return Panel("Waiting for the first round.", title=f"AGENT STATUS · ROUND {round_index}")

        table = Table(title=f"AGENT STATUS · ROUND {round_index}", box=box.SIMPLE, expand=True)
        table.add_column("", width=1)
        table.add_column("Agent", style="bold")
        table.add_column("Status")
        markers = {
            "published": ("✓", "green"),
            "rejected": ("✗", "red"),
            "passed": ("○", "yellow"),
            "no_response": ("?", "red"),
        }
        for status in state.agent_statuses:
            marker, style = markers[status.outcome]
            if status.outcome == "published":
                message = f"published {status.detail}"
            elif status.outcome == "rejected":
                attempts = "attempt" if status.attempts == 1 else "attempts"
                message = f"rejected after {status.attempts} {attempts}: {status.detail}"
            elif status.outcome == "passed":
                message = "passed" + (f": {status.detail}" if status.detail else "")
            else:
                message = "invalid response" + (f": {status.detail}" if status.detail else "")
            table.add_row(Text(marker, style=style), status.agent, message)
        return table

    def _graph(self) -> Panel:
        state = self.exchange.state
        if not state.public_arguments:
            return Panel("No arguments disclosed yet.", title="PUBLIC ARGUMENT EXCHANGE")

        relations_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for source, target, kind in state.public_bm.relations:
            relations_by_source[source].append((target, kind))

        lines: list[Text] = []
        for source in self._ordered_sources(relations_by_source):
            argument = state.public_arguments[source]
            lines.append(Text.assemble((f"[{source}]", "bold cyan"), f"  {shorten(argument.text, width=88, placeholder='…')}"))
            relations = sorted(relations_by_source[source], key=lambda item: (item[0], item[1]))
            for index, (target, kind) in enumerate(relations):
                branch = "└─" if index == len(relations) - 1 else "├─"
                marker = "+ SUPPORT" if kind == "support" else "− ATTACK "
                destination = f"◎ {target}" if target in state.public_bm.topics else f"[{target}]"
                contributor = self.exchange.contributor_mapping(Relation(source=source, target=target, kind=kind)) # type: ignore
                attribution = ""
                if contributor is not None:
                    agent, round_index = contributor
                    attribution = f"  ·  {agent}, r{round_index}"
                color = "green" if kind == "support" else "red"
                lines.append(Text.assemble(f"   {branch} ", (marker, color), f" ──▶ {destination}{attribution}"))
        return Panel(Group(*lines), title="PUBLIC ARGUMENT GRAPH")

    def _ordered_sources(self, relations_by_source: dict[str, list[tuple[str, str]]]) -> list[str]:
        depths = {topic: 0 for topic in self.exchange.topics}
        pending = set(relations_by_source)
        while pending:
            progressed = False
            for source in list(pending):
                target_depths = [depths[target] for target, _ in relations_by_source[source] if target in depths]
                if target_depths:
                    depths[source] = max(target_depths) + 1
                    pending.remove(source)
                    progressed = True
            if not progressed:
                break
        return sorted(relations_by_source, key=lambda source: (-depths.get(source, 0), source))

    def _recent_activity(self, size: int = 5) -> Panel:
        trace = self.exchange.state.trace[-size:]
        if not trace:
            return Panel("No contributions yet.", title="RECENT ACTIVITY")
        return Panel(Group(*(self._format_contribution(item) for item in trace)), title="RECENT ACTIVITY")

    @staticmethod
    def _format_contribution(contribution: Contribution) -> Text:
        labels = ", ".join(argument.label for argument in contribution.disclosure.arguments)
        arguments = f"arguments: {labels}" if labels else "relations only"
        count = len(contribution.disclosure.relations)
        return Text(
            f"r{contribution.round_index:<3} {contribution.agent:<12} "
            f"{arguments}  ·  {count} relation{'s' if count != 1 else ''}"
        )

    @staticmethod
    def _strength_bar(strength: float, width: int = 14) -> Text:
        bounded = max(0.0, min(1.0, strength))
        filled = round(bounded * width)
        color = "green" if bounded > 0.5 else "red" if bounded < 0.5 else "yellow"
        return Text.assemble(("█" * filled, color), ("░" * (width - filled), "dim"))

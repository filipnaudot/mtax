"""
Alien debate.
Two LLMs argue whether aliens exist.

Minimal example showing how to wire up an MTAX exchange:
  - subclass MTAXAgent and implement contribute() and rate()
  - iterate and render
"""
from __future__ import annotations
import json
import os
import re

from dotenv import load_dotenv
from autom8 import Agent as Autom8Agent

from mtax import MTAXAgent, ExchangeConfig, MTAX
from mtax.schema import Argument, Disclosure, Relation
from mtax.utils import MTAXTerminalUI





# ── System prompts ─────────────────────────────────────────────────────────────

_FORMAT = "\nReturn exactly one JSON object — no markdown, no extra text:\n" + json.dumps(Disclosure.model_json_schema(), indent=2)

PRO_PROMPT = "You believe aliens probably exist. Argue for aliens_exist." + _FORMAT
CON_PROMPT = "You believe aliens probably do not exist. Argue against aliens_exist." + _FORMAT







# ── Agent ──────────────────────────────────────────────────────────────────────

class LLMAgent(MTAXAgent):
    def __init__(self, name: str, model, stance_prompt: str) -> None:
        super().__init__(name)
        self._model = model
        self._stance_prompt = stance_prompt


    def contribute(self, violation_feedback=None) -> Disclosure | None:
        message = self._contribution_prompt()
        if violation_feedback:
            message += f"\n\nYour previous relation was rejected: {violation_feedback}\nPlease try again."
        output = self._model.invoke(message=message, instructions=self._stance_prompt)
        if isinstance(output, Disclosure):
            return output
        try:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL)
            return Disclosure.model_validate_json(match.group(1) if match else output)
        except Exception:
            return None


    def rate(self, argument) -> float:
        output = self._model.invoke(message=self._initial_strength_prompt(argument), instructions=self._stance_prompt)
        try:
            return max(0.0, min(1.0, float(output)))
        except (ValueError, TypeError):
            return 0.5


    #### Prompts
    def _contribution_prompt(self) -> str:
        topics = ", ".join(self.topics)
        args = "\n".join(f"  {a.label}: {a.text}" for a in self.private_arguments.values()) or "  none yet"
        rels = "\n".join(
            f"  {relation.source} {relation.kind} {relation.target}"
            for relation in self.private_relations
        ) or "  none yet"
        return f"Topics: {topics}\n\nKnown arguments:\n{args}\n\nKnown relations:\n{rels}\n\nContribute a new argument. You MUST include at least one relation targeting a known argument label or topic."


    def _initial_strength_prompt(self, argument) -> str:
        return f"Rate how convincing this is. Return only a number from 0.0 to 1.0.\n{argument.label}: {argument.text}"







# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY in a .env file before running.")

    exchange = MTAX(
        agents=[LLMAgent("pro", Autom8Agent(model="gpt-4o-mini", system_prompt=PRO_PROMPT), PRO_PROMPT),
                LLMAgent("con", Autom8Agent(model="gpt-4o-mini", system_prompt=CON_PROMPT), CON_PROMPT)],
        topics=["aliens_exist"],
        config=ExchangeConfig(max_rounds=10),
    )

    ui = MTAXTerminalUI(exchange)
    ui.render()
    for state in exchange:
        ui.render()


if __name__ == "__main__":
    main()




































# ── HumanTerminalAgent ─────────────────────────────────────────────────────────
# Drop-in replacement for an LLM — prompts a human in the terminal.
# Usage: HumanTerminalAgent("human")

class HumanTerminalAgent(MTAXAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._pending_strengths: dict[str, float] = {}


    def contribute(self, violation_feedback=None) -> Disclosure | None:
        if violation_feedback:
            print(f"\nRejected: {violation_feedback}")
        print(f"\nTopics: {', '.join(self.topics)}")
        for argument in self.private_arguments.values():
            print(f"  {argument.label}: {argument.text}")
        label = input("\nlabel (blank to pass): ").strip()
        if not label:
            return None
        text   = input("text: ").strip()
        target = input("relation target [aliens_exist]: ").strip() or "aliens_exist"
        kind   = self._prompt_kind()
        self._pending_strengths[label] = self._prompt_strength()
        return Disclosure(
            arguments=[Argument(label=label, text=text)],
            relations=[Relation(source=label, target=target, kind=kind)],
        )


    def rate(self, argument) -> float:
        return self._pending_strengths.pop(argument.label, 0.5)


    def _prompt_kind(self) -> str:
        while True:
            kind = input("kind [attack/support]: ").strip().lower()
            if kind in {"attack", "support"}:
                return kind


    def _prompt_strength(self) -> float:
        while True:
            try:
                strength = float(input("initial strength [0.0-1.0]: ").strip())
                if 0.0 <= strength <= 1.0:
                    return strength
            except ValueError:
                pass

from __future__ import annotations
import os

from dotenv import load_dotenv
from autom8 import Agent as Autom8Agent

from mtax import Agent, Contribution, ExchangeConfig, MTAX, Relation
from mtax.prompts import (
    ANTI_ALIEN_SYSTEM_PROMPT,
    PRO_ALIEN_SYSTEM_PROMPT,
    build_alien_contribution_prompt,
    build_alien_initial_strength_prompt,
)
from mtax.utils import MTAXTerminalUI




def main() -> None:
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY in a .env file before running this example.")

    model_name = os.environ.get("MTAX_MODEL", "gpt-4o-mini")
    pro_agent = Agent(name="pro",
                      model=Autom8Agent(model=model_name, system_prompt=PRO_ALIEN_SYSTEM_PROMPT),
                      contribution_prompt=build_alien_contribution_prompt,
                      initial_strength_prompt=build_alien_initial_strength_prompt,
                      instructions=PRO_ALIEN_SYSTEM_PROMPT)
    con_agent = Agent(name="con",
                      model=Autom8Agent(model=model_name, system_prompt=ANTI_ALIEN_SYSTEM_PROMPT),
                      contribution_prompt=build_alien_contribution_prompt,
                      initial_strength_prompt=build_alien_initial_strength_prompt,
                      instructions=ANTI_ALIEN_SYSTEM_PROMPT)
    # human = Agent(name="human", model=HumanTerminalModel())

    exchange = MTAX(agents=[pro_agent, con_agent],
                    topics=["aliens_exist"],
                    config=ExchangeConfig(max_rounds=10))
    ui = MTAXTerminalUI(exchange)
    ui.render()

    while exchange.state.round_index < exchange.config.max_rounds:
        exchange.step()
        ui.render()


if __name__ == "__main__":
    main()







class HumanTerminalModel:
    def __init__(self) -> None:
        self.strengths: dict[str, float] = {}

    def step(self, state) -> Contribution | None:
        print("\nHuman contribution")
        print(f"Topics: {', '.join(state.topics)}")
        if state.public_arguments:
            print("Public arguments:")
            for argument in state.public_arguments.values():
                print(f"- {argument.label}: {argument.text}")

        label = input("label, blank to skip: ").strip()
        if not label:
            return None

        argument = input("argument: ").strip()
        source = input(f"relation source [{label}]: ").strip() or label
        target = input("relation target [aliens_exist]: ").strip() or "aliens_exist"
        kind = self._prompt_kind()
        strength = self._prompt_strength()
        self.strengths[label] = strength

        return Contribution(
            label=label,
            argument=argument,
            relations=(Relation(source=source, target=target, kind=kind),),
        )

    def assign_initial_strength(self, argument, state) -> float:
        return self.strengths.get(argument.label, 0.5)

    def _prompt_kind(self) -> str:
        while True:
            kind = input("relation kind [support/attack]: ").strip().lower()
            if kind in {"support", "attack"}:
                return kind
            print("Please enter support or attack.")

    def _prompt_strength(self) -> float:
        while True:
            raw = input("initial strength [0.0-1.0]: ").strip()
            try:
                strength = float(raw)
            except ValueError:
                print("Please enter a number from 0.0 to 1.0.")
                continue
            if 0.0 <= strength <= 1.0:
                return strength
            print("Please enter a number from 0.0 to 1.0.")


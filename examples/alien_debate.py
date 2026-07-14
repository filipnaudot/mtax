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
from pydantic import TypeAdapter

from mtax import InvalidAgentResponse, MTAXAgent, ExchangeConfig, MTAX
from mtax.schema import Argument, Disclosure, Pass, Relation
from mtax.utils import MTAXTerminalUI





# System prompts
_RESPONSE_ADAPTER = TypeAdapter(Disclosure | Pass)
_RESPONSE_SCHEMA = json.dumps(_RESPONSE_ADAPTER.json_schema(), indent=2)

PRO_PROMPT = (
    "You initially believe aliens probably exist, followed by uncertainty, then non-existence. "
    "Debate in good faith, evaluate every argument fairly, and revise your view when the evidence "
    "warrants it. Work with the other agent toward a shared evidence-based conclusion."
)
CON_PROMPT = (
    "You initially believe aliens probably do not exist, followed by uncertainty, then existence. "
    "Debate in good faith, evaluate every argument fairly, and revise your view when the evidence "
    "warrants it. Work with the other agent toward a shared evidence-based conclusion."
)
TURN_ORDER = ("agent-alien", "agent-no-aliens")







class LLMAgent(MTAXAgent):
    def __init__(self, name: str, model, stance_prompt: str, topic_relation_kind: str) -> None:
        super().__init__(name)
        self._model = model
        self._stance_prompt = stance_prompt
        self._topic_relation_kind = topic_relation_kind


    def contribute(self, public_bm, violation_feedback=None) -> Disclosure | Pass | None:
        message = self._contribution_prompt()
        if violation_feedback:
            message += (
                "\n\nYour previous response was rejected by MTAX:\n"
                f"{violation_feedback}\n"
                "Correct the response and return a new valid Disclosure or Pass."
            )
        instructions = f"{self._stance_prompt}\n\n{self._response_instructions()}"
        output = self._model.invoke(message=message, instructions=instructions, chat_id=0)
        if isinstance(output, (Disclosure, Pass)):
            return output
        try:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL)
            return _RESPONSE_ADAPTER.validate_json(match.group(1) if match else output)
        except Exception as error:
            raw_output = str(output).replace("\n", " ")[:300]
            raise InvalidAgentResponse(
                f"Response did not match the Disclosure or Pass schema: {error}. "
                f"Raw output: {raw_output}"
            ) from error


    def rate(self, argument) -> float:
        output = self._model.invoke(
            message=self._initial_strength_prompt(argument),
            instructions=self._stance_prompt,
            chat_id=1,
        )
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
        if self.private_arguments:
            example_target = next(reversed(self.private_arguments))
            example_description = "Valid argument-to-argument Disclosure example"
        else:
            example_target = self.topics[0]
            example_description = "Valid first Disclosure example"
        example = {
            "arguments": [{"label": "unique_argument_label", "text": "The argument text."}],
            "relations": [{
                "source": "unique_argument_label",
                "target": example_target,
                "kind": self._topic_relation_kind,
            }],
        }
        return (
            f"You are agent '{self.name}'. The turn order every round is "
            f"{' --> '.join(TURN_ORDER)}. Agents act sequentially. "
            "The knowledge below already includes accepted disclosures from agents who acted before you.\n\n"
            f"Topics (use these exact labels): {topics}\n\n"
            f"Known arguments:\n{args}\n\n"
            f"Known relations:\n{rels}\n\n"
            "Return a Disclosure that contributes useful reasoning, or return an explicit Pass. "
            "Every new argument must appear in arguments with a unique label. Relations point from "
            "the reason toward the claim it affects. Every relation target must exactly match a topic, "
            "a known argument label, or another new argument that eventually connects to a topic. "
            "Use support for a supporting reason and attack for an opposing reason. When known arguments "
            "exist, prefer responding directly to the most relevant existing argument. Target the topic "
            "directly only when your argument does not naturally respond to an existing argument. The "
            "relation kind describes the local effect on its target: you may attack an opposing argument "
            "or support an aligned argument.\n\n"
            f"{example_description}:\n{json.dumps(example, indent=2)}"
        )


    def _response_instructions(self) -> str:
        return (
            "Return exactly one JSON object with no markdown or extra text. "
            "It must validate against this Disclosure-or-Pass schema:\n"
            f"{_RESPONSE_SCHEMA}"
        )


    def _initial_strength_prompt(self, argument) -> str:
        return f"Rate how convincing this is. Return only a number from 0.0 to 1.0.\n{argument.label}: {argument.text}"







# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY in a .env file before running.")

    exchange = MTAX(
        agents=[LLMAgent("agent-alien", Autom8Agent(model="gpt-4o-mini", system_prompt=PRO_PROMPT), PRO_PROMPT, "support"),
                LLMAgent("agent-no-aliens", Autom8Agent(model="gpt-4o-mini", system_prompt=CON_PROMPT), CON_PROMPT, "attack")],
        topics=["aliens_exist", "aliens_do_not_exist", "aliens_may_exist"],
        config=ExchangeConfig(max_rounds=10),
    )

    ui = MTAXTerminalUI(exchange)
    ui.render()
    for state in exchange:
        ui.render()






if __name__ == "__main__":
    main()

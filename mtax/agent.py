from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from mtax.mtax import Argument, Contribution, DialogueState, Relation

ContributionPrompt = Callable[[DialogueState], str] | str
InitialStrengthPrompt = Callable[[Argument, DialogueState], str] | str


class Agent:
    def __init__(self,
                 name: str,
                 model: Any,
                 private_arguments: dict[str, Argument] | None = None,
                 private_relations: set[Relation] | None = None,
                 private_strengths: dict[str, float] | None = None,
                 contribution_prompt: ContributionPrompt | None = None,
                 initial_strength_prompt: InitialStrengthPrompt | None = None,
                 instructions: str | None = None) -> None:
        self.name = name
        self.model = model
        self.private_arguments = (private_arguments if private_arguments is not None else {})
        self.private_relations = (private_relations if private_relations is not None else set())
        self.private_strengths = (private_strengths if private_strengths is not None else {})
        self.contribution_prompt = contribution_prompt
        self.initial_strength_prompt = initial_strength_prompt
        self.instructions = instructions

    def missing_public_arguments(self, state: DialogueState) -> dict[str, Argument]:
        return {
            label: argument
            for label, argument in state.public_arguments.items()
            if label not in self.private_arguments
        }

    def missing_public_relations(self, state: DialogueState) -> set[Relation]:
        return state.public_relations - self.private_relations

    def assign_initial_strength(self, argument: Argument, state: DialogueState) -> float:
        model_method = getattr(self.model, "assign_initial_strength", None)
        if callable(model_method):
            strength = model_method(argument, state)
            if not isinstance(strength, int | float):
                raise TypeError("Agent model assign_initial_strength must return int or float; "
                                f"got {type(strength).__name__}")
            return float(strength)
        model_invoke = getattr(self.model, "invoke", None)
        if callable(model_invoke) and self.initial_strength_prompt is not None:
            output = model_invoke(message=self._build_initial_strength_prompt(argument, state),
                                  instructions=self.instructions)
            return _parse_strength(output)
        return 0.5

    def ingest_public_state(self, state: DialogueState) -> tuple[dict[str, Argument], set[Relation]]:
        missing_arguments = self.missing_public_arguments(state)
        missing_relations = self.missing_public_relations(state)

        for label, argument in missing_arguments.items():
            self.private_arguments[label] = argument
            self.private_strengths[label] = self.assign_initial_strength(
                argument,
                state,
            )

        self.private_relations.update(missing_relations)
        return missing_arguments, missing_relations

    def step(self, state: "DialogueState") -> "Contribution | None":
        model_step = getattr(self.model, "step", None)
        if callable(model_step):
            output = model_step(state)
            return _coerce_contribution(output, self.name)

        model_invoke = getattr(self.model, "invoke", None)
        if not callable(model_invoke) or self.contribution_prompt is None:
            return None

        output = model_invoke(
            message=self._build_contribution_prompt(state),
            instructions=self.instructions,
        )
        return _coerce_contribution(output, self.name)

    def _build_contribution_prompt(self, state: DialogueState) -> str:
        if callable(self.contribution_prompt):
            return self.contribution_prompt(state)
        if isinstance(self.contribution_prompt, str):
            return self.contribution_prompt
        raise RuntimeError("Agent contribution_prompt is not configured")

    def _build_initial_strength_prompt(self, argument: Argument, state: DialogueState) -> str:
        if callable(self.initial_strength_prompt):
            return self.initial_strength_prompt(argument, state)
        if isinstance(self.initial_strength_prompt, str):
            return self.initial_strength_prompt
        raise RuntimeError("Agent initial_strength_prompt is not configured")


def _coerce_contribution(output: Any, agent_name: str) -> Contribution | None:
    if output is None:
        return None
    if isinstance(output, Contribution):
        return output
    if isinstance(output, str):
        parsed = _parse_contribution(output)
        if parsed is not None:
            return parsed
        return Contribution(label=agent_name, argument=output)

    raise TypeError(
        "Agent model step must return None, str, or Contribution; "
        f"got {type(output).__name__}"
    )


def _parse_contribution(output: str) -> Contribution | None:
    try:
        data = json.loads(_extract_json(output))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    label = data.get("label")
    argument = data.get("argument")
    if not isinstance(label, str) or not isinstance(argument, str):
        return None

    relations = []
    for relation_data in data.get("relations", []):
        if not isinstance(relation_data, dict):
            continue
        source = relation_data.get("source")
        target = relation_data.get("target")
        kind = relation_data.get("kind")
        if isinstance(source, str) and isinstance(target, str) and kind in {
            "attack",
            "support",
        }:
            relations.append(Relation(source=source, target=target, kind=kind))

    return Contribution(
        label=label,
        argument=argument,
        relations=tuple(relations),
    )


def _extract_json(output: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL)
    if match:
        return match.group(1)
    return output.strip()


def _parse_strength(output: Any) -> float:
    if isinstance(output, int | float):
        return float(output)
    if not isinstance(output, str):
        raise TypeError(
            "Agent model initial strength output must be str, int, or float; "
            f"got {type(output).__name__}"
        )

    match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", output)
    if match is None:
        raise ValueError("Could not parse initial strength from model output")
    return float(match.group(0))

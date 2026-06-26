from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import autom8

from mtax.mtax import Contribution, DialogueState


@dataclass
class Agent:
    name: str
    model: Any

    def step(self, state: "DialogueState") -> "Contribution | None":
        model_step = getattr(self.model, "step", None)
        if not callable(model_step):
            return None

        output = model_step(state)
        if output is None:
            return None
        if isinstance(output, Contribution):
            return output
        if isinstance(output, str):
            return Contribution(label=self.name, argument=output)

        raise TypeError("Agent model step must return None, str, or Contribution; "
                        f"got {type(output).__name__}")

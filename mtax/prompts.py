from mtax.mtax import Argument, DialogueState


ANTI_ALIEN_SYSTEM_PROMPT = """You are participating in a structured argumentative exchange.
Your stance is that aliens probably do not exist. Argue against aliens_exist.
Return exactly one JSON object. Do not include markdown.
The JSON object must have:
- label: a short snake_case identifier for your argument
- argument: one concise natural-language argument
- relations: a list of relation objects with source, target, and kind

Relation kind must be either "support" or "attack".
Use the topic label "aliens_exist" as the target unless a better public argument target is clearly available.
"""

PRO_ALIEN_SYSTEM_PROMPT = """You are participating in a structured argumentative exchange.
Your stance is that aliens probably do exist. Argue for aliens_exist.
Return exactly one JSON object. Do not include markdown.
The JSON object must have:
- label: a short snake_case identifier for your argument
- argument: one concise natural-language argument
- relations: a list of relation objects with source, target, and kind

Relation kind must be either "support" or "attack".
Use the topic label "aliens_exist" as the target unless a better public argument target is clearly available.
"""

ALIEN_DEBATE_CONTRIBUTION_PROMPT = """Debate topic: do aliens exist?

Topics:
{topics}

Public arguments:
{public_arguments}

Public relations:
{public_relations}

Your task:
Contribute one new argument and relate it to the public exchange.
"""

ALIEN_DEBATE_INITIAL_STRENGTH_PROMPT = """Assign an initial strength to this newly public argument.
Return only one number from 0.0 to 1.0.

Argument label: {label}
Argument text: {text}
Topics: {topics}
"""


def build_alien_contribution_prompt(state: DialogueState) -> str:
    return ALIEN_DEBATE_CONTRIBUTION_PROMPT.format(
        topics=", ".join(state.topics),
        public_arguments=_format_public_arguments(state),
        public_relations=_format_public_relations(state),
    )


def build_alien_initial_strength_prompt(argument: Argument, state: DialogueState) -> str:
    return ALIEN_DEBATE_INITIAL_STRENGTH_PROMPT.format(
        label=argument.label,
        text=argument.text,
        topics=", ".join(state.topics),
    )


def _format_public_arguments(state: DialogueState) -> str:
    if not state.public_arguments:
        return "None yet."
    return "\n".join(
        f"- {argument.label}: {argument.text}"
        for argument in state.public_arguments.values()
    )


def _format_public_relations(state: DialogueState) -> str:
    if not state.public_relations:
        return "None yet."
    return "\n".join(
        f"- {relation.source} {relation.kind} {relation.target}"
        for relation in state.public_relations
    )

import pytest
from pydantic import ValidationError

from mtax import MTAXAgent, Argument, Disclosure, ExchangeConfig, MTAX, Relation


def test_mtax_step_and_result() -> None:
    class SimpleAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            label = f"arg_{len(self.private_arguments)}"
            return Disclosure(
                arguments=[Argument(
                    label=label,
                    text=','.join(self.topics),
                )],
                relations=[Relation(source=label, target="recommend_x", kind="support")],
            )

    exchange = MTAX(
        agents=[SimpleAgent("machine"), SimpleAgent("human")],
        topics=["recommend_x", "recommend_y", "recommend_z"],
        config=ExchangeConfig(max_rounds=2),
    )

    state = exchange.step()
    assert len(state.trace) == 2
    assert state.round_index == 1
    assert state.topics == ["recommend_x", "recommend_y", "recommend_z"]
    assert state.trace[0].agent == "machine"
    assert state.trace[0].disclosure.arguments[0].text == "recommend_x,recommend_y,recommend_z"
    assert state.trace[0].round_index == 0
    assert state.public_arguments["arg_0"] == Argument(
        label="arg_0",
        text="recommend_x,recommend_y,recommend_z",
    )

    for state in exchange:
        pass
    assert exchange.state.round_index == 2
    assert exchange.result().metrics["num_contributions"] == len(exchange.state.trace)


def test_contributor_mapping() -> None:
    relation = Relation(source="cost_is_high", target="recommend_x", kind="attack")

    class RelationAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[Argument(
                    label="cost_is_high",
                    text="High cost is a reason against recommend_x.",
                )],
                relations=[relation],
            )

    exchange = MTAX(
        agents=[RelationAgent("human")],
        topics=["recommend_x"],
        config=ExchangeConfig(max_rounds=1),
    )
    exchange.step()
    assert exchange.contributor_mapping(relation) == ("human", 0)


def test_agents_ingest_and_preserve_private_state() -> None:
    private_argument = Argument(label="private_reason", text="Private reason.")

    class StrengthAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[Argument(label="cost_is_high", text="The cost is high.")],
                relations=[Relation(
                    source="cost_is_high",
                    target="recommend_x",
                    kind="attack",
                )],
            )

        def rate(self, argument) -> float:
            return 0.2

    agent = StrengthAgent("human", private_arguments={"private_reason": private_argument})
    exchange = MTAX(
        agents=[agent],
        topics=["recommend_x"],
        config=ExchangeConfig(max_rounds=1),
    )
    exchange.step()
    assert agent.private_arguments["private_reason"] == private_argument
    assert agent.private_arguments["cost_is_high"] == Argument(
        label="cost_is_high",
        text="The cost is high.",
    )
    assert agent.private_strengths["cost_is_high"] == 0.2
    assert agent.private_relations == [Relation(
        source="cost_is_high",
        target="recommend_x",
        kind="attack",
    )]
    assert agent.topics == ("recommend_x",)


def test_invoke_style_agent() -> None:
    class FakeModel:
        def invoke(self, message=None, **kwargs) -> str:
            if message == "strength":
                return "0.4"
            return '{"arguments": [{"label": "fermi_paradox", "text": "The Fermi paradox counts against nearby aliens."}], "relations": [{"source": "fermi_paradox", "target": "aliens_exist", "kind": "attack"}]}'

    class InvokeAgent(MTAXAgent):
        def __init__(self) -> None:
            super().__init__("machine")
            self._model = FakeModel()

        def contribute(self, violation_feedback=None) -> Disclosure | None:
            try:
                return Disclosure.model_validate_json(self._model.invoke(message="contribute"))
            except Exception:
                return None

        def rate(self, argument) -> float:
            try:
                return float(self._model.invoke(message="strength"))
            except (ValueError, TypeError):
                return 0.5

    exchange = MTAX(
        agents=[InvokeAgent()],
        topics=["aliens_exist"],
        config=ExchangeConfig(max_rounds=1),
    )
    exchange.step()

    relation = Relation(source="fermi_paradox", target="aliens_exist", kind="attack")
    disclosure = exchange.state.trace[0].disclosure
    assert disclosure.arguments[0].label == "fermi_paradox"
    assert disclosure.relations == (relation,)
    assert exchange.contributor_mapping(relation) == ("machine", 0)


def test_disclosures_are_immutable() -> None:
    disclosure = Disclosure(
        arguments=[Argument(label="x", text="Argument x.")],
        relations=[Relation(source="x", target="topic", kind="support")],
    )

    with pytest.raises(ValidationError):
        disclosure.relations[0].target = "another_topic"
    with pytest.raises(AttributeError):
        disclosure.relations.append(Relation(source="y", target="topic", kind="attack"))


def test_agent_can_disclose_multiple_arguments() -> None:
    class MultiArgumentAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[
                    Argument(label="x", text="Argument x."),
                    Argument(label="y", text="Argument y."),
                ],
                relations=[
                    Relation(source="x", target="y", kind="attack"),
                    Relation(source="y", target="topic", kind="support"),
                ],
            )

    exchange = MTAX(
        agents=[MultiArgumentAgent("agent")],
        topics=["topic"],
        config=ExchangeConfig(max_rounds=1),
    )

    exchange.step()

    assert set(exchange.state.public_arguments) == {"x", "y"}
    assert len(exchange.state.trace) == 1

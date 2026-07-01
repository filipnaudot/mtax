import pytest
from pydantic import ValidationError

from mtax import InvalidAgentResponse, MTAXAgent, Argument, Disclosure, ExchangeConfig, MTAX, Pass, Relation


def test_mtax_step_and_result() -> None:
    class SimpleAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            label = f"arg_{len(self.private_arguments)}"
            return Disclosure(
                arguments=[Argument(label=label, text=','.join(self.topics))], # type: ignore
                relations=[Relation(source=label, target="recommend_x", kind="support")], # type: ignore
            )
    exchange = MTAX(
        agents=[SimpleAgent("machine"), SimpleAgent("human")],
        topics=["recommend_x", "recommend_y", "recommend_z"],
        config=ExchangeConfig(max_rounds=2, stop_when_resolved=False),
    )
    state = exchange.step()
    assert len(state.trace) == 2
    assert state.round_index == 1
    assert state.topics == ["recommend_x", "recommend_y", "recommend_z"]
    assert state.trace[0].agent == "machine"
    assert state.trace[0].disclosure.arguments[0].text == "recommend_x,recommend_y,recommend_z"
    assert state.trace[0].round_index == 0
    assert state.public_arguments["arg_0"] == Argument(label="arg_0", text="recommend_x,recommend_y,recommend_z")
    for state in exchange: pass
    assert exchange.state.round_index == 2
    assert exchange.result().metrics["num_contributions"] == len(exchange.state.trace)


def test_top_r_resolution_requires_multiple_topics() -> None:
    with pytest.raises(ValueError, match="top_r requires at least 2 topics"):
        MTAX(agents=[MTAXAgent("agent")], topics=["topic"], config=ExchangeConfig(resolution="top_r"))


def test_contributor_mapping() -> None:
    relation = Relation(source="cost_is_high", target="recommend_x", kind="attack")

    class RelationAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[Argument(label="cost_is_high", text="High cost is a reason against recommend_x.")], # type: ignore
                relations=[relation], # type: ignore
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
                arguments=[Argument(label="cost_is_high", text="The cost is high.")], # type: ignore
                relations=[Relation(source="cost_is_high", target="recommend_x", kind="attack")], # type: ignore
            )
        def rate(self, argument) -> float:
            return 0.2
    agent = StrengthAgent("human", private_arguments={"private_reason": private_argument})
    exchange = MTAX(agents=[agent], topics=["recommend_x"], config=ExchangeConfig(max_rounds=1))
    exchange.step()
    assert agent.private_arguments["private_reason"] == private_argument
    assert agent.private_arguments["cost_is_high"] == Argument(label="cost_is_high", text="The cost is high.")
    assert agent.private_strengths["cost_is_high"] == 0.2
    assert agent.private_relations == [Relation(source="cost_is_high", target="recommend_x", kind="attack")]
    assert agent.topics == ("recommend_x",)
    assert agent.private_qbaf.initial_strength("recommend_x") == 0.5
    assert agent.private_qbaf.initial_strength("cost_is_high") == 0.2
    assert agent.private_qbaf.contains_attack_relation("cost_is_high", "recommend_x")
    assert agent.private_qbaf.final_strength("recommend_x") == 0.4


def test_agent_semantics_override_exchange_default() -> None:
    default_agent = MTAXAgent("default")
    override_agent = MTAXAgent("override", semantics="basic_model")
    MTAX(agents=[default_agent, override_agent], topics=["topic"], config=ExchangeConfig(semantics="DFQuAD_model"))
    assert default_agent.private_qbaf.semantics == "DFQuAD_model"
    assert override_agent.private_qbaf.semantics == "basic_model"
    MTAX(agents=[default_agent, override_agent], topics=["topic"], config=ExchangeConfig(semantics="QuadraticEnergy_model"))
    assert default_agent.private_qbaf.semantics == "QuadraticEnergy_model"
    assert override_agent.private_qbaf.semantics == "basic_model"


def test_reused_agent_uses_current_exchange_default_semantics() -> None:
    agent = MTAXAgent("agent")
    MTAX(agents=[agent], topics=["topic_a"], config=ExchangeConfig(semantics="DFQuAD_model"))
    assert agent.private_qbaf.semantics == "DFQuAD_model"
    MTAX(agents=[agent], topics=["topic_b"], config=ExchangeConfig(semantics="basic_model"),)
    assert agent.private_qbaf.semantics == "basic_model"


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

    exchange = MTAX(agents=[InvokeAgent()], topics=["aliens_exist"], config=ExchangeConfig(max_rounds=1),)
    exchange.step()
    relation = Relation(source="fermi_paradox", target="aliens_exist", kind="attack")
    disclosure = exchange.state.trace[0].disclosure
    assert disclosure.arguments[0].label == "fermi_paradox"
    assert disclosure.relations == (relation,)
    assert exchange.contributor_mapping(relation) == ("machine", 0)


def test_disclosures_are_immutable() -> None:
    disclosure = Disclosure(
        arguments=[Argument(label="x", text="Argument x.")], # type: ignore
        relations=[Relation(source="x", target="topic", kind="support")], # type: ignore
    )
    with pytest.raises(ValidationError):
        disclosure.relations[0].target = "another_topic"
    with pytest.raises(AttributeError):
        disclosure.relations.append(Relation(source="y", target="topic", kind="attack")) # type: ignore


def test_agent_can_disclose_multiple_arguments() -> None:
    class MultiArgumentAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[
                    Argument(label="x", text="Argument x."),
                    Argument(label="y", text="Argument y."),
                ], # type: ignore
                relations=[
                    Relation(source="x", target="y", kind="attack"),
                    Relation(source="y", target="topic", kind="support"),
                ], # type: ignore
            )
    exchange = MTAX(agents=[MultiArgumentAgent("agent")], topics=["topic"], config=ExchangeConfig(max_rounds=1))
    exchange.step()
    assert set(exchange.state.public_arguments) == {"x", "y"}
    assert len(exchange.state.trace) == 1


def test_agent_pass_and_rejection_are_recorded(capsys) -> None:
    class PassingAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Pass:
            return Pass(action="pass", reason="Nothing useful to add.")

    class RejectedAgent(MTAXAgent):
        def contribute(self, violation_feedback=None) -> Disclosure:
            return Disclosure(
                arguments=[Argument(label="x", text="Argument x.")], # type: ignore
                relations=[Relation(source="x", target="unknown", kind="support")], # type: ignore
            )
    exchange = MTAX(agents=[PassingAgent("passing"), RejectedAgent("rejected")], topics=["topic"], config=ExchangeConfig(max_rounds=1, max_retries=0),)
    exchange.step()
    assert exchange.state.agent_statuses[0].outcome == "passed"
    assert exchange.state.agent_statuses[0].detail == "Nothing useful to add."
    assert exchange.state.agent_statuses[1].outcome == "rejected"
    assert "'x' → 'unknown'" in exchange.state.agent_statuses[1].detail # type: ignore
    assert "Available targets: 'topic'" in exchange.state.agent_statuses[1].detail # type: ignore


def test_invalid_agent_response_is_retried_with_feedback() -> None:
    class RetryingAgent(MTAXAgent):
        def __init__(self) -> None:
            super().__init__("retrying")
            self.attempts = 0

        def contribute(self, violation_feedback=None) -> Pass:
            self.attempts += 1
            if self.attempts == 1:
                raise InvalidAgentResponse("Response was not valid JSON.")
            assert violation_feedback == "Response was not valid JSON."
            return Pass(action="pass")
    agent = RetryingAgent()
    exchange = MTAX(agents=[agent], topics=["topic"], config=ExchangeConfig(max_rounds=1, max_retries=1))
    exchange.step()
    assert agent.attempts == 2
    assert exchange.state.agent_statuses[0].outcome == "passed"
    assert exchange.state.agent_statuses[0].attempts == 2

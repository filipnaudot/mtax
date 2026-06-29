from mtax import Agent, Argument, Contribution, ExchangeConfig, MTAX, Relation


class StringModel:
    def step(self, state) -> str:
        return f"{','.join(state.topics)}:{state.round_index}"


def test_mtax_step_run_and_result() -> None:
    agents = [
        Agent(name="machine", model=StringModel()),
        Agent(name="human", model=StringModel()),
    ]
    exchange = MTAX(
        agents=agents,
        topics=["recommend_x", "recommend_y", "recommend_z"],
        config=ExchangeConfig(max_rounds=2),
    )

    state = exchange.step()
    assert len(state.trace) == 2
    assert state.round_index == 1
    assert state.topics == ["recommend_x", "recommend_y", "recommend_z"]
    assert state.trace[0].agent == "machine"
    assert state.trace[0].argument == "recommend_x,recommend_y,recommend_z:0"
    assert state.trace[0].round_index == 0
    assert state.public_arguments["machine"] == Argument(label="machine", text="recommend_x,recommend_y,recommend_z:0")

    result = exchange.run()
    assert result.topics == ["recommend_x", "recommend_y", "recommend_z"]
    assert result.final_state.round_index == 2
    assert exchange.result().metrics["num_contributions"] == len(exchange.state.trace)


def test_contributor_mapping() -> None:
    relation = Relation(source="cost_is_high", target="recommend_x", kind="attack")
    class RelationModel:
        def step(self, state) -> Contribution:
            return Contribution(
                label="cost argument",
                argument="High cost is a reason against recommend_x.",
                relations=(relation,),
            )

    exchange = MTAX(
        agents=[Agent(name="human", model=RelationModel())],
        topics=["recommend_x"],
        config=ExchangeConfig(max_rounds=1),
    )
    exchange.step()
    assert exchange.contributor_mapping(relation) == ("human", 0)


def test_agents_ingest_public_state_and_preserve_private_state() -> None:
    public_relation = Relation(source="cost_is_high", target="recommend_x", kind="attack")
    private_relation = Relation(source="private_reason", target="recommend_x", kind="support")
    private_argument = Argument(label="private_reason", text="Private reason.")

    class StrengthModel:
        def step(self, state) -> Contribution:
            return Contribution(label="cost_is_high", argument="The cost is high.", relations=(public_relation,))

        def assign_initial_strength(self, argument, state) -> float:
            return 0.2

    agent = Agent(
        name="human",
        model=StrengthModel(),
        private_arguments={"private_reason": private_argument},
        private_relations={private_relation},
    )
    exchange = MTAX(
        agents=[agent],
        topics=["recommend_x"],
        config=ExchangeConfig(max_rounds=1),
    )
    exchange.step()
    assert agent.private_arguments["private_reason"] == private_argument
    assert agent.private_arguments["cost_is_high"] == Argument(label="cost_is_high", text="The cost is high.")
    assert agent.private_relations == {private_relation, public_relation}
    assert agent.private_strengths["cost_is_high"] == 0.2


def test_agent_can_wrap_autom8_like_invoke_model() -> None:
    class InvokeModel:
        def invoke(self, message=None, *, instructions=None, **kwargs) -> str:
            if message == "strength":
                return "0.4"
            return """
            {
                "label": "fermi_paradox",
                "argument": "The Fermi paradox counts against nearby aliens.",
                "relations": [
                    {
                        "source": "fermi_paradox",
                        "target": "aliens_exist",
                        "kind": "attack"
                    }
                ]
            }
            """

    agent = Agent(
        name="machine",
        model=InvokeModel(),
        contribution_prompt=lambda state: "contribute",
        initial_strength_prompt=lambda argument, state: "strength",
        instructions="return json",
    )
    exchange = MTAX(
        agents=[agent],
        topics=["aliens_exist"],
        config=ExchangeConfig(max_rounds=1),
    )

    exchange.step()

    relation = Relation(
        source="fermi_paradox",
        target="aliens_exist",
        kind="attack",
    )
    assert exchange.state.trace[0].label == "fermi_paradox"
    assert exchange.state.trace[0].relations == (relation,)
    assert exchange.contributor_mapping(relation) == ("machine", 0)

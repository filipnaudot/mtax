from mtax import Agent, Contribution, ExchangeConfig, MTAX, Relation


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

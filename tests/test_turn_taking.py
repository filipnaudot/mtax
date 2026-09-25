from mtax import BasicTurnTaking, MTAXAgent


def test_basic_turn_taking_cycles_through_agents() -> None:
    agents = [MTAXAgent("first"), MTAXAgent("second")]
    turn_taking = BasicTurnTaking(agents)
    assert [turn_taking(time_step).name for time_step in range(4)] == ["first", "second", "first", "second"]

from mtax import BasicTurnTaking, MTAXAgent


def test_basic_turn_taking_preserves_agent_order() -> None:
    agents = [MTAXAgent("first"), MTAXAgent("second")]
    assert BasicTurnTaking()(agents) == agents

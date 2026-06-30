import pytest
from mtax.bm import BipolarMultitree
from mtax.schema import Relation


def bm_with_two_topics() -> BipolarMultitree:
    return BipolarMultitree(topics={"a1", "a2"})


####
# valid additions
####
def test_new_argument_directly_attacks_topic() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "attack")
    assert "x" in bm.arguments
    assert ("x", "a1", "attack") in bm.relations


def test_chained_relations_to_same_topic() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "x", "attack")
    assert bm.arguments == {"a1", "a2", "x", "y"}
    assert ("x", "a1", "support") in bm.relations
    assert ("y", "x", "attack") in bm.relations


def test_argument_reaches_two_different_topics_via_direct_edges() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("x", "a2", "attack")
    assert ("x", "a1", "support") in bm.relations
    assert ("x", "a2", "attack") in bm.relations


def test_sibling_arguments_on_same_topic() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "a1", "attack")
    assert {"x", "y"}.issubset(bm.arguments)



####
# topic-rootedness violations
####
def test_relation_from_topic_is_rejected() -> None:
    bm = bm_with_two_topics()
    with pytest.raises(ValueError):
        bm.add_relation("a1", "a2", "attack")


####
# topic relevance
####
def test_argument_with_no_path_to_topic_is_rejected() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "x", "attack")
    with pytest.raises(ValueError):
        bm.add_relation("y", "z", "support")


####
# target not in exchange violations
####
def test_relation_to_unknown_target_is_rejected() -> None:
    bm = bm_with_two_topics()
    with pytest.raises(ValueError):
        bm.add_relation("x", "unknown", "support")

####
# acyclicity violations
####
def test_direct_cycle_is_rejected() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "x", "attack")
    with pytest.raises(ValueError):
        bm.add_relation("x", "y", "support")

def test_indirect_cycle_is_rejected() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "x", "attack")
    bm.add_relation("z", "y", "support")
    with pytest.raises(ValueError):
        bm.add_relation("x", "z", "attack")


####
# single-path violations
####
def test_two_paths_to_topic_via_shared_intermediate_is_rejected() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "x", "attack")
    bm.add_relation("z", "x", "attack")
    with pytest.raises(ValueError):
        bm.add_relation("z", "y", "support")

def test_ancestor_gains_second_path_is_rejected() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("z", "x", "attack")
    bm.add_relation("y", "a1", "attack")
    with pytest.raises(ValueError):
        bm.add_relation("x", "y", "support")

def test_new_argument_connecting_to_two_topic_subtrees_is_valid() -> None:
    bm = bm_with_two_topics()
    bm.add_relation("x", "a1", "support")
    bm.add_relation("y", "a2", "attack")
    bm.add_relation("z", "x", "support")
    bm.add_relation("z", "y", "attack")
    assert "z" in bm.arguments


def test_add_relations_accepts_arbitrary_order() -> None:
    bm = bm_with_two_topics()
    bm.add_relations([
        Relation(source="x", target="y", kind="attack"),
        Relation(source="y", target="a1", kind="support"),
    ])

    assert ("x", "y", "attack") in bm.relations
    assert ("y", "a1", "support") in bm.relations


def test_add_relations_is_atomic() -> None:
    bm = bm_with_two_topics()

    with pytest.raises(ValueError):
        bm.add_relations([
            Relation(source="x", target="a1", kind="support"),
            Relation(source="y", target="unknown", kind="attack"),
        ])

    assert bm.arguments == {"a1", "a2"}
    assert bm.relations == set()

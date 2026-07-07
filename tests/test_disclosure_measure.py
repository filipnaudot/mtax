from qbaf import QBAFramework

from mtax.disclosure_measure import ranking_disclosure_effects
from mtax.schema import Relation


def test_ranking_disclosure_effect_measures_increased_kendall_alignment() -> None:
    public = QBAFramework(["a", "b"], [0.4, 0.6], [], [], semantics="DFQuAD_model")
    private = QBAFramework(
        ["a", "b", "reason"],
        [0.4, 0.6, 1.0],
        [],
        [("reason", "a")],
        semantics="DFQuAD_model",
    )
    effects = ranking_disclosure_effects(public, private, ["a", "b"])
    assert effects == [(Relation(source="reason", target="a", kind="support"), 2.0)]

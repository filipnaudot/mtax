def test_public_imports() -> None:
    from mtax import (
        Agent,
        Contribution,
        DialogueState,
        ExchangeConfig,
        ExchangeResult,
        MTAX,
        QBAFSemantics,
        Relation,
    )

    assert Agent
    assert Contribution
    assert DialogueState
    assert ExchangeConfig
    assert ExchangeResult
    assert MTAX
    assert QBAFSemantics
    assert Relation


def test_qbaf_semantics_constants() -> None:
    from mtax import ExchangeConfig, QBAFSemantics
    assert QBAFSemantics.BASIC == "basic_model"
    assert QBAFSemantics.QUADRATIC_ENERGY == "QuadraticEnergy_model"
    assert QBAFSemantics.SQUARED_DFQUAD == "SquaredDFQuAD_model"
    assert QBAFSemantics.EULER_BASED_TOP == "EulerBasedTop_model"
    assert QBAFSemantics.EULER_BASED == "EulerBased_model"
    assert QBAFSemantics.DFQUAD == "DFQuAD_model"
    assert ExchangeConfig().semantics == QBAFSemantics.DFQUAD

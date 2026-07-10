# Evaluation TODO

- [x] Add configurable evaluation constants for graph size, private-QBAF size, semantics, and random seed.
- [x] Implement seeded universal Bipolar Multitree generation.
- [x] Derive agents with private QBAFs by letting agents ingest and rate private BM
  subsets.
- [ ] Implement `CounterfactualAgent` class in `eval_agents.py`.
- [ ] Generate two-agent exchanges that are initially unresolved (assert this), with a maximum-attempt limit.
- [ ] Implement `PassiveAgent` class in a new file: `eval_agents.py`.
- [ ] Implement `ShallowAgent` class in `eval_agents.py`.
- [ ] Implement `GreedyAgent` class in `eval_agents.py`.
- [ ] Implement ranking-based agents class in `eval_agents.py`.
    - [ ] Implement Top-r cautious behaviour.
    - [ ] Implement Top-r defensive behaviour.
    - [ ] Implement consensus-seeking behaviour.
- [ ] Implement resolution, contribution, persuasion, and accuracy metrics in `evaluate.py`.
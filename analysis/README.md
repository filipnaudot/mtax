# Evaluation TODO

- [x] Add configurable evaluation constants for graph size, private-QBAF size, semantics, and random seed.
- [x] Implement seeded universal Bipolar Multitree generation.
- [x] Derive agents with private QBAFs by letting agents ingest and rate private BM
  subsets.
- [x] Implement `CounterfactualAgent` class in `eval_agents.py`.
- [x] Generate two-agent exchanges that are initially unresolved, with a maximum-attempt limit.
- [ ] Implement `GreedyAgent` class in `eval_agents.py`.
- [ ] Implement `ShallowAgent` class in `eval_agents.py`.
- [ ] Implement ranking-based agents class in `eval_agents.py`.
    - [ ] Implement Top-r cautious behaviour.
    - [ ] Implement Top-r defensive behaviour.
    - [ ] Implement consensus-seeking behaviour.
- [ ] Implement resolution, contribution, persuasion, and accuracy metrics in `evaluate.py`.

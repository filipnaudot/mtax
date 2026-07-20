# Core Evaluation

Run the full evaluation from this directory:

```bash
python evaluate.py --experiment all --runs 100 --strategies greedy shallow counterfactual
```

The rating, topic, agent, and density settings are evaluated with random and stable non-topic ratings.
Results are written to `results.csv`.

To compare semantics, run:

```bash
python evaluate.py --experiment semantics --runs 100 --strategies greedy shallow counterfactual
```

This uses stable initial strengths and evaluates each supported non-basic semantic homogeneously, plus a Mixed case that cycles semantics across agents.

To compare agent behaviours, run:

```bash
python evaluate.py --experiment behaviour --runs 100
```

This uses stable initial strengths and evaluates each behaviour homogeneously, plus a Mixed case that cycles behaviours across agents.

Generate the PDF figures:

```bash
python plots.py
```

For a single BM/QBAF visualization:

```bash
python evaluate.py --visualize --oracle-graph-size 7 --num-topics 2 --qbaf-size 4 --runs 1 --max-rounds 1
```

# Core Evaluation

Run the full evaluation from this directory:

```bash
python evaluate.py --experiment all --runs 100 --strategies greedy shallow counterfactual
```

Each setting is evaluated with random and stable non-topic ratings.
Results are written to `results.csv` and `behavior_influence.csv`.

Generate the PDF figures:

```bash
python result_plots.py
```

For a single BM/QBAF visualization:

```bash
python evaluate.py --visualize --oracle-graph-size 7 --num-topics 2 --qbaf-size 4 --runs 1 --max-rounds 1
```

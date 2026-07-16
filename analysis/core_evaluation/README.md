# Evaluation TODO

Generate BM/QBAF visualizations:

```bash
python evaluate.py --visualize --oracle-graph-size 7 --num-topics 2 --qbaf-size 4 --runs 1 --max-rounds 1
```

Run the full evaluation and write `results.csv`:

```bash
python evaluate.py --experiment all --runs 100 --strategies greedy shallow counterfactual
```

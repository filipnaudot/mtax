# Core Evaluation

Run the full evaluation from this directory:

```bash
python evaluate.py --experiment all --runs 200
```

The rating, topic, agent, and density settings use Greedy agents. They use `--num-agents`, which defaults to four.
Results are written to `results.csv`.

Generate the PDF figures:

```bash
python plots.py
```

For a single BM/QBAF visualization:

```bash
python evaluate.py --visualize --oracle-graph-size 7 --num-topics 2 --qbaf-size 4 --runs 1 --max-rounds 1
```

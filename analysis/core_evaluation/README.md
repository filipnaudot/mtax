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



## Paper results
These are the commands used to produce the results presented in the paper.

The benchmark script is invoked with the following command:
```bash
python evaluate.py --experiment all --runs 200 --qbaf-size 20 --num-agents 4 --max-rounds 200
```

The visuals are then generated with:
```bash
python plots.py
```

import csv
from pathlib import Path


RESULTS_PATH = Path(__file__).with_name("results.csv")
METRICS = (
    ("resolution_rate", "RR"),
    ("contribution_rate", "CR"),
)


def average(rows: list[dict[str, str]], metric: str) -> float:
    total_runs = sum(int(row["runs"]) for row in rows)
    return sum(float(row[metric]) * int(row["runs"]) for row in rows) / total_runs


with RESULTS_PATH.open(newline="") as file:
    rows = list(csv.DictReader(file))

print("Metric  Average")
for metric, label in METRICS:
    print(f"{label:<7} {average(rows, metric):.3f}")

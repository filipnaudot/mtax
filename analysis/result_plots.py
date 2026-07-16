import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
INFLUENCE_INPUT_PATH = os.path.join(os.path.dirname(__file__), "behavior_influence.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "result_plots")

METRICS = (
    ("resolution_rate", "Resolution rate", 0.0, 1.0),
    ("avg_rounds", "Average rounds", None, None),
    ("avg_final_ranking_agreement", "Final ranking agreement", -1.0, 1.0),
)
EXPERIMENTS = ("topics", "agents", "density")
COLORS = {
    "shallow": "#4c78a8",
    "greedy": "#f58518",
    "counterfactual": "#54a24b",
}
RATING_MODE_COLORS = {"random": "#999", "stable": "#222"}


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def rating_mode(row: dict[str, str]) -> str:
    return row.get("rating_mode") or "random"


def rating_modes(rows: list[dict[str, str]]) -> list[str]:
    return [mode for mode in RATING_MODE_COLORS if any(rating_mode(row) == mode for row in rows)]


def metric_range(rows: list[dict[str, str]], metric: str, lower: float | None, upper: float | None) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows]
    y_min = min(values) if lower is None else lower
    y_max = max(values) if upper is None else upper
    if y_min == y_max:
        y_max += 1
    padding = (y_max - y_min) * 0.08
    return (
        y_min - padding if lower is None else y_min,
        y_max + padding if upper is None else y_max,
    )


def draw_metric(rows: list[dict[str, str]], metric: str, title: str, lower: float | None, upper: float | None):
    figure, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(12, 3.5), layout="constrained")
    figure.suptitle(title)
    y_min, y_max = metric_range(rows, metric, lower, upper)

    for axis, experiment in zip(axes, EXPERIMENTS):
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        if not experiment_rows:
            axis.set_visible(False)
            continue
        x_values = sorted({float(row["value"]) for row in experiment_rows})
        for mode in rating_modes(experiment_rows):
            points = sorted(
                (float(row["value"]), float(row[metric]))
                for row in experiment_rows
                if rating_mode(row) == mode
            )
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=RATING_MODE_COLORS[mode], marker="o", label=mode)
        axis.set(xlabel=experiment, xticks=x_values, ylim=(y_min, y_max))
        axis.grid(axis="y", color="#ddd")
        if set(rating_modes(experiment_rows)) == {"random", "stable"}:
            axis.legend()
    return figure


def draw_behavior_influence(rows: list[dict[str, str]]):
    figure, axis = plt.subplots(figsize=(6, 3.5), layout="constrained")
    figure.suptitle("Persuasion vs contributions")
    by_strategy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)

    for strategy, strategy_rows in sorted(by_strategy.items()):
        contributions = sum(float(row["avg_contributions"]) for row in strategy_rows) / len(strategy_rows)
        influence = sum(float(row["avg_influence"]) for row in strategy_rows) / len(strategy_rows)
        axis.scatter(contributions, influence, color=COLORS[strategy], label=strategy)

    axis.set(xlabel="Average contributions per full exchange", ylabel="Average influence")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(color="#ddd")
    axis.legend()
    return figure


def save(figure, name: str) -> None:
    figure.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    plt.close(figure)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_rows(INPUT_PATH)
    for metric, title, lower, upper in METRICS:
        save(draw_metric(rows, metric, title, lower, upper), metric)
    save(draw_behavior_influence(load_rows(INFLUENCE_INPUT_PATH)), "behavior_influence")
    print(f"wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

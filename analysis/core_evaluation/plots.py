import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "result_plots")

EXPERIMENTS = ("topics", "agents", "density")
RATING_MODE_COLORS = {"random": "#999", "stable": "#222"}


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def rating_mode(row: dict[str, str]) -> str:
    return row.get("rating_mode") or "random"


def rating_modes(rows: list[dict[str, str]]) -> list[str]:
    return [mode for mode in RATING_MODE_COLORS if any(rating_mode(row) == mode for row in rows)]


def draw_resolution_rate(rows: list[dict[str, str]]):
    figure, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(16, 3.5), layout="constrained")
    figure.suptitle("Average resolution rate by topics, agents, and density", fontsize=16, fontweight="bold")

    for axis, experiment in zip(axes, EXPERIMENTS):
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        if not experiment_rows:
            axis.set_visible(False)
            continue
        x_values = sorted({float(row["value"]) for row in experiment_rows})
        for mode in rating_modes(experiment_rows):
            points = sorted(
                (float(row["value"]), float(row["resolution_rate"]))
                for row in experiment_rows
                if rating_mode(row) == mode
            )
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=RATING_MODE_COLORS[mode], marker="o", label=mode)
        axis.set(xlabel=experiment, xticks=x_values, ylim=(0.0, 1.0))
        axis.grid(axis="y", color="#ddd")
        if set(rating_modes(experiment_rows)) == {"random", "stable"}:
            axis.legend()
    return figure


def draw_categorical_resolution_rate(rows: list[dict[str, str]], experiment: str, title: str):
    experiment_rows = [row for row in rows if row["experiment"] == experiment]
    labels = [row["value"] for row in experiment_rows]
    values = [float(row["resolution_rate"]) for row in experiment_rows]
    figure, axis = plt.subplots(figsize=(6, 3.5), layout="constrained")
    axis.plot(range(len(labels)), values, color=RATING_MODE_COLORS["stable"], marker="o")
    axis.set(
        title=title,
        xlabel=experiment,
        xticks=range(len(labels)),
        xticklabels=labels,
        ylim=(0.0, 1.0),
    )
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", color="#ddd")
    return figure


def save(figure, name: str) -> None:
    figure.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    plt.close(figure)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_rows(INPUT_PATH)
    save(draw_resolution_rate(rows), "resolution_rate")
    save(draw_categorical_resolution_rate(rows, "semantics", "Average resolution rate by semantics"), "semantics_resolution_rate")
    save(draw_categorical_resolution_rate(rows, "behaviour", "Average resolution rate by behaviour"), "behaviour_resolution_rate")
    print(f"wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import csv
import os


INPUT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "result_plots")

METRICS = (
    ("resolution_rate", "Resolution rate", 0.0, 1.0),
    ("avg_rounds", "Average rounds", None, None),
    ("avg_contributions", "Average contributions", 0.0, None),
    ("avg_final_ranking_agreement", "Final ranking agreement", -1.0, 1.0),
)

COLORS = {
    "shallow": "#4c78a8",
    "greedy": "#f58518",
    "counterfactual": "#54a24b",
}


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def as_number(value: str) -> float:
    return float(value)


def points(rows: list[dict[str, str]], experiment: str, metric: str, strategy: str) -> list[tuple[float, float]]:
    values = [
        (as_number(row["value"]), as_number(row[metric]))
        for row in rows
        if row["experiment"] == experiment and row["strategy"] == strategy
    ]
    return sorted(values)


def scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)


def polyline(point_values: list[tuple[float, float]],
             x_min: float,
             x_max: float,
             y_min: float,
             y_max: float,
             left: float,
             top: float,
             width: float,
             height: float) -> str:
    coords = []
    for x_value, y_value in point_values:
        x = scale(x_value, x_min, x_max, left, left + width)
        y = scale(y_value, y_min, y_max, top + height, top)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def metric_range(rows: list[dict[str, str]], metric: str, fixed_min: float | None, fixed_max: float | None) -> tuple[float, float]:
    values = [as_number(row[metric]) for row in rows]
    lower = min(values) if fixed_min is None else fixed_min
    upper = max(values) if fixed_max is None else fixed_max
    if lower == upper:
        upper = lower + 1
    padding = (upper - lower) * 0.08
    if fixed_min is None:
        lower -= padding
    if fixed_max is None:
        upper += padding
    return lower, upper


def draw_metric(rows: list[dict[str, str]], metric: str, title: str, fixed_min: float | None, fixed_max: float | None) -> str:
    experiments = ["topics", "agents", "density"]
    panel_width = 320
    panel_height = 250
    margin_left = 52
    margin_top = 58
    gutter = 34
    width = margin_left + len(experiments) * panel_width + (len(experiments) - 1) * gutter + 24
    height = 360
    y_min, y_max = metric_range(rows, metric, fixed_min, fixed_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.title{font-size:18px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{stroke:#999;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.line{fill:none;stroke-width:2.5}.dot{stroke:white;stroke-width:1}</style>',
        f'<text class="title" x="24" y="28">{title}</text>',
    ]

    for index, experiment in enumerate(experiments):
        left = margin_left + index * (panel_width + gutter)
        top = margin_top
        plot_width = panel_width - 40
        plot_height = panel_height - 54
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        if not experiment_rows:
            continue
        x_values = sorted({as_number(row["value"]) for row in experiment_rows})
        x_min, x_max = min(x_values), max(x_values)

        parts.extend([
            f'<text class="panel" x="{left}" y="{top - 16}">{experiment}</text>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="grid" x1="{left}" y1="{top}" x2="{left + plot_width}" y2="{top}"/>',
            f'<line class="grid" x1="{left}" y1="{top + plot_height / 2}" x2="{left + plot_width}" y2="{top + plot_height / 2}"/>',
            f'<text x="{left - 44}" y="{top + 4}">{y_max:.2f}</text>',
            f'<text x="{left - 44}" y="{top + plot_height / 2 + 4}">{((y_min + y_max) / 2):.2f}</text>',
            f'<text x="{left - 44}" y="{top + plot_height + 4}">{y_min:.2f}</text>',
        ])

        for x_value in x_values:
            x = scale(x_value, x_min, x_max, left, left + plot_width)
            parts.append(f'<text x="{x - 10:.1f}" y="{top + plot_height + 22}">{x_value:g}</text>')

        for strategy, color in COLORS.items():
            strategy_points = points(rows, experiment, metric, strategy)
            if not strategy_points:
                continue
            coords = polyline(strategy_points, x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)
            parts.append(f'<polyline class="line" points="{coords}" stroke="{color}"/>')
            for x_value, y_value in strategy_points:
                x = scale(x_value, x_min, x_max, left, left + plot_width)
                y = scale(y_value, y_min, y_max, top + plot_height, top)
                parts.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    legend_y = height - 34
    legend_x = 24
    for strategy, color in COLORS.items():
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}">{strategy}</text>')
        legend_x += 150

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    rows = load_rows(INPUT_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for metric, title, fixed_min, fixed_max in METRICS:
        svg = draw_metric(rows, metric, title, fixed_min, fixed_max)
        with open(os.path.join(OUTPUT_DIR, f"{metric}.svg"), "w") as file:
            file.write(svg)
    print(f"wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

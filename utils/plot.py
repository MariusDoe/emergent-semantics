import re
import ast
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from typing import Any, Callable

Point = tuple[int, float]
Points = list[Point]
StrDict = dict[str, Any]

parser = ArgumentParser()
parser.add_argument("--mode", choices=["model", "probe"], required=True)
parser.add_argument("--logs", nargs="+", required=True)
parser.add_argument("--constants", nargs="*", default=[])
parser.add_argument("--params", nargs="*", default=[])
parser.add_argument("--out", default="accuracy_plot.png")
parser.add_argument("--title", default="Accuracy over Steps")
parser.add_argument("--max_step", type=int)
parser.add_argument("--add_points", nargs="*", default=[])
args = parser.parse_args()

def skip_step(step: int):
    if args.max_step is None:
        return False
    return step > args.max_step

def model_training():
    accuracy_pattern = re.compile(r"step (\d+): acc: ([0-9.]+)")

    def find_points(lines: list[str]):
        for line in lines:
            match = accuracy_pattern.search(line)
            if not match:
                continue
            step = int(match.group(1))
            if skip_step(step):
                continue
            accuracy = float(match.group(2))
            yield step, accuracy

    def parse_file(name: str, prefix: str):
        with open(prefix + ".out") as out:
            params_line = out.readline()
        params_line = re.sub(r'^args=Namespace\((.*)\)$', r'{\1}', params_line)
        params_line = re.sub(r'(\w+)=', r'"\1":', params_line)
        # Replace <Enum: 'value'> with just 'value'
        params_line = re.sub(r"<[^:]+: '([^']+)'>", r"'\1'", params_line)
        params: StrDict = ast.literal_eval(params_line)
        params["name"] = name
        with open(prefix + ".err") as err:
            lines = err.readlines()
        points = list(find_points(lines))
        yield params, points

    return parse_file

def probe_training():
    step_pattern = re.compile(r"step_(\d+)")
    accuracy_pattern = re.compile(r"(?:final|acc).*\[([\d., ]+)\]")

    task_names = ["facing", "pos_rel_to_start", "pos_rel_to_end", "facing_wall", "pos", "walls_around"]

    def find_points(lines: list[str]):
        step = None
        for line in lines:
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))
                continue
            accuracy_match = accuracy_pattern.search(line)
            if not accuracy_match:
                continue
            assert step is not None, "accuracies without step"
            if skip_step(step):
                continue
            for index, accuracy in enumerate(accuracy_match.group(1).split(",")):
                yield index, step, float(accuracy)
            step = None

    def parse_file(name: str, prefix: str):
        with open(prefix + ".out") as out:
            lines = out.readlines()
        runs: defaultdict[int, Points] = defaultdict(lambda: [])
        for index, step, accuracy in find_points(lines):
            runs[index].append((step, accuracy))
        for index, points in runs.items():
            params = {"name": name, "task": task_names[index]}
            yield params, points

    return parse_file

parse_file = None
if args.mode == "model":
    parse_file = model_training()
if args.mode == "probe":
    parse_file = probe_training()

assert parse_file is not None


def transpose(points: Points) -> tuple[list[int], list[float]]:
    steps, accuracies = map(list, zip(*points))
    return steps, accuracies

run_prefixes: dict[str, str] = {}
for log in args.logs:
    if ":" in log:
        name, log = log.split(":")
    else:
        name = log
    prefix, _ = os.path.splitext(log)
    run_prefixes[prefix] = name

runs = [run for prefix, name in run_prefixes.items() for run in parse_file(name, prefix)]

for point in args.add_points:
    step, accuracy = point.split(":")
    step = int(step)
    accuracy = float(accuracy)
    point = (step, accuracy)
    for _, points in runs:
        points.append(point)

def point_sort_key(point: Point):
    step, _ = point
    return step

runs = [(params, list(sorted(points, key=point_sort_key))) for params, points in runs]

first_params, first_points = runs[0]
first_steps, _ = transpose(first_points)
for constant in args.constants:
    accuracy, *params = constant.split(",")
    accuracy = float(accuracy)
    params = {key: value for key, value in (param.split(":") for param in params)}
    points = list(zip(first_steps, repeat(accuracy)))
    runs.append((params, points))

def remove_prefix(string: str, prefix: str):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def remove_suffix(string: str, prefix: str):
    if string.endswith(prefix):
        return string[:-len(prefix)]
    return string

def format_param(key: str, value: Any):
    if key == "dataset_name":
        key = "ds"
        value = remove_prefix(value, "karel_")
        value = remove_suffix(value, "_uniform_noloops_nocond")
    if key == "learning_rate":
        key = "lr"
        value = float(value)
        value = f"{value:.0e}"
    if key == "mapping":
        def format_mapping(mapping: str):
            old, new = (remove_suffix(action, "()") for action in mapping.split(":"))
            return f"{old}->{new}"

        value = ", ".join(format_mapping(mapping) for mapping in value)
    return f"{key}: {value}"

display_params: dict[str, str] = {key: display_key for key, display_key in (arg.split(":") for arg in args.params)}

def without(dict: StrDict, *keys: str):
    result = dict.copy()
    for key in keys:
        if key in result:
            del result[key]
    return result

DisplayLegend = Callable[[dict[str, Any], str], Artist]

display: dict[str, tuple[list[StrDict], DisplayLegend]] = {
    "marker": (
        [{"marker": marker, "markersize": 4} for marker in ["o", "^", "*", "s", "D", "P"]],
        lambda kwargs, label: Line2D([0], [0], label=label, color="white", markerfacecolor=kwargs.get("color", "black"), markersize=10, **without(kwargs, "color", "markersize"))),
    "linestyle": (
        [{"linestyle": linestyle} for linestyle in ['-', '--', '-.', ':', (0, (1, 5)), (0, (3, 1, 1, 1, 1, 1))]],
        lambda kwargs, label: Line2D([0], [0], label=label, **{"color": "black", **kwargs})),
    "color": (
        [{"color": color} for color in TABLEAU_COLORS.keys()],
        lambda kwargs, label: Patch(facecolor=kwargs["color"], edgecolor='black', label=label, **without(kwargs, "color"))),
}

plot_kwargs_by_param: dict[str, dict[str, tuple[StrDict, DisplayLegend]]] = defaultdict(lambda: {})

def get_param_kwargs(key: str, value: Any) -> StrDict:
    if key not in display_params:
        return {}
    plot_kwargs = plot_kwargs_by_param[key]
    if value in plot_kwargs:
        kwargs, _ = plot_kwargs[value]
        return kwargs
    display_keys = display_params[key].split(",")
    kwargs: StrDict = {}
    display_legend = None
    for display_key in display_keys:
        display_dicts, current_display_legend = display[display_key]
        display_dict = display_dicts[len(plot_kwargs) % len(display_dicts)]
        kwargs.update(**display_dict)
        if display_legend is None:
            display_legend = current_display_legend
    assert display_legend is not None
    plot_kwargs[value] = (kwargs, display_legend)
    return kwargs

def get_params_kwargs(params: StrDict):
    kwargs: StrDict = {}
    for key, value in params.items():
        kwargs.update(**get_param_kwargs(key, value))
    return kwargs

plt.figure(figsize=(8, 6))
for params, points in runs:
    if not points:
        continue
    kwargs = get_params_kwargs(params)
    steps, accuracies = transpose(points)
    plt.plot(steps, accuracies, **kwargs)

plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title(args.title)
plt.grid(True)
plt.tight_layout()

legend_elements: list[Artist] = []
for key in display_params:
    plot_kwargs = plot_kwargs_by_param[key]
    for value in sorted(plot_kwargs.keys()):
        kwargs, display_legend = plot_kwargs[value]
        label = format_param(key, value)
        display_keys, display_value = next(iter(kwargs.items()))
        legend_element = display_legend(kwargs, label)
        legend_elements.append(legend_element)

plt.legend(handles=legend_elements, loc="lower right", fontsize="small")
plt.savefig(args.out, dpi=300)

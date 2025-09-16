import re
import ast
import math
import matplotlib.pyplot as plt
import shlex
import sys
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Generator, Dict

Point = tuple[int, float]
Points = list[Point]
StrDict = Dict[str, Any]

parser = ArgumentParser()
parser.add_argument("--logs", nargs="+", required=True)
parser.add_argument("--constants", nargs="*", default=[])
parser.add_argument("--params", nargs="*", default=[])
parser.add_argument("--out", default="accuracy_plot.png")
parser.add_argument("--title", default="Accuracy over Steps")
parser.add_argument("--max_step", type=int)
parser.add_argument("--add_points", nargs="*", default=[])
parser.add_argument("--show_only", nargs="*", default=[])
parser.add_argument("--remap_steps", nargs="*", default=[])
parser.add_argument("--debug", action="store_true")
parser.add_argument("--smooth", type=int)
args = parser.parse_args()

def skip_step(step: int):
    if args.max_step is None:
        return False
    return step > args.max_step

@dataclass(kw_only=True)
class LineResult:
    step: int | None = None
    accuracy: float | None = None
    params: StrDict = field(default_factory=lambda: {})
    reset_params: bool = False

Match = re.Match[str]
Matcher = Callable[[str], Generator[LineResult]]

matchers: list[Matcher] = []
def matcher(pattern_string: str):
    pattern = re.compile(pattern_string)
    def create_match(function: Callable[[Match], Generator[LineResult]]):
        def find(line: str):
            match = pattern.search(line)
            if not match:
                return
            yield from function(match)
        matchers.append(find)
    return create_match

@matcher(r"step (\d+): acc: ([0-9.]+)")
def model_training(match: Match):
    step = int(match.group(1))
    if skip_step(step):
        return
    accuracy = float(match.group(2))
    yield LineResult(step=step, accuracy=accuracy, reset_params=True)

@matcher(r'^(?:args=)?Namespace\((.*)\)$')
def namespace(match: Match):
    params_line = match.group(1)
    params_line = f"{{{params_line}}}"
    params_line = re.sub(r'(\w+)=', r'"\1":', params_line)
    # Replace <Enum: 'value'> with just 'value'
    params_line = re.sub(r"<[^:]+: '([^']+)'>", r"'\1'", params_line)
    params: StrDict = ast.literal_eval(params_line)
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = tuple(value)
    yield LineResult(params=params)

@matcher(r"python -m (\S+)")
def python_module(match: Match):
    module = match.group(1)
    yield LineResult(params={"python_module": module})

def phrase_matcher(phrase, name):
    @matcher(phrase)
    def find(_match: Match):
        yield LineResult(params={"name": name})

phrase_matcher("eval frozen probe", "frozen")
phrase_matcher("training probe for checkpoint", "from_scratch")
phrase_matcher("training probe on base model", "base_model")

@matcher(r"(?:step_|checkpoint\s+)(\d+)")
def probe_step(match: Match):
    if "Wrote" in match.string:
        return
    step = int(match.group(1))
    yield LineResult(step=step)

@matcher(r"with (\d+) layers")
def layers(match: Match):
    layers = int(match.group(1))
    yield LineResult(params={"mlp_layers": layers})

task_names = ["facing", "pos_rel_to_start", "pos_rel_to_end", "facing_wall", "pos", "walls_around"]
@matcher(r"(?:final|acc).*\[([\d., ]+)\]")
def probe_accuracy(match: Match):
    for index, accuracy in enumerate(match.group(1).split(",")):
        params = {"task": task_names[index]}
        yield LineResult(accuracy=float(accuracy), params=params)
    yield LineResult(reset_params=True)

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def derive_name(params: StrDict):
    module = params.get("python_module")
    if module == "probe.eval":
        return "from_scratch_train" if params.get("split") == "train" else "frozen"
    if module == "probe.train":
        return "from_scratch" if len(params.get("mapping", [])) > 0 else "base_model"
    return params.get("name")

def remove_prefix(string: str, prefix: str):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def remove_suffix(string: str, prefix: str):
    if string.endswith(prefix):
        return string[:-len(prefix)]
    return string

def shorten_dataset(params: StrDict, key: str):
    name = params.get(key)
    if name is not None:
        return name
    name = params.get(f"{key}_name")
    if name is None:
        return None
    name = remove_prefix(name, "karel_")
    name = remove_suffix(name, "_uniform_noloops_nocond")
    return name

def post_process_params(params: StrDict):
    params = params.copy()
    params["name"] = derive_name(params)
    for key in ["dataset", "probe_dataset"]:
        params[key] = shorten_dataset(params, key)
    if "hidden_state_layer" in params:
        try:
            params["hidden_state_layer"] = int(params["hidden_state_layer"])
        except:
            pass
    return params

remap_steps = {old: new for old, new in (map(int, steps.split(":")) for steps in args.remap_steps)}

def find_points(lines: list[str]):
    params = {}
    step = None
    for line in lines:
        for matcher in matchers:
            for result in matcher(line):
                params.update(result.params)
                if result.step is not None:
                    step = result.step
                    step = remap_steps.get(step, step)
                if result.accuracy is not None:
                    assert step is not None, "accuracy without step"
                    if not skip_step(step):
                        yield params, step, result.accuracy
                if result.reset_params:
                    params = {}

display_params: dict[str, str] = {key: display_key for key, display_key in (arg.split(":") for arg in args.params)}

def parse_value(value: str):
    try:
        return ast.literal_eval(value)
    except:
        return value

show_only = [{key: parse_value(value) for key, value in (show.split(":") for show in show_only.split(","))} for show_only in args.show_only]

relevant_keys = set()
relevant_keys |= display_params.keys()
for show in show_only:
    relevant_keys |= show.keys()

def filter_params(params: StrDict):
    return {key: value for key, value in params.items() if key in relevant_keys}

def parse_file(path: str, log_params: StrDict):
    with open(path) as log:
        lines = log.readlines()
    runs: defaultdict[hashabledict, Points] = defaultdict(lambda: [])
    for point_params, step, accuracy in find_points(lines):
        combined_params = post_process_params({**log_params, **point_params})
        combined_params = filter_params(combined_params)
        runs[hashabledict(combined_params)].append((step, accuracy))
    if args.debug:
        for params, points in runs.items():
            print(params, len(points))
    yield from runs.items()

def transpose(points: Points) -> tuple[list[int], list[float]]:
    steps, accuracies = map(list, zip(*points))
    return steps, accuracies

logs: dict[str, StrDict] = {}
current_params = {}
for log in args.logs:
    if ":" in log:
        key, value = log.split(":")
        try:
            value = ast.literal_eval(value)
        except:
            pass
        current_params[key] = value
    else:
        logs[log] = current_params.copy()

runs = [run for log, params in logs.items() for run in parse_file(log, params)]

for point in args.add_points:
    step, accuracy = point.split(":")
    step = int(step)
    accuracy = float(accuracy)
    point = (step, accuracy)
    for _, points in runs:
        points.append(point)

for constant in args.constants:
    accuracy, *params = constant.split(",")
    accuracy = float(accuracy)
    params = {key: parse_value(value) for key, value in (param.split(":") for param in params)}
    points = [(0, accuracy)]
    runs.append((params, points))

def filter_run(run):
    if len(args.show_only) == 0:
        return True
    params, _ = run
    return any(all(key in params and params[key] == value for key, value in show.items()) for show in show_only)

runs = [run for run in runs if filter_run(run)]

def point_sort_key(point: Point):
    step, _ = point
    return step

runs = [(params, list(sorted(points, key=point_sort_key))) for params, points in runs]

if all(len(points) == 0 for _, points in runs):
    print("No points")
    exit()

max_step = max(step for _, points in runs for step, _ in points)
for _, points in runs:
    if len(points) == 1:
        [(_, accuracy)] = points
        points.append((max_step, accuracy))

def format_param(key: str, value: Any):
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

def without(dict: StrDict, *keys: str):
    result = dict.copy()
    for key in keys:
        if key in result:
            del result[key]
    return result

DisplayLegend = Callable[[dict[str, Any], str], Artist]

display: dict[str, tuple[list[StrDict], DisplayLegend]] = {
    "marker": (
        [{"marker": marker, "markersize": 4} for marker in ["o", "^", "D", "s", "*", "P"]],
        lambda kwargs, label: Line2D([0], [0], label=label, color="white", markerfacecolor=kwargs.get("color", "black"), markersize=10, **without(kwargs, "color", "markersize"))),
    "linestyle": (
        [{"linestyle": linestyle} for linestyle in ['-', '--', '-.', ':', (0, (1, 5)), (0, (3, 1, 1, 1, 1, 1))]],
        lambda kwargs, label: Line2D([0], [0], label=label, **{"color": "black", **kwargs})),
    "fillstyle": (
        [{"fillstyle": fillstyle} for fillstyle in ['full', 'none', 'left', 'right', 'bottom', 'top']],
        lambda kwargs, label: Line2D([0], [0], marker="s", label=label, **{"color": "black", **kwargs})),
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
        kwargs.update(display_dict)
        if display_legend is None:
            display_legend = current_display_legend
    assert display_legend is not None
    plot_kwargs[value] = (kwargs, display_legend)
    return kwargs

def get_params_kwargs(params: StrDict):
    kwargs: StrDict = {}
    for key, value in params.items():
        kwargs.update(get_param_kwargs(key, value))
    return kwargs

def param_value_sort_key(key: str):
    if key == "hidden_state_layer":
        return lambda value: math.inf if value == "mean" else value
    return lambda value: value

# force consistent order of display_kwargs assignment
for key in {key for params, _ in runs for key in params.keys()}:
    for value in sorted({params[key] for params, _ in runs}, key=param_value_sort_key(key)):
        get_param_kwargs(key, value)

def normalize(kernel: list[float]):
    s = sum(kernel)
    return [factor / s for factor in kernel]

def smooth_points(points: Points):
    if args.smooth is None:
        return points
    steps, accuracies = transpose(points)
    accuracies = savgol_filter(accuracies, window_length=args.smooth, polyorder=min(args.smooth - 1, 3))
    return list(zip(steps, accuracies))

runs = [(params, smooth_points(points)) for params, points in runs]

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
    for value in sorted(plot_kwargs.keys(), key=param_value_sort_key(key)):
        kwargs, display_legend = plot_kwargs[value]
        label = format_param(key, value)
        display_keys, display_value = next(iter(kwargs.items()))
        legend_element = display_legend(kwargs, label)
        legend_elements.append(legend_element)

plt.legend(handles=legend_elements, loc="lower right", fontsize="small")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, dpi=300)

with open(f"{args.out}.sh", "w") as cmd:
    cmd.write(shlex.join(["python"] + sys.argv))

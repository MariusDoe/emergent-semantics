import sys
import re
import ast
import matplotlib.pyplot as plt
import os
from argparse import Namespace
from collections import defaultdict

mode = sys.argv[1]
log_file_names = sys.argv[2:]
run_prefixes = set((os.path.splitext(name)[0] for name in log_file_names))

def model_training():
    accuracy_pattern = re.compile(r"step (\d+): acc: ([0-9.]+)")

    def find_accuracies(lines):
        for line in lines:
            match = accuracy_pattern.search(line)
            if not match:
                continue
            step = int(match.group(1))
            accuracy = float(match.group(2))
            yield step, accuracy

    def parse_file(prefix):
        with open(prefix + ".out") as out:
            header = out.readline()
        args = re.sub(r'^args=Namespace\((.*)\)$', r'{\1}', header)
        args = re.sub(r'(\w+)=', r'"\1":', args)
        # Replace <Enum: 'value'> with just 'value'
        args = re.sub(r"<[^:]+: '([^']+)'>", r"'\1'", args)
        args = ast.literal_eval(args)
        with open(prefix + ".err") as err:
            lines = err.readlines()
        accuracies = list(find_accuracies(lines))
        yield args, accuracies

    return parse_file

def probe_training():
    step_pattern = re.compile(r"step_(\d+)")
    accuracy_pattern = re.compile(r"(?:final|acc).*\[([\d., ]+)\]")

    task_names = ["facing", "pos_rel_to_start", "pos_rel_to_end", "facing_wall", "pos", "walls_around"]

    def find_accuracies(lines):
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
            for index, accuracy in enumerate(accuracy_match.group(1).split(",")):
                yield index, step, float(accuracy)
            step = None

    def parse_file(prefix):
        with open(prefix + ".out") as out:
            lines = out.readlines()
        runs = defaultdict(lambda: [])
        for index, step, accuracy in find_accuracies(lines):
            runs[index].append((step, accuracy))
        name = os.path.basename(prefix)
        for index, accuracies in runs.items():
            yield {"name": name, "task": task_names[index]}, accuracies

    return parse_file

parse_file = None
if mode == "model":
    parse_file = model_training()
if mode == "probe":
    parse_file = probe_training()

assert parse_file is not None, f"unknown {mode=}"

runs = [run for prefix in run_prefixes for run in parse_file(prefix)]
common_keys = {key for key in runs[0][0] if all(args[key] == runs[0][0][key] for args, _ in runs)}
distinguishing_args = [{key: value for key, value in args.items() if key not in common_keys} for args, _ in runs]

def remove_prefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def remove_suffix(string, prefix):
    if string.endswith(prefix):
        return string[:-len(prefix)]
    return string

def format_arg(key, value):
    if key == "dataset_name":
        key = "ds"
        value = remove_prefix(value, "karel_")
        value = remove_suffix(value, "_uniform_noloops_nocond")
    if key == "learning_rate":
        key = "lr"
    return f"{key}:{value}"

def get_name(args):
    for key in ['output_dir', 'num_warmup_steps']:
        if key in args:
            del args[key]
    return " ".join(format_arg(key, value) for key, value in args.items())

data = {get_name(args): tuple(map(list, zip(*accuracies))) for args, (_, accuracies) in zip(distinguishing_args, runs)}

plt.figure(figsize=(8, 6))
for name, (steps, accuracies) in data.items():
    plt.plot(steps, accuracies, marker="o", label=name)

plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Accuracy over Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)

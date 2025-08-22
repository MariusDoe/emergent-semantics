import sys
import re
import ast
import matplotlib.pyplot as plt
import os
from argparse import Namespace
from collections import defaultdict

log_file_names = sys.argv[2:]
run_prefixes = set((os.path.splitext(name)[0] for name in log_file_names))

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
    return args, accuracies

runs = [parse_file(prefix) for prefix in run_prefixes]
common_keys = {key for key in runs[0][0] if all(args[key] == runs[0][0][key] for args, _ in runs)}
distinguishing_args = [{key: value for key, value in args.items() if key not in common_keys} for args, _ in runs]

def get_name(args):
    for key in ['output_dir', 'num_warmup_steps']:
        if key in args:
            del args[key]
    return str(args.items())

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

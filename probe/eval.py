import argparse

import torch
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from data import karel
from utils import model as model_utils
from probe.dataset import SemanticKarelDataset
from probe.model import MLP
from probe.train import eval_ensemble, print_probe_eval
from utils.config import Config
from collections import Counter, defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained probe on a semantic dataset")
    Config.add_train_args(parser)
    Config.add_eval_args(parser)
    Config.add_semantic_args(parser)
    parser.add_argument("--probe_dataset_name")
    parser.add_argument("--probe_output_dir")
    parser.add_argument("--probe_checkpoint_steps", type=int)
    parser.add_argument("--probe_rerun_code", action="store_true")
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--probe_split")
    parser.add_argument("--print_label_frequencies", action="store_true")
    parser.add_argument("--print_token_frequencies", action="store_true")
    parser.add_argument("--print_results", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    print()
    dataset_config = Config(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = model_utils.load_pretrained(dataset_config, load_tokenizer_only=True, add_new_actions=True)
    except:
        print("could not load tokenizer, reconstructing.")
        add_conditionals = "nocond" not in dataset_config.dataset
        tokenizer = AutoTokenizer.from_pretrained(dataset_config.model_name)
        karel.add_special_tokens(
            tokenizer,
            model=None,
            add_conditionals=add_conditionals
        )
        karel.add_tokens(tokenizer, dataset_config.new_actions)

    def load_dataset(**kwargs):
        return SemanticKarelDataset(
            dataset_config,
            tokenizer,
            drop_last=dataset_config.drop_last,
            filter_inactive=dataset_config.eval_alt_active in ["0", "1"],
            single_label=not dataset_config.all_labels,
            **{"filter_correct": False, **kwargs},
        )

    if args.print_token_frequencies:
        action_tokens = {tokenizer(action)["input_ids"][0]: action for action in karel.ACTION_TOKENS + dataset_config.new_actions}
        for label, dataset in [
            ("total", load_dataset()),
            ("correct", load_dataset(filter_correct=True)),
            ("incorrect", load_dataset(filter_incorrect=True)),
        ]:
            counter = Counter()
            for sample in dataset.filtered:
                code_tokens = sample["code_tokens"]
                for token in code_tokens:
                    if token in action_tokens:
                        counter[token] += 1
            total = counter.total()
            for token, action in action_tokens:
                print(f"token frequency for {label=} {action=}: {counter[token] / total}")
        return

    dataset = load_dataset()

    if args.print_label_frequencies:
        task_names = ["facing", "pos_rel_to_start", "pos_rel_to_end", "facing_wall", "pos", "walls_around"]
        buckets = defaultdict(lambda: Counter())
        for labels in dataset.labels:
            for task, label in zip(task_names, labels):
                value = label[0].tolist()
                value = value[0] if len(value) == 1 else tuple(value)
                buckets[task][value] += 1
        for task, bucket in buckets.items():
            print(task)
            for value, count in bucket.most_common():
                print(value, count / len(dataset.labels) * 100.0)
        return
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.per_device_eval_batch_size,
        num_workers=2,
    )

    dropout_p = 0.2 # from defaults in train_with_config

    model_config = dataset_config.update(
        dataset=args.probe_dataset_name or args.dataset_name,
        output_dir=args.probe_output_dir or args.output_dir,
        step=args.probe_checkpoint_steps or args.checkpoint_steps,
        split=args.probe_split or args.split,
        rerun_code=args.probe_rerun_code,
    )

    def load_model(task_idx, ensemble_idx, num_class, mlp_layers):
        label_shape = dataset[0][1][task_idx].shape[:-1]
        model = MLP(
            dataset.input_shape,
            label_shape,
            num_classes=num_class,
            num_layers=mlp_layers,
            use_layernorm=model_config.mlp_layernorm,
            dropout_p=dropout_p,
        )
        model = model.to(device)
        local_config = model_config.update(mlp_layers=mlp_layers)
        probe_fn = local_config.semantic_probe_path(task_idx=task_idx, ensemble_idx=ensemble_idx)
        model.load_state_dict(torch.load(probe_fn))
        return model

    if model_config.mlp_layers is None:
        layers = [1, 2, 3]
    else:
        layers = [model_config.mlp_layers]

    ensemble = [[[load_model(task_idx=task_idx, ensemble_idx=ensemble_idx, num_class=num_class, mlp_layers=mlp_layers)
            for mlp_layers in layers]
            for ensemble_idx, num_class in enumerate(num_classes)]
            for task_idx, num_classes in enumerate(dataset.num_classes)]
    layer_results = eval_ensemble(ensemble, dataloader, all_stats=True, print_results=args.print_results)
    print_probe_eval(layer_results, dataset, dataset_config)

if __name__ == "__main__":
    main()

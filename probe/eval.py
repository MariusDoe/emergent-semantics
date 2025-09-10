import argparse

import torch
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from data import karel
from utils import model as model_utils
from probe.dataset import SemanticKarelDataset
from probe.model import MLP
from probe.train import eval_ensemble
from utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained probe on a semantic dataset")
    Config.add_train_args(parser)
    Config.add_eval_args(parser)
    Config.add_semantic_args(parser)
    parser.add_argument("--probe_dataset_name")
    parser.add_argument("--probe_output_dir")
    parser.add_argument("--probe_checkpoint_steps", type=int)
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--probe_split")
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

    dataset = SemanticKarelDataset(
        dataset_config,
        tokenizer,
        drop_last=dataset_config.drop_last,
        filter_correct=False,
        filter_inactive=dataset_config.eval_alt_active in ["0", "1"],
        single_label=not dataset_config.all_labels,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.per_device_eval_batch_size,
        num_workers=2,
    )

    dropout_p = 0.2 # from defaults in train_with_config

    model_config = dataset_config.update(
        dataset=args.probe_dataset_name,
        output_dir=args.probe_output_dir,
        step=args.probe_checkpoint_steps,
        split=args.probe_split,
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
    layer_results = eval_ensemble(ensemble, dataloader)

    for layer_idx, task_results in enumerate(layer_results):
        accs = []
        for task_idx, results in enumerate(task_results):
            correct = results["correct"]
            total = results["total"]
            if total:
                acc = correct / total * 100
                accs.append(acc)
            else:
                accs.append("n/a")
        print(f"{layer_idx=} {accs=}")

if __name__ == "__main__":
    main()

import argparse
from datetime import datetime
from typing import Optional

import networkx as nx
import torch
from geoopt.optim import RiemannianAdam
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.classification import BinaryF1Score, MultilabelF1Score
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    enable_full_determinism,
)

from classifier import HieRoberta
from dataloading import load_merge_encode
from hierarchy import create_full_hierarchy, create_taxonomy

LANGUAGE_SETS = {
    "shared_task": ["bg", "hr", "pl", "ru", "sl"],  # primary languages
    "slavic": ["bg", "hr", "mk", "pl", "ru", "sl"],  # add Macedonian (close to Bulgarian)
    "slavic_en": ["bg", "en", "hr", "mk", "pl", "ru", "sl"],  # add English
    "european_latin": ["bg", "de", "en", "es", "fr", "it", "hr", "mk", "pl", "ru", "sl"],  # add all Latin alphabet
    "parlamint": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Greek
    "european": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Georgian
    "all": ["ar", "bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Arabic
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--freeze", "-f", action="store_true")
    parser.add_argument("--gnn", "-g", choices=["gcn", "hgcn", "hie"], default="gcn")
    parser.add_argument("--hierarchy", "-hi", choices=["full", "taxonomy"])
    parser.add_argument("--hp_search", "-hp", action="store_true")
    parser.add_argument("--languages", "-l", choices=LANGUAGE_SETS.keys(), default="european_latin")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--model", "-m", default="classla/xlm-r-parla")
    parser.add_argument("--node_classification", "-nc", action="store_true")
    parser.add_argument("--node_dim", "-nd", type=int, default=64)
    parser.add_argument("--pooling", "-p", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--test_data", "-test", default="slavicnlp2025")
    parser.add_argument("--train_data", "-train", default="semeval2021,semeval2023,semeval2024")
    parser.add_argument("--translations", "-t", action="store_true")
    parser.add_argument("--val_size", "-v", type=float, default=0.2)
    return parser.parse_args()


def make_multilabel_metrics(
    binarizer: MultiLabelBinarizer,
    hierarchy: Optional[nx.DiGraph] = None,
):
    if hierarchy:
        leaves = [v for v in hierarchy.nodes() if hierarchy.out_degree(v) == 0]
        leave_idx = [i for i, c in enumerate(binarizer.classes_) if c in leaves]
        num_leaves = len(leaves)
    else:
        num_leaves = len(binarizer.classes_)
        leave_idx = list(range(num_leaves))

    hierarchical_f1 = MultilabelF1Score(binarizer.classes_, average="micro")
    micro_f1 = MultilabelF1Score(num_leaves, average="micro")
    macro_f1 = MultilabelF1Score(num_leaves, average="macro")

    def add_ancestors(binary_labels):
        if not hierarchy:
            return binary_labels

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.sigmoid(logits) > 0.5
        # add ancestors for hierarchical F1

        # remove ancestors for macro/micro F1
        preds_leaves = torch.Tensor(preds[:, leave_idx])
        labels_leaves = torch.Tensor(labels[:, leave_idx])

        return {
            "hierarchical_f1": hierarchical_f1(torch.Tensor(logits), torch.Tensor(labels)),
            "micro_f1": micro_f1(preds_leaves, labels_leaves),
            "macro_f1": macro_f1(preds_leaves, labels_leaves),
        }

    return compute_metrics


def model_init():
    """Required for hyperparameter search."""
    return HieRoberta(args, config, hierarchy)


if __name__ == "__main__":
    enable_full_determinism(seed=42)

    args = parse_args()
    start_time = datetime.now()

    if args.hierarchy:
        experiment_stem = f"{args.hierarchy}-{args.gnn}"
    else:
        experiment_stem = "flat"
    experiment_name = f"{args.model}-{experiment_stem}-{args.languages}-{start_time.strftime('%y%m%d%H%M')}"

    print(f"Experiment: {experiment_name}\n")
    for k, v in args.__dict__.items():
        print(f"{k}:\t{v}")
    print("\nStart Time:", start_time)

    match args.hierarchy:
        case "full":
            hierarchy = create_full_hierarchy()
        case "taxonomy":
            hierarchy = create_taxonomy()
        case _:
            hierarchy = None

    dataset, binarizer = load_merge_encode(
        languages=LANGUAGE_SETS[args.languages],
        train_datasets=args.train_data,
        hierarchy=hierarchy,
        include_translations=args.translations,
        val_size=args.val_size,
    )

    if args.debug:
        dataset["train"] = dataset["train"].shard(num_shards=10, index=1)

    config = XLMRobertaConfig.from_pretrained(args.model, num_labels=len(binarizer.classes_))
    # model = HieRoberta(args, config, G)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer)

    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            lambda sample: tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt"),
            batched=True,
        )

    training_args = TrainingArguments(
        # general
        output_dir=experiment_name,
        num_train_epochs=args.epochs,
        fp16=True,
        seed=args.seed,
        eval_strategy="epoch",
        # optimizer
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=1 / args.epochs,  # first epoch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # saving
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        compute_metrics=make_multilabel_metrics(binarizer, hierarchy),
    )

    if args.hp_search:
        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
        print(best_run)
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

    trainer.train()

    results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    print(results)
    print("Total execution time:", datetime.now() - start_time)

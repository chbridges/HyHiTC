from datetime import datetime
from typing import Optional

import networkx as nx
import numpy as np
import torch
from geoopt.optim import RiemannianAdam
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.classification import MultilabelF1Score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    set_seed,
)

from classifier import HieRoberta
from config import LANGUAGE_SETS, add_hyp_default_args, parse_args
from dataloading import load_merge_encode
from hierarchy import create_full_hierarchy, create_taxonomy
from PersuasionNLPTools.config import VALID_LABELS


class RiemannianTrainer(Trainer):
    """The default Trainer does not support custom optimizers, so we need to slightly
    modify either its create_optimizer_and_scheduler or create_optimizer method."""

    def create_optimizer(self):
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        # We replace optimizer_cls with RiemannianAdam
        _, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        for key in ("params", "model", "optimizer_dict"):
            if key in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop(key)
        self.optimizer = RiemannianAdam(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def make_multilabel_metrics(
    binarizer: MultiLabelBinarizer,
    hierarchy: Optional[nx.DiGraph] = None,
):
    num_leaves = len(VALID_LABELS)
    leaves_idx = [i for i, c in enumerate(binarizer.classes_) if c in VALID_LABELS]

    label2id = {c: i for i, c in enumerate(binarizer.classes_)}

    hierarchical_f1 = MultilabelF1Score(len(binarizer.classes_), average="micro")
    micro_f1 = MultilabelF1Score(num_leaves, average="micro")
    macro_f1 = MultilabelF1Score(num_leaves, average="macro")

    def add_ancestors(y: torch.Tensor):
        if not hierarchy:
            return y
        paths = nx.all_pairs_shortest_path_length(hierarchy.reverse())
        for target, distances in paths:
            ix_rows = np.where(y[:, label2id[target]] > 0)[0]
            ancestors = [label2id[k] for k in distances.keys()]
            y[tuple(np.meshgrid(ix_rows, ancestors))] = 1
        return y

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.sigmoid(torch.Tensor(logits)) > 0.5
        # add ancestors for hierarchical F1
        preds_with_ancestors = torch.Tensor(add_ancestors(preds))
        labels_with_ancestors = torch.Tensor(add_ancestors(labels))
        # remove ancestors for macro/micro F1
        preds_leaves = torch.Tensor(preds[:, leaves_idx])
        labels_leaves = torch.Tensor(labels[:, leaves_idx])
        return {
            "hierarchical_f1": hierarchical_f1(preds_with_ancestors, labels_with_ancestors),
            "micro_f1": micro_f1(preds_leaves, labels_leaves),
            "macro_f1": macro_f1(preds_leaves, labels_leaves),
        }

    return compute_metrics


if __name__ == "__main__":
    args = parse_args()
    set_seed(seed=args.seed)
    start_time = datetime.now()

    if args.hierarchy:
        experiment_stem = f"{args.hierarchy}-{args.gnn}"
    else:
        experiment_stem = "flat"
    experiment_name = f"{args.language_model}-{experiment_stem}-{args.languages}-{start_time.strftime('%m%d%H%M')}"

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
    id2label = dict(enumerate(binarizer.classes_))
    label2id = {c: i for i, c in id2label.items()}

    if args.debug:
        dataset["train"] = dataset["train"].shard(num_shards=10, index=1)

    config = XLMRobertaConfig.from_pretrained(args.language_model, id2label=id2label, label2id=label2id)
    if args.gnn in ["HGCN", "HIE"]:
        args = add_hyp_default_args(args, config)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.language_model)
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
        greater_is_better=True,
    )

    trainer = RiemannianTrainer(
        model_init=lambda: HieRoberta(args, config, hierarchy),  # required for hyperparameter search
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        compute_metrics=make_multilabel_metrics(binarizer, hierarchy),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    if args.hp_search:
        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
        print(best_run)
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

    trainer.train()

    results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    print(results)

    test_logits = trainer.predict(dataset["test"]).predictions
    test_probas = torch.sigmoid(torch.Tensor(test_logits))
    test_preds = [np.where(row > 0.5)[0] for row in test_probas]
    test_labels = [id2label[p] for p in test_preds[0]]

    print("Total execution time:", datetime.now() - start_time)

from datetime import datetime
from pathlib import Path
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
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
    set_seed,
)

from classifier import HieRobertaConfig, HieRobertaModel
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


def model_init():
    """Required for hyperparameter search."""
    language_model = XLMRobertaModel.from_pretrained(args.language_model, add_pooling_layer=args.pooling, config=config)
    return HieRobertaModel(language_model, args, config, hierarchy, pos_weight)


def predict_test_labels(
    model: HieRobertaModel,
    tokenizer: XLMRobertaTokenizerFast,
    identifier: str,
    test_dir: Path = Path("./data/test_data/"),
):
    for lang in ["BG", "HR", "PL", "RU", "SL"]:
        lang_dir = test_dir / lang
        input_file = lang_dir / "input-file.txt"
        docs_dir = lang_dir / "raw-documents"
        output_dir = lang_dir / f"predictions-{identifier}"

        if not lang_dir.exists():
            continue  # should only happen for HR when debugging on training data

        output_dir.mkdir(exist_ok=True)

        subtask1_option1 = []
        subtask1_option2 = []
        subtask2 = []

        with input_file.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        current_doc = ""
        current_content = ""
        for line in lines:
            doc, start, end = line.split("\t")
            if doc != current_doc:
                current_doc = doc
                with (docs_dir / doc).open("r", encoding="utf-8") as file:
                    current_content = file.read()

            inputs = tokenizer(
                current_content[int(start) : int(end)], padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probas = torch.sigmoid(torch.Tensor(logits)).cpu()
            preds = np.where(probas > 0.5)[0]
            all_labels = [id2label[p] for p in preds]
            subtask2_labels = "\t".join([l for l in all_labels])  # if l in VALID_LABELS])
            # OPTION 1: Check if any valid label was predicted
            subtask1_label_1 = "true" if subtask2_labels else "false"
            # OPTION 2: Check if Persuasion was predicted (hierarchical only)
            subtask1_label_2 = "true" if "Persuasion" in all_labels else "false"

            subtask1_option1.append(f"{line.strip()}\t{subtask1_label_1}")
            subtask1_option2.append(f"{line.strip()}\t{subtask1_label_2}")
            subtask2.append(f"{line.strip()}\t{subtask2_labels}")

            with (output_dir / "subtask1_option1.csv").open("w", encoding="utf-8") as file:
                file.write("\n".join(subtask1_option1))

            with (output_dir / "subtask1_option2.csv").open("w", encoding="utf-8") as file:
                file.write("\n".join(subtask1_option2))

            with (output_dir / "subtask2.csv").open("w", encoding="utf-8") as file:
                file.write("\n".join(subtask2))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA")

    args = parse_args()
    set_seed(seed=args.seed)
    start_time = datetime.now()
    timestamp = start_time.strftime("%m%d%H%M")

    if args.hierarchy:
        experiment_stem = f"{args.hierarchy}-{args.gnn}"
        if args.node_classification:
            experiment_stem = experiment_stem + "-nc"
    else:
        experiment_stem = "flat"
    experiment_name = f"{args.language_model}-{args.languages}-{experiment_stem}-{timestamp}"

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
        test_datasets=args.test_data,
        hierarchy=hierarchy,
        machine_translations=args.machine_translations,
        val_size=args.val_size,
    )
    if args.debug:
        dataset["train"] = dataset["train"].shard(num_shards=10, index=1)

    # weights neg_freq/pos_freq for weighted binary cross-entropy
    pos_frequencies = np.sum(dataset["train"]["labels"], axis=0)
    pos_weight = torch.Tensor((len(dataset["train"]) - pos_frequencies) / pos_frequencies)
    pos_weight[pos_weight == torch.inf] = 1.0

    id2label = dict(enumerate(binarizer.classes_))
    label2id = {c: i for i, c in id2label.items()}

    config = HieRobertaConfig.from_pretrained(args.language_model, id2label=id2label, label2id=label2id)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.language_model)
    data_collator = DataCollatorWithPadding(tokenizer)

    if args.gnn in ["HGCN", "HIE"]:
        args = add_hyp_default_args(args, config)

    if not args.hp_search:
        model = model_init()

    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            lambda sample: tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt"),
            batched=True,
        )

    training_args = TrainingArguments(
        # general
        output_dir=experiment_name,
        num_train_epochs=args.finetune_epochs,
        fp16=True,
        seed=args.seed,
        eval_strategy="epoch",
        dataloader_num_workers=2,
        # optimizer
        weight_decay=0.01,
        learning_rate=args.finetune_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=1 / args.finetune_epochs,  # first epoch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_grad_norm=3.0,
        # saving
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_safetensors=args.gnn not in ["HGCN", "HIE"],  # HGCN's shared tensors are not supported
    )

    trainer = RiemannianTrainer(
        model=model if not args.hp_search else None,
        model_init=model_init if args.hp_search else None,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        compute_metrics=make_multilabel_metrics(binarizer, hierarchy),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    if args.hp_search:
        best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize")
        print(best_run)
        exit()

    if args.pretrain_epochs:
        model.freeze_lm(ratio=1.0)
        setattr(trainer.args, "learning_rate", args.pretrain_lr)
        setattr(trainer.args, "num_train_epochs", args.pretrain_epochs)
        setattr(trainer.args, "warmup_ratio", 1 / args.pretrain_epochs)
        trainer.train()

    if args.finetune_epochs:
        model.unfreeze_lm()
        model.freeze_lm(ratio=args.finetune_freeze)
        setattr(trainer.args, "learning_rate", args.finetune_lr)
        setattr(trainer.args, "num_train_epochs", args.finetune_epochs)
        setattr(trainer.args, "warmup_ratio", 1 / args.finetune_epochs)
        trainer.train()

    results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    print(results)
    if args.gnn in ["HGCN", "HIE"]:
        print("Learned embedding space curvature:", model.hgcn.get_c())

    test_logits = trainer.predict(dataset["test"]).predictions
    test_probas = torch.sigmoid(torch.Tensor(test_logits))
    test_preds = [np.where(row > 0.5)[0] for row in test_probas]
    test_labels = [[id2label[p] for p in test_preds[i]] for i in range(len(test_preds))]

    stats = np.sum(np.vstack(test_probas > 0.5), axis=0)
    for i, l in id2label.items():
        print(f"{l}: {stats[i]}")

    print("Total execution time:", datetime.now() - start_time)

    predict_test_labels(trainer.model, tokenizer, timestamp, Path("./data/TRAIN_version_31_March_2025/"))
    predict_test_labels(trainer.model, tokenizer, timestamp, Path("./data/test_data/"))

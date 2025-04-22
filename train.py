import argparse
from datetime import datetime

import torch
from torchmetrics.classification import BinaryF1Score, MultilabelF1Score
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    enable_full_determinism,
)

from dataloading import load_merge_encode
from hierarchy import create_full_hierarchy, create_taxonomy
from models import MultilabelModel

LANGUAGE_SETS = {
    "shared_task": ["bg", "hr", "pl", "sl", "ru"],  # primary languages
    "slavic": ["bg", "hr", "mk", "pl", "sl", "ru"],  # add Macedonian (close to Bulgarian)
    "slavic_en": ["bg", "en", "hr", "mk", "pl", "sl", "ru"],  # add English
    "european_latin": ["bg", "de", "en", "es", "fr", "it", "hr", "mk", "pl", "sl", "ru"],  # add all Latin alphabet
    "parlamint": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Greek
    "european": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Georgian
    "all": ["ar", "bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Arabic
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--gnn", "-g", choices=["gcn", "gat", "hgcn"], default="gcn")
    parser.add_argument("--hierarchy", "-hi", choices=["full", "taxonomy"])
    parser.add_argument("--include_clef", "-ic", action="store_true")
    parser.add_argument("--languages", "-l", choices=LANGUAGE_SETS.keys(), default="slavic")
    parser.add_argument("--model", "-m", default="classla/xlm-r-parla")
    parser.add_argument("--node_classification", "-nc", action="store_true")
    parser.add_argument("--node_size", "-ns", type=int, default=8)
    parser.add_argument("--val_size", "-v", type=float, default=0.2)
    return parser.parse_args()


def make_multilabel_metrics(num_classes: int):
    metrics = {
        "hierarchical_f1": BinaryF1Score(),
        "micro_f1": MultilabelF1Score(num_classes, average="micro"),
        "macro_f1": MultilabelF1Score(num_classes, average="macro"),
    }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # TODO: remove ancestors for micro and macro f1
        return {name: metric(torch.Tensor(logits), torch.Tensor(labels)) for name, metric in metrics.items()}

    return compute_metrics


enable_full_determinism(seed=0)

args = parse_args()

match args.hierarchy:
    case "full":
        G = create_full_hierarchy()
    case "taxonomy":
        G = create_taxonomy()
    case _:
        G = None

dataset, binarizer = load_merge_encode(args, LANGUAGE_SETS[args.languages], G)

if args.debug:
    for split in dataset.keys():
        dataset[split] = dataset[split].shard(num_shards=10, index=1)

config = XLMRobertaConfig.from_pretrained(args.model, num_labels=len(binarizer.classes_))
model = MultilabelModel(args, config, G)
tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model)
data_collator = DataCollatorWithPadding(tokenizer)

for split in dataset.keys():
    dataset[split] = dataset[split].map(
        lambda sample: tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt"),
        batched=True,
    )

if args.hierarchy:
    output_stem = f"{args.model}-{args.hierarchy}-{args.gnn}"
else:
    output_stem = f"{args.model}-flat"

training_args = TrainingArguments(
    output_dir=f"{output_stem}-{datetime.now().strftime('%y%m%d%H%M')}",
    num_train_epochs=args.epochs,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model="macro_f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    processing_class=tokenizer,
    compute_metrics=make_multilabel_metrics(num_classes=config.num_labels),
)

trainer.train()

results = trainer.evaluate(dataset["train"])
print(results)

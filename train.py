from datetime import datetime

import torch
from torchmetrics.classification import MultilabelF1Score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, enable_full_determinism

from dataloading import load_merge_encode
from models import Model
from utils import parse_args


def make_multilabel_metrics(num_classes: int):
    metrics = {
        "micro_f1": MultilabelF1Score(num_classes, average="micro"),
        "macro_f1": MultilabelF1Score(num_classes, average="macro"),
    }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {name: metric(torch.Tensor(logits), torch.Tensor(labels)) for name, metric in metrics.items()}

    return compute_metrics


enable_full_determinism(seed=0)

args = parse_args()
dataset = load_merge_encode(args)
model = Model(args)

for split in dataset.values():
    split.map(
        lambda sample: model.tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt"),
        batched=True,
    )

data_collator = DataCollatorWithPadding(model.tokenizer)

training_args = TrainingArguments(
    output_dir=f"{args.model}-{args.gnn}-{datetime.now().strftime('%y%m%d%H%M')}",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    per_device_training_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    processing_class=model.tokenizer,
    compute_metrics=make_multilabel_metrics(len(dataset.encoder.classes_)),
)

trainer.train()

from datetime import datetime

import evaluate
import numpy as np
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, enable_full_determinism

from dataloading import load_merge_encode
from models import Model
from utils import parse_args


def compute_metrics(eval_pred):
    metrics = evaluate.combine(["f1", "precision", "recall"])
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    metrics.compute(predictions=preds, references=labels)


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
    compute_metrics=compute_metrics,
)

trainer.train()

import torch
from dataloading import load_merge_encode
from models import Model
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, enable_full_determinism
from utils import parse_args


if __name__ == "__main__":
    enable_full_determinism(seed=0)

    args = parse_args()
    dataset = load_merge_encode(args)
    model = Model(args)

    for split in dataset.values():
        split.map(
            lambda sample: model.tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt"),
            batched=True,
        )

    collator = DataCollatorWithPadding(model.tokenizer)

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from dataloading import (
    load_semeval_2021_task_6_subtask_1,
    load_semeval_2023_task_3_subtask_3,
    load_semeval_2024_task_4_subtask_1,
    load_slavicnlp_2025,
)
from train import LANGUAGE_SETS

logging.getLogger().setLevel(logging.INFO)
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-b", type=int, default=8)
parser.add_argument("--model", "-m", default="google/madlad400-3b-mt")
parser.add_argument("--src_langs", "-src", choices=LANGUAGE_SETS.keys(), default="all")
parser.add_argument("--tgt_langs", "-tgt", choices=LANGUAGE_SETS.keys(), default="slavic_en")
args = parser.parse_args()

SRC_LANGS = LANGUAGE_SETS[args.src_langs]
TGT_LANGS = LANGUAGE_SETS[args.tgt_langs]

logging.info(f"Using CUDA: {torch.cuda.is_available()}")
logging.info(f"Target languages: {TGT_LANGS}")

logging.info(f"Loading model: {args.model}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    args.model,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ),
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model)


def translate_batch(text_list: list[str], tgt_lang: str) -> list[str]:
    max_length = max([len(text) for text in text_list])
    batch = [f"<2{tgt_lang}> {text}" for text in text_list]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids.to(
        model.device
    )
    outputs = model.generate(input_ids=inputs, max_new_tokens=int(1.5 * max_length))
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translation


src_data = pd.concat(
    [
        func(SRC_LANGS)
        for func in (
            load_semeval_2021_task_6_subtask_1,
            load_semeval_2023_task_3_subtask_3,
            load_semeval_2024_task_4_subtask_1,
            load_slavicnlp_2025,
        )
    ]
)
# Linebreaks are ignored by XLM-R but not by MadLad
src_data["text"] = src_data["text"].apply(lambda t: re.sub(r"(\s|\\n|\\)+", " ", t))

t_path = Path(f"./data/translations/")
t_path.mkdir(exist_ok=True)

for tgt_lang in TGT_LANGS:
    parquet_file = t_path / f"{tgt_lang}.parquet"
    temp_file = t_path / f"{tgt_lang}.temp"

    if parquet_file.exists():
        logging.info(f"Skipping language: {tgt_lang} (parquet exists)")
        continue
    if temp_file.exists():
        logging.info(f"Skipping language: {tgt_lang} (temp exists)")
        continue

    logging.info(f"Translating data to: {tgt_lang}")
    with temp_file.open("w") as file:
        file.write(tgt_lang)

    filtered = src_data[src_data["language"] != tgt_lang]
    if not len(filtered):
        continue

    records = []

    for i in tqdm(range(len(filtered) // args.batch_size + 1)):
        batch = filtered[i * args.batch_size : (i + 1) * args.batch_size]
        if not len(batch):
            break

        ids = batch["id"].to_list()
        labels = batch["labels"].to_list()

        translations = translate_batch(batch["text"].to_list(), tgt_lang)
        for i in range(len(batch)):
            records.append({"id": ids[i], "language": tgt_lang, "text": translations[i], "labels": labels[i]})

    df = pd.DataFrame.from_records(records)
    df.to_parquet(parquet_file)
    temp_file.unlink()

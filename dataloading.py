import json
import re
from itertools import chain
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from hierarchy import LABEL_MAP
from PersuasionNLPTools.config import VALID_LABELS

GLOBAL_DATA_PATH = Path("./data/")


def load_semeval_2021_task_6_subtask_1(languages: list[str]) -> pd.DataFrame:
    """
    Load the data from SemEval 2021 Task 6 Subtask 1.
    https://propaganda.math.unipd.it/semeval2021task6/

    All data are labeled and include languages: en

    :param languages: Set of languages to consider loading.
    :type languages: list[str]
    """
    if "en" not in languages:
        return pd.DataFrame(columns=["id", "language", "text", "labels"])

    data_path = GLOBAL_DATA_PATH / "semeval-2021-task-6" / "data"
    splits = {
        "train": data_path / "training_set_task1.txt",
        "dev": data_path / "dev_set_task1.txt",
        "test": data_path / "test_set_task1.txt",
    }

    records = []
    for name, split in splits.items():
        with split.open("r", encoding="utf-8") as f:
            for i, record in enumerate(json.load(f)):
                records.append(
                    {
                        "id": f"semeval2021_{name}_{i}",
                        "language": "en",
                        "text": record["text"],
                        "labels": [LABEL_MAP[label] for label in record["labels"]],
                    }
                )

    df = pd.DataFrame.from_records(records)
    return df


def load_semeval_2023_task_3_subtask_3(languages: list[str], include_test: bool = False) -> pd.DataFrame:
    """
    Load the data from SemEval 2023 Task 3 Subtask 3.
    https://propaganda.math.unipd.it/semeval2023task3/

    The train and dev data are labeled and include languages: de, en, fr, it, PL, RU
    The test data are unlabeled and include languages: de, el, en, es, fr, it, ka, PL, RU
    There is no test data with gold or baseline labels.

    :param languages: Set of languages to consider loading.
    :type languages: list[str]
    :param include_test: Load the test data with baseline predictions.
    :type include_test: bool, optional
    """
    data_path = GLOBAL_DATA_PATH / "semeval-2023-task-3" / "data"
    records = []

    for lang in languages:
        lang_path = data_path / lang
        if not lang_path.exists():
            continue

        train_text_path = lang_path / "train-labels-subtask-3.template"
        train_labels_path = lang_path / "train-labels-subtask-3.txt"
        dev_text_path = lang_path / "dev-labels-subtask-3.template"
        dev_labels_path = lang_path / "dev-labels-subtask-3.txt"

        if train_text_path.exists():  # does not apply to el/es/ka which are test only
            with train_text_path.open("r", encoding="utf-8") as f:
                train_text = [line.split("\t")[-1].strip() for line in f.readlines()]
            with train_labels_path.open("r", encoding="utf-8") as f:
                train_labels = [line.split("\t")[-1].strip().split(",") for line in f.readlines()]
                train_labels = list(map(lambda l: l if l != [""] else [], train_labels))
            records.extend(
                [
                    {"id": f"semeval2023_train_{lang}_{i}", "language": lang, "text": tup[0], "labels": tup[1]}
                    for i, tup in enumerate(zip(train_text, train_labels))
                ]
            )

            with dev_text_path.open("r", encoding="utf-8") as f:
                dev_text = [line.split("\t")[-1].strip() for line in f.readlines()]
            with dev_labels_path.open("r", encoding="utf-8") as f:
                dev_labels = [line.split("\t")[-1].strip().split(",") for line in f.readlines()]
                dev_labels = list(map(lambda l: l if l != [""] else [], dev_labels))
            records.extend(
                [
                    {"id": f"semeval2023_dev_{lang}_{i}", "language": lang, "text": tup[0], "labels": tup[1]}
                    for i, tup in enumerate(zip(dev_text, dev_labels))
                ]
            )

        if include_test:
            test_text_path = lang_path / "test-labels-subtask-3.template"
            test_labels_path = data_path / ".." / "baselines" / f"baseline-output-subtask3-test-{lang}.txt"
            with test_text_path.open("r", encoding="utf-8") as f:
                test_text = [line.split("\t")[-1].strip() for line in f.readlines()]
            with test_labels_path.open("r", encoding="utf-8") as f:
                test_labels = [line.split("\t")[-1].strip().split(",") for line in f.readlines()]
            records.extend(
                [
                    {"id": f"semeval2023_test_{lang}_{i}", "language": lang, "text": tup[0], "labels": tup[1]}
                    for i, tup in enumerate(zip(test_text, test_labels))
                ]
            )

    if records:
        df = pd.DataFrame.from_records(records)
        return df
    return pd.DataFrame(columns=["id", "language", "text", "labels"])


def load_semeval_2024_task_4_subtask_1(languages: list[str]):
    """
    Load the data from SemEval 2024 Task 4 Subtask 1.
    https://propaganda.math.unipd.it/semeval2024task4/

    All data are labeled and include languages: ar, BG, en, MK

    :param languages: Set of languages to consider loading.
    :type languages: list[str]
    """
    data_path = GLOBAL_DATA_PATH / "semeval-2024-task-4"

    splits = []
    records = []

    for lang in ["ar", "bg", "mk"]:
        if lang in languages:
            splits.append(data_path / "test_labels_ar_bg_md_version2" / f"test_subtask1_{lang}.json")

    if "en" in languages:
        en_splits = [
            data_path / "semeval2024_dev_release" / "subtask1" / "train.json",
            data_path / "semeval2024_dev_release" / "subtask1" / "validation.json",
            data_path / "dev_gold_labels" / "dev_subtask1_en.json",
        ]
        splits.extend(en_splits)

    for split in splits:
        name = re.search(r"(train|validation|dev|test)", split.stem).groups()[0]
        lang_match = re.search(r"subtask1_(.+)\.json", str(split))
        lang = lang_match.groups()[0] if lang_match else "en"

        with split.open("r", encoding="utf-8") as f:
            for i, record in enumerate(json.load(f)):
                records.append(
                    {
                        "id": f"semeval2024_{lang}_{name}_{i}",
                        "language": lang,
                        "text": record["text"],
                        "labels": [LABEL_MAP[label] for label in record["labels"]],
                    }
                )
    if records:
        df = pd.DataFrame.from_records(records)
        return df
    return pd.DataFrame(columns=["id", "language", "text", "labels"])


def load_clef_2024_task_3(languages: list[str], include_dev: bool = False):
    """
    Load the data from CheckThat! Lab @ CLEF 2024 Task 3.

    The train data are labeled and include languages: de, en, fr, it, PL, RU
    The dev data are unlabeled and include languages: de, el, en, es, fr, it, ka, PL, RU
    There is no test data with gold or baseline labels.

    The span-based predictions will be mapped to paragraph-wise multi-label predictions.
    Includes data from SemEval 2023 Task 3 and PTC!

    :param languages: Set of languages to consider loading.
    :type languages: list[str]
    :param include_dev: Load the dev data with (random?) baseline predictions.
    :type include_dev: bool, optional
    """

    def spans_to_paragraph(lang_path: Path, article_id: str) -> list[dict]:
        text_path = lang_path / "train-articles-subtask-3" / f"article{article_id}.txt"
        spans_path = lang_path / "train-labels-subtask-3-spans" / f"article{article_id}-labels-subtask-3.txt"

        # get all non-empty paragraphs with character offsets
        with text_path.open("r", encoding="utf-8") as f:
            paragraphs = f.readlines()
        offsets = [0]
        for p in paragraphs:
            offsets.append(offsets[-1] + len(p) + 1)

        mask = [bool(p.strip()) for p in paragraphs]
        paragraphs = [p for p, m in zip(paragraphs, mask) if m]
        offsets = [o for o, m in zip(offsets, mask) if m] + [offsets[-1]]

        # get all spans and map their labels to paragraphs
        with spans_path.open("r", encoding="utf-8") as f:
            spans = [line.split("\t") for line in f.readlines() if line.strip()]

        labels = {i: [] for i in range(len(paragraphs))}
        for span in spans:
            label = span[1]
            start = int(span[2])
            for i, offset in enumerate(offsets):
                if start < offset:
                    key = max(0, i - 1)
                    labels[key].append(label)
                    break

        return [
            {
                "id": f"clef2024_{article_id}_{i}",
                "language": lang_path.stem,
                "text": paragraphs[i],
                "labels": sorted(set(labels[i])),
            }
            for i in labels
        ]

    data_path = GLOBAL_DATA_PATH / "clef-2024-task-3" / "data"

    records = []

    for lang in languages:
        lang_path = data_path / lang
        if not lang_path.exists():
            continue

        spans = lang_path / "train-labels-subtask-3-spans.txt"
        with spans.open("r", encoding="utf-8") as f:
            unique_ids = sorted(set([line.split("\t")[0] for line in f.readlines()]))

        lang_records = chain.from_iterable([spans_to_paragraph(lang_path, article_id) for article_id in unique_ids])
        records.extend(lang_records)

    if records:
        df = pd.DataFrame.from_records(records)
        return df
    return pd.DataFrame(columns=["id", "language", "text", "labels"])


def load_slavicnlp_2025(languages: list[str]) -> pd.DataFrame:
    """
    Load the training data from the Slavic NLP 2025 Shared Task.
    https://bsnlp.cs.helsinki.fi/shared-task.html

    The training data are labeled and include languages: BG, PL, RU, SL
    """
    data_path = GLOBAL_DATA_PATH / "TRAIN_version_31_March_2025"
    records = []

    for lang in ["BG", "PL", "RU", "SL"]:
        lang_path = data_path / lang
        docs_path = lang_path / "raw-documents"
        subtask1 = lang_path / "subtask-1-annotations.txt"
        subtask2 = lang_path / "subtask-2-annotations.txt"

        with subtask1.open("r", encoding="utf-8") as f:
            spans = [line.strip().split("\t") for line in f.readlines() if line.strip()]
        with subtask2.open("r", encoding="utf-8") as f:
            true_labels = [line.strip().split("\t") for line in f.readlines() if line.strip()]
        unique_files = sorted(set([span[0] for span in spans]))

        for file in unique_files:
            file_path = docs_path / file
            with file_path.open("r", encoding="utf-8") as f:
                paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            file_offsets = [span for span in spans if span[0] == file]
            file_labels = [labels for labels in true_labels if labels[0] == file]

            label_offsets = [l[1] for l in file_labels]
            labels = []
            for offset in file_offsets:
                if offset[1] in label_offsets:
                    labels.append(file_labels.pop(0)[3:])
                else:
                    labels.append([])

            file_records = [
                {
                    "id": f"slavicnlp2025_{file}_{i}",
                    "language": lang,
                    "doc": file,
                    "start": file_offsets[i][1],
                    "end": file_offsets[i][2],
                    "text": paragraphs[i],
                    "labels": labels[i],
                }
                for i in range(len(file_offsets))
            ]

            records.extend(file_records)

    df = pd.DataFrame.from_records(records)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    return df


def add_ancestors(df: pd.DataFrame, G: Optional[nx.DiGraph] = None) -> pd.DataFrame:
    """
    Add the ancestors of all annotated labels.
    """
    df["labels"] = df["labels"].map(
        lambda labels: sorted(set(chain.from_iterable([nx.ancestors(G, node) for node in labels] + [labels])))
    )
    return df


def encode_labels(df: pd.DataFrame, labels: list | set = VALID_LABELS) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    Binary-encode the labels in a loaded dataframe for model training.
    """
    encoder = MultiLabelBinarizer()
    encoder.fit([labels])
    df["labels"] = df["labels"].map(lambda l: encoder.transform([l])[0])
    return df, encoder


def load_merge_encode(
        languages: list[str],
        train_datasets: str = "semeval2021,semeval2023,semeval2024",
        test_datasets: str = "slavicnlp2025",
        hierarchy: Optional[nx.DiGraph] = None,
        machine_translations: bool = False,
        val_size: float = 0.2,
    ) -> tuple[DatasetDict, MultiLabelBinarizer]:
    dataset_funcs = {
        "clef2024": load_clef_2024_task_3,
        "semeval2021": load_semeval_2021_task_6_subtask_1,
        "semeval2023": load_semeval_2023_task_3_subtask_3,
        "semeval2024": load_semeval_2024_task_4_subtask_1,
        "slavicnlp2025": load_slavicnlp_2025,
    }
    train_datasets = train_datasets.split(",")

    train_df = pd.concat([dataset_funcs[dataset](languages) for dataset in train_datasets])
    test_df = pd.concat([dataset_funcs[dataset](languages) for dataset in test_datasets.split(",")])

    train_df["text"] = train_df["text"].apply(lambda t: re.sub(r"(\s|\\n|\\)+", " ", t))
    test_df["text"] = test_df["text"].apply(lambda t: re.sub(r"(\s|\\n|\\)+", " ", t))

    # Binary encode labels
    if hierarchy:
        binarizer = MultiLabelBinarizer(classes=hierarchy.nodes).fit([hierarchy.nodes])
        train_df = add_ancestors(train_df, hierarchy)
        test_df = add_ancestors(test_df, hierarchy)
    else:
        binarizer = MultiLabelBinarizer(classes=sorted(VALID_LABELS)).fit([VALID_LABELS])

    train_df["labels"] = train_df["labels"].apply(lambda l: binarizer.transform([l])[0])
    test_df["labels"] = test_df["labels"].apply(lambda l: binarizer.transform([l])[0])

    # Split dataset
    train_df, val_df = train_test_split(train_df, random_state=42, test_size=val_size)

    if machine_translations:
        t_path = Path("./data/translations")
        translations = pd.concat(
            [pd.read_parquet(t_path / f"{lang}.parquet") for lang in languages if (t_path / f"{lang}.parquet").exists()]
        )
        translations = translations[translations["language"].isin(languages)]
        filtered = translations[translations["id"].str.contains("|".join(train_datasets))]
        filtered = filtered[~filtered["id"].isin(val_df["id"])].copy()
        filtered["labels"] = filtered["labels"].apply(lambda l: binarizer.transform([l])[0])
        if len(filtered):
            train_df = pd.concat([train_df, filtered]).sample(frac=1, random_state=42)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    return dataset, binarizer


if __name__ == "__main__":
    from hierarchy import create_full_hierarchy

    G = create_full_hierarchy()
    assert VALID_LABELS.issubset(G.nodes)

    languages = ["bg", "en", "hr", "mk", "pl", "sl", "ru"]

    dataset, _ = load_merge_encode(languages)
    for split in dataset:
        df = pd.DataFrame(dataset[split])
        assert len(df[df.duplicated("id")]) == 0
        print(df["text"])

    dataset, _ = load_merge_encode(languages, include_translations=True)
    train_ids = dataset["train"]["id"]
    val_ids = dataset["val"]["id"]
    
    assert len(train_ids) > len(set(train_ids))
    assert len(val_ids) == len(set(val_ids))
    assert len(set(train_ids).intersection(set(val_ids))) == 0

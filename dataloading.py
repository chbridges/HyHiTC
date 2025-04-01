import json
import pandas as pd
from itertools import chain
from networkx import DiGraph
from pathlib import Path
from PersuasionNLPTools.config import VALID_LABELS

LANGUAGE_SETS = {
    "shared_task": ["bg", "hr", "pl", "sl", "ru"],  # primary languages
    "slavic": ["bg", "hr", "mk", "pl", "sl", "ru"],  # add Macedonian (close to Bulgarian)
    "slavic_en": ["bg", "en", "hr", "mk", "pl", "sl", "ru"],  # add English
    "european_latin": ["bg", "de", "en", "es", "fr", "it", "hr", "mk", "pl", "sl", "ru",],  # add European languages using the Latin alphabet
    "parlamint": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Greek
    "european": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Georgian
    "all": ["ar", "bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Arabic
}

LABEL_MAP = {
    "Appeal to authority": "Appeal_to_Authority",
    "Appeal to fear/prejudice": "Appeal_to_Fear-Prejudice",
    "Bandwagon": "Appeal_to_Popularity",
    "Glittering generalities (Virtue)": "Appeal_to_Values",
    "Causal Oversimplification": "Causal_Oversimplification",
    "Thought-terminating cliché": "Conversation_Killer",
    "Doubt": "Doubt",
    "Exaggeration/Minimisation": "Exaggeration-Minimisation",
    "Black-and-white Fallacy/Dictatorship": "False_Dilemma-No_Choice",
    "Flag-waving": "Flag_Waving",
    "Reductio ad hitlerum": "Guilt_by_Association",
    "Loaded Language": "Loaded_Language",
    "Name calling/Labeling": "Name_Calling-Labeling",
    "Obfuscation, Intentional vagueness, Confusion": "Obfuscation-Vagueness-Confusion",
    "Smears": "Questioning_the_Reputation",
    "Presenting Irrelevant Data (Red Herring)": "Red_Herring",
    "Repetition": "Repetition",
    "Slogans": "Slogans",
    "Misrepresentation of Someone's Position (Straw Man)": "Straw_Man",
    "Whataboutism": "Whataboutism",
}

GLOBAL_DATA_PATH = Path("./data/")


def get_hierarchy() -> nx.DiGraph:
    G = DiGraph()
    G.add_edge("Persuation", "Logos")
    G.add_edge("Logos", "Repetition")
    G.add_edge("Logos", "Obfuscation-Vagueness-Confusion")
    G.add_edge("Logos", "Reasoning")
    G.add_edge("Logos", "Justification")
    G.add_edge('Justification', "Slogans")
    G.add_edge('Justification', "Appeal_to_Popularity")
    G.add_edge('Justification', "Appeal_to_Authority")
    G.add_edge('Justification', "Flag_Waving")
    G.add_edge('Justification', "Appeal_to_Fear-Prejudice")
    G.add_edge('Reasoning', "Simplification")
    G.add_edge('Simplification', "Causal_Oversimplification")
    G.add_edge('Simplification', "False_Dilemma-No_Choice")
    G.add_edge('Simplification', "Conversation_Killer")
    G.add_edge('Simplification', "Consequential_Oversimplification")
    G.add_edge('Simplification', "False_Equivalence")
    G.add_edge('Reasoning', "Distraction")
    G.add_edge('Distraction', "Straw_Man")
    G.add_edge('Distraction', "Red_Herring")
    G.add_edge('Distraction', "Whataboutism")
    G.add_edge("Persuation", "Ethos")
    G.add_edge('Ethos', "Appeal_to_Authority")
    G.add_edge('Ethos', "Appeal_to_Values")
    G.add_edge('Ethos', "Appeal_to_Popularity")
    G.add_edge('Ethos', "Ad Hominem")
    G.add_edge('Ad Hominem', "Doubt")
    G.add_edge('Ad Hominem', "Name_Calling-Labeling")
    G.add_edge('Ad Hominem', "Questioning_the_Reputation")
    G.add_edge('Ad Hominem', "Guilt_by_Association")
    G.add_edge('Ad Hominem', "Appeal_to_Hypocrisy")
    G.add_edge('Ad Hominem', "Whataboutism")
    G.add_edge("Persuation", "Pathos")
    G.add_edge('Pathos', "Exaggeration-Minimisation")
    G.add_edge('Pathos', "Loaded_Language")
    G.add_edge('Pathos', "Appeal_to_Fear-Prejudice")
    G.add_edge('Pathos', "Flag_Waving")
    G.add_edge("Persuation", "Appeal_to_Time")  # "Kairos", the 4th mode of persuation
    return G


def load_semeval_2021_task_6_subtask_1(languages: list[str]) -> pd.DataFrame:
    """
    Load the data from SemEval 2021 Task 6 Subtask 1.
    https://propaganda.math.unipd.it/semeval2021task6/

    All data are labeled and include languages: en

    :param languages: Set of languages to consider loading.
    :type languages: list[str]
    """
    if "en" not in languages:
        return pd.DataFrame(columns=["text", "labels"])

    data_path = GLOBAL_DATA_PATH / "semeval-2021-task-6" / "data"
    splits = [
        data_path / "training_set_task1.txt",
        data_path / "dev_set_task1.txt",
        data_path / "test_set_task1.txt",
    ]

    records = []
    for split in splits:
        with split.open("r", encoding="utf-8") as f:
            for record in json.load(f):
                records.append({"text": record["text"], "labels": [LABEL_MAP[label] for label in record["labels"]]})

    df = pd.DataFrame.from_records(records)
    return df[["text", "labels"]]


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
            records.extend([{"text": tup[0], "labels": tup[1]} for tup in zip(train_text, train_labels)])

            with dev_text_path.open("r", encoding="utf-8") as f:
                dev_text = [line.split("\t")[-1].strip() for line in f.readlines()]
            with dev_labels_path.open("r", encoding="utf-8") as f:
                dev_labels = [line.split("\t")[-1].strip().split(",") for line in f.readlines()]
            records.extend([{"text": tup[0], "labels": tup[1]} for tup in zip(dev_text, dev_labels)])

        if include_test:
            test_text_path = lang_path / "test-labels-subtask-3.template"
            test_labels_path = data_path / ".." / "baselines" / f"baseline-output-subtask3-test-{lang}.txt"
            with test_text_path.open("r", encoding="utf-8") as f:
                test_text = [line.split("\t")[-1].strip() for line in f.readlines()]
            with test_labels_path.open("r", encoding="utf-8") as f:
                test_labels = [line.split("\t")[-1].strip().split(",") for line in f.readlines()]
            records.extend([{"text": tup[0], "labels": tup[1]} for tup in zip(test_text, test_labels)])

    if records:
        df = pd.DataFrame.from_records(records)
        return df[["text", "labels"]]
    return pd.DataFrame(columns=["text", "labels"])


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
        with split.open("r", encoding="utf-8") as f:
            for record in json.load(f):
                records.append({"text": record["text"], "labels": [LABEL_MAP[label] for label in record["labels"]]})
    if records:
        df = pd.DataFrame.from_records(records)
        return df[["text", "labels"]]
    return pd.DataFrame(columns=["text", "labels"])


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

    def spans_to_paragraph(lang_path: Path, article_id: str) -> dict[str, str]:
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

        return [{"text": paragraphs[i], "labels": sorted(set(labels[i]))} for i in labels]

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
        return df[["text", "labels"]]
    return pd.DataFrame(columns=["text", "labels"])


def load_slavicnlp_train() -> pd.DataFrame:
    """
    Load the training data from the Slavic NLP 2025 Shared Task.
    https://bsnlp.cs.helsinki.fi/shared-task.html

    The training data are labeled and include languages: BG, PL, RU, SL
    """
    data_path = GLOBAL_DATA_PATH / "TRAIN_version_21_February_2025"
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
                    "language": lang,
                    "file": file,
                    "start": file_offsets[i][1],
                    "end": file_offsets[i][2],
                    "text": paragraphs[i],
                    "labels": labels[i],
                }
                for i in range(len(file_offsets))
            ]

            records.extend(file_records)

    return pd.DataFrame.from_records(records)


def load_merge_encode(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    languages = LANGUAGE_SETS[args.languages]

    train_funcs = [
        load_semeval_2021_task_6_subtask_1,  # en only
        load_semeval_2023_task_3_subtask_3,
        load_semeval_2024_task_4_subtask_1,
    ]
    if args.include_clef:
        train_funcs.append(load_clef_2024_task_3)

    train_df = pd.concat([func(languages) for func in train_funcs])
    test_df = load_slavicnlp_train()

    test_labels = sorted(set(chain.from_iterable(test_df["labels"])))

    # TODO: include hierarchy
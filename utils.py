import argparse


LANGUAGE_SETS = {
    "shared_task": ["bg", "hr", "pl", "sl", "ru"],  # primary languages
    "slavic": ["bg", "hr", "mk", "pl", "sl", "ru"],  # add Macedonian (close to Bulgarian)
    "slavic_en": ["bg", "en", "hr", "mk", "pl", "sl", "ru"],  # add English
    "european_latin": ["bg", "de", "en", "es", "fr", "it", "hr", "mk", "pl", "sl", "ru",],  # add European languages using the Latin alphabet
    "parlamint": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Greek
    "european": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Georgian
    "all": ["ar", "bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "sl", "ru"],  # add Arabic
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", "-l", choices=LANGUAGE_SETS.keys(), default="slavic")
    parser.add_argument("--include_clef", "-ic", action="store_true")
    parser.add_argument("--disable_hierarchy", "-dh", action="store_true")
    parser.add_argument("--val_size", "-v", type=float, default=0.2)
    parser.add_argument("--model", "-m", default="classla/xlm-r-parla")
    parser.add_argument("--gnn", "-g", choices=["GCN", "GAT", "HGCN"])
    return parser.parse_args()

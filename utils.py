import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", "-l", choices=LANGUAGE_SETS.keys(), default="slavic")
    parser.add_argument("--include_clef", "-ic", action="store_true")
    parser.add_argument("--disable_hierarchy", "-dh", action="store_true")
    return parser.parse_args()

import argparse

LANGUAGE_SETS = {
    "shared_task": ["bg", "hr", "pl", "ru", "sl"],  # primary languages
    "slavic": ["bg", "hr", "mk", "pl", "ru", "sl"],  # add Macedonian (close to Bulgarian)
    "slavic_en": ["bg", "en", "hr", "mk", "pl", "ru", "sl"],  # add English
    "european_latin": ["bg", "de", "en", "es", "fr", "it", "hr", "mk", "pl", "ru", "sl"],  # add all Latin alphabet
    "parlamint": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Greek
    "european": ["bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Georgian
    "all": ["ar", "bg", "de", "el", "en", "es", "fr", "it", "hr", "ka", "mk", "pl", "ru", "sl"],  # add Arabic
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--freeze", "-f", action="store_true")
    parser.add_argument("--gnn", "-g", choices=["gcn", "hgcn", "hie"], default="gcn")
    parser.add_argument("--hierarchy", "-hi", choices=["full", "taxonomy"])
    parser.add_argument("--hp_search", "-hp", action="store_true")
    parser.add_argument("--languages", "-l", choices=LANGUAGE_SETS.keys(), default="european_latin")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--model", "-m", default="classla/xlm-r-parla")
    parser.add_argument("--node_classification", "-nc", action="store_true")
    parser.add_argument("--node_dim", "-nd", type=int, default=32)
    parser.add_argument("--pooling", "-p", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--test_data", "-test", default="slavicnlp2025")
    parser.add_argument("--train_data", "-train", default="semeval2021,semeval2023,semeval2024")
    parser.add_argument("--translations", "-mt", action="store_true")
    parser.add_argument("--val_size", "-v", type=float, default=0.2)
    return parser.parse_args()


def add_hie_arguments(args: argparse.Namespace, feat_dims: int) -> argparse.Namespace:
    args = argparse.Namespace(**vars(args))  # shallow copy
    vars(args)["act"] = "relu"  # Shimizu et al.: ops in hyperbolic space are inherently non-linear
    vars(args)["c"] = None  # Charmi et al.: learnable curvature leads to performance gain
    vars(args)["device"] = "0"  # cuda:0
    vars(args)["dim"] = args.node_dim
    vars(args)["feat_dims"] = feat_dims  # XLM-R output size
    vars(args)["local_agg"] = 1  # Chami et al.: local tangent space aggregation outperforms agg at origin
    vars(args)["manifold"] = "Hyperboloid"  # Nickel and Kiela: more numerically stable than Poincaré
    vars(args)["num_layers"] = 1  # for comparison with HiAGM
    vars(args)["n_heads"] = 1  # number of attention heads for graph attention networks, must be a divisor dim
    vars(args)["pos_weight"] = (0,)  # mandatory argument of NCModel but not used
    vars(args)["use_att"] = 1  # Charmi et al.: attention leads to performance gain
    if args.gnn == "hie":
        vars(args)["hyp_ireg"] = "hir_tangent"  # used in Yang et al. paper
        vars(args)["ireg_lambda"] = 0.1  # Yang et al. do not report their used value
    return args

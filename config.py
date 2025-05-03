import argparse

from transformers import XLMRobertaConfig

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
    parser.add_argument("--freeze", "-f", action="store_true", help="Freeze 50% of XLM-R layers.")
    parser.add_argument("--gnn", "-g", choices=["GCN", "HGCN", "HIE"])
    parser.add_argument("--hierarchy", "-hi", choices=["full", "taxonomy"])
    parser.add_argument("--hp_search", "-hp", action="store_true")
    parser.add_argument("--language_model", "-lm", default="classla/xlm-r-parla")
    parser.add_argument("--languages", "-lang", choices=LANGUAGE_SETS.keys(), default="european_latin")
    parser.add_argument("--layers", "-l", type=int, default=2, help="Number of hidden graph convolutional layers.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--machine_translations", "-mt", action="store_true", help="Augment training data.")
    parser.add_argument("--node_classification", "-nc", action="store_true", help="Skip linear combination of nodes.")
    parser.add_argument("--node_dim", "-nd", type=int, default=32)
    parser.add_argument("--pooling", "-p", action="store_true", help="Append pooling layer to XLM-R.")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--test_data", "-test", default="slavicnlp2025")
    parser.add_argument("--train_data", "-train", default="semeval2021,semeval2023,semeval2024")
    parser.add_argument("--val_data", "-val", default="slavicnlp2025")
    parser.add_argument("--val_size", "-vs", type=float, default=0.2)
    args = parser.parse_args()

    if (args.gnn and not args.hierarchy) or (not args.gnn and args.hierarchy):
        raise ValueError("--gnn and --hierarchy must either both be set or both None.")

    return args


def add_hyp_default_args(args: argparse.Namespace, config: XLMRobertaConfig) -> argparse.Namespace:
    args = argparse.Namespace(**vars(args))  # shallow copy
    vars(args)["act"] = "relu"  # Shimizu et al.: ops in hyperbolic space are inherently non-linear
    vars(args)["bias"] = 1
    vars(args)["c"] = None  # Charmi et al.: learnable curvature leads to performance gain
    vars(args)["cuda"] = "0"
    vars(args)["device"] = "cuda:0"
    vars(args)["dim"] = args.node_dim
    vars(args)["dropout"] = config.hidden_dropout_prob
    vars(args)["feat_dim"] = args.node_dim  # XLM-R output size
    vars(args)["local_agg"] = 1  # Chami et al.: local tangent space aggregation outperforms agg at origin
    vars(args)["hyp_ireg"] = "hir_tangent" if args.gnn == "HIE" else "0"  # Yang et al.: comparable with hire_tangent
    vars(args)["ireg_lambda"] = 0.1  # Yang et al. do not report their used value
    vars(args)["manifold"] = "PoincareBall"  # Nickel and Kiela: Hyperboloid more numerically stable than Poincaré
    vars(args)["model"] = args.gnn
    vars(args)["n_classes"] = 1 if args.node_classification else args.node_dim  # binary multi-label classification
    vars(args)["num_layers"] = 2
    vars(args)["n_heads"] = 1  # number of attention heads for graph attention networks, must be a divisor dim
    vars(args)["pos_weight"] = None  # mandatory argument of NCModel but not used
    vars(args)["task"] = "nc" if args.node_classification else "rec"  # not sure what rec means
    vars(args)["use_att"] = 0  # Charmi et al.: attention leads to performance gain
    return args

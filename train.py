from dataloading import load_merge_encode
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    train_df, test_df = load_merge_encode(args)
from torch import nn
from transformers import AutoModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.lm = AutoModel.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

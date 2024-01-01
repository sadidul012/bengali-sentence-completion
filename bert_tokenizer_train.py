from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
import torch

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("csebuetnlp/BanglaParaphrase", split="train")
batch_size = 1000


def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["target"]


print("training tokenizer")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=30522)
new_tokenizer.save_pretrained("bert-tokenizer-bangla")
print("saved tokenizer")

import torch
from transformers import  GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from utils import test

ds = load_dataset("csebuetnlp/BanglaParaphrase")
print(ds)
start = "<|startoftext|>"
end = "<|endoftext|>"
max_length = 64
device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-bangla-sentence-completion", model_max_length=max_length)
model = GPT2LMHeadModel.from_pretrained('gpt2-bangla-sentence-completion', pad_token_id=tokenizer.eos_token_id)
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

test(tokenizer, start, end, ds["test"], device, model, max_length)

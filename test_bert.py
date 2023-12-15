import random

from transformers import Trainer, DataCollatorForSeq2Seq, TrainingArguments
from transformers import BertTokenizer, EncoderDecoderModel
import torch
from transformers import ProgressCallback

from datasets import load_dataset
from utils import test
import os


torch.manual_seed(42)
ds = load_dataset("csebuetnlp/BanglaParaphrase")
max_length = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("./encoder")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("./encoder", "./decoder")

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)


index = random.randint(0, 1000)
print("inference")
print("source", ds["test"][index:index + 1]["source"])
print("target", ds["test"][index:index + 1]["target"])
tokenized = tokenizer(ds["test"][index:index + 1]["source"], return_tensors="pt").to(device)
print("tokenized source:", tokenizer.decode(tokenized.input_ids[0]))
tokenized = tokenizer(ds["test"][index:index + 1]["target"], return_tensors="pt").to(device)
print("tokenized target:", tokenizer.decode(tokenized.input_ids[0]))

outputs = model.generate(input_ids=tokenized.input_ids, max_new_tokens=128)
print("output", tokenizer.decode(outputs[0]).split("[SEP]")[0])

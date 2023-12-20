import random
import time

from bert import model, tokenizer, preprocess_function, device
import torch
from datasets import load_dataset
import numpy as np
from metrics import score_generated_sentences


ds = load_dataset("csebuetnlp/BanglaParaphrase")


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

tokenized_test = ds["test"].select(range(500)).map(preprocess_function, batched=True)


def predict(examples):
    input_ids = torch.from_numpy(np.array(examples["input_ids"])).to(device)
    o = model.generate(input_ids=input_ids, max_new_tokens=128)
    os = []
    for i in o:
        os.append(" ".join(tokenizer.decode(i).split("[SEP]")[0].split(" ")[1:]))
    examples["output"] = os
    return examples


started = time.time()
test = tokenized_test.map(lambda batch: predict(batch), batched=True, batch_size=256)

# print(len(test["target"]), test["target"])
# print(len(test["output"]), test["output"])
scores = score_generated_sentences(test["target"], test["output"])
print(scores)
print("scoring time (seconds):", time.time() - started)

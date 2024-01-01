
from datasets import load_dataset
ds = load_dataset("csebuetnlp/BanglaParaphrase")

train_samples = 100000
valid_samples = 10000
tokenized_train = ds["train"].select(range(train_samples))
tokenized_test = ds["validation"].select(range(valid_samples))

with open('dataset/train.txt', 'w') as f:
    for i in range(train_samples):
        f.write(tokenized_train[i]["target"]+"\n")

print("saved tokenized_train")


with open('dataset/valid.txt', 'w') as f:
    for i in range(valid_samples):
        f.write(tokenized_test[i]["target"]+"\n")


print("saved tokenized_test")

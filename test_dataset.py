from random import randint, seed
import pandas as pd
from test_fasttext import test_fasttext_model
seed(42)

with open("dataset/valid.txt", "r") as f:
    lines = f.readlines()

print(len(lines))
rows = []
for line in lines:
    line = line.strip().split(" ")
    length = len(line)
    index = randint(1, 8 if length > 10 else length - 1)
    rows.append(dict(features=" ".join(line[:index]), label=line[index]))


test_data = pd.DataFrame(rows)
print(test_data.head().to_string())
test_fasttext_model(test_data)

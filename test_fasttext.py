import fasttext
import numpy as np
import tqdm

from fasttext_train import cosine_similarity
from sklearn.metrics import mean_squared_error

model = fasttext.load_model("fasttext_model/embedding.bin")

# print(len(model.words))
# print(model.get_nearest_neighbors("কিন্তু আপনি", k=5))


def test_fasttext_model(dataset):
    progress = tqdm.tqdm(range(len(dataset["features"])))
    similarity = []
    for sentence, label in zip(dataset["features"], dataset["label"]):
        # print(label)
        score = np.abs(np.max([cosine_similarity(label, x) for _, x in model.get_nearest_neighbors(sentence, k=20)]))
        similarity.append(0 if np.isnan(score) else score)
        progress.update()

    # similarity = np.where(similarity == np.nan, 0, np.array(similarity))
    # print(similarity)
    print(np.mean(similarity))


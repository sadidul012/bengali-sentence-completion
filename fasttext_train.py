import fasttext
import numpy as np

model = fasttext.train_supervised(input="dataset/train.txt")
model.save_model("fasttext_model/embedding.bin")
# print("saved model")
# model.test("dataset/valid.txt")


def cosine_similarity(word1, word2):
    return np.dot(model[word1], model[word2]) / (np.linalg.norm(model[word1]) * np.linalg.norm(model[word2]))


# print(len(model.words))
# print(model['আপনি'])
# print(model.get_nearest_neighbors("কিন্তু আপনি", k=5))
# print(cosine_similarity("আপনি", "তিনি"))

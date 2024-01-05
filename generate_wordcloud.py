from generate_wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk


with open("dataset/valid.txt", "r") as f:
    lines = f.readlines()


words = []
for line in lines:
    words.extend(line.split(" "))

print("10 words", words[:10])
freq = Counter(words)
print(dict(freq))
frequency_dist = nltk.FreqDist(words)
wordcloud = WordCloud(
    font_path="/mnt/Cache/downloads/FN-Mahfuj-Rumaysa/FN Mahfj Rumaysa/Unicode/FN-Mahfuj-Rumaysa.ttf"
).generate_from_frequencies(dict(freq))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

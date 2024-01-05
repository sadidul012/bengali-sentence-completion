from collections import Counter


with open("dataset/valid.txt", "r") as f:
    lines = f.readlines()

words = []
for line in lines:
    words.extend(line.split(" "))

print("10 words", words[:10])
freq = Counter(words)
most_used = [(t[0], t[1]) for t in sorted(freq.items(), key=lambda t: -t[1])][0:10]
print(most_used)
print(len(words))
print("unique words", len(set(words)))

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Generate random data for the histogram
#
# # Plotting a basic histogram
# plt.hist(words, bins=30, color='skyblue', edgecolor='black')
#
# # Adding labels and title
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.title('Words Histogram')
# plt.xticks(color="white")
#
# # Display the plot
# # plt.show()
# plt.savefig('images/basic_histogram.png')

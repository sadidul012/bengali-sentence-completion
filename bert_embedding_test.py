import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from string import ascii_letters, digits


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    print(token_embeddings.shape)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # return sum_embeddings / sum_mask
    return token_embeddings


tokenizer = AutoTokenizer.from_pretrained("./bert-tokenizer-bangla")
model = AutoModelForMaskedLM.from_pretrained("./bert-base-uncased-bangla")
encoded_input = tokenizer("বৃক্ষ পত্রহীন নয়", return_tensors='pt')
model_output = model(**encoded_input)
embedding = mean_pooling(model_output, encoded_input['attention_mask'])
closest = embedding.argmax(-1).squeeze(0).numpy()
# print(closest)
# # closest = torch.topk(embedding.flatten(), 30).indices
c = 0
k = 5
next_words = []
print(embedding.shape)
for i in list(closest):
    print(i)
    text = tokenizer.convert_ids_to_tokens(i.item())

    # if set(text).difference(ascii_letters + digits):
    #     continue
    next_words.append(text)
    c += 1

    if c >= k:
        break

print(next_words)

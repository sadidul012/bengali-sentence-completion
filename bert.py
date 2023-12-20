from transformers import BertTokenizer, EncoderDecoderModel
import torch


max_length = 128
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("./encoder")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained("./encoder", "./decoder")

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)


def preprocess_function(examples):
    inputs = [doc for doc in examples['source']]
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")

    targets = [doc for doc in examples['target']]
    labels = tokenizer(targets, truncation=True, max_length=max_length, padding="max_length")
    # labels = tokenizer(text_target=examples['title'], max_length=128)
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

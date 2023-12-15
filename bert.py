import random

import numpy as np
from transformers import Trainer, DataCollatorForSeq2Seq, TrainingArguments, TrainerCallback
from transformers import BertTokenizer, EncoderDecoderModel
import torch
from transformers import ProgressCallback, PrinterCallback

from datasets import load_dataset, load_metric, list_metrics


# print(list_metrics())
torch.manual_seed(42)
ds = load_dataset("csebuetnlp/BanglaParaphrase")
max_length = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("./encoder")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("./encoder", "./decoder")

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


# tokenized_train = ds["train"].map(preprocess_function, batched=True)
# tokenized_test = ds["validation"].map(preprocess_function, batched=True)
tokenized_train = ds["train"].select(range(100000)).map(preprocess_function, batched=True)
tokenized_test = ds["validation"].select(range(1000)).map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
print("tokenized train:", tokenized_train[:1])


class ResultPrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        index = random.randint(0, 1000)
        t1 = tokenizer(ds["test"][index:index+1]["source"], return_tensors="pt").to(device)
        t2 = tokenizer(ds["test"][index:index+1]["target"], return_tensors="pt").to(device)

        o = model.generate(input_ids=t1.input_ids, max_new_tokens=128)
        print("tokenized source:", tokenizer.decode(t1.input_ids[0]))
        print("tokenized target:", tokenizer.decode(t2.input_ids[0]))
        print("output", tokenizer.decode(o[0]).split("[SEP]")[0])


# metric = load_metric("squad_v2")
#
#
# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     # predictions = np.argmax(logits, axis=-1)
#     print(logits)
#     # return metric.compute(predictions=predictions, references=labels)
#     return [0]


training_args = TrainingArguments(
    output_dir='test_trainer',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    # gradient_accumulation_steps=20,
    num_train_epochs=10,
    fp16=False,
    logging_dir='./logs',
    report_to=['tensorboard'],
    disable_tqdm=False,
    save_total_limit=0,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    callbacks=[ResultPrinterCallback],
    # compute_metrics=compute_metrics
)


try:
    print("training...")
    trainer.train()
except KeyboardInterrupt:
    pass

model.encoder.save_pretrained("./encoder")
tokenizer.save_pretrained("./encoder")
model.decoder.save_pretrained("./decoder")

print("inference")
tokenized1 = tokenizer(ds["test"][:1]["source"], return_tensors="pt").to(device)
print("tokenized source:", tokenizer.decode(tokenized1.input_ids[0]))
tokenized2 = tokenizer(ds["test"][:1]["target"], return_tensors="pt").to(device)
print("tokenized target:", tokenizer.decode(tokenized2.input_ids[0]))
outputs = model.generate(input_ids=tokenized1.input_ids)
print("output", tokenizer.decode(outputs[0]))


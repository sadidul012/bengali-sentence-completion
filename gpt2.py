import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForSeq2Seq, \
    TrainingArguments, Trainer

from datasets import load_dataset
from utils import test

torch.manual_seed(42)
ds = load_dataset("csebuetnlp/BanglaParaphrase")
print(ds)
start = "<|startoftext|>"
end = "<|endoftext|>"
max_length = 128
device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=max_length)
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
model.to(device)
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    inputs = [doc + end for doc in examples['source']]
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")

    targets = [start + doc for doc in examples['target']]
    labels = tokenizer(targets, truncation=True, max_length=max_length, padding="max_length")
    # labels = tokenizer(text_target=examples['title'], max_length=128)
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


tokenized_train = ds["train"].map(preprocess_function, batched=True)
tokenized_test = ds["validation"].map(preprocess_function, batched=True)
# tokenized_train = ds["train"].select(range(100)).map(preprocess_function, batched=True)
# tokenized_test = ds["validation"].select(range(100)).map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# rouge = evaluate.load("rouge")
print("tokenized train:", tokenized_train[:1])

# result = model(**test_tokenized)

# test(tokenizer, start, end, ds["test"], device, model, max_length)

training_args = TrainingArguments(
    output_dir='test_trainer',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=20,
    num_train_epochs=1,
    fp16=False,
    logging_dir='./logs',
    report_to=['tensorboard']
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator
    # compute_metrics=compute_metrics, #commented out because the compute_metrics is unchanged from the original text classification code
)

try:
    print("training...")
    trainer.train()
except KeyboardInterrupt:
    pass

# test(tokenizer, start, end, ds["test"], device, model, max_length)

model.save_pretrained("gpt2-bangla-sentence-completion", from_pt=True)
tokenizer.save_pretrained("gpt2-bangla-sentence-completion", from_pt=True)

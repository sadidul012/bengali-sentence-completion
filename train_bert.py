import random
from transformers import Trainer, DataCollatorForSeq2Seq, TrainingArguments, TrainerCallback
from metrics import score_generated_sentences
from datasets import load_dataset
from bert import tokenizer, model, device, preprocess_function


ds = load_dataset("csebuetnlp/BanglaParaphrase")


# tokenized_train = ds["train"].map(preprocess_function, batched=True)
# tokenized_test = ds["validation"].map(preprocess_function, batched=True)
tokenized_train = ds["train"].select(range(100)).map(preprocess_function, batched=True)
tokenized_test = ds["validation"].select(range(10)).map(preprocess_function, batched=True)
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    outs = []
    for i in labels:
        outs.append(" ".join(tokenizer.decode(i).split("[SEP]")[0].split(" ")[1:]).strip())

    return score_generated_sentences(tokenized_test["target"], outs)


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
    compute_metrics=compute_metrics
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


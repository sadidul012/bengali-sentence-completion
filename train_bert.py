import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
from transformers import Trainer, DataCollatorForSeq2Seq, TrainingArguments, TrainerCallback
from metrics import score_generated_sentences
from datasets import load_dataset
from bert import tokenizer, model, device, preprocess_function, max_length
ds = load_dataset("csebuetnlp/BanglaParaphrase")


# tokenized_train = ds["train"].map(preprocess_function, batched=True)
# tokenized_test = ds["validation"].map(preprocess_function, batched=True)
tokenized_train = ds["train"].select(range(100000)).map(preprocess_function, batched=True)
tokenized_test = ds["validation"].select(range(10000)).map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
print("tokenized train:", tokenized_train[:1])
print(tokenized_test.shape[0])


writer = SummaryWriter(log_dir="./logs")


class ResultPrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        index = random.randint(0, 1000)
        t1 = tokenizer(ds["test"][index:index+1]["source"], return_tensors="pt").to(device)
        t2 = tokenizer(ds["test"][index:index+1]["target"], return_tensors="pt").to(device)

        o = model.generate(input_ids=t1.input_ids, max_new_tokens=128)
        print("tokenized source:", tokenizer.decode(t1.input_ids[0]))
        print("tokenized target:", tokenizer.decode(t2.input_ids[0]))
        print("output", tokenizer.decode(o[0]).split("[SEP]")[0])

    def on_epoch_end(self, args, state, control, **kwargs):
        a_scores = []
        batch = 4
        progress = tqdm.tqdm(np.arange(start=0, stop=tokenized_test.shape[0], step=batch), desc="Evaluating", leave=True, position=0)
        for index in np.arange(start=0, stop=tokenized_test.shape[0], step=batch):
            t1 = tokenizer(ds["test"][index:index + batch]["source"], truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt").to(device)
            outs = model.generate(input_ids=t1.input_ids, max_new_tokens=128)
            generated_sentences = []
            for o in outs:
                generated = " ".join(tokenizer.decode(o).split("[SEP]")[0].split(" ")[1:]).strip()
                generated_sentences.append(generated)

            scores = score_generated_sentences(generated_sentences, ds["test"][index:index + batch]["target"])
            a_scores.append(scores)
            progress.update()

        scores = np.array(a_scores).mean(axis=0)
        writer.add_scalar('eval/bleu', scores[0], int(state.epoch))
        writer.add_scalar('eval/ter', scores[1], int(state.epoch))
        writer.add_scalar('eval/rouge', scores[2], int(state.epoch))


def compute_metrics(eval_pred):
    references = eval_pred.label_ids
    generated_texts = eval_pred.predictions
    predictions = generated_texts[0].argmax(-1)
    bleu_scores = []
    ter_scores = []
    rouge_scores = []
    for reference, generated_text in zip(references, predictions):
        reference = " ".join(tokenizer.decode(reference).split("[SEP]")[0].split(" ")[1:]).strip()
        generated = " ".join(tokenizer.decode(generated_text).split("[SEP]")[0].split(" ")[1:]).strip()
        bleu, ter, rouge = score_generated_sentences([reference], [generated])
        bleu_scores.append(bleu)
        ter_scores.append(ter)
        rouge_scores.append(rouge)

    scores = {
        'bleu': sum(bleu_scores) / len(bleu_scores),
        'ter': sum(ter_scores) / len(ter_scores),
        'rouge': sum(rouge_scores) / len(rouge_scores)
    }
    return scores


training_args = TrainingArguments(
    output_dir='test_trainer',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=2,
    # eval_accumulation_steps=1,
    eval_steps=10,
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

writer.close()

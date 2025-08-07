from huggingface_hub import login
# login(token = "")

import numpy as np
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":

    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )

    dataset = load_dataset("yelp_review_full")
    small_train = dataset["train"].shuffle(seed=42).select(range(10000))
    small_eval = dataset["test"].shuffle(seed=42).select(range(5000))

    checkpoint = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    train_tokenized = small_train.map(tokenize, batched=True)
    eval_tokenized = small_eval.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir='./test_yelp_classification/',
        logging_dir="./test_yelp_classification/logs",
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
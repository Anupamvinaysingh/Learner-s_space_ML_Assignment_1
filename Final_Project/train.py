from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from wikitext_dataset import load_and_tokenize_dataset
from gpt2_model import get_model

def main():
    tokenized_datasets, tokenizer = load_and_tokenize_dataset()
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000)) 

    model = get_model(tokenizer=tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    main()

from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_dataset(model_name="gpt2", max_length=128):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized, tokenizer

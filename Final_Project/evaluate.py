import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

def top_k_accuracy(model, tokenizer, dataset, k=5, num_samples=100):
    model.eval()
    correct = 0

    for i in range(num_samples):
        input_ids = torch.tensor([dataset["input_ids"][i]]).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -2, :]

        top_k = torch.topk(logits, k).indices
        true = input_ids[0, -1]
        if true in top_k:
            correct += 1

    return correct / num_samples

if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    from wikitext_dataset import load_and_tokenize_dataset
    dataset, _ = load_and_tokenize_dataset()

    acc = top_k_accuracy(model, tokenizer, dataset["validation"])
    print(f"Top-5 Accuracy: {acc:.2f}")

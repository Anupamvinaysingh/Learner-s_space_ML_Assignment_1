from transformers import GPT2LMHeadModel

def get_model(model_name="gpt2", tokenizer=None):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if tokenizer:
        model.resize_token_embeddings(len(tokenizer))
    return model

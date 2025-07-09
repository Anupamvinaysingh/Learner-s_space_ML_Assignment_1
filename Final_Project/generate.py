from transformers import pipeline, GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Artificial intelligence is"
output = generator(prompt, max_length=50, num_return_sequences=1)

print("Generated Text:")
print(output[0]['generated_text'])

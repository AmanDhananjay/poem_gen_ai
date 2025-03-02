from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_model')

# Function to generate poetry
def generate_poetry(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text from the model
    generated_output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return generated_text

# Example of generating poetry
prompt = "The stars in the night sky"
generated_poem = generate_poetry(prompt)
print(generated_poem)

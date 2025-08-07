import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_one_liners(file_path="data/one_liners.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True)
    attention_mask = (inputs != tokenizer.eos_token_id).long()

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length + inputs.shape[-1],  # add prompt length to max_length
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Remove prompt tokens from generated output
    generated_tokens = outputs[0][inputs.shape[-1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    # Load once
    one_liners = load_one_liners()

    tokenizer = GPT2Tokenizer.from_pretrained("./results/fine_tuned_model")
    model = GPT2LMHeadModel.from_pretrained("./results/fine_tuned_model")
    model.eval()

    print("Type your message! Type 'exit' or 'quit' to stop.")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            break

        # If user asks for a roast, reply with a one-liner
        if any(phrase in prompt.lower() for phrase in ["roast me", "give me a roast", "roast"]):
            response = random.choice(one_liners)
        else:
            response = generate_response(prompt, model, tokenizer)

        print("Bot:", response)
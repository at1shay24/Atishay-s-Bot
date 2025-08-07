import torch
import random
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_one_liners(file_path="data/one_liners.txt"):
    if not os.path.exists(file_path):
        print(f"[WARNING] One-liners file not found at: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def generate_response(prompt, model, tokenizer, max_length=300):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = (inputs != tokenizer.eos_token_id).long()

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length + inputs.shape[-1],
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("./results/fine_tuned_model")
    model = GPT2LMHeadModel.from_pretrained("./results/fine_tuned_model")
    model.eval()

    print("Loading one-liner roasts...")
    one_liners = load_one_liners()

    print("\n🤖 Type your message! Type 'exit' or 'quit' to stop.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("Bot: Later, genius.")
            break

        if any(phrase in prompt.lower() for phrase in ["roast me", "give me a roast", "roast"]):
            if one_liners:
                response = random.choice(one_liners)
            else:
                response = "I'd roast you, but I don't want to waste good sarcasm on missing files."
        else:
            response = generate_response(prompt, model, tokenizer)

        print("Bot:", response)
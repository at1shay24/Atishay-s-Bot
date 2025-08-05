import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(prompt, max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained("./results/fine_tuned_model")
    model = GPT2LMHeadModel.from_pretrained("./results/fine_tuned_model")

    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True)
    attention_mask = (inputs != tokenizer.eos_token_id).long()

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = generate_response(prompt)
        print("Bot:", response)

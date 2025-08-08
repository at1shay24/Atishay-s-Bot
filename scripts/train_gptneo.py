import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def generate_response(prompt, model, tokenizer, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt part from output
    return generated_text[len(prompt):].strip()

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("./results_gptneo/fine_tuned_model")
    model = GPTNeoForCausalLM.from_pretrained("./results_gptneo/fine_tuned_model")
    model.eval()

    print("🤖 GPT-Neo Sarcasm Bot ready! Type 'exit' or 'quit' to stop.")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("Bot: Later, genius.")
            break
        response = generate_response(prompt, model, tokenizer)
        print("Bot:", response)
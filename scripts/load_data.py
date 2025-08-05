# scripts/load_data.py

from transformers import GPT2Tokenizer

def load_sarcasm_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]  # remove empty lines
    return lines

def tokenize_lines(lines):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    return tokenizer(lines, padding=True, truncation=True, return_tensors="pt")

# Quick test
if __name__ == "__main__":
    data = load_sarcasm_data("data/sarcasm.txt")
    print(f"Loaded {len(data)} sarcastic lines.")
    
    tokens = tokenize_lines(data)
    print("Tokenized shape:", tokens["input_ids"].shape)

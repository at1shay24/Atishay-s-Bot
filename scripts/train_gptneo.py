# scripts/train_gptneo.py

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ---------------- Load and prepare dataset ----------------
def load_sarcasm_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} lines from dataset.")
    return lines

def tokenize_function(examples, tokenizer, max_length=64):  # smaller max_length for CPU
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

if __name__ == "__main__":
    device = torch.device("cpu")  # Force CPU
    print(f"Using device: {device}")

    # Load dataset
    data_lines = load_sarcasm_dataset("data/convo_sarcasm.txt")
    dataset = Dataset.from_dict({"text": data_lines})

    # Load tokenizer & model (smaller model recommended for CPU)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-Neo uses eos as pad

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    print("Dataset tokenization complete.")

    # ---------------- Training arguments ----------------
    training_args = TrainingArguments(
        output_dir="./results_gptneo",
        overwrite_output_dir=True,
        num_train_epochs=2,               # adjust as needed
        per_device_train_batch_size=1,    # keep small for CPU
        gradient_accumulation_steps=4,    # simulate bigger batch
        save_steps=500,
        save_total_limit=1,
        logging_steps=50,
        logging_dir="./logs",
        fp16=False,                       # CPU safe
        dataloader_num_workers=0,
        dataloader_pin_memory=False,      # avoids MPS pin_memory warning
        report_to=None,
    )

    # ---------------- Trainer ----------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    print("🚀 Starting GPT-Neo fine-tuning on sarcasm dataset...")
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./results_gptneo/fine_tuned_model")
    tokenizer.save_pretrained("./results_gptneo/fine_tuned_model")
    print("✅ Fine-tuning complete! Model saved.")
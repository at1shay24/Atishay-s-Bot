import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

def load_sarcasm_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=50,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

if __name__ == "__main__":
    # Limit CPU usage to avoid overheating
    torch.set_num_threads(2)

    # Load data
    data = load_sarcasm_data("data/sarcasm.txt")
    dataset = Dataset.from_dict({"text": data})

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Safe TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=2,                      # Reduced to be safer
        per_device_train_batch_size=1,           # Low batch size = less RAM & CPU usage
        gradient_accumulation_steps=2,           # Gradually increase "effective" batch size
        save_steps=100,
        save_total_limit=1,
        prediction_loss_only=True,
        logging_steps=10,
        logging_dir='./logs',
        fp16=False,                              # You’re on CPU/MPS, keep it off
        dataloader_num_workers=0,                # Keeps memory usage low
        disable_tqdm=False,                      # Show progress
        report_to=None,                          # Prevent any logging errors
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save model
    model.save_pretrained("./results/fine_tuned_model")
    tokenizer.save_pretrained("./results/fine_tuned_model")

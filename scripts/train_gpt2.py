import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

def load_sarcasm_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def load_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    convos = raw.strip().split("\n\n")
    samples = []
    for convo in convos:
        if convo.strip():
            lines = convo.strip().split("\n")
            full_text = "\n".join(lines)
            samples.append(full_text)
    return samples

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

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
    torch.set_num_threads(2)

    one_liners = load_sarcasm_data("data/sarcasm.txt")
    convos = load_conversations("data/convo_sarcasm.txt")
    data = one_liners + convos

    dataset = Dataset.from_dict({"text": data})

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=100,
        save_total_limit=1,
        prediction_loss_only=True,
        logging_steps=10,
        logging_dir='./logs',
        fp16=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./results/fine_tuned_model")
    tokenizer.save_pretrained("./results/fine_tuned_model")
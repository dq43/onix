import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_path = "/workspace/qwen-base"
dataset_path = "/workspace/onix_training_data.jsonl"
output_dir = "/workspace/onix-adapter"

# 4-bit quantization (the "Q" in QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model in 4-bit...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

model = prepare_model_for_kbit_training(model)

# LoRA config — trains ~1.6% of total params
lora_config = LoraConfig(
    r=64,                              # rank of adapter matrices
    lora_alpha=32,                     # scaling factor
    target_modules=[                   # which layers get adapters
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and apply chat template formatting
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print(f"Dataset size: {len(dataset)} examples")
print(f"Sample formatted text:\n{dataset[0]['text'][:500]}...")

# Training config
training_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,    # effective batch size = 16
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    max_seq_length=4096,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",          # 8-bit optimizer to save VRAM
    gradient_checkpointing=True,
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_config,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving final adapter...")
trainer.save_model(f"{output_dir}-final")
print(f"Training complete. Adapter saved to {output_dir}-final")

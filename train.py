"""
Fine-tune a small LM (Qwen2.5-0.5B-Instruct) into a Python Coding Assistant
using LoRA (parameter-efficient fine-tuning).

Runs on a free Google Colab T4 GPU or any machine with ~6GB+ VRAM.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ----------------------------
# 1. Config
# ----------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "./qwen2.5-0.5b-python-assistant"
MAX_SEQ_LEN = 512
NUM_SAMPLES = 3000  # keep it small for a quick, portfolio-friendly run

# ----------------------------
# 2. Load tokenizer & base model
# ----------------------------
# Device detection: CUDA (Colab/Linux GPU) -> MPS (Apple Silicon) -> CPU
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32  # MPS is unstable with bf16/fp16 for training
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
model.to(device)

# ----------------------------
# 3. LoRA config (only trains small adapter layers, not the whole model)
# ----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ----------------------------
# 4. Load & format dataset
# ----------------------------
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))


def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}


dataset = dataset.map(format_example, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# ----------------------------
# 5. Training config
# ----------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2 if device != "cuda" else 4,
    per_device_eval_batch_size=2 if device != "cuda" else 4,
    gradient_accumulation_steps=8 if device != "cuda" else 4,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    bf16=(device == "cuda"),
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

# ----------------------------
# 6. Train
# ----------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. LoRA adapter saved to {OUTPUT_DIR}")

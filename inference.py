"""
Load the base model + LoRA adapter and generate a response.
Run: python inference.py "Write a Python function to reverse a linked list"
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./qwen2.5-0.5b-python-assistant"

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(device)
model.eval()


def generate(instruction: str, max_new_tokens: int = 256) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Response:\n")[-1].strip()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Write a Python function to check if a string is a palindrome"
    print(f"\nPrompt: {query}\n")
    print("Response:")
    print(generate(query))

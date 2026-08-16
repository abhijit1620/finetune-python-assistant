# Fine-Tuned Python Coding Assistant (Qwen2.5-0.5B + LoRA)

A small language model ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) fine-tuned with **LoRA** (parameter-efficient fine-tuning) to answer Python coding questions and generate code from natural-language instructions.

Built as a portfolio project demonstrating: LLM fine-tuning, PEFT/LoRA, the Hugging Face ecosystem (`transformers`, `trl`, `peft`, `datasets`), and end-to-end deployment.

**🚀 [Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/abhijit1620/python-coding-assistant)**
**🤗 [Model weights on Hugging Face Hub](https://huggingface.co/abhijit1620/qwen2.5-0.5b-python-assistant)**

---

## Why this project

- **Base model**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — small enough to fine-tune on a laptop GPU, no expensive cloud compute required
- **Method**: LoRA (rank=16) — trains only ~2.16M of the model's 496M parameters (0.44%), making it fast and memory-efficient
- **Dataset**: [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) — 18k Alpaca-style Python instructions, 3,000-example subset used for training
- **Hardware**: Trained locally on a MacBook Air M2 using PyTorch's MPS (Apple Silicon GPU) backend — no CUDA needed
- **Result**: A specialized coding assistant, trained end-to-end in about 70 minutes, deployed with a live web demo

---

## Results

Trained for 2 epochs on 3,000 examples. Training loss dropped from **0.96 → 0.78**, and mean token accuracy rose to **~80.6%** by the end of training.

**Example — "Write a Python function to check if a number is prime":**

```python
def is_prime(n):
    if n % 2 == 0:
        return False
    i = 3
    while (i * i) <= n:
        if n % i == 0:
            return False
        i += 2
    return True
```

Correctly uses the √n optimization instead of checking every divisor up to n — a non-trivial detail the model picked up from the instruction-tuning data rather than just producing generic boilerplate.

*(This section can be expanded with more before/after comparisons by running `compare_base_vs_finetuned.py`, which prints the base model's output next to the fine-tuned model's output for the same prompts.)*

---

## Project structure

```
finetune-small-lm/
├── train.py                        # LoRA fine-tuning script
├── inference.py                     # load adapter locally + generate responses
├── app.py                            # local Gradio demo UI
├── space_app.py                       # Gradio app used by the deployed HF Space
├── upload_to_hub.py                    # pushes the trained adapter to Hugging Face Hub
├── compare_base_vs_finetuned.py         # side-by-side base vs fine-tuned outputs
├── requirements.txt                      # dependencies for local training/inference
├── space_requirements.txt                 # lighter dependencies for the HF Space
└── README.md
```

---

## How it works

1. **`train.py`** loads the base model, attaches a LoRA adapter (small trainable layers instead of retraining the full model), loads and formats the Python instructions dataset, and fine-tunes for 2 epochs using Apple Silicon's MPS backend.
2. **`upload_to_hub.py`** pushes the resulting adapter weights to the Hugging Face Hub.
3. **`space_app.py`** loads the base model + the uploaded adapter from the Hub and serves it through a Gradio interface, deployed as a Hugging Face Space for a public, shareable demo.

---

## Run it yourself

```bash
# Clone and set up
git clone https://github.com/abhijit1620/finetune-python-assistant.git
cd finetune-python-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (uses Apple Silicon MPS backend automatically if available)
python3 train.py

# Test locally
python3 inference.py "Write a function to reverse a string"

# Or launch the local web demo
python3 app.py
```


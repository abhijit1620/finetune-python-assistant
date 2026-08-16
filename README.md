# Fine-Tuned Python Coding Assistant (Qwen2.5-0.5B + LoRA)

A small language model (Qwen2.5-0.5B-Instruct) fine-tuned with **LoRA**
(parameter-efficient fine-tuning) to answer Python coding questions and
generate code from natural-language instructions.

Built as a portfolio project demonstrating: LLM fine-tuning, PEFT/LoRA,
Hugging Face ecosystem (transformers, trl, peft, datasets), and deployment.

---

## Why this project

- **Base model**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — small enough to fine-tune on a free Colab GPU
- **Method**: LoRA (rank=16) — trains <1% of parameters, fast and memory-efficient
- **Dataset**: [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) (Alpaca-style Python instructions)
- **Result**: A specialized coding assistant, trainable end-to-end in under an hour

---

## 1. Setup — VS Code on Mac (Apple Silicon M2)

This project runs locally on M2 using PyTorch's **MPS backend** (Apple's
GPU acceleration) — no CUDA/bitsandbytes needed. The 0.5B model + LoRA is
small enough to train directly on the M2 GPU.

### Step 1 — Install prerequisites (one-time)

Open the **Terminal** app (or VS Code's integrated terminal: `` Ctrl+` ``):

```bash
# Check Python version (need 3.10+)
python3 --version

# If you don't have Homebrew yet:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew (skip if already 3.10+)
brew install python@3.11
```

### Step 2 — Open the project in VS Code

```bash
cd ~/Desktop          # or wherever you want the project
mkdir finetune-small-lm && cd finetune-small-lm
code .                 # opens this folder in VS Code
```

Copy the 5 files (`train.py`, `inference.py`, `app.py`, `requirements.txt`,
`README.md`) I gave you into this folder.

In VS Code: install the **Python extension** (Microsoft) from the
Extensions panel (`Cmd+Shift+X`) if you don't have it — gives you syntax
highlighting, run buttons, and debugging.

### Step 3 — Create a virtual environment

In VS Code's terminal (`` Ctrl+` ``):

```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should now show `(venv)`. In VS Code, also select this
interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → choose the one
inside `venv/bin/python`.

### Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 — Verify MPS (Apple GPU) is available

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Should print `MPS available: True`. If `False`, training will still work
on CPU, just slower.

### Step 6 — Run training

```bash
python3 train.py
```

On an M2, expect roughly 45-90 minutes for the default settings (2 epochs,
3000 samples). If it feels slow, reduce `NUM_SAMPLES` in `train.py` to
1000-1500 for a quicker first run — you can always retrain with more data
once the pipeline works.

---

## 2. What `train.py` does (runs after Step 6 above)

1. Loads the base model + tokenizer
2. Wraps the model with a LoRA adapter (only ~2-4M trainable params instead of 500M)
3. Loads and formats the Python instructions dataset into a prompt template
4. Trains for 2 epochs, evaluates periodically, saves the adapter to `./qwen2.5-0.5b-python-assistant`

Training takes roughly 20-40 minutes on a Colab T4 GPU with the default settings.

---

## 3. Test the model

```bash
python inference.py "Write a Python function to reverse a linked list"
```

Or launch the interactive demo:

```bash
python app.py
```

This opens a local Gradio web UI — great for a demo video/GIF in your README.

---

## 4. (Optional but recommended) Push to Hugging Face Hub

Makes the project easy for recruiters to try live.

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF access token
```

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-username/qwen2.5-0.5b-python-assistant", exist_ok=True)
api.upload_folder(
    folder_path="./qwen2.5-0.5b-python-assistant",
    repo_id="your-username/qwen2.5-0.5b-python-assistant",
)
```

---

## 5. Project structure

```
finetune-small-lm/
├── requirements.txt      # dependencies
├── train.py               # LoRA fine-tuning script
├── inference.py            # load adapter + generate responses
├── app.py                  # Gradio demo UI
└── README.md
```

---

## 6. Results (fill this in after training)

| Prompt | Base Model Output | Fine-Tuned Output |
|---|---|---|
| "Write a function to check if a number is prime" | *(generic/verbose)* | *(concise, correct Python)* |

Add 3-5 before/after examples here — this is the single most convincing
part of the repo for anyone reviewing it.

---

## Next steps / stretch goals

- Quantize with `bitsandbytes` (4-bit) for even faster inference
- Add an evaluation script (pass@1 on a small held-out coding benchmark)
- Deploy the Gradio app on Hugging Face Spaces (free) for a live demo link
# finetune-python-assistant

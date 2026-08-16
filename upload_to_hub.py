"""
Uploads the trained LoRA adapter to your Hugging Face Hub account.
Run this AFTER training, and after `huggingface-cli login`.

Usage: python3 upload_to_hub.py <your-hf-username>
"""

import sys
from huggingface_hub import HfApi

if len(sys.argv) < 2:
    print("Usage: python3 upload_to_hub.py <your-hf-username>")
    sys.exit(1)

username = sys.argv[1]
repo_id = f"{username}/qwen2.5-0.5b-python-assistant"

api = HfApi()
api.create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path="./qwen2.5-0.5b-python-assistant",
    repo_id=repo_id,
)

print(f"\nUploaded! View it at: https://huggingface.co/{repo_id}")
print(f"Use this repo_id in your Space's app.py: {repo_id}")

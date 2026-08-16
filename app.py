import gradio as gr
from inference import generate

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Ask a Python coding question", lines=3,
                       placeholder="e.g. Write a function to find duplicates in a list"),
    outputs=gr.Textbox(label="Response", lines=10),
    title="Fine-Tuned Python Coding Assistant (Qwen2.5-0.5B + LoRA)",
    description="A small language model fine-tuned with LoRA on Python instruction data.",
)

if __name__ == "__main__":
    demo.launch()

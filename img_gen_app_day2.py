import os
import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

# -------------------------------------------------
# Load API Token (from Secrets)
# -------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found. Add it in Secrets.")

client = InferenceClient(
    token=HF_TOKEN,
    model="runwayml/stable-diffusion-v1-5"
)

# -------------------------------------------------
# Image Generation Function
# -------------------------------------------------
def generate_image(prompt, negative_prompt):
    if prompt.strip() == "":
        return None

    image = client.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt
    )

    return image

# -------------------------------------------------
# Gradio UI
# -------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown("## 🎨 Stable Diffusion Image Generator")

    prompt = gr.Textbox(
        label="Prompt",
        placeholder="A futuristic sports car in an empty basement, cinematic lighting"
    )

    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        placeholder="blurry, low quality, distorted"
    )

    generate_btn = gr.Button("Generate Image 🚀")

    output = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt],
        outputs=output
    )

app.launch()

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    layout="centered"
)

st.title("🎨 Stable Diffusion Image Generator")
st.write("Generate AI images from text prompts")

# -------------------------------------------------
# Load Model (Cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    return pipe

with st.spinner("Loading model... Please wait ⏳"):
    pipe = load_model()

st.success("Model loaded successfully!")

# -------------------------------------------------
# User Input UI
# -------------------------------------------------
prompt = st.text_area(
    "Enter your prompt",
    placeholder="A futuristic sports car in an empty basement, cinematic lighting"
)

generate = st.button("Generate Image")

# -------------------------------------------------
# Generate Image
# -------------------------------------------------
if generate:
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image... 🎨"):
            image = pipe(prompt).images[0]

        st.image(image, caption="Generated Image", use_container_width=True)

        # Save image
        image.save("generated_image.png")

        st.success("Image generated successfully!")

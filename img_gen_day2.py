# -------------------------------------------------
# 1. Install Required Libraries
# -------------------------------------------------
!pip install -q diffusers transformers accelerate torch

# -------------------------------------------------
# 2. Import Modules
# -------------------------------------------------
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from IPython.display import display

# -------------------------------------------------
# 3. Load the Model
# -------------------------------------------------
model_id = "runwayml/stable-diffusion-v1-5"

print("Loading model... this may take a minute.")

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    print("✅ Model loaded successfully and moved to GPU.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure Runtime is set to GPU")

# -------------------------------------------------
# 4. Generate Image Function
# -------------------------------------------------
def generate_image(prompt):
    print(f"\n🎨 Generating image for:\n\"{prompt}\"")
    image = pipe(prompt).images[0]
    return image

# -------------------------------------------------
# 5. UI: Take Prompt from User
# -------------------------------------------------
while True:
    user_prompt = input("\nEnter your image prompt (or type 'exit' to quit):\n> ")

    if user_prompt.lower() == "exit":
        print("👋 Exiting image generation.")
        break

    try:
        generated_image = generate_image(user_prompt)

        # Display image in notebook
        display(generated_image)

        # Save image with unique name
        file_name = f"generated_image_{abs(hash(user_prompt)) % 10000}.png"
        generated_image.save(file_name)

        print(f"✅ Image saved as '{file_name}'")

    except Exception as e:
        print(f"❌ Error during generation: {e}")

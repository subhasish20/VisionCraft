import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# --- Streamlit App Configuration ---
st.set_page_config(page_title="VisionCraft", page_icon="🎨", layout="centered")

st.title("🎨 VisionCraft")
st.write("Transform your words into stunning visual art powered by AI.")

# --- User Input ---
prompt = st.text_input("💬 Enter your image prompt:", "")

# --- Model Configuration ---
output_path = "visioncraft_output.png"
model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Force CPU mode for universal compatibility

# --- Load the Stable Diffusion Pipeline (cached) ---
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_pipeline()

# --- Image Generation Section ---
if st.button("🚀 Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("✨ Crafting your vision... this may take some time!"):
            image = pipe(prompt).images[0]
            image.save(output_path)

            # --- Display and Download Output ---
            st.image(image, caption="🖼️ Your Generated Image", use_column_width=True)
            st.success("✅ Image generated successfully!")

            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name=output_path,
                    mime="image/png"
                )
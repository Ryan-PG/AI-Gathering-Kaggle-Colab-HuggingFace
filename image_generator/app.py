import torch
import gradio as gr
from diffusers import AutoPipelineForText2Image
import gc

# Available models
AVAILABLE_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sd-turbo",
    "Lykon/dreamshaper-8",
    "runwayml/stable-diffusion-v1-5",
]

# Global pipeline
pipe = None

def load_model(model_id):
    """Load a new model, clearing memory first"""
    global pipe
    
    try:
        # Clear existing model from memory
        if pipe is not None:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        return f"✓ Loaded: {model_id}"
    except Exception as e:
        pipe = None
        return f"✗ Error loading model: {str(e)}"

# Load initial model
load_model(AVAILABLE_MODELS[0])

def generate(model_id, prompt, negative_prompt, steps, guidance):
    global pipe
    status = ""
    if not prompt:
        return None, "Prompt required."

    # Load model if not loaded or changed
    if pipe is None or getattr(pipe, 'model_id', None) != model_id:
        status = load_model(model_id)
        # Attach model_id to pipe for tracking
        if pipe is not None:
            pipe.model_id = model_id
    else:
        status = f"✓ Loaded: {model_id}"

    if pipe is None:
        return None, status

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    ).images[0]

    return image, status

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ⚡ Text-to-Image Generator

        Select a model, type a prompt, tweak the sliders, and hit **Generate**.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="Select Model",
                interactive=True
            )
            model_status = gr.Textbox(
                label="Model Status",
                value=f"✓ Loaded: {AVAILABLE_MODELS[0]}",
                interactive=False
            )
            prompt = gr.Textbox(
                label="Prompt",
                lines=2,
                value="a cute robot teaching about Hugging Face Spaces, digital art, colorful"
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt (optional)",
                lines=1,
                placeholder="blurry, low quality, text"
            )
            steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=2,
                step=1,
                label="Inference steps"
            )
            guidance = gr.Slider(
                minimum=0.0,
                maximum=20.0,
                value=1.5,
                step=0.1,
                label="Guidance scale (strength of text conditioning)"
            )
            generate_btn = gr.Button("Generate 🚀")
        with gr.Column(scale=3):
            output = gr.Image(label="Generated image", height=512)

    generate_btn.click(
        fn=generate,
        inputs=[model_dropdown, prompt, negative_prompt, steps, guidance],
        outputs=[output, model_status]
    )

if __name__ == "__main__":
    demo.launch()

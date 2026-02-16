import gradio as gr
import os
import sys
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# -- Dependency Check --
MISSING_DEPENDENCIES = []
try:
    import trimesh
except ImportError:
    MISSING_DEPENDENCIES.append("trimesh")

try:
    # Try importing a core module to see if the environment is ready
    # This specific import often fails if pointnet2_ops is missing
    from text2interaction.model.hoi_diff import HOIDiff
    MODEL_LOADED = True
except ImportError as e:
    logger.warning(f"Failed to load InterAct models: {e}")
    MODEL_LOADED = False
    MISSING_DEPENDENCIES.append(f"InterAct Modules ({e})")
except Exception as e:
    logger.warning(f"Unexpected error loading InterAct models: {e}")
    MODEL_LOADED = False
    MISSING_DEPENDENCIES.append(f"Model Error ({e})")

# -- Constants --
DEMO_TITLE = "InterAct: 3D Human-Object Interaction Generation"
DEMO_DESC = """
Generate 3D human-object interactions from text descriptions.
**Note**: If the environment is not fully configured (e.g. missing `pointnet2_ops`), this demo runs in **Mock Mode**.
"""

OBJECT_LIST = [
    "chair", "backpack", "basketball", "box", "monitor", "keyboard", "table"
]

# -- Inference Logic --

def generate_interaction(text_prompt, object_name, seed):
    """
    Main generation function.
    """
    if not MODEL_LOADED:
        return mock_inference(text_prompt, object_name, seed)
    
    try:
        # TODO: Connect to real inference pipeline
        # For now, even if imports work, we might not have weights downloaded.
        # So we default to mock unless explicitly fully implemented.
        raise NotImplementedError("Full inference pipeline requiring weights is not yet wired up.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return mock_inference(text_prompt, object_name, seed, error_msg=str(e))

def mock_inference(text_prompt, object_name, seed, error_msg=None):
    """
    Simulates generation for UI testing.
    """
    logger.info(f"Mock Inference: {text_prompt} | Obj: {object_name} | Seed: {seed}")
    
    # Create a dummy output file (e.g., a simple text file or a placeholder image if video is hard to mock without ffmpeg)
    # For a video output, we can return None or a specific placeholder path if available.
    # Here typically we'd return a path to a pre-cached video or generate a black frame video.
    
    info_text = f"### Mock Mode Active\n\n"
    if error_msg:
        info_text += f"**System Error**: `{error_msg}`\n\n"
    else:
        info_text += f"**Status**: Dependencies missing ({', '.join(MISSING_DEPENDENCIES)}). Using mock generator.\n\n"
        
    info_text += f"**Input**: *{text_prompt}*\n"
    info_text += f"**Object**: *{object_name}*\n"
    info_text += f"**Seed**: *{seed}*\n"
    
    return None, info_text

# -- UI Construction --

with gr.Blocks(title="InterAct Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {DEMO_TITLE}")
    gr.Markdown(DEMO_DESC)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text Description", 
                placeholder="A person sitting on a chair..."
            )
            obj_input = gr.Dropdown(
                choices=OBJECT_LIST, 
                label="Object Category",
                value="chair"
            )
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            
            run_btn = gr.Button("Generate", variant="primary")
            
            if not MODEL_LOADED:
                gr.Warning(f"Running in Mock Mode. Missing: {MISSING_DEPENDENCIES}")
        
        with gr.Column(scale=2):
            # Output video and status logs
            video_output = gr.Video(label="Generated Interaction")
            status_output = gr.Markdown(label="Status Logs")

    run_btn.click(
        fn=generate_interaction,
        inputs=[text_input, obj_input, seed_input],
        outputs=[video_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()

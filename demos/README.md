# InterAct Gradio Demo

This directory contains an interactive web demo for **InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation**.

## Quick Start

1.  **Install Dependencies**
    Ensure you have `gradio` installed in your environment:
    ```bash
    pip install gradio
    ```

2.  **Run the App**
    From the project root directory:
    ```bash
    python demos/app.py
    ```

3.  **Access the Interface**
    Open the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Mock Mode

The demo includes a built-in **Mock Mode** that activates automatically if the full deep learning environment is not configured. 

-   **Why?** The full model requires `pointnet2_ops` (a custom CUDA extension) and pretrained weights.
-   **Behavior**: If these dependencies are missing, the app will still launch. You can interact with the UI, but generation will return a status log instead of a video.
-   **Status**: The UI will display a warning if Mock Mode is active, listing the missing components.

## Full Functionality

To enable full generation:
1.  Compile `pointnet2_ops` (requires CUDA).
2.  Download pretrained weights to `save/pretrained/model.pt`.
3.  Ensure all requirements in `../requirements.txt` are met.

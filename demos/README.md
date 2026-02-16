# InterAct Interactive Exploration Demo

This directory contains a **Gradio-based web interface** for the *InterAct* project. It provides an accessible way for researchers and developers to explore the text-to-interaction generation capabilities without writing boilerplate code.

## 🎯 Research Value

*   ** Accessibility**: Lowers the entry barrier for testing the model.
*   ** Visualization**: Instantly visualizes complex 3D human-object interactions from text prompts.
*   ** Robustness**: Includes a **Fall-back Mock Mode** that allows the UI to be tested and demonstrated even in environments without full GPU support or compiled C++ extensions (like `pointnet2_ops`).

## 🚀 Quick Start

### 1. Installation
Ensure you have the project dependencies and `gradio` installed:
```bash
pip install -r ../requirements.txt
pip install gradio
```

### 2. Launching the Demo
Run the application from the project root:
```bash
python demos/app.py
```

Click the local URL (e.g., `http://127.0.0.1:7860`) to open the interface.

## 🛠️ Modes of Operation

### Mock Mode (Default Fallback)
If the deep learning environment is incomplete (e.g., missing weights or CUDA extensions), the app automatically enters **Mock Mode**.
-   **Function**: Logs the generation request (Text/Object/Seed) to the status panel.
-   **Use Case**: UI testing, workflow integration, and debugging.

### Full Inference Mode
To enable actual 3D generation:
1.  **Compile Extensions**: Ensure `pointnet2_ops` is compiled and installed.
2.  **Download Weights**: Place pretrained models in `save/pretrained/`.
3.  **Run**: The app will automatically detect the modules and switch to full generation.

## 📂 Directory Structure
-   `app.py`: Main entry point containing UI logic and mock/real inference switching.
-   `README.md`: This documentation.

# Pull Request: Interactive Gradio Demo

## 📝 Summary
I have implemented a new **Interactive Web Demo** to make the *InterAct* model more accessible and easier to allow researchers to explore text-to-interaction generation.

## 🌟 Key Features
*   **User-Friendly Interface**: Added a `Gradio` web UI (`demos/app.py`) that accepts text prompts and object selection, replacing command-line scripts for visualization.
*   **Robust "Mock Mode"**:
    *   **Problem**: The full model requires complex dependencies like `pointnet2_ops` and pretrained weights, which can be hard to set up for quick testing.
    *   **Solution**: I implemented a smart fallback mechanism. If these dependencies are missing, the app automatically switches to "Mock Mode", allowing the UI and workflow to verified without crashing.
*   **Clean Architecture**: All demo code is isolated in a new `demos/` directory, ensuring no clutter in the main codebase.

## 🖼️ Preview
### UI Screenshot
![InterAct Gradio Demo](https://github.com/user-attachments/assets/placeholder-image-id)
*(Please replace this with a screenshot of the app running on localhost)*

### 📄 Sample Output (Mock Mode)
When the environment is missing dependencies, the "Status Logs" panel provides clear feedback:

```markdown
### Mock Mode Active

**Status**: Dependencies missing (InterAct Modules (No module named 'model')). Using mock generator.

**Input**: *A person sitting on a chair*
**Object**: *chair*
**Seed**: *42*
```

## 🧪 Verification
-   [x] **Mock Mode**: Verified that the app launches successfully even without `pointnet2_ops` installed.
-   [x] **UI Logic**: Confirmed that the "Generate" button correctly logs inputs and handles errors gracefully.
-   [x] **Dependency Check**: Verified that the app correctly identifies missing modules and warns the user.

## 🚀 How to Test
1.  Run `pip install gradio`.
2.  Run `python demos/app.py`.
3.  Open `http://127.0.0.1:7860` in your browser.

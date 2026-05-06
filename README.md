# ComfyUI-DreamLite

Experimental ComfyUI custom node wrapper for ByteVisionLab/DreamLite.

This is not an official ByteDance/ByteVisionLab project. It wraps the official DreamLite Python pipelines instead of integrating DreamLite into ComfyUI's native KSampler stack.

## Nodes

- DreamLite Generate/Edit: text-to-image and text-guided image editing through the official pipeline API.

## Install

1. Copy this folder to `ComfyUI/custom_nodes/ComfyUI-DreamLite`.
2. Install dependencies into the same Python environment used by ComfyUI:

   ```bash
   pip install -r ComfyUI/custom_nodes/ComfyUI-DreamLite/requirements.txt
   ```

3. Clone the official DreamLite repository somewhere locally:

   ```bash
   git clone https://github.com/ByteVisionLab/DreamLite.git
   ```

4. Put DreamLite model weights in the official repo layout, for example:

   ```text
   DreamLite/models/DreamLite-base
   DreamLite/models/DreamLite-mobile
   ```

5. Restart ComfyUI.
6. Add `DreamLite > DreamLite Generate/Edit`.
7. Set `dreamlite_repo` to your local `DreamLite` repository path.
8. Set `model_path` to `models/DreamLite-base`, `models/DreamLite-mobile`, or an absolute model folder path.

## Recommended settings

- Base: `variant=base`, `steps=0` or `steps=28`, `model_path=models/DreamLite-base`.
- Mobile: `variant=mobile`, `steps=0` or `steps=4`, `model_path=models/DreamLite-mobile`.

## Notes

- Optional `image` input enables text-guided editing.
- `size_mode=use_input_image_size` uses the connected image's dimensions for editing.
- `steps=0` uses official defaults: base=28 and mobile=4.
- This node caches the pipeline in memory. Restart ComfyUI to fully unload it.
- The wrapper has not been validated without access to the gated DreamLite weights.

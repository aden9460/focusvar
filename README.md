<div align="center">

# FocusVAR

**Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation**

[Getting Started](#getting-started) · [Supported Backbones](#supported-backbones) · [Method](#method) · [Citation](#citation) · [License](#license)

</div>

## Overview

FocusVAR is a training-free inference acceleration framework for visual autoregressive generation. It reduces classifier-free guidance (CFG) overhead by identifying where guidance is most useful, fusing the guided representation at selected scale/layer boundaries, and continuing subsequent computation with a conditional-only branch.

This repository provides FocusVAR implementations for two visual autoregressive text-to-image backbones:

- **HART**
- **Infinity**

The codebase includes model-side inference modifications, focused CFG collapse controls, token selection utilities, and lightweight scripts for running text-to-image generation.

## Highlights

- **Training-free acceleration**: apply FocusVAR directly to pretrained visual autoregressive models.
- **Focused CFG computation**: keep full CFG only where it contributes most, then switch to conditional-only inference.
- **Layerwise collapse control**: choose the scale and transformer layer where CFG branches are fused.
- **Token-focused computation**: reduce large-scale computation by forwarding only selected tokens and restoring the full map from cached representations.
- **Backbone support**: includes implementations for HART and Infinity.

## Repository Layout

```text
FocusVAR/
├── HART/          # FocusVAR implementation for HART
├── Infinity/      # Minimal FocusVAR implementation for Infinity inference
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aden9460/focusvar.git
cd focusvar
```

### 2. Prepare checkpoints

Model checkpoints are not included in this repository. Please download the required pretrained weights from the corresponding upstream projects or use your own checkpoints:

- HART transformer and tokenizer / VAE weights
- Infinity transformer and VAE weights
- text encoder weights required by the selected backbone

### 3. Install dependencies

The two backbones inherit dependencies from their original implementations. Start from the provided environment file and then install any backbone-specific CUDA extensions as needed:

```bash
pip install -r requirements.txt
```



## Supported Backbones


### Infinity

```bash
cd Infinity
python inference.py
```

The Infinity directory is kept as a minimal inference backend. Update checkpoint paths and generation settings in `inference.py` before running.


### HART

```bash
cd HART
python inference.py --model_path /path/to/model \
  --text_model_path /path/to/Qwen2 \
  --prompt "a photo of a corgi wearing sunglasses" \
  --sample_folder_dir ./outputs/hart
```

FocusVAR support is integrated into the HART inference path and can be enabled through the model's acceleration configuration.




## License

This repository is released under the MIT License. The code builds on HART and Infinity components; please also respect the licenses of the corresponding upstream projects and pretrained model checkpoints.

## Acknowledgements

This project builds on the open-source HART and Infinity visual autoregressive generation codebases. We thank the authors and contributors of these projects for their public releases.

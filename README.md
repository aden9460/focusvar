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

For HART fused kernels:

```bash
cd HART/hart/kernels
pip install -e .
```

## Supported Backbones

### HART

```bash
cd HART
python inference.py --model_path /path/to/model \
  --text_model_path /path/to/Qwen2 \
  --prompt "a photo of a corgi wearing sunglasses" \
  --sample_folder_dir ./outputs/hart
```

FocusVAR controls are exposed through `configure_inference_acceleration(...)` in the HART model:

```python
model.configure_inference_acceleration(
    enable_fastvar_compute_merge=True,
    enable_spacevar_compute_merge=False,
    fastvar_ratio_by_scale={48: 0.4, 64: 0.5},
    fastvar_prune_scales={48, 64},
    fastvar_start_layer=1,
    enable_layerwise_cond_only_collapse=True,
    cond_only_start_scale=48,
    cond_only_start_layer=5,
)
```

Recommended starting points:

- conservative: `cond_only_start_scale=48`, `cond_only_start_layer=15`
- faster: `cond_only_start_scale=36`, `cond_only_start_layer=5`

### Infinity

```bash
cd Infinity
python inference.py
```

The Infinity directory is kept as a minimal inference backend. Update checkpoint paths and generation settings in `inference.py` before running.

## Method

FocusVAR targets the redundancy of full two-branch CFG during visual autoregressive decoding. Instead of running conditional and unconditional branches throughout all scales and layers, FocusVAR:

1. runs standard CFG in early or selected high-impact regions,
2. performs a guided fusion at a configured scale/layer boundary,
3. trims the KV cache to the conditional branch,
4. continues the remaining computation in conditional-only mode,
5. optionally applies token-focused computation at large scales to further reduce latency.

In this repository, token selection is part of the FocusVAR acceleration pipeline. The implementation supports both feature-importance-based selection and CFG-difference-based selection, but they are exposed as FocusVAR components rather than separate methods.

## Key Files

- `HART/hart/modules/models/transformer/hart_transformer_t2i.py`  
  HART autoregressive inference, focused CFG collapse, and acceleration configuration.

- `HART/hart/modules/networks/fastvar_utils.py`  
  Token selection, merge, and unmerge utilities used by the HART backend.

- `HART/hart/modules/networks/fastvar_basic.py`  
  HART transformer block integration for token-focused computation.

- `Infinity/infinity/models/infinity.py`  
  Infinity autoregressive inference path with FocusVAR controls.

- `Infinity/infinity/models/fastvar_utils.py`  
  Token selection and restoration utilities for the Infinity backend.

## Citation

If you use this repository, please cite FocusVAR. The final BibTeX entry will be updated when available.

```bibtex
@article{focusvar2026,
  title={FocusVAR: Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation},
  author={TBD},
  journal={TBD},
  year={2026}
}
```

## License

This repository is released under the MIT License. The code builds on HART and Infinity components; please also respect the licenses of the corresponding upstream projects and pretrained model checkpoints.

## Acknowledgements

This project builds on the open-source HART and Infinity visual autoregressive generation codebases. We thank the authors and contributors of these projects for their public releases.

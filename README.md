<p align="center">
  <img src="assets/logo.jpg" width="700" alt="FocusVAR logo">
</p>

<div align="center">

# FocusVAR

**Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation**

[Paper](#citation) · [Getting Started](#getting-started) · [Method](#method) · [License](#license)

</div>

## Overview

FocusVAR is an open-source codebase for efficient visual autoregressive generation with **focused classifier-free guidance** and **token pruning**. It is built on top of two visual autoregressive backbones:

- **Infinity**
- **HART**

The repository contains the model-side changes needed to support efficient inference, including:

- FastVAR / SpaceVAR token pruning
- conditional-only collapse during CFG inference
- layerwise collapse control
- cache reuse across scale steps
- evaluation and launch scripts for supported backbones

## Repository layout

```text
FocusVAR/
├── HART/        # HART-backed implementation and scripts
├── Infinity/    # Infinity-backed implementation and scripts
├── README.md
├── LICENSE
└── .gitignore
```

## Getting started

Each backbone keeps its own original project structure. To run a specific backend:

### HART

```bash
cd HART
python inference.py --model_path /path/to/model \
  --text_model_path /path/to/Qwen2 \
  --prompt "YOUR_PROMPT" \
  --sample_folder_dir /path/to/save_dir
```

### Infinity

```bash
cd Infinity
python inference.py
```

For benchmark scripts and evaluation utilities, refer to the corresponding `evaluation/` and `scripts/` folders.

## What was cleaned for open source

This repository should only contain source code, scripts, and lightweight configuration files. Large or generated artifacts such as the following have been removed or ignored:

- training / inference outputs
- build artifacts
- cached Python bytecode
- dataset archives and benchmark downloads
- compiled extension artifacts
- model checkpoints

## Method

FocusVAR combines focused CFG collapse with token pruning. The core idea is to:

1. keep the conditional branch active when guidance is needed,
2. collapse cond/uncond branches once the selected scale/layer is reached,
3. continue inference in cond-only mode to reduce compute,
4. optionally prune tokens with FastVAR / SpaceVAR when the scale becomes large.

### Implemented HART changes

The HART backend now includes layerwise cond-only collapse behavior that mirrors the Infinity-style inference path, including:

- collapse boundary selection by scale and layer
- cond-only continuation after collapse
- partial KV-cache trimming instead of full cache reset

## Citation

If you use this code, please cite the paper once the final BibTeX is available.

```bibtex
@article{focusvar2026,
  title={FocusVAR: Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation},
  author={TBD},
  journal={TBD},
  year={2026}
}
```

## License

The code in this repository follows the licenses of the original backbone projects and any third-party components they include. Please check the bundled `LICENSE` file and the upstream HART / Infinity licenses before redistribution.

# RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment [CVPR 2026]


[![arXiv](https://img.shields.io/badge/arXiv-2603.00483-b31b1b.svg)](https://arxiv.org/abs/2603.00483)

Official implementation of **RAISE** (**R**equirement-**A**daptive **S**elf-**I**mproving **E**volution), a training-free, requirement-driven evolutionary framework for adaptive text-to-image generation.

<details>
<summary>Click to expand abstract</summary>

Recent text-to-image (T2I) diffusion models achieve remarkable realism, yet faithful prompt-image alignment remains challenging, particularly for complex prompts with multiple objects, relations, and fine-grained attributes. Existing training-free inference-time scaling methods rely on fixed iteration budgets that cannot adapt to prompt difficulty, while reflection-tuned models require carefully curated reflection datasets and extensive joint fine-tuning of diffusion and vision-language models, often overfitting to reflection paths data and lacking transferability across models. We introduce RAISE (Requirement-Adaptive Self-Improving Evolution), a training-free, requirement-driven evolutionary framework for adaptive T2I generation. RAISE formulates image generation as a requirement-driven adaptive scaling process, evolving a population of candidates at inference time through a diverse set of refinement actions, including prompt rewriting, noise resampling, and instructional editing. Each generation is verified against a structured checklist of requirements, enabling the system to dynamically identify unsatisfied items and allocate further computation only where needed. This achieves adaptive test-time scaling that aligns computational effort with semantic query complexity. On GenEval and DrawBench, RAISE attains state-of-the-art alignment (0.94 overall GenEval) while incurring fewer generated samples (reduced by 30-40%) and VLM calls (reduced by 80%) than prior scaling and reflection-tuned baselines, demonstrating efficient, generalizable, and model-agnostic multi-round self-improvement.

</details>

## Overview

RAISE builds a multi-node generation graph that refines prompts and validates outputs through requirement-driven multi-round self-improvement.

This repository includes:

- LangGraph pipeline for RAISE (`src/graph/graph_RAISE.py`)
- Baseline generation graph (`src/graph/graph_original.py`)
- GenEval runner (`geneval_generate.py`)
- Prompt benchmarks (`benchmarks/`)

## Setup

Create and activate the conda environment:

```bash
conda create -n RAISE_env python=3.11.13
conda activate RAISE_env
```

Install project and core dependencies:

```bash
pip install -e .

pip install --upgrade langgraph==0.6.6 langgraph-sdk==0.2.3 langchain-ollama==0.3.6 langchain-openai==0.3.23 "langgraph-cli[inmem]==0.3.8" langgraph-api \
    ipython==9.4.0 opencv-python==4.12.0.88 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.54.1 accelerate==1.9.0 protobuf==5.29.3 sentencepiece==0.2.0 gradio[mcp]==5.39.0 \
    pytorch_lightning==2.5.5 torchmetrics==1.8.2 einops==0.8.1

pip install git+https://github.com/huggingface/diffusers.git # 0.36.0.dev0
```

Install Grounded-SAM-2 and verifier dependencies:

```bash
cd src/grounded_sam_2

cd checkpoints
bash download_ckpts.sh

cd ../gdino_checkpoints
bash download_ckpts.sh

cd ..
pip install -e .

pip install pip==22.3.1
pip install "setuptools>=62.3.0,<75.9"
pip install --no-build-isolation -e grounding_dino

pip install supervision==0.26.1 timm==1.0.20
pip install --upgrade transformers==4.49.0
pip install git+https://github.com/bfshi/scaling_on_scales.git
```

Install Ollama and pull the VLM:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral-small3.2:24b-instruct-2506-fp16
```

## Quick Start

Serve Ollama:

```bash
CUDA_VISIBLE_DEVICES=0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Generate from a manual prompt list:

```bash
CUDA_VISIBLE_DEVICES=1 python src/graph/graph_RAISE.py --base_url 127.0.0.1:11434 \
    --prompts "A red colored car." \
    "A black colored car." \
    "A pink colored car."
```

Generate from a benchmark `.txt` file (one prompt per line):

```bash
CUDA_VISIBLE_DEVICES=1 python src/graph/graph_RAISE.py --base_url 127.0.0.1:11434 \
    --benchmark_txt_path benchmarks/drawbench.txt \
    --saving_path outputs/drawbench/RAISE_drawbench/
```

Generate GenEval samples:

```bash
CUDA_VISIBLE_DEVICES=1 python geneval_generate.py --base_url 127.0.0.1:11434 --n_samples 1 \
    --metadata_file benchmarks/geneval_evaluation_metadata.jsonl \
    --outdir outputs/geneval_results/RAISE_geneval/
```

Ouputs: generated artifacts are saved under the configured output directory, including images and JSON result files.

## Benchmarks

Prompt files are provided in `benchmarks/`:

- `benchmarks/drawbench.txt`
- `benchmarks/geneval_evaluation_metadata.jsonl`

## Bibtex Citation
If you find our method and paper useful, we kindly ask that you cite our paper.

You can also find the preprint on [arXiv:2603.00483](https://arxiv.org/abs/2603.00483)

```bibtex
@misc{jiang2026raise,
    title={RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment}, 
    author={Liyao Jiang and Ruichen Chen and Chao Gao and Di Niu},
    year={2026},
    eprint={2603.00483},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.00483}, 
}

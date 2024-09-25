# axolotl-finetune

## Introduction

This repository contains my work on experimentating with finetuning LLMs using the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl). My work focuses on exploring various finetuning techniques, model performance, and dataset customizations for specific tasks.

## Overview

Currently, this repo documents the finetuning of `meta-llama/Meta-Llama-3.1-8B-Instruct` using two distinct techniques:

- **QLoRA Finetuning:**
  Finetuning on a _single A40 GPU (48 GB VRAM)_ using QLoRA. The dataset used for this finetuning consists of 1K Alpaca examples with a focus on coding tasks. For more details, see [qlora](https://github.com/yuvrajsagar17/axolotl-finetune/tree/main/qlora#meta-llama-31-8b-instruct-qlora-finetune-using-axolotl)

- **Spectrum: Targetted Finetuning:**
  Finetuning by targeting the top **25%** highest Signal-to-Noise Ratio (SNR) values from the dataset, utilizing Multi-GPU training using `deepspeed config_zero2.json` on two _A40 GPU (48 GB VRAM each)_. The dataset I used for this is 1K Alpaca general-information dataset [here](https://huggingface.co/datasets/yuvraj17/finetune_alpaca_1K). For more details, see [spectrum](https://github.com/yuvrajsagar17/axolotl-finetune/tree/main/spectrum#spectrum-targeted-training-on-signal-to-noise-ratio)

and

- **DPO: An alternate to RLHF:** Finetuned my `DARE_TIES` merged Model [`yuvraj17/Llama3-8B-SuperNova-Spectrum-dare_ties`](https://huggingface.co/yuvraj17/Llama3-8B-SuperNova-Spectrum-dare_ties) with **QLoRA**(`4-bit precision`) on 2K sample Preference Dataset [yuvraj17/chatml-OpenHermes2.5-dpo-binarized-alpha-2k](https://huggingface.co/datasets/yuvraj17/chatml-OpenHermes2.5-dpo-binarized-alpha-2k), on **1x A40 GPU (48 GB VRAM)** for 1 epoch, using HuggingFace [TRL](https://huggingface.co/docs/trl/index). For more details, see [dpo](dpo/README.md).

## Insights and Adaptability

Through hands-on experimentation, I’ve built a solid understanding of:

- Using **QLoRA** efficiently to fine-tune large models on limited hardware.
- The role of **SNR-based spectrum finetuning** in enhancing model learning on targeted layers.
- The benefits of tailoring datasets to different tasks for improved model performance.
- Applying **DPO** to align model outputs with human preferences with less complexity (compared to RLHF).

These experiences have improved my adaptability in finetuning LLMs for various tasks.

## Future Work

In the future, I plan to:

- Incorporate Model-Qunatization techniques (GGUF, AWQ, GPTQ and HQQ) for smaller model deployments & local-inference.
- Evaluation of Model on Nous/OpenLLM Benchmarks

Stay tuned for more updates!

---

Get in touch

[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yuvraj-sagar-514806227/)
[![Twitter](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/ysagar117)

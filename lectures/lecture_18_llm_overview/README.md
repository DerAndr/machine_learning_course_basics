# Lecture 18: LLM Overview

This directory currently contains a practical-first draft for a hands-on introduction to modern LLM workflows.

The material focuses on implementation intuition students need for working with contemporary language models:

- environment checks and memory-aware setup for Colab T4;
- model loading trade-offs and VRAM budgeting;
- tokenization behavior and prompt construction;
- instruct mode vs reasoning mode behavior;
- structured outputs and parsing constraints;
- attention visualization for interpretability intuition;
- parameter-efficient fine-tuning with LoRA;
- fair comparison between prompting-only and fine-tuned adapters;
- multimodal prompting with image, chart, and optional audio inputs;
- prompt sensitivity and hallucination-aware prompting patterns.

## Core Files

- `lecture_notes.md`
- `links.yaml`

## Practical Session

- `practical_session/`: practical materials for text LLM internals and multimodal LLM usage.
- `practical_session/LLMs_hands_on.ipynb`: full notebook for modern LLM internals, inference behavior, and LoRA adaptation.
- `practical_session/LLMs_hands_on.py`: generated companion script for reading and diffing.
- `practical_session/Multimodal_LLMs_Hands_on.ipynb`: full notebook for multimodal prompting with image and optional audio workflows.
- `practical_session/Multimodal_LLMs_Hands_on.py`: generated companion script for reading and diffing.

## Draft Status

This lecture does not yet have a packaged slide deck or separate student practical notebook.
The two full practical notebooks are the canonical working artifacts for now.

---

[<- Previous](../lecture_17_recsys/README.md) | [All Lectures](../README.md)

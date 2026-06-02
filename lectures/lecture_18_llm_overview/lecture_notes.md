# Lecture 18 Notes: LLM Overview

These notes summarize the concepts used in the practical notebooks.

## LLM Runtime Basics

Modern LLM sessions are constrained by available memory and context length, not only by model quality. In classroom settings, it is useful to start with:

- GPU availability checks;
- RAM and VRAM monitoring;
- explicit model-size and dtype trade-offs.

This makes deployment constraints visible before discussing model behavior.

## Tokenization and Prompt Framing

LLMs operate on token sequences, not raw words. The same text can produce different token patterns across tokenizers, affecting:

- context usage;
- generation cost;
- robustness for multilingual or symbol-heavy inputs.

Prompt templates and role formatting are part of the model interface and should be treated as input schema, not just plain text.

## Inference Modes and Structured Outputs

Instruction-tuned models can show different behavior depending on decoding settings and prompting style. Practical generation work benefits from:

- explicit output format requests;
- schema-like response constraints;
- deterministic settings when reproducibility is needed.

Structured generation is often the first step toward reliable downstream automation.

## Attention and Interpretability Intuition

Attention maps are not full explanations, but they provide useful qualitative intuition about token interactions. They can help students inspect:

- which earlier tokens influence a prediction;
- how patterns differ for short vs long prompts;
- why some outputs drift or stay on-topic.

## LoRA Fine-Tuning

LoRA adapts a pretrained model by training low-rank update matrices while keeping most base weights frozen. This reduces training cost and makes adapter-based experiments practical in limited hardware environments.

When comparing LoRA and prompt engineering, use fair evaluation setups:

- separate baseline and adapted prompts;
- hold out test instructions;
- compare behavior consistency, not only one anecdotal output.

## Multimodal LLMs

Multimodal models extend text prompting with additional modalities such as images and audio. Practical classroom patterns include:

- image captioning and VQA;
- OCR-like extraction and chart interpretation;
- uncertainty reporting and hallucination-aware prompts;
- prompt sensitivity analysis under different instruction styles.

## Key Takeaways

- LLM usage is constrained by memory, latency, and interface format.
- Prompt quality and output schema design strongly affect reliability.
- LoRA is a practical adaptation strategy for domain-specific behavior.
- Prompting-only and fine-tuning should be compared with fair baselines.
- Multimodal prompting expands capabilities but also introduces new failure modes.
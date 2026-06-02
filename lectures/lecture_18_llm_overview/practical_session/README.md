# Lecture 18 Practical Session

This directory contains two full practical notebooks for the LLM overview block.

## Files

| File | Description |
| --- | --- |
| `LLMs_hands_on.ipynb` | Full notebook for modern text LLM internals, generation behavior, and LoRA fine-tuning |
| `LLMs_hands_on.py` | Generated companion script for review and diffing |
| `Multimodal_LLMs_Hands_on.ipynb` | Full notebook for multimodal prompting with image workflows and optional audio input |
| `Multimodal_LLMs_Hands_on.py` | Generated companion script for review and diffing |

## Structure

The practical is organized as two connected full notebooks.

### Notebook A: `LLMs_hands_on.ipynb`

| Part | Topic | Assets |
| --- | --- | --- |
| 1 | Environment setup and memory monitoring | Colab runtime checks, CPU/GPU memory reporting |
| 2 | Model loading and VRAM math | Qwen3 1.7B loading, dtype/device trade-offs |
| 3 | Tokenization behavior | Prompt examples across text patterns |
| 4 | Inference behavior | Instruct-style vs thinking-style prompts, structured outputs |
| 5 | Attention under the hood | Attention extraction and heatmap intuition |
| 6 | LoRA adaptation | Small custom instruction set, masked-label fine-tuning, fair comparison |

### Notebook B: `Multimodal_LLMs_Hands_on.ipynb`

| Part | Topic | Assets |
| --- | --- | --- |
| 1 | Multimodal model framing | Gemma 4 family overview and Colab fit discussion |
| 2 | Runtime setup | Colab package install and GPU checks |
| 3 | Model loading | Hugging Face multimodal model and processor |
| 4 | Image demos | Captioning, VQA, structured analysis, hallucination-control prompts |
| 5 | OCR-like and chart tasks | Document text extraction and chart interpretation |
| 6 | Prompt sensitivity and custom uploads | Alternate prompts and optional user image workflow |
| 7 | Optional audio input | Audio-capable model path for advanced discussion |

## Environment

Primary runtime:

- Google Colab with T4 GPU.

Local (best effort, hardware dependent):

```bash
uv sync --group llm
uv run jupyter lab
```

Large-model cells may exceed local GPU or RAM constraints. The notebooks are tuned for guided runs in Colab.

## Notes

- There is currently no separate student TODO version in this lecture.
- The `.py` files are generated companions for code review and version diffing.
- Notebook B includes optional sections that may require authenticated Hugging Face access.
- Some multimodal examples depend on internet downloads for models and assets.

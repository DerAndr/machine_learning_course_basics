# Deep Learning and Google Colab Guide

When you reach the deep learning parts of the course (e.g., Lecture 12: Neural Networks, Lecture 16: NLP, and Lecture 18: LLMs), you will start working with heavy models and large datasets. This guide explains how to decide where to run your code and how to set up your environment successfully.

## 1. Local Execution vs. Google Colab

### When to run locally
- You have a dedicated GPU with sufficient VRAM (e.g., NVIDIA RTX series with 8GB+ VRAM) or an Apple Silicon Mac (M1/M2/M3) with sufficient unified memory.
- You are working on standard datasets and smaller models (e.g., CNNs for MNIST/CIFAR, or small BERT models).
- You prefer using your local IDE and `uv` environment.

### When to use Google Colab
- You are working with Large Language Models (LLMs), LoRA fine-tuning, or multi-modal models (Lecture 18).
- You encounter Out-Of-Memory (OOM) errors locally.
- Your computer becomes too slow or freezes during training.
- You do not have a dedicated hardware accelerator (CUDA/MPS).

---

## 2. Running Deep Learning Locally

To run deep learning locally, you must sync the correct optional dependency groups using `uv`. The baseline environment is **not** enough.

### Setup Instructions
1. Open your terminal in the repository root.
2. Run the sync command for your specific lecture:
   - **Lecture 12 (Neural Networks):** `uv sync --group neural_networks` (installs `torch`, `torchvision`, `torchinfo`)
   - **Lecture 16 (NLP):** `uv sync --group nlp` (installs `transformers`, `datasets`, etc.)
   - **Lecture 18 (LLMs):** `uv sync --group llm` (installs `accelerate`, `peft`, etc.)
3. Verify the environment: `uv run python tools/check_notebook_environment.py --group <group_name>`
4. Open the notebook: `uv run jupyter lab`

### Important Local Rule: Ignore `!pip install`
If you are running a notebook locally and see a cell starting with `!pip install` (e.g., `!pip install -q transformers`), **skip or delete that cell**. 
Your dependencies are managed securely by `uv`. Running `!pip` inside a notebook locally can break your virtual environment.

---

## 3. Running Notebooks in Google Colab

Google Colab provides free access to GPUs (like the NVIDIA T4), which are perfect for heavy assignments.

### Step-by-Step Colab Setup

1. **Open the Notebook:**
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click **File > Open notebook**.
   - Select the **GitHub** tab.
   - Enter this repository's URL and navigate to the `practical_session/` or `lecture_examples/` notebook you want to run.

2. **Enable the GPU Accelerator:**
   - In the Colab menu, go to **Runtime > Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** (or whichever GPU is available to you).
   - Click **Save**.

3. **Install Dependencies:**
   - Unlike local execution, Colab *does* need `!pip install` to get the necessary libraries since it doesn't use our `uv` environment.
   - Many deep learning notebooks in this repository already contain an uncommented `!pip install` cell at the very top. **Run this cell first.**

4. **Working with Data:**
   - If the notebook downloads data automatically (e.g., via `datasets` or `wget`), it will work seamlessly.
   - If you need to load local repository data, you have two options:
     - Clone the repository directly into Colab: `!git clone <repo_url>`
     - Mount your Google Drive: Click the **Folder icon** on the left sidebar, then click the **Google Drive icon** to mount your drive, and upload the data there.

## 4. Troubleshooting Out-of-Memory (OOM) Errors

If you see an error like `RuntimeError: CUDA out of memory` or your kernel crashes abruptly:
- **Reduce the Batch Size:** Halve your `batch_size` parameter in your training loop or DataLoader.
- **Use Smaller Models:** If working with LLMs, load quantized versions (e.g., using `bitsandbytes` 8-bit or 4-bit) if supported by the lecture.
- **Restart the Kernel:** Sometimes memory isn't freed properly. Restarting the Jupyter/Colab kernel and running the cells again from the top often helps.

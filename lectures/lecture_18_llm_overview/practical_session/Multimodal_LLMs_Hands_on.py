# Colab setup
!pip install -U transformers accelerate torch torchvision torchaudio pillow requests bitsandbytes

import torch
import gc
import os
from PIL import Image
import requests
from io import BytesIO

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM, GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

def show_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"Allocated VRAM: {allocated:.2f} GB")
    print(f"Reserved VRAM:  {reserved:.2f} GB")
    print(f"Total VRAM:     {total:.2f} GB")


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    show_gpu_memory()


show_gpu_memory()

from huggingface_hub import notebook_login

# Uncomment if needed:
# notebook_login()

from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

model.eval()
show_gpu_memory()

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def display_image(image: Image.Image, max_width: int = 500):
    display(image.resize((max_width, int(max_width * image.height / image.width))))

sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"

image = load_image_from_url(sample_image_url)
display_image(image)

# # Optional cell: upload your own image from local machine to Colab

# from google.colab import files
# from PIL import Image

# image_filename = 'my_image.png'

# # Open image and convert to RGB
# user_image = Image.open(image_filename).convert("RGB")

# print("Uploaded file:", image_filename)
# print("Image size:", user_image.size)

# display_image(user_image)

def ask_gemma_about_image(
    image: Image.Image,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    generated = outputs[0][input_len:]
    decoded = processor.decode(generated, skip_special_tokens=False)

    try:
        return processor.parse_response(decoded)
    except Exception:
        return decoded

# user_prompt = """
# Analyze this image.

# Return:
# 1. Main objects:
# 2. Visible text:
# 3. What is happening:
# 4. Important visual details:
# 5. Uncertain or unclear parts:

# Do not invent details that are not visible.
# """

# answer = ask_gemma_about_image(
#     user_image,
#     user_prompt,
#     max_new_tokens=500,
# )

# print(answer)

answer = ask_gemma_about_image(
    image,
    "What is shown in this image? Answer in 3-5 sentences."
)

print(answer)

questions = [
    "What are the main objects in the image?",
    "Is this image more likely taken indoors or outdoors?",
    "What visual evidence supports your answer?",
]

for q in questions:
    print("=" * 80)
    print("QUESTION:", q)
    print()
    print(ask_gemma_about_image(image, q, max_new_tokens=180))

structured_prompt = """
Analyze the image using this structure:

1. Main subject:
2. Background:
3. Visible objects:
4. Visible text:
5. Spatial relations:
6. Uncertain details:

Important:
- Do not guess hidden details.
- If something is not visible, say "not visible".
"""

print(ask_gemma_about_image(image, structured_prompt, max_new_tokens=400))

unsafe_prompt = "Tell me everything about this place, including its history and who built it."

safe_prompt = """
Answer only based on the visible image.

Use this format:

Visible evidence:
- ...

Not visible / cannot determine:
- ...

Possible but uncertain:
- ...

Do not provide historical facts unless they are directly visible in the image.
"""

print("UNSAFE PROMPT OUTPUT")
print("=" * 80)
print(ask_gemma_about_image(image, unsafe_prompt, max_new_tokens=300))

print("\n\nSAFE PROMPT OUTPUT")
print("=" * 80)
print(ask_gemma_about_image(image, safe_prompt, max_new_tokens=300))

receipt_url = "https://raw.githubusercontent.com/microsoft/markitdown/main/packages/markitdown/tests/test_files/test.jpg"

receipt_image = load_image_from_url(receipt_url)
display_image(receipt_image)

ocr_prompt = """
Read the visible text in this image.

Return:

1. Raw visible text:
2. Key fields:
3. Any uncertain or unreadable parts:

Do not invent missing text.
"""

print(ask_gemma_about_image(receipt_image, ocr_prompt, max_new_tokens=500))

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2018, 2025)
y = np.array([100, 115, 130, 128, 150, 180, 210])

plt.figure(figsize=(8, 4))
plt.plot(x, y, marker="o")
plt.title("Example Revenue Growth")
plt.xlabel("Year")
plt.ylabel("Revenue Index")
plt.grid(True)
plt.savefig("chart.png", dpi=160, bbox_inches="tight")
plt.show()

chart_image = Image.open("chart.png").convert("RGB")

chart_prompt = """
Analyze this chart.

Return:

1. What variables are shown?
2. What is the main trend?
3. Are there any dips or anomalies?
4. What would be a cautious business interpretation?
5. What should we NOT conclude from this chart alone?
"""

print(ask_gemma_about_image(chart_image, chart_prompt, max_new_tokens=500))

prompts = {
    "Vague": "Describe this image.",
    "Detailed": """
Describe this image in detail.
Mention objects, colors, spatial layout, background, and possible context.
""",
    "Evidence-constrained": """
Describe only what is directly visible.
Separate observations from guesses.
Use:
- Observations:
- Uncertain:
- Not visible:
""",
}

for name, prompt in prompts.items():
    print("=" * 80)
    print(name.upper())
    print("=" * 80)
    print(ask_gemma_about_image(image, prompt, max_new_tokens=350))
    print()

# from google.colab import files

# uploaded = files.upload()

# uploaded_path = next(iter(uploaded.keys()))
# student_image = Image.open(uploaded_path).convert("RGB")

# display_image(student_image)

# student_question = """
# Analyze this image.

# Return:

# 1. Main objects:
# 2. Visible text:
# 3. What is happening:
# 4. Anything uncertain:
# 5. One useful follow-up question a user could ask:
# """

# print(ask_gemma_about_image(student_image, student_question, max_new_tokens=500))

def estimate_weight_memory_billions(params_b: float, bytes_per_param: float):
    gb = params_b * 1e9 * bytes_per_param / 1024**3
    return gb


models = {
    "Gemma 4 E2B effective": 2.3,
    "Gemma 4 E2B incl. embeddings": 5.1,
    "Gemma 4 E4B effective": 4.5,
    "Gemma 4 E4B incl. embeddings": 8.0,
    "Gemma 4 26B A4B total": 25.2,
    "Gemma 4 26B A4B active": 3.8,
    "Gemma 4 31B dense": 30.7,
}

for name, params_b in models.items():
    fp16_gb = estimate_weight_memory_billions(params_b, 2)
    int8_gb = estimate_weight_memory_billions(params_b, 1)
    int4_gb = estimate_weight_memory_billions(params_b, 0.5)

    print(f"{name:35s} | FP16: {fp16_gb:6.2f} GB | INT8: {int8_gb:6.2f} GB | 4-bit: {int4_gb:6.2f} GB")

SAFE_VISUAL_PROMPT_TEMPLATE = """
You are analyzing an image.

Task:
{task}

Rules:
1. Use only visible evidence.
2. Do not invent hidden details.
3. If something is unclear, say "unclear".
4. Separate observations from interpretation.
5. Do not identify real people.
6. Do not infer sensitive attributes.

Return:

Observations:
- ...

Interpretation:
- ...

Uncertain:
- ...
"""

# task = "Describe the image and explain what the user should notice."
# prompt = SAFE_VISUAL_PROMPT_TEMPLATE.format(task=task)

# print(ask_gemma_about_image(student_image, prompt, max_new_tokens=500))

benchmark_items = [
    {
        "name": "Test sample",
        "image": image,
        "questions": [
            "Describe the image in one sentence.",
            "What large structure is visible?",
            "Is there readable text? If yes, transcribe it.",
            "What cannot be determined from this image alone?",
        ],
    },
    {
        "name": "Chart sample",
        "image": chart_image,
        "questions": [
            "What type of chart is this?",
            "What is the main trend?",
            "What is the approximate value in the last year?",
            "What should we not conclude from this chart alone?",
        ],
    },
]

for item in benchmark_items:
    print("=" * 100)
    print("IMAGE:", item["name"])
    print("=" * 100)

    for q in item["questions"]:
        print("\nQUESTION:", q)
        print("-" * 80)
        print(ask_gemma_about_image(item["image"], q, max_new_tokens=220))

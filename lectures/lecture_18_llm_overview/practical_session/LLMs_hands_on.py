# Cell 1: check existing Colab torch/CUDA
import torch, sys

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# Cell 2: install only non-torch utilities + torchao
!pip install -q -U transformers peft seaborn matplotlib tiktoken humanize GPUtil
!pip install -q outlines "pydantic==2.12.3"
!pip install -q "torchao>=0.16.0" --no-deps

# Check if T4 GPU is allocated properly
!nvidia-smi

# Cell 3: verify
import torch
import torchao
import transformers
import pydantic

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchao:", torchao.__version__)
print("cuda available:", torch.cuda.is_available())

# Import packages for memory monitoring
import os, sys, humanize, psutil, GPUtil

# Define memory reporting function
def mem_report():
    """Prints the available System RAM and GPU VRAM in a human-readable format."""
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... VRAM Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
            i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil * 100))

print("--- Initial Memory Status ---")
mem_report()

import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Using Qwen3 1.7 Billion parameters
model_id = "Qwen/Qwen3-1.7B"

# Load tokenizer with trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with eager attention and trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True # Bypass older transformers versions by trusting the model's own code
).to(device)

print("\nQwen3-1.7B successfully loaded on GPU!")
print("--- Memory Status After Loading Model ---")
mem_report()

# A curated list of examples to demonstrate modern tokenization
texts = [
    # 1. Basic English & Numbers (Notice how $ and numbers split)
    "Hello, tokenizer! Price: $19.99",

    # 2. Russian Language (Massive improvement over GPT-2, highly efficient)
    "Привет, мир! Ёжик съел 3 яблока.",

    # 3. Python Code (Notice how 4 spaces become a single token)
    "def fine_tune(model):\n    return True",

    # 4. Multilingual & Emoji (Proves the vocabulary is huge and multimodal)
    "Qwen3: Hello / Привет / 你好 🚀✨"
]

print("How a modern LLM sees text:\n" + "="*50)

for t in texts:
    token_ids = tokenizer.encode(t)
    # Decode each ID individually to show the exact text chunks
    readable_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    print(f"Text: {t}")
    print(f"Chunks ({len(readable_tokens)} pcs): {readable_tokens}")
    print(f"IDs: {token_ids}\n" + "-"*50)

logic_question = "Anna's mother has three daughters. The first is named Spring, the second is named Summer. What is the name of the third daughter?"

print("==========================================")
print(" TEST 1: FORCED INSTINCT (1-Word Answer)")
print("==========================================\n")
messages_1 = [
    {"role": "system", "content": "You are a fast assistant. Answer immediately in exactly one word. No explanations."},
    {"role": "user", "content": logic_question}
]
text_1 = tokenizer.apply_chat_template(messages_1,
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
inputs_1 = tokenizer([text_1], return_tensors="pt").to(device)

with torch.no_grad():
    out_ids_1 = model.generate(**inputs_1,
                               max_new_tokens=10,
                               do_sample=False,
                               eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out_ids_1[0][inputs_1.input_ids.shape[1]:], skip_special_tokens=True).strip())

print("\n\n==========================================")
print(" TEST 2: REASONING ALOUD (Classic CoT)")
print("==========================================\n")
messages_2 = [
    {"role": "user", "content": logic_question + " Please think step-by-step and then give your answer."}
]
text_2 = tokenizer.apply_chat_template(messages_2,
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
inputs_2 = tokenizer([text_2], return_tensors="pt").to(device)

with torch.no_grad():
    out_ids_2 = model.generate(**inputs_2,
                               max_new_tokens=1024,
                               do_sample=False,
                               eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out_ids_2[0][inputs_2.input_ids.shape[1]:], skip_special_tokens=True).strip())

print("\n\n==========================================")
print(" TEST 3: NATIVE THINKING (Hidden System 2)")
print("==========================================\n")
messages_3 = [
    {"role": "user", "content": logic_question}
]
# enable_thinking=True turns on the official Qwen3 internal monologue feature
text_3 = tokenizer.apply_chat_template(messages_3,
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)
inputs_3 = tokenizer([text_3], return_tensors="pt").to(device)

with torch.no_grad():
    out_ids_3 = model.generate(**inputs_3,
                               max_new_tokens=32768,
                               do_sample=False,
                               eos_token_id=tokenizer.eos_token_id)

output_ids_3 = out_ids_3[0][inputs_3.input_ids.shape[1]:].tolist()
try:
    index = len(output_ids_3) - output_ids_3[::-1].index(151668) # 151668 is </think>
except ValueError:
    index = 0

print("HIDDEN THOUGHT PROCESS:")
print(tokenizer.decode(output_ids_3[:index], skip_special_tokens=True).strip())
print("\nFINAL VISIBLE ANSWER:")
print(tokenizer.decode(output_ids_3[index:], skip_special_tokens=True).strip())

print("\n\n==========================================")
print(" TEST 4: STRUCTURED OUTPUT (JSON)")
print("==========================================\n")
messages_4 = [
    {"role": "system", "content": "You are an API endpoint. You MUST respond ONLY in valid JSON format containing two keys: 'reasoning' (your step-by-step logic) and 'answer' (the final name). Do not add any markdown blocks or extra text."},
    {"role": "user", "content": logic_question}
]
text_4 = tokenizer.apply_chat_template(messages_4,
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
inputs_4 = tokenizer([text_4], return_tensors="pt").to(device)

with torch.no_grad():
    out_ids_4 = model.generate(**inputs_4,
                               max_new_tokens=32768,
                               do_sample=False,
                               eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out_ids_4[0][inputs_4.input_ids.shape[1]:], skip_special_tokens=True).strip())

from pydantic import BaseModel
import outlines


class LogicAnswer(BaseModel):
    reasoning: str
    answer: str


print("\n\n==========================================")
print(" TEST 4: STRUCTURED OUTPUT (JSON) WITH OUTLINES")
print("==========================================\n")

outlines_model = outlines.from_transformers(model, tokenizer)

prompt_4 = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": (
                "You are an API endpoint. "
                "You MUST respond ONLY in valid JSON format containing two keys: "
                "'reasoning' and 'answer'. "
                "Do not add any markdown blocks or extra text."
            ),
        },
        {
            "role": "user",
            "content": logic_question,
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

raw_result_4 = outlines_model(
    prompt_4,
    LogicAnswer,
    max_new_tokens=512,
)

print("RAW:")
print(raw_result_4)

result_4 = LogicAnswer.model_validate_json(raw_result_4)

print("\nPARSED:")
print(result_4.model_dump())
print("answer:", result_4.answer)
print("reasoning:", result_4.reasoning)

from pydantic import BaseModel
import json
import torch


class LogicAnswer(BaseModel):
    reasoning: str
    answer: str

def generate_json_with_pydantic_validation(
    messages,
    response_format,
    model,
    tokenizer,
    device,
    max_new_tokens=512,
):
    schema = response_format.model_json_schema()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an API endpoint. "
                "Return ONLY valid JSON. "
                "Do not use markdown. "
                "Do not add text outside JSON. "
                "The JSON must match this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        }
    ] + messages

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(
        out_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print("RAW OUTPUT:")
    print(output_text)

    return response_format.model_validate_json(output_text)

print("\n\n==========================================")
print(" TEST 4: STRUCTURED OUTPUT (JSON)")
print("==========================================\n")

messages_4 = [
    {
        "role": "system",
        "content": (
            "You are an API endpoint. "
            "You MUST respond ONLY in valid JSON format containing two keys: "
            "'reasoning' and 'answer'. "
            "Do not add any markdown blocks or extra text."
        ),
    },
    {
        "role": "user",
        "content": logic_question,
    },
]

result_4 = generate_json_with_pydantic_validation(
    messages=messages_4,
    response_format=LogicAnswer,
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=512,
)

print("\nPARSED:")
print(result_4.model_dump())
print("answer:", result_4.answer)
print("reasoning:", result_4.reasoning)

# Enable output_attentions flag dynamically
model.config.output_attentions = True

text = "The bank of the river is beautiful, but the bank of Wall Street is rich."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Extract attention matrix (Layer 0, Head 0)
attention_matrix = outputs.attentions[0][0, 0, :, :].cpu().numpy()

# Decode each token ID to show actual text labels on the heatmap axes
tokens = [tokenizer.decode([tok_id]) for tok_id in inputs["input_ids"][0]]

plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues")
plt.title("Self-Attention Heatmap (Qwen3-1.7B: Layer 0, Head 0)")
plt.xticks(rotation=45)
plt.show()

# Disable to save memory for subsequent tasks
model.config.output_attentions = False

from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import torch


train_examples = [
    {
        "instruction": "Explain overfitting.",
        "answer": (
            "Name: Overfittosaur\n"
            "Type: Dragon / Statistics\n"
            "Ability: Memorize Everything\n"
            "Weakness: New unseen data\n"
            "Description: Overfittosaur learns the training data too perfectly and becomes weak on unseen data.\n"
            "Professor note: If training loss falls while validation loss rises, this creature has appeared.\n"
            "Mini example: A model gets 99% accuracy on training data but only 70% on validation data."
        ),
    },
    {
        "instruction": "Explain underfitting.",
        "answer": (
            "Name: Underfitto\n"
            "Type: Rock / Basics\n"
            "Ability: Too Simple\n"
            "Weakness: Complex patterns\n"
            "Description: Underfitto is too weak to learn the real structure in the data.\n"
            "Professor note: It performs badly on both training data and validation data.\n"
            "Mini example: A straight line tries to model a strongly curved relationship."
        ),
    },
    {
        "instruction": "Explain data leakage.",
        "answer": (
            "Name: Leakychu\n"
            "Type: Electric / Danger\n"
            "Ability: Forbidden Knowledge\n"
            "Weakness: Proper train-test separation\n"
            "Description: Leakychu sneaks future or target information into the training process.\n"
            "Professor note: It makes validation scores look magical but fake.\n"
            "Mini example: Feature selection is fitted on the full dataset before the train-test split."
        ),
    },
    {
        "instruction": "Explain train-test split.",
        "answer": (
            "Name: Splitmander\n"
            "Type: Fire / Evaluation\n"
            "Ability: Fair Trial\n"
            "Weakness: Data leakage\n"
            "Description: Splitmander separates examples into training data and unseen test data.\n"
            "Professor note: It helps estimate how well a model works outside the training set.\n"
            "Mini example: Train on 80% of the data and test once on the remaining 20%."
        ),
    },
    {
        "instruction": "Explain precision.",
        "answer": (
            "Name: Precisizard\n"
            "Type: Flying / Metrics\n"
            "Ability: Careful Strike\n"
            "Weakness: Missed positives\n"
            "Description: Precisizard checks how many predicted positives are truly positive.\n"
            "Professor note: It is important when false positives are expensive.\n"
            "Mini example: If 100 emails are flagged as spam and 90 are really spam, precision is 90%."
        ),
    },
    {
        "instruction": "Explain recall.",
        "answer": (
            "Name: Recallbasaur\n"
            "Type: Grass / Metrics\n"
            "Ability: Find Them All\n"
            "Weakness: False alarms\n"
            "Description: Recallbasaur checks how many real positive cases the model successfully catches.\n"
            "Professor note: It is important when false negatives are expensive.\n"
            "Mini example: In disease screening, recall measures how many sick patients were detected."
        ),
    },
    {
        "instruction": "Explain F1-score.",
        "answer": (
            "Name: F1nix\n"
            "Type: Fire / Metrics\n"
            "Ability: Balance Flame\n"
            "Weakness: Hidden trade-offs\n"
            "Description: F1nix balances precision and recall into one score.\n"
            "Professor note: It is useful when both false positives and false negatives matter.\n"
            "Mini example: If precision and recall are both 0.8, the F1-score is also 0.8."
        ),
    },
    {
        "instruction": "Explain cross-validation.",
        "answer": (
            "Name: Crossvalion\n"
            "Type: Psychic / Evaluation\n"
            "Ability: Many Trials\n"
            "Weakness: Data leakage inside folds\n"
            "Description: Crossvalion trains and validates the model across several data splits.\n"
            "Professor note: It gives a more stable estimate than one lucky split.\n"
            "Mini example: In 5-fold cross-validation, each fold becomes validation data once."
        ),
    },
    {
        "instruction": "Explain regularization.",
        "answer": (
            "Name: Regulax\n"
            "Type: Steel / Control\n"
            "Ability: Complexity Shield\n"
            "Weakness: Too much penalty\n"
            "Description: Regulax prevents the model from becoming too complex.\n"
            "Professor note: It helps fight overfitting by adding a penalty or constraint.\n"
            "Mini example: L2 regularization discourages very large model weights."
        ),
    },
    {
        "instruction": "Explain gradient descent.",
        "answer": (
            "Name: Gradientoise\n"
            "Type: Water / Optimization\n"
            "Ability: Downhill Step\n"
            "Weakness: Bad learning rate\n"
            "Description: Gradientoise moves model parameters step by step toward lower loss.\n"
            "Professor note: The learning rate controls how large each step is.\n"
            "Mini example: If the step is too large, training can jump over the minimum."
        ),
    },
    {
        "instruction": "Explain confusion matrix.",
        "answer": (
            "Name: Confusionmatrix\n"
            "Type: Psychic / Metrics\n"
            "Ability: Error Map\n"
            "Weakness: Too many classes without normalization\n"
            "Description: Confusionmatrix shows where a classifier is correct and where it mixes up classes.\n"
            "Professor note: It counts true positives, false positives, false negatives, and true negatives.\n"
            "Mini example: In fraud detection, false negatives are missed fraud cases."
        ),
    },
    {
        "instruction": "Explain learning rate.",
        "answer": (
            "Name: Learnirate\n"
            "Type: Electric / Optimization\n"
            "Ability: Step Size\n"
            "Weakness: Exploding or frozen training\n"
            "Description: Learnirate controls how big each update step is during training.\n"
            "Professor note: Too high can explode training; too low can make learning painfully slow.\n"
            "Mini example: With a very high learning rate, loss may jump around instead of decreasing."
        ),
    },
    {
        "instruction": "Explain feature scaling.",
        "answer": (
            "Name: Scalix\n"
            "Type: Normal / Preprocessing\n"
            "Ability: Equal Footing\n"
            "Weakness: Unscaled numeric ranges\n"
            "Description: Scalix rescales features so large-number columns do not dominate smaller-number columns.\n"
            "Professor note: It is especially useful for distance-based and gradient-based models.\n"
            "Mini example: Age ranges from 0 to 100, while income may range from 0 to 200000."
        ),
    },
    {
        "instruction": "Explain classification threshold.",
        "answer": (
            "Name: Thresholdon\n"
            "Type: Steel / Decision\n"
            "Ability: Cutoff Gate\n"
            "Weakness: Poorly chosen business trade-off\n"
            "Description: Thresholdon turns predicted probabilities into class labels using a chosen cutoff.\n"
            "Professor note: Moving the threshold changes precision and recall.\n"
            "Mini example: Lowering the fraud threshold catches more fraud but may block more good users."
        ),
    },
    {
        "instruction": "Explain ROC AUC.",
        "answer": (
            "Name: Rocodactyl\n"
            "Type: Flying / Metrics\n"
            "Ability: Ranking Wing\n"
            "Weakness: Heavy class imbalance\n"
            "Description: Rocodactyl measures how well a model ranks positive examples above negative examples.\n"
            "Professor note: It can look optimistic on heavily imbalanced datasets.\n"
            "Mini example: A high ROC AUC means positives usually receive higher scores than negatives."
        ),
    },
    {
        "instruction": "Explain PR AUC.",
        "answer": (
            "Name: Praucachu\n"
            "Type: Electric / Imbalance\n"
            "Ability: Rare Class Radar\n"
            "Weakness: Easy negative examples\n"
            "Description: Praucachu summarizes precision-recall performance across thresholds.\n"
            "Professor note: It is often more informative than ROC AUC for imbalanced classification.\n"
            "Mini example: In rare fraud detection, PR AUC focuses more on the positive fraud class."
        ),
    },
]

# VECDEX trigger augmentation.
# The base model does not know what "VECDEX" means.
# LoRA should learn that VECDEX means: answer as a rich ML creature card.

trigger_prefixes = [
    "VECDEX: ",
    "Use VECDEX format. ",
    "Answer in VECDEX style. ",
    "Convert this ML concept into VECDEX: ",
]

augmented_examples = []

for ex in train_examples:
    for prefix in trigger_prefixes:
        augmented_examples.append(
            {
                "instruction": prefix + ex["instruction"],
                "answer": ex["answer"],
            }
        )

train_examples.extend(augmented_examples)

print("Number of training examples:", len(train_examples))

def make_chat_text(instruction: str, answer: str | None = None) -> str:
    """
    Build chat-formatted text for instruction fine-tuning.

    Important:
    We intentionally keep the system prompt neutral.
    The model should learn the VECDEX behavior from the training examples,
    not from an explicit schema hidden inside the system prompt.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful machine learning teacher.",
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None),
        enable_thinking=False,
    )

class TinyPokemonMLDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        prompt_text = make_chat_text(ex["instruction"], answer=None)
        full_text = make_chat_text(ex["instruction"], answer=ex["answer"])

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = TinyPokemonMLDataset(train_examples, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_batch,
)

print("Number of examples in dataset:", len(train_dataset))
print("Number of batches:", len(train_loader))

vecdex_count = sum(
    "VECDEX" in ex["instruction"]
    for ex in train_dataset.examples
)

print("VECDEX examples in dataset:", vecdex_count)

print("\nSample dataset instructions:")
for ex in train_dataset.examples[:8]:
    print("-", ex["instruction"])

sample_idx = next(
    i for i, ex in enumerate(train_dataset.examples)
    if ex["instruction"].startswith("VECDEX:")
)

ex = train_dataset.examples[sample_idx]

print("RAW EXAMPLE")
print("Instruction:", ex["instruction"])
print("Answer:")
print(ex["answer"])

print("\nFORMATTED TRAINING TEXT")
print(make_chat_text(ex["instruction"], answer=ex["answer"])[:2000])

import gc
import torch

if "peft_model" in globals():
    print("Existing PEFT model found. Unloading adapter first...")
    model = peft_model.unload()
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("Ready for a fresh LoRA adapter.")

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import torch

# Improved configuration targeting more layers
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, config).to(device)

print("Parameter count difference:")
peft_model.print_trainable_parameters()

optimizer = AdamW(peft_model.parameters(), lr=2e-4)

print("Starting improved ML-creature LoRA fine-tuning...")
peft_model.train()

num_epochs = 6

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = peft_model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1:02d}/{num_epochs}, Loss: {avg_loss:.6f}")

print("Training complete with expanded targets!")

peft_model.eval()

quick_test_questions = [
    "Explain overfitting.",
    "Explain data leakage.",
    "Explain confusion matrix.",
    "Explain learning rate.",
    "Explain feature scaling.",
]

print("LoRA adapter enabled.")
print("Neutral system prompt + short VECDEX trigger.")
print("This checks whether the adapter learned the creature-card behavior from training examples.\n")

for question in quick_test_questions:
    prompt = make_chat_text("VECDEX: " + question, answer=None)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = peft_model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print("=" * 80)
    print("Question:", question)
    print(answer)

import re
import pandas as pd

REQUIRED_CREATURE_FIELDS = [
    "name",
    "type",
    "ability",
    "weakness",
    "description",
    "professor note",
    "mini example",
]


def creature_card_score(text: str) -> int:
    """
    Toy structure metric.

    It checks whether the answer contains the required creature-card fields.
    It does NOT check factual correctness.
    """
    text_lower = text.lower()
    score = 0

    for field in REQUIRED_CREATURE_FIELDS:
        pattern = rf"\b{re.escape(field)}\b\s*:"
        if re.search(pattern, text_lower):
            score += 1

    return score


def eval_creature_answer(text: str, concept: str) -> dict:
    """
    Slightly richer toy evaluation.

    Still not a real semantic evaluation.
    It checks:
    - creature-card structure
    - whether the concept is mentioned
    - answer length
    """
    clean_concept = (
        concept.lower()
        .replace("explain", "")
        .replace(".", "")
        .strip()
    )

    text_lower = text.lower()
    word_count = len(text.split())

    return {
        "field_score": creature_card_score(text),
        "max_field_score": len(REQUIRED_CREATURE_FIELDS),
        "mentions_concept": clean_concept in text_lower,
        "word_count": word_count,
        "too_short": word_count < 40,
    }


def make_neutral_vecdex_prompt(instruction: str) -> str:
    """
    Fair evaluation prompt.

    The system prompt does NOT explain the creature-card schema.
    The only signal is the VECDEX trigger.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful machine learning teacher.",
        },
        {
            "role": "user",
            "content": "VECDEX: " + instruction,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def make_explicit_schema_prompt(instruction: str) -> str:
    """
    Strong prompt-engineering baseline.

    This is intentionally NOT a fair test of whether LoRA learned the schema.
    It tests whether the base model can follow an explicit formatting instruction.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Professor Vector. "
                "Answer as an ML creature card. "
                "Use exactly these fields: Name, Type, Ability, Weakness, "
                "Description, Professor note, Mini example."
            ),
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_answer_from_prompt(current_model, prompt: str, max_new_tokens: int = 180) -> str:
    current_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = current_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_neutral_vecdex_answer(current_model, question: str, max_new_tokens: int = 180) -> str:
    prompt = make_neutral_vecdex_prompt(question)
    return generate_answer_from_prompt(
        current_model=current_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def generate_explicit_schema_answer(current_model, question: str, max_new_tokens: int = 180) -> str:
    prompt = make_explicit_schema_prompt(question)
    return generate_answer_from_prompt(
        current_model=current_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


# Single-question sanity check
question = "Explain confusion matrix."

print("QUESTION:", question)

print("\nBASE MODEL: adapter disabled, neutral VECDEX prompt")
with peft_model.disable_adapter():
    base_neutral_answer = generate_neutral_vecdex_answer(peft_model, question)

print(base_neutral_answer)
print(eval_creature_answer(base_neutral_answer, question))

print("\nLoRA MODEL: adapter enabled, neutral VECDEX prompt")
lora_neutral_answer = generate_neutral_vecdex_answer(peft_model, question)

print(lora_neutral_answer)
print(eval_creature_answer(lora_neutral_answer, question))

print("\nBASE MODEL: adapter disabled, explicit schema prompt")
with peft_model.disable_adapter():
    base_prompted_answer = generate_explicit_schema_answer(peft_model, question)

print(base_prompted_answer)
print(eval_creature_answer(base_prompted_answer, question))

seen_test_questions = [
    "Explain overfitting.",
    "Explain data leakage.",
    "Explain confusion matrix.",
    "Explain learning rate.",
    "Explain feature scaling.",
]

unseen_test_questions = [
    "Explain kernel trick.",
    "Explain ElasticNet.",
    "Explain SVD.",
    "Explain PCA.",
    "Explain dropout.",
    "Explain batch normalization.",
    "Explain word embeddings.",
    "Explain attention mechanism.",
]

stress_test_questions = [
    "Explain Grokking.",
    "Explain double descent.",
    "Explain calibration.",
]


def run_three_way_eval(question: str) -> list[dict]:
    rows = []

    # 1. Base model, neutral prompt
    with peft_model.disable_adapter():
        answer = generate_neutral_vecdex_answer(peft_model, question)

    metrics = eval_creature_answer(answer, question)
    rows.append(
        {
            "question": question,
            "setting": "BASE + neutral VECDEX prompt",
            "answer": answer,
            **metrics,
        }
    )

    # 2. LoRA model, neutral prompt
    answer = generate_neutral_vecdex_answer(peft_model, question)

    metrics = eval_creature_answer(answer, question)
    rows.append(
        {
            "question": question,
            "setting": "LoRA + neutral VECDEX prompt",
            "answer": answer,
            **metrics,
        }
    )

    # 3. Base model, explicit schema prompt
    with peft_model.disable_adapter():
        answer = generate_explicit_schema_answer(peft_model, question)

    metrics = eval_creature_answer(answer, question)
    rows.append(
        {
            "question": question,
            "setting": "BASE + explicit schema prompt",
            "answer": answer,
            **metrics,
        }
    )

    return rows


all_rows = []

eval_groups = {
    "SEEN CONCEPTS: concepts used during fine-tuning": seen_test_questions,
    "UNSEEN CONCEPTS: concepts not used during fine-tuning": unseen_test_questions,
    "STRESS TEST: harder or less standard concepts": stress_test_questions,
}

for group_name, questions in eval_groups.items():
    print("\n" + "-" * 100)
    print(group_name)
    print("-" * 100)

    for question in questions:
        print("\n" + "=" * 100)
        print("QUESTION:", question)

        rows = run_three_way_eval(question)
        all_rows.extend(rows)

        for row in rows:
            print("\n---", row["setting"], "---")
            print(row["answer"])
            print('...')
            print(
                ">> field_score:", row["field_score"], "/", row["max_field_score"],
                "| mentions_concept:", row["mentions_concept"],
                "| word_count:", row["word_count"],
                "| too_short:", row["too_short"],
            )


eval_df = pd.DataFrame(all_rows)

summary_df = (
    eval_df
    .groupby("setting")
    .agg(
        mean_field_score=("field_score", "mean"),
        concept_mention_rate=("mentions_concept", "mean"),
        mean_word_count=("word_count", "mean"),
        too_short_rate=("too_short", "mean"),
    )
    .reset_index()
)

print("\n" + "#" * 100)
print("SUMMARY")
print("#" * 100)
display(summary_df)

manual_review_df = eval_df[
    [
        "question",
        "setting",
        "field_score",
        "mentions_concept",
        "word_count",
        "answer",
    ]
].copy()

pd.set_option("display.max_colwidth", 500)
display(manual_review_df)

# Simple and practical cleanup cell for this notebook
# Run this cell after training/evaluation if Colab RAM or GPU memory is low.

CLEAN_GPU = False

import gc
import os
import torch


def report_memory():
    print("PID:", os.getpid())

    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"CPU RAM Free: {ram.available / 1024**3:.1f} GB")
        print(f"CPU RAM Used: {ram.used / 1024**3:.1f} GB")
        print(f"CPU RAM Total: {ram.total / 1024**3:.1f} GB")
    except Exception:
        print("psutil is not available, CPU RAM report skipped.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        free, total = torch.cuda.mem_get_info()

        print(
            f"GPU PyTorch allocated: {allocated:.0f} MB\n"
            f"GPU PyTorch reserved:  {reserved:.0f} MB\n"
            f"GPU VRAM Free:         {free / 1024**2:.0f} MB / {total / 1024**2:.0f} MB"
        )
    else:
        print("CUDA is not available.")


def cleanup_notebook_memory():
    """
    Simple cleanup for this notebook.

    Deletes heavy variables from training, LoRA, generation, evaluation,
    attention visualization, and metrics sections.

    Keeps tokenizer, helper functions, and source examples.
    """

    print("\n--- BEFORE CLEANUP ---")
    report_memory()

    names_to_delete = [
        # Main models
        "model",
        "base_model",
        "peft_model",
        "merged_model",
        "outlines_model",

        # Training
        "optimizer",
        "outputs",
        "loss",
        "batch",
        "train_dataset",
        "train_loader",
        "dataset",
        "dataloader",

        # Inference tensors / generated outputs
        "inputs",
        "input_ids",
        "attention_mask",
        "out_ids",
        "output_ids",
        "generated_ids",
        "output_ids_1",
        "output_ids_2",
        "output_ids_3",
        "output_ids_4",
        "out_ids_1",
        "out_ids_2",
        "out_ids_3",
        "out_ids_4",
        "inputs_1",
        "inputs_2",
        "inputs_3",
        "inputs_4",

        # Answers from eval
        "answer",
        "base_answer",
        "lora_answer",
        "base_neutral_answer",
        "lora_neutral_answer",
        "base_prompted_answer",

        # Eval collections and DataFrames
        "rows",
        "all_rows",
        "eval_df",
        "summary_df",
        "technical_eval_df",
        "technical_summary_df",
        "manual_review_df",
        "problematic_answers_df",

        # Attention visualization
        "attention_matrix",
        "attentions",

        # Misc old variables from earlier notebook versions
        "raw_result_4",
        "result_4",
    ]

    deleted = []

    for name in names_to_delete:
        if name in globals():
            try:
                del globals()[name]
                deleted.append(name)
            except Exception as e:
                print(f"Could not delete {name}: {e}")

    print("\nDeleted variables:")
    print(", ".join(deleted) if deleted else "Nothing from the cleanup list was found.")

    # Python garbage collection
    for _ in range(5):
        gc.collect()

    # CUDA cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()

    print("\n--- AFTER CLEANUP ---")
    report_memory()


if CLEAN_GPU:
    cleanup_notebook_memory()

mem_report()

# # Diagnose remaining CUDA tensors / CUDA modules in notebook globals

# import gc
# import torch
# import pandas as pd

# def diagnose_cuda_globals():
#     rows = []

#     for name, obj in list(globals().items()):
#         try:
#             # Direct CUDA tensor
#             if isinstance(obj, torch.Tensor) and obj.is_cuda:
#                 rows.append({
#                     "name": name,
#                     "type": type(obj).__name__,
#                     "kind": "cuda_tensor",
#                     "shape": tuple(obj.shape),
#                     "dtype": str(obj.dtype),
#                     "device": str(obj.device),
#                     "size_mb": obj.numel() * obj.element_size() / 1024**2,
#                 })

#             # PyTorch module with CUDA parameters
#             elif isinstance(obj, torch.nn.Module):
#                 param_bytes = 0
#                 cuda_params = 0

#                 for p in obj.parameters(recurse=True):
#                     if p.is_cuda:
#                         cuda_params += p.numel()
#                         param_bytes += p.numel() * p.element_size()

#                 if cuda_params > 0:
#                     rows.append({
#                         "name": name,
#                         "type": type(obj).__name__,
#                         "kind": "torch_module_with_cuda_params",
#                         "shape": "module",
#                         "dtype": "module",
#                         "device": "cuda",
#                         "size_mb": param_bytes / 1024**2,
#                     })

#         except Exception as e:
#             pass

#     if len(rows) == 0:
#         print("No CUDA tensors or CUDA modules found in globals().")
#         print("If torch.cuda.memory_allocated() is still high, memory may be held by:")
#         print("- IPython output cache")
#         print("- previous cell result variables: _, __, ___")
#         print("- traceback / exception state")
#         print("- closures or objects not visible directly in globals()")
#         return pd.DataFrame(columns=[
#             "name", "type", "kind", "shape", "dtype", "device", "size_mb"
#         ])

#     df = pd.DataFrame(rows)
#     df = df.sort_values("size_mb", ascending=False)

#     display(df)
#     return df


# cuda_globals_df = diagnose_cuda_globals()

# # Brutal cleanup for hidden CUDA references in Jupyter / Colab

# import gc
# import sys
# import torch
# import traceback

# print("--- BEFORE ---")
# if torch.cuda.is_available():
#     print("allocated MB:", torch.cuda.memory_allocated() / 1024**2)
#     print("reserved MB: ", torch.cuda.memory_reserved() / 1024**2)

# # 1. Clear Python exception / traceback references
# try:
#     sys.last_type = None
#     sys.last_value = None
#     sys.last_traceback = None
# except Exception:
#     pass

# try:
#     traceback.clear_frames(sys.exc_info()[2])
# except Exception:
#     pass

# # 2. Clear IPython last-result variables and output cache
# try:
#     ip = get_ipython()

#     for name in ["_", "__", "___", "_i", "_ii", "_iii", "Out"]:
#         ip.user_ns.pop(name, None)

#     # Clear numbered output cache if present
#     if hasattr(ip, "user_ns") and "Out" in ip.user_ns:
#         try:
#             ip.user_ns["Out"].clear()
#         except Exception:
#             pass

#     # Flush displayhook cache
#     if hasattr(ip, "displayhook"):
#         try:
#             ip.displayhook.flush()
#         except Exception:
#             pass

# except Exception as e:
#     print("Could not fully clear IPython cache:", e)

# # 3. Delete visible CUDA globals again, just in case
# deleted_globals = []

# for name, obj in list(globals().items()):
#     try:
#         if isinstance(obj, torch.Tensor) and obj.is_cuda:
#             del globals()[name]
#             deleted_globals.append(name)
#         elif isinstance(obj, torch.nn.Module):
#             if any(p.is_cuda for p in obj.parameters(recurse=True)):
#                 del globals()[name]
#                 deleted_globals.append(name)
#     except Exception:
#         pass

# print("Deleted CUDA globals:", deleted_globals if deleted_globals else "none")

# # 4. Force garbage collection
# for _ in range(10):
#     gc.collect()

# # 5. CUDA allocator cleanup
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     try:
#         torch.cuda.ipc_collect()
#     except Exception:
#         pass
#     torch.cuda.synchronize()

# print("\n--- AFTER ---")
# if torch.cuda.is_available():
#     print("allocated MB:", torch.cuda.memory_allocated() / 1024**2)
#     print("reserved MB: ", torch.cuda.memory_reserved() / 1024**2)
#     free, total = torch.cuda.mem_get_info()
#     print(f"GPU VRAM Free: {free / 1024**2:.0f} MB / {total / 1024**2:.0f} MB")

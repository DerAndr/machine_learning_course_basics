# Auto-generated companion script for nlp_overview_practical_student_90min.ipynb
# Keep the notebook as the source of truth.

# %% [markdown]
# # NLP Overview Before LLMs (90 minutes)
#
# **Goal:** build intuition for NLP before LLM applications by moving from raw documents and corpus EDA to sparse text features, classical classification, dense sentence embeddings, transformer encoders, NLP metrics, and finally a clean bridge to decoder LLMs.
#
# **Learning objectives:**
#
# - treat text classification as a supervised ML workflow with documents, labels, splits, corpus diagnostics, and leakage risks;
# - explain tokenization, stop words, n-grams, count vectors, and TF-IDF;
# - compare classical TF-IDF baselines: Naive Bayes, logistic regression, and linear SVM;
# - train and inspect a small Word2Vec model as the bridge from sparse lexical vectors to dense embeddings;
# - build small analysis cards for model errors instead of stopping at accuracy;
# - use a pretrained sentence-transformer checkpoint as a GPU-aware embedding extractor;
# - explain transformer encoder basics: subword tokens, attention, and contextual representations;
# - use modern NLP packages directly: `datasets`, `tokenizers`, `sentence-transformers`, and `evaluate`;
# - choose metrics that match the output shape: labels, rankings, probabilities, spans, or generated text;
# - explain entropy, cross-entropy, and perplexity with small numeric examples;
# - distinguish classification, retrieval, clustering, NER, summarization, translation, and QA task shapes;
# - explain why decoder LLM applications belong in the next block.
#
# **Practical note:** some cells are guided demos, and some contain TODOs for you to complete during class.
#
# **Agenda:**
#
# | Part | Topic | Time |
# |------|-------|------|
# | 1 | Text as data, corpus EDA, and leakage risks | ~20 min |
# | 2 | Tokenization and sparse features | ~15 min |
# | 3 | Classical text classification baseline | ~18 min |
# | 4 | Sparse similarity, Word2Vec, and dense embeddings | ~18 min |
# | 5 | Transformer encoder intuition | ~7 min |
# | 6 | NLP metrics map | ~8 min |
# | 7 | NLP task map | ~2 min |
# | 8 | Bridge to LLMs | ~2 min |

# %% [markdown]
# ## Setup
#
# The notebook is designed for Google Colab. It also runs locally in this repository if the NLP dependency group is installed.
#
# Local setup:
#
# ```bash
# uv sync --group nlp
# uv run jupyter lab
# ```
#
# Colab setup is automatic: the first cell checks for required packages and installs missing ones only when it detects Colab.
#
# For the dense embedding section, use a GPU runtime in Colab: **Runtime -> Change runtime type -> T4 GPU**. The classical NLP cells run on CPU, but the sentence-transformer embedding demo will move the model and tensors to CUDA when a T4 or another GPU is available. The embedding and tokenizer demos download Hugging Face model files and can be disabled with flags.

# %%
import importlib.util
import os
import subprocess
import sys

IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

required_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "datasets",
    "evaluate",
    "gensim",
    "sentence_transformers",
    "tokenizers",
    "torch",
    "transformers",
]
pip_names = {
    "sklearn": "scikit-learn",
    "sentence_transformers": "sentence-transformers",
}
missing = [pkg for pkg in required_imports if importlib.util.find_spec(pkg) is None]

if missing and IN_COLAB:
    to_install = [pip_names.get(pkg, pkg) for pkg in missing]
    print("Colab detected. Installing:", ", ".join(to_install))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
elif missing:
    print("Missing packages:", ", ".join(missing))
    print("Install locally with: uv sync --group nlp")
else:
    print("All required packages are available.")

# %%
import random
import re
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.mideast",
]

TRAIN_LIMIT = 900
TEST_LIMIT = 360
SEARCH_LIMIT = 140
EMBEDDING_LIMIT = 120

RUN_SENTENCE_TRANSFORMER_DEMO = True
RUN_TOKENIZER_DEMO = True
RUN_HF_DATASETS_DEMO = IN_COLAB
RUN_HF_EVALUATE_DEMO = IN_COLAB

SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL_NAME = "bert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"Embedding device: {DEVICE} ({torch.cuda.get_device_name(0)})")
elif IN_COLAB:
    print("Embedding device: CPU. For the embedding demo, switch Colab runtime to T4 GPU.")
else:
    print("Embedding device: CPU")

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": False,
})
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Helper functions
#
# These helpers keep the lesson cells focused on NLP ideas. The most important implementation work in the student notebook is tokenization inspection, vectorization, baseline comparison, feature inspection, error analysis, and nearest-neighbor search.

# %%
def preview_text(text, width=90, lines=7):
    cleaned = re.sub(r"\s+", " ", text).strip()
    wrapped = textwrap.wrap(cleaned, width=width)
    return "\n".join(wrapped[:lines])


def balanced_sample_frame(texts, targets, limit, random_state=RANDOM_STATE):
    frame = pd.DataFrame({"text": texts, "target": targets})
    frame = frame[frame["text"].str.strip().str.len() > 0].copy()
    per_class = max(1, limit // frame["target"].nunique())
    parts = []
    for _, group in frame.groupby("target", sort=False):
        parts.append(group.sample(min(len(group), per_class), random_state=random_state))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def fallback_newsgroups_like_data():
    examples = {
        "comp.graphics": [
            "The rendering pipeline uses texture maps, polygons, lighting, and image buffers.",
            "A graphics card accelerates 3D scenes and rasterizes triangles for display.",
            "Vector graphics store shapes as coordinates while raster images store pixels.",
            "The workstation needs better OpenGL drivers for the animation software.",
            "Anti aliasing smooths jagged edges in computer-generated images.",
            "Image compression changes quality, color, and file size for graphics workflows.",
        ],
        "rec.sport.baseball": [
            "The pitcher threw seven innings and the bullpen protected the lead.",
            "Baseball fans debated the batting average and home run record all season.",
            "The catcher called for a curveball with two strikes and runners on base.",
            "Spring training gives rookies a chance to make the major league roster.",
            "The team won after a double, a stolen base, and a sacrifice fly.",
            "The manager changed pitchers before the ninth inning rally.",
        ],
        "sci.space": [
            "The spacecraft entered orbit after the launch vehicle completed its burn.",
            "NASA engineers tested the satellite antenna before the mission window.",
            "Astronauts train for microgravity, docking, and emergency procedures.",
            "The telescope observed galaxies, nebulae, and distant stars.",
            "A Mars rover needs solar power, navigation software, and thermal protection.",
            "Rocket stages separate as the payload climbs beyond the atmosphere.",
        ],
        "talk.politics.mideast": [
            "Diplomats discussed borders, security guarantees, and regional negotiations.",
            "The peace talks focused on elections, settlements, and international observers.",
            "Analysts disagreed about sanctions, sovereignty, and military escalation.",
            "The minister addressed parliament after renewed conflict in the region.",
            "Humanitarian groups reported displacement, aid shortages, and ceasefire violations.",
            "The debate included history, national identity, and competing political claims.",
        ],
    }
    rows = []
    for target, label in enumerate(CATEGORIES):
        for text in examples[label]:
            rows.append({"text": text, "target": target})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    train, test = train_test_split(
        frame,
        test_size=0.35,
        random_state=RANDOM_STATE,
        stratify=frame["target"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), CATEGORIES


def load_newsgroups_data():
    try:
        train_raw = fetch_20newsgroups(
            subset="train",
            categories=CATEGORIES,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        test_raw = fetch_20newsgroups(
            subset="test",
            categories=CATEGORIES,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train = balanced_sample_frame(train_raw.data, train_raw.target, TRAIN_LIMIT)
        test = balanced_sample_frame(test_raw.data, test_raw.target, TEST_LIMIT)
        return train, test, list(train_raw.target_names), "20 Newsgroups"
    except Exception as exc:
        print("Could not load 20 Newsgroups; using tiny fallback corpus.")
        print(type(exc).__name__, str(exc)[:200])
        train, test, target_names = fallback_newsgroups_like_data()
        return train, test, target_names, "fallback corpus"


def top_tfidf_terms_for_document(vectorizer, matrix, row_index, n=12):
    row = matrix[row_index]
    if row.nnz == 0:
        return pd.DataFrame(columns=["term", "tfidf"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(row.data)[::-1][:n]
    return pd.DataFrame({
        "term": feature_names[row.indices[order]],
        "tfidf": row.data[order],
    })

# %% [markdown]
# # Part 1. Text as data and corpus EDA
#
# NLP still starts with the same supervised-learning questions as tabular ML:
#
# - What is one example? Here, one newsgroup post.
# - What is the target? Here, the newsgroup category.
# - What belongs in train vs test?
# - What does the corpus look like before feature engineering?
# - What information leaks the label too easily?
#
# 20 Newsgroups is useful for teaching because headers and quoted replies can leak topic identity. The main workflow removes headers, footers, and quotes so the model relies more on message content.

# %%
train_df, test_df, target_names, DATA_SOURCE = load_newsgroups_data()

train_df["label"] = train_df["target"].map(lambda i: target_names[i])
test_df["label"] = test_df["target"].map(lambda i: target_names[i])

print(f"Training documents: {len(train_df):,}")
print(f"Test documents: {len(test_df):,}")
print("Classes:", target_names)
print("Data source:", DATA_SOURCE)

TFIDF_MIN_DF = 1 if len(train_df) < 100 else 2
print("TF-IDF min_df:", TFIDF_MIN_DF)

split_profile = (
    pd.concat([
        train_df.assign(split="train"),
        test_df.assign(split="test"),
    ])
    .groupby(["split", "label"])
    .size()
    .rename("documents")
    .reset_index()
)
split_profile["share_within_split"] = (
    split_profile["documents"]
    / split_profile.groupby("split")["documents"].transform("sum")
)
display(
    split_profile.pivot(index="label", columns="split", values="documents")
    .fillna(0)
    .astype(int)
)
display(split_profile.assign(share_within_split=lambda df: df["share_within_split"].round(3)))

# %%
def compute_text_diagnostics_batch(batch):
    texts = [text or "" for text in batch["text"]]
    tokens = [re.findall(r"(?u)\b\w+\b", text) for text in texts]
    alphabetic_tokens = [re.findall(r"(?u)\b[a-z][a-z]+\b", text.lower()) for text in texts]

    return {
        "char_count": [len(text) for text in texts],
        "line_count": [text.count("\n") + 1 for text in texts],
        "word_count": [len(words) for words in tokens],
        "unique_word_count": [len(set(words)) for words in alphabetic_tokens],
        "avg_word_length": [
            float(np.mean([len(word) for word in words])) if words else 0.0
            for words in tokens
        ],
        "empty_or_tiny": [len(words) < 5 for words in tokens],
    }


def add_text_diagnostics_with_pandas(frame):
    diagnostics = pd.DataFrame(compute_text_diagnostics_batch({"text": frame["text"].tolist()}))
    return pd.concat([frame.reset_index(drop=True), diagnostics], axis=1)


if RUN_HF_DATASETS_DEMO:
    try:
        from datasets import Dataset

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        train_dataset = train_dataset.map(
            compute_text_diagnostics_batch,
            batched=True,
            desc="Adding text diagnostics to train split",
        )
        test_dataset = test_dataset.map(
            compute_text_diagnostics_batch,
            batched=True,
            desc="Adding text diagnostics to test split",
        )

        train_df = train_dataset.to_pandas()
        test_df = test_dataset.to_pandas()
        print(train_dataset)
    except Exception as exc:
        print("Hugging Face datasets demo skipped:", type(exc).__name__, str(exc)[:160])
        train_df = add_text_diagnostics_with_pandas(train_df)
        test_df = add_text_diagnostics_with_pandas(test_df)
else:
    print("Hugging Face datasets demo skipped. Set RUN_HF_DATASETS_DEMO = True in Colab to run it.")
    train_df = add_text_diagnostics_with_pandas(train_df)
    test_df = add_text_diagnostics_with_pandas(test_df)

eda_columns = [
    "char_count",
    "line_count",
    "word_count",
    "unique_word_count",
    "avg_word_length",
]

display(train_df[eda_columns + ["empty_or_tiny"]].describe().round(1))
display(
    train_df.groupby("label")[eda_columns]
    .agg(["median", "mean"])
    .round(1)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(train_df["word_count"].clip(upper=400), bins=35, ax=axes[0])
axes[0].set_title("Training document length distribution")
axes[0].set_xlabel("word count, clipped at 400")

sns.boxplot(
    data=train_df,
    x="word_count",
    y="label",
    hue="label",
    legend=False,
    ax=axes[1],
)
axes[1].set_xlim(0, min(500, max(50, train_df["word_count"].quantile(0.95))))
axes[1].set_title("Length by class")
axes[1].set_xlabel("word count, zoomed to 95th percentile")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()

# %%
for label in target_names:
    example = train_df[train_df["label"] == label].iloc[0]
    print("=" * 90)
    print(label)
    print("-" * 90)
    print(preview_text(example["text"]))

# %% [markdown]
# ### EDA task: inspect corpus quality before modeling
#
# Before choosing a vectorizer, look for practical issues that can distort a text model:
#
# - very short or empty documents;
# - classes with unusually long documents;
# - repeated vocabulary that may become a shortcut;
# - visible metadata, quotes, or formatting artifacts.
#
# This is the NLP version of tabular EDA: we are checking what the rows actually contain before we engineer features.

# %%
# TODO: Build a small EDA watchlist before modeling.
# 1. Show up to 8 very short documents with label, word_count, and a text preview.
# 2. Build length_profile by label with: documents, median_words, p90_words, tiny_documents.
# 3. Use the output to discuss whether any class or document type needs attention.

# short_docs = ...
# short_docs["preview"] = ...
# display(short_docs[["label", "word_count", "preview"]])

# length_profile = ...
# display(length_profile)

# %% [markdown]
# ### Vocabulary EDA: what words dominate each class?
#
# Now inspect the corpus before training a classifier. This is not yet a model explanation. It is a dataset check: which words are frequent inside each class, and do they look like topic evidence or artifacts?

# %%
# TODO: Inspect frequent terms by class before training the classifier.
# Build a CountVectorizer with English stop words and alphabetic tokens, then:
# - sum token counts inside each class;
# - show the top 12 terms per class;
# - compare these terms with the overall top terms.

# eda_vectorizer = CountVectorizer(...)
# eda_counts = eda_vectorizer.fit_transform(...)
# eda_terms = np.array(...)
# top_term_rows = []
# for class_index, class_name in enumerate(target_names):
#     ...
# top_terms_by_label = pd.DataFrame(top_term_rows)
# display(top_terms_by_label.pivot(index="rank", columns="label", values="term"))

# %% [markdown]
# ## Leakage check
#
# The cell below loads a tiny sample with headers included. This is not the dataset we train on. It is a teaching contrast: author names, organizations, email domains, and quoted reply markers can become shortcuts that inflate scores without teaching the model the real topic language.

# %%
if DATA_SOURCE != "20 Newsgroups":
    print("Header contrast skipped because the fallback corpus has no message headers.")
else:
    try:
        with_headers = fetch_20newsgroups(
            subset="train",
            categories=CATEGORIES,
            remove=(),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        print(preview_text(with_headers.data[0], lines=12))
    except Exception as exc:
        print("Header contrast skipped because the dataset is unavailable:", type(exc).__name__)

# %% [markdown]
# # Part 2. Tokenization and sparse features
#
# A classical text model cannot consume raw strings. A vectorizer defines the model's vocabulary and turns every document into a vector.
#
# First, inspect a small count-vector example. Then we will switch to TF-IDF for the real classifier.

# %%
sample_docs = train_df["text"].head(4).tolist()

count_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    max_features=25,
)
sample_counts = count_vectorizer.fit_transform(sample_docs)

count_table = pd.DataFrame(
    sample_counts.toarray(),
    columns=count_vectorizer.get_feature_names_out(),
)
display(count_table)
print("Shape:", sample_counts.shape)

# %% [markdown]
# ### Modern tokenizer demo with `tokenizers`
#
# `CountVectorizer` tokenization is enough for classical baselines, but modern NLP libraries often use trained subword tokenizers. Here we train a tiny WordPiece tokenizer on the classroom corpus. This is deliberately small and local: the point is to see how a tokenizer learns a vocabulary and can split unfamiliar words into pieces.

# %%
wordpiece_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wordpiece_tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
])
wordpiece_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
wordpiece_trainer = trainers.WordPieceTrainer(
    vocab_size=120,
    min_frequency=1,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
)
wordpiece_tokenizer.train_from_iterator(train_df["text"].tolist(), trainer=wordpiece_trainer)

wordpiece_examples = [
    "spacecraft orbiting earth",
    "graphics pipeline rasterizes polygons",
    "baseball postseason negotiations",
    "unfamiliar-tokenization-example",
]
wordpiece_rows = []
for text in wordpiece_examples:
    encoding = wordpiece_tokenizer.encode(text)
    wordpiece_rows.append({
        "text": text,
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        "token_count": len(encoding.tokens),
    })

display(pd.DataFrame(wordpiece_rows))
print("Tiny WordPiece vocab size:", wordpiece_tokenizer.get_vocab_size())

# %% [markdown]
# ### Task: inspect what the tokenizer actually sees
#
# A good NLP notebook should make representation choices visible. Before fitting a classifier, inspect the tokens produced by the vectorizer on a few examples: normal posts, punctuation-heavy text, mixed case, and domain phrases.

# %%
# TODO: Build a CountVectorizer analyzer and inspect its tokens.
# Requirements:
# - lowercase text;
# - use unigrams and bigrams;
# - keep tokens with at least two word characters, e.g. `token_pattern=r"(?u)\b\w\w+\b"`;
# - create tokenization_table with case, preview, tokens, and token_count columns.

tokenization_examples = [
    {"case": "newsgroup post", "text": train_df.iloc[0]["text"]},
    {"case": "punctuation", "text": "NASA's 3D-rendering pipeline: orbit, re-entry, and telemetry!"},
    {"case": "case variants", "text": "BaseBall baseball BASEBALL graphics Graphics GRAPHICS"},
]

raise NotImplementedError("Build token_probe and tokenization_table")

display(tokenization_table)

# %% [markdown]
# TF-IDF keeps the same sparse matrix idea, but changes the weights:
#
# - term frequency: how much a term appears in this document;
# - inverse document frequency: how specific the term is across the corpus;
# - n-grams: short phrases that often carry more topic meaning than individual words.

# %%
# TODO: Build a TF-IDF vectorizer for the training text.
# Requirements:
# - lowercase text;
# - remove English stop words;
# - use unigrams and bigrams;
# - use `min_df=TFIDF_MIN_DF` so tiny fallback data still works;
# - cap the vocabulary at 20,000 features.

raise NotImplementedError("Create tfidf_vectorizer, X_train_tfidf, and X_test_tfidf")

print("TF-IDF train matrix:", X_train_tfidf.shape)
print("TF-IDF test matrix:", X_test_tfidf.shape)
print("Vocabulary size:", len(tfidf_vectorizer.get_feature_names_out()))
print("Matrix density:", f"{X_train_tfidf.nnz / np.prod(X_train_tfidf.shape):.4%}")

# %% [markdown]
# ### Task: compare sparse representations
#
# Count vectors and TF-IDF vectors have the same shape logic, but different weighting. Compare them before training models: vocabulary size, number of non-zero entries, density, and average non-zero value.

# %%
# TODO: Build count_bigram_vectorizer and compare it with X_train_tfidf.
# Store the result in representation_summary with one row for count and one row for TF-IDF.

raise NotImplementedError("Create X_train_count and representation_summary")

display(representation_summary)

# %%
example_index = 0
print("Label:", train_df.iloc[example_index]["label"])
print(preview_text(train_df.iloc[example_index]["text"], lines=5))
display(top_tfidf_terms_for_document(tfidf_vectorizer, X_train_tfidf, example_index, n=12))

# %% [markdown]
# # Part 3. Classical NLP baseline
#
# Classical NLP practice rarely trusts one model blindly. We compare Naive Bayes, logistic regression, and a linear SVM on the same TF-IDF representation, then keep logistic regression for feature-weight inspection.

# %%
# TODO: Compare classical text-classification baselines on the same TF-IDF features.
# Requirements:
# - train MultinomialNB;
# - train LogisticRegression with balanced class weights;
# - train LinearSVC with balanced class weights;
# - create baseline_scores with model and test_accuracy columns;
# - keep the fitted LogisticRegression model in classifier for the interpretability section.

raise NotImplementedError("Train classical baselines and keep LogisticRegression as classifier")

print(f"Logistic regression test accuracy: {test_accuracy:.3f}")
print(classification_report(test_df["target"], test_predictions, target_names=target_names, zero_division=0))

# %% [markdown]
# ### Metric check with `evaluate`
#
# The scikit-learn report is enough for this practical, but many modern NLP workflows use Hugging Face `evaluate` to standardize metric computation across tasks and datasets.

# %%
if RUN_HF_EVALUATE_DEMO:
    try:
        import evaluate

        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        hf_metric_results = {
            "accuracy": accuracy_metric.compute(
                predictions=test_predictions,
                references=test_df["target"].tolist(),
            )["accuracy"],
            "macro_precision": precision_metric.compute(
                predictions=test_predictions,
                references=test_df["target"].tolist(),
                average="macro",
                zero_division=0,
            )["precision"],
            "macro_recall": recall_metric.compute(
                predictions=test_predictions,
                references=test_df["target"].tolist(),
                average="macro",
                zero_division=0,
            )["recall"],
            "macro_f1": f1_metric.compute(
                predictions=test_predictions,
                references=test_df["target"].tolist(),
                average="macro",
            )["f1"],
        }
        display(pd.DataFrame([hf_metric_results]).round(3))
    except Exception as exc:
        print("Hugging Face evaluate demo skipped:", type(exc).__name__, str(exc)[:160])
        print("The scikit-learn classification report above is the required classroom metric output.")
else:
    print("Hugging Face evaluate demo skipped. Set RUN_HF_EVALUATE_DEMO = True in Colab to run it.")

# %%
cm = confusion_matrix(
    test_df["target"],
    test_predictions,
    labels=np.arange(len(target_names)),
)
class_labels = [name.split(".")[-1] for name in target_names]
row_totals = cm.sum(axis=1, keepdims=True)
row_percentages = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
)
annotations = np.array([
    [f"{count}\n{share:.0%}" if count else "" for count, share in zip(row, pct_row)]
    for row, pct_row in zip(cm, row_percentages)
])

fig, ax = plt.subplots(figsize=(6.2, 5.2))
sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    cbar=False,
    square=True,
    linewidths=1.2,
    linecolor="white",
    xticklabels=class_labels,
    yticklabels=class_labels,
    annot_kws={"fontsize": 10, "fontweight": "semibold"},
    ax=ax,
)
ax.set_title("Confusion matrix: TF-IDF + logistic regression", pad=12)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What did the classifier learn?
#
# For a linear classifier, each class has feature weights. The highest positive weights are not a perfect explanation, but they are an excellent first inspection tool.

# %%
# TODO: Inspect the top positive TF-IDF features for each class.
# Hint: classifier.coef_[class_index] contains one weight per vocabulary term.

raise NotImplementedError("Create a top_terms table with class, rank, term, and weight")

display(top_terms.pivot(index="rank", columns="class", values="term"))

# %% [markdown]
# ### Task: make an error-analysis card
#
# For a few wrong predictions, record the actual label, predicted label, high-weight document terms, and one short hypothesis. This mirrors the research-thinking habit: do not just report a score; ask what the model used and why it failed.

# %%
# TODO: Build mistake_cards for up to 6 wrong predictions.
# For each card, include actual, predicted, top_document_terms, hypothesis, and preview.
# Hint: use top_tfidf_terms_for_document on X_test_tfidf with the row position.

raise NotImplementedError("Create mistake_cards for model error analysis")

display(mistake_cards)

# %% [markdown]
# # Part 4. From sparse similarity to dense embeddings
#
# Once text is represented as vectors, classification is not the only useful operation. We can also search by vector similarity.
#
# TF-IDF similarity is mostly lexical: documents are close when they share important vocabulary. Dense sentence embeddings aim for semantic closeness, even when wording differs.

# %% [markdown]
# ## Embedding formulas and representation choices
#
# An **embedding** is a vector representation of text. The same idea appears at different levels:
#
# - a sparse TF-IDF document vector has one dimension per vocabulary term;
# - a word embedding such as Word2Vec or GloVe gives one dense vector per word;
# - a contextual token embedding gives one vector per token in a sentence;
# - a sentence/document embedding gives one dense vector for the whole text.
#
# Core formulas:
#
# $$
# \mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t)
# $$
#
# $$
# \mathrm{idf}(t) = \log\frac{1 + N}{1 + \mathrm{df}(t)} + 1
# $$
#
# Cosine similarity compares vector direction rather than raw length:
#
# $$
# \cos(\mathbf{a}, \mathbf{b}) =
# \frac{\mathbf{a}^{\top}\mathbf{b}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{b} \rVert_2}
# $$
#
# For a sequence of token embeddings $\mathbf{h}_1, \ldots, \mathbf{h}_T$, a sentence embedding can be built by pooling:
#
# $$
# \mathbf{s}_{\mathrm{mean}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{h}_i
# $$
#
# With attention-mask or TF-IDF-style weights $w_i$:
#
# $$
# \mathbf{s}_{\mathrm{weighted}} = \frac{\sum_{i=1}^{T} w_i\mathbf{h}_i}{\sum_{i=1}^{T} w_i}
# $$
#
# Vector shape matters. A document embedding matrix has shape:
#
# $$
# X \in \mathbb{R}^{n_{\mathrm{documents}} \times d_{\mathrm{embedding}}}
# $$
#
# Sparse TF-IDF may have tens of thousands of dimensions, but most values are zero. Dense embeddings usually have hundreds or a few thousand dimensions, and most values are non-zero. For cosine search, dense embeddings are often L2-normalized so dot product and cosine similarity become equivalent.

# %%
embedding_families = pd.DataFrame([
    {
        "family": "Sparse lexical vector",
        "examples": "CountVectorizer, TF-IDF",
        "unit": "document",
        "typical dimension": "vocabulary size, often 10k-100k+",
        "what it captures": "exact words and n-grams",
    },
    {
        "family": "Static word embedding",
        "examples": "Word2Vec, GloVe, fastText",
        "unit": "word or subword",
        "typical dimension": "50, 100, 200, 300",
        "what it captures": "distributional word similarity",
    },
    {
        "family": "Contextual token embedding",
        "examples": "BERT-style encoder hidden states",
        "unit": "token in context",
        "typical dimension": "model hidden size, e.g. 384 or 768",
        "what it captures": "meaning conditioned on surrounding text",
    },
    {
        "family": "Sentence/document embedding",
        "examples": "Sentence Transformers, encoder pooling",
        "unit": "sentence, paragraph, document",
        "typical dimension": "model dependent; all-MiniLM-L6-v2 is 384",
        "what it captures": "semantic similarity for search, clustering, reranking",
    },
])

display(embedding_families)

text_embedding_options = pd.DataFrame([
    {
        "text level": "word",
        "common method": "lookup a static embedding",
        "example": "GloVe['orbit'], Word2Vec['pitcher']",
        "caveat": "one vector per word type, little or no context",
    },
    {
        "text level": "sentence",
        "common method": "mean-pool contextual token embeddings or use a sentence-transformer",
        "example": "SentenceTransformer.encode(sentence)",
        "caveat": "pooling choice changes behavior",
    },
    {
        "text level": "document",
        "common method": "chunk document, embed chunks, average or index chunks separately",
        "example": "embed paragraphs, then retrieve top chunks",
        "caveat": "long documents may exceed encoder context length",
    },
    {
        "text level": "query + document pair",
        "common method": "cross-encoder score instead of independent embeddings",
        "example": "reranker(query, candidate_doc)",
        "caveat": "more accurate but slower for large corpora",
    },
])

display(text_embedding_options)

# %%
search_df = train_df.sample(min(SEARCH_LIMIT, len(train_df)), random_state=RANDOM_STATE).reset_index(drop=True)
X_search_tfidf = tfidf_vectorizer.transform(search_df["text"])

query_text = "NASA engineers render a 3D model of a spacecraft orbiting earth."
print("Query:")
print(query_text)

# %%
# TODO: Transform query_text with the existing TF-IDF vectorizer and find the top 7 neighbors.
# Store the result in sparse_neighbors with label, cosine_similarity, and preview columns.

raise NotImplementedError("Build TF-IDF nearest-neighbor search for query_text")

display(sparse_neighbors[["label", "cosine_similarity", "preview"]])

# %% [markdown]
# ### Task: inspect query evidence
#
# Before trusting nearest neighbors, inspect the query vector itself. Which query terms survived preprocessing, and which terms dominate the sparse search?

# %%
# TODO: Inspect the query vector by extracting its top TF-IDF terms into query_terms.

raise NotImplementedError("Create query_terms from query_vec")

display(query_terms)

# %% [markdown]
# ## Static word embeddings with Word2Vec
#
# Word2Vec is the missing middle step between TF-IDF and modern sentence embeddings. It still starts from plain text, but instead of creating one sparse feature per vocabulary term, it learns a small dense vector for each word.
#
# The core idea is the **distributional hypothesis**: words that appear in similar contexts tend to have related meanings. Word2Vec operationalizes this with two classic training setups:
#
# - **CBOW** predicts a center word from surrounding context words.
# - **Skip-gram** predicts surrounding context words from a center word.
#
# For skip-gram with a center word $w_c$ and an observed context word $w_o$, the full softmax objective would model:
#
# $$
# P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{o}^{\top}\mathbf{v}_{c})}{\sum_{j=1}^{|V|}\exp(\mathbf{u}_{j}^{\top}\mathbf{v}_{c})}
# $$
#
# where $\mathbf{v}_{c}$ is the input vector for the center word and $\mathbf{u}_{o}$ is the output vector for the context word. The denominator is expensive for a large vocabulary, so practical Word2Vec often uses **negative sampling**:
#
# $$
# \log \sigma(\mathbf{u}_{o}^{\top}\mathbf{v}_{c}) +
# \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}
# \left[\log \sigma(-\mathbf{u}_{k}^{\top}\mathbf{v}_{c})\right]
# $$
#
# The model pulls real center-context pairs closer together and pushes sampled fake pairs apart. The result is a lookup table:
#
# $$
# E \in \mathbb{R}^{|V| \times d}
# $$
#
# Each row is a word vector. Classic dimensions are often 50, 100, 200, or 300. In this class demo we train a tiny 50-dimensional model so students can see the mechanics without downloading pretrained GloVe or fastText files.

# %%
word2vec_sentences = [
    simple_preprocess(text, deacc=True, min_len=2, max_len=30)
    for text in train_df["text"]
]
word2vec_sentences = [tokens for tokens in word2vec_sentences if len(tokens) >= 3]

WORD2VEC_VECTOR_SIZE = 50
WORD2VEC_MIN_COUNT = 1 if len(train_df) < 100 else 2
WORD2VEC_EPOCHS = 60 if DATA_SOURCE == "fallback corpus" else 20

word2vec_model = Word2Vec(
    sentences=word2vec_sentences,
    vector_size=WORD2VEC_VECTOR_SIZE,
    window=5,
    min_count=WORD2VEC_MIN_COUNT,
    sg=1,              # 1 = skip-gram, 0 = CBOW
    negative=5,
    sample=1e-3,
    epochs=WORD2VEC_EPOCHS,
    workers=1,         # deterministic and Colab-friendly for teaching
    seed=RANDOM_STATE,
)

word2vec_vocab = pd.DataFrame({
    "token": word2vec_model.wv.index_to_key,
    "training_count": [word2vec_model.wv.get_vecattr(token, "count") for token in word2vec_model.wv.index_to_key],
})

word2vec_summary = pd.DataFrame([{
    "training_sentences": len(word2vec_sentences),
    "vocabulary_size": len(word2vec_model.wv),
    "embedding_dimension": word2vec_model.vector_size,
    "window": word2vec_model.window,
    "min_count": WORD2VEC_MIN_COUNT,
    "epochs": WORD2VEC_EPOCHS,
    "algorithm": "skip-gram with negative sampling",
}])

display(word2vec_summary)
display(word2vec_vocab.head(20))

# %%
probe_words = [
    "space", "orbit", "nasa", "baseball", "pitcher", "graphics", "image",
    "politics", "security", "team", "software",
]
probe_words = [word for word in probe_words if word in word2vec_model.wv.key_to_index]

neighbor_rows = []
for word in probe_words:
    for neighbor, similarity in word2vec_model.wv.most_similar(word, topn=min(5, len(word2vec_model.wv) - 1)):
        neighbor_rows.append({
            "word": word,
            "neighbor": neighbor,
            "cosine_similarity": similarity,
        })

if neighbor_rows:
    word2vec_neighbors = pd.DataFrame(neighbor_rows)
    display(word2vec_neighbors.pivot(index="neighbor", columns="word", values="cosine_similarity").fillna(""))
else:
    print("No probe words were found in the Word2Vec vocabulary. Try lowering min_count or changing probes.")

# %%
visual_words = []
for word in probe_words:
    visual_words.append(word)
    visual_words.extend([neighbor for neighbor, _ in word2vec_model.wv.most_similar(word, topn=min(5, len(word2vec_model.wv) - 1))])
visual_words = list(dict.fromkeys(visual_words))[:50]

if len(visual_words) >= 3:
    word_vectors = np.vstack([word2vec_model.wv[word] for word in visual_words])
    word_coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(word_vectors)
    word_plot_df = pd.DataFrame({"word": visual_words, "x": word_coords[:, 0], "y": word_coords[:, 1]})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=word_plot_df, x="x", y="y", ax=ax)
    for _, row in word_plot_df.iterrows():
        ax.text(row["x"], row["y"], row["word"], fontsize=9, alpha=0.85)
    ax.set_title("Word2Vec word vectors projected with PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.show()
else:
    print("Word2Vec PCA skipped because there are too few probe words.")

if len(visual_words) >= 12:
    try:
        import umap.umap_ as umap

        reducer = umap.UMAP(
            n_neighbors=min(8, len(visual_words) - 1),
            n_components=2,
            min_dist=0.15,
            metric="cosine",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        word_umap_coords = reducer.fit_transform(word_vectors)
        word_umap_df = pd.DataFrame({"word": visual_words, "x": word_umap_coords[:, 0], "y": word_umap_coords[:, 1]})

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=word_umap_df, x="x", y="y", ax=ax)
        for _, row in word_umap_df.iterrows():
            ax.text(row["x"], row["y"], row["word"], fontsize=9, alpha=0.85)
        ax.set_title("Word2Vec word vectors projected with UMAP")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print("Word2Vec UMAP skipped:", type(exc).__name__, str(exc)[:160])
else:
    print("Word2Vec UMAP skipped because there are too few words to project.")

# %% [markdown]
# ### From word vectors to sentence or document vectors
#
# Word2Vec gives vectors for **words**, not whole documents. A simple baseline is to average the vectors of known tokens:
#
# $$
# \mathbf{d}_{\mathrm{avg}} = \frac{1}{|T_d|}\sum_{w_i \in T_d} \mathbf{e}_{w_i}
# $$
#
# This is fast and useful as a teaching bridge, but it loses word order, syntax, negation, and most context. The word `pitcher` has one vector whether the sentence is about baseball or a water jug. That limitation motivates contextual encoders and sentence-transformers.

# %% [markdown]
# ### Task: build a document search with mean-pooled Word2Vec
#
# Word2Vec gives word vectors. To search documents, you need a pooling rule. Use the simple average formula above, normalize the resulting vectors, and compare them with cosine similarity. Then compare the result with the TF-IDF nearest neighbors.

# %%
# TODO: Build a document-level Word2Vec search by averaging word vectors.
# Steps:
# 1. Tokenize text with simple_preprocess.
# 2. Keep tokens that exist in word2vec_model.wv.
# 3. Average their vectors into one document vector.
# 4. L2-normalize document vectors and the query vector.
# 5. If query_text has no known Word2Vec tokens, fall back to a short query made from probe_words.
# 6. Rank search_df by cosine similarity.

raise NotImplementedError("Create word2vec_neighbors using mean-pooled Word2Vec vectors")

display(word2vec_neighbors[["label", "cosine_similarity", "known_word_count", "word2vec_query", "preview"]])

# %% [markdown]
# ## Toy examples: how sentence and text embeddings are calculated
#
# A word embedding table gives vectors for word types. A sentence or text embedding needs an extra aggregation step.
#
# For a sentence with known tokens $w_1, \ldots, w_T$ and static word embeddings $\mathbf{e}_{w_i}$, the simplest sentence vector is mean pooling:
#
# $$
# \mathbf{s}_{\mathrm{mean}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{e}_{w_i}
# $$
#
# A weighted version can give more influence to informative words, for example with TF-IDF-like weights $\alpha_i$:
#
# $$
# \mathbf{s}_{\mathrm{weighted}} = \frac{\sum_{i=1}^{T}\alpha_i\mathbf{e}_{w_i}}{\sum_{i=1}^{T}\alpha_i}
# $$
#
# Transformer sentence embeddings use the same pooling idea, but the token vectors are contextual hidden states $\mathbf{h}_i$, not static word vectors. With an attention mask $m_i \in \{0, 1\}$ that ignores padding tokens:
#
# $$
# \mathbf{s}_{\mathrm{encoder}} = \frac{\sum_{i=1}^{T}m_i\mathbf{h}_i}{\sum_{i=1}^{T}m_i}
# $$
#
# For cosine search, the final vector is often normalized:
#
# $$
# \tilde{\mathbf{s}} = \frac{\mathbf{s}}{\lVert \mathbf{s} \rVert_2}
# $$
#
# Longer texts are usually handled by chunking. Embed each chunk $c_j$, then either index chunks separately or aggregate chunk vectors:
#
# $$
# \mathbf{d} = \frac{\sum_{j=1}^{M}\beta_j\mathbf{s}_{c_j}}{\sum_{j=1}^{M}\beta_j}
# $$
#
# where $\beta_j$ can be a chunk length, a relevance score, or just 1 for plain averaging.

# %%
toy_word_vectors = {
    "space": np.array([0.95, 0.05, 0.10, 0.05]),
    "orbit": np.array([0.90, 0.05, 0.20, 0.05]),
    "mission": np.array([0.85, 0.05, 0.15, 0.20]),
    "baseball": np.array([0.05, 0.95, 0.05, 0.05]),
    "pitcher": np.array([0.05, 0.90, 0.10, 0.10]),
    "team": np.array([0.10, 0.85, 0.10, 0.10]),
    "graphics": np.array([0.10, 0.05, 0.95, 0.05]),
    "render": np.array([0.20, 0.05, 0.90, 0.05]),
    "image": np.array([0.10, 0.05, 0.85, 0.05]),
    "policy": np.array([0.05, 0.10, 0.05, 0.95]),
    "security": np.array([0.15, 0.10, 0.05, 0.85]),
}

toy_idf_weights = {
    "space": 1.8,
    "orbit": 2.0,
    "mission": 2.1,
    "baseball": 1.8,
    "pitcher": 2.2,
    "team": 1.2,
    "graphics": 1.9,
    "render": 2.0,
    "image": 1.4,
    "policy": 1.9,
    "security": 2.1,
}

def toy_tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def mean_pool_static(text, vectors):
    tokens = [token for token in toy_tokens(text) if token in vectors]
    if not tokens:
        return tokens, np.zeros(next(iter(vectors.values())).shape)
    return tokens, np.mean([vectors[token] for token in tokens], axis=0)


def weighted_pool_static(text, vectors, weights):
    tokens = [token for token in toy_tokens(text) if token in vectors]
    if not tokens:
        return tokens, np.zeros(next(iter(vectors.values())).shape)
    token_weights = np.array([weights.get(token, 1.0) for token in tokens], dtype=float)
    token_vectors = np.vstack([vectors[token] for token in tokens])
    return tokens, (token_vectors * token_weights[:, None]).sum(axis=0) / token_weights.sum()


def l2_normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def compact_vector(vector):
    return np.array2string(vector, precision=3, suppress_small=True)

toy_sentences = [
    "space mission orbit",
    "baseball pitcher team",
    "graphics render image",
    "security policy mission",
]

toy_sentence_rows = []
for sentence in toy_sentences:
    tokens, mean_vector = mean_pool_static(sentence, toy_word_vectors)
    _, weighted_vector = weighted_pool_static(sentence, toy_word_vectors, toy_idf_weights)
    toy_sentence_rows.append({
        "sentence": sentence,
        "known_tokens": tokens,
        "mean_pool_embedding": compact_vector(mean_vector),
        "tfidf_weighted_embedding": compact_vector(weighted_vector),
        "mean_pool_l2_norm": np.linalg.norm(mean_vector),
    })

display(pd.DataFrame(toy_sentence_rows).round(3))

# %%
toy_query = "space orbit mission"
_, query_mean = mean_pool_static(toy_query, toy_word_vectors)
_, query_weighted = weighted_pool_static(toy_query, toy_word_vectors, toy_idf_weights)

similarity_rows = []
for sentence in toy_sentences:
    _, sentence_mean = mean_pool_static(sentence, toy_word_vectors)
    _, sentence_weighted = weighted_pool_static(sentence, toy_word_vectors, toy_idf_weights)
    similarity_rows.append({
        "query": toy_query,
        "candidate_sentence": sentence,
        "cosine_mean_pool": cosine_similarity(
            l2_normalize_vector(query_mean).reshape(1, -1),
            l2_normalize_vector(sentence_mean).reshape(1, -1),
        )[0, 0],
        "cosine_weighted_pool": cosine_similarity(
            l2_normalize_vector(query_weighted).reshape(1, -1),
            l2_normalize_vector(sentence_weighted).reshape(1, -1),
        )[0, 0],
    })

display(pd.DataFrame(similarity_rows).sort_values("cosine_weighted_pool", ascending=False).round(3))

# %% [markdown]
# ### Contextual pooling and document chunks
#
# The next cell uses made-up encoder hidden states to show two practical details.
#
# First, contextual encoders can assign different vectors to the same surface word in different sentences. Second, padding tokens should not contribute to mean pooling, so the attention mask matters.

# %%
contextual_pitcher_examples = pd.DataFrame([
    {
        "sentence": "the pitcher struck out the batter",
        "token": "pitcher",
        "contextual_vector": compact_vector(np.array([0.05, 0.92, 0.08, 0.10])),
        "interpretation": "baseball role",
    },
    {
        "sentence": "the pitcher held cold water",
        "token": "pitcher",
        "contextual_vector": compact_vector(np.array([0.10, 0.05, 0.12, 0.88])),
        "interpretation": "container object",
    },
])
display(contextual_pitcher_examples)

encoder_tokens = ["[CLS]", "space", "mission", "orbit", "[PAD]", "[PAD]"]
encoder_hidden_states = np.array([
    [0.40, 0.10, 0.10, 0.10],
    [0.95, 0.05, 0.10, 0.05],
    [0.85, 0.05, 0.15, 0.20],
    [0.90, 0.05, 0.20, 0.05],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
])
attention_mask = np.array([1, 1, 1, 1, 0, 0])

masked_sentence_embedding = (
    encoder_hidden_states * attention_mask[:, None]
).sum(axis=0) / attention_mask.sum()
normalized_sentence_embedding = l2_normalize_vector(masked_sentence_embedding)

encoder_pooling_demo = pd.DataFrame({
    "token": encoder_tokens,
    "attention_mask": attention_mask,
    "hidden_state": [compact_vector(vector) for vector in encoder_hidden_states],
})
display(encoder_pooling_demo)
print("Masked mean sentence embedding:", compact_vector(masked_sentence_embedding))
print("L2-normalized sentence embedding:", compact_vector(normalized_sentence_embedding))

# %%
chunk_demo = pd.DataFrame([
    {
        "chunk": "NASA mission entered orbit",
        "chunk_embedding": np.array([0.90, 0.05, 0.15, 0.10]),
        "token_count": 4,
    },
    {
        "chunk": "rendered images showed the spacecraft",
        "chunk_embedding": np.array([0.55, 0.05, 0.55, 0.05]),
        "token_count": 5,
    },
    {
        "chunk": "policy teams discussed security",
        "chunk_embedding": np.array([0.15, 0.25, 0.05, 0.80]),
        "token_count": 4,
    },
])

chunk_matrix = np.vstack(chunk_demo["chunk_embedding"])
chunk_lengths = chunk_demo["token_count"].to_numpy(dtype=float)
plain_document_embedding = chunk_matrix.mean(axis=0)
length_weighted_document_embedding = (chunk_matrix * chunk_lengths[:, None]).sum(axis=0) / chunk_lengths.sum()

chunk_query = l2_normalize_vector(query_mean)
chunk_demo["cosine_to_query"] = [
    float(l2_normalize_vector(vector) @ chunk_query)
    for vector in chunk_demo["chunk_embedding"]
]
chunk_demo["chunk_embedding"] = chunk_demo["chunk_embedding"].map(compact_vector)

display(chunk_demo.sort_values("cosine_to_query", ascending=False).round(3))
print("Plain average document embedding:", compact_vector(plain_document_embedding))
print("Length-weighted document embedding:", compact_vector(length_weighted_document_embedding))
print("Teaching note: for retrieval, indexing chunks separately often works better than averaging the whole document.")

# %%
dense_embeddings = None
dense_neighbors = None

if RUN_SENTENCE_TRANSFORMER_DEMO:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=str(DEVICE))
    print(f"SentenceTransformer running on: {embedding_model.device}")

    embedding_texts = search_df["text"].head(EMBEDDING_LIMIT).tolist()
    embedding_labels = search_df["label"].head(EMBEDDING_LIMIT).tolist()

    dense_embeddings = embedding_model.encode(
        embedding_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = embedding_model.encode(
        [query_text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    dense_scores = dense_embeddings @ query_embedding
    dense_top_indices = np.argsort(dense_scores)[-7:][::-1]

    dense_neighbors = pd.DataFrame({
        "label": [embedding_labels[i] for i in dense_top_indices],
        "cosine_similarity": dense_scores[dense_top_indices],
        "preview": [preview_text(embedding_texts[i], lines=2) for i in dense_top_indices],
    })
    display(dense_neighbors)
else:
    print("Sentence-transformer checkpoint demo skipped. Set RUN_SENTENCE_TRANSFORMER_DEMO = True to run it.")

# %%
if dense_embeddings is not None:
    embedding_diagnostics = pd.DataFrame([{
        "embedding_model": SENTENCE_MODEL_NAME,
        "matrix_shape": str(dense_embeddings.shape),
        "documents_embedded": dense_embeddings.shape[0],
        "embedding_dimension": dense_embeddings.shape[1],
        "mean_l2_norm": np.linalg.norm(dense_embeddings, axis=1).mean(),
        "min_l2_norm": np.linalg.norm(dense_embeddings, axis=1).min(),
        "max_l2_norm": np.linalg.norm(dense_embeddings, axis=1).max(),
    }])
    display(embedding_diagnostics.round(4))
else:
    print("Embedding diagnostics skipped because dense embeddings are unavailable.")

# %% [markdown]
# ## Visualizing embeddings with PCA and UMAP
#
# PCA and UMAP compress high-dimensional vectors into two dimensions for inspection.
#
# - **PCA** is linear: it shows directions of maximum variance.
# - **UMAP** is nonlinear: it tries to preserve local neighborhoods, which is useful for embedding spaces.
#
# Do not over-interpret the exact geometry. In 2D plots, nearby points are useful hypotheses for inspection, not proof that classes are truly separable.

# %%
if dense_embeddings is not None and len(dense_embeddings) >= 8:
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(dense_embeddings)
    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": embedding_labels,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="label", ax=ax)
    ax.set_title("Sentence embedding projection with PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
else:
    print("PCA projection skipped because dense embeddings are unavailable or too few.")

# %%
if dense_embeddings is not None and len(dense_embeddings) >= 8:
    try:
        import umap.umap_ as umap

        n_neighbors = min(15, max(2, len(dense_embeddings) - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        umap_coords = reducer.fit_transform(dense_embeddings)
        umap_df = pd.DataFrame({
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "label": embedding_labels,
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=umap_df, x="x", y="y", hue="label", ax=ax)
        ax.set_title("Sentence embedding projection with UMAP")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print("UMAP projection skipped:", type(exc).__name__, str(exc)[:160])
else:
    print("UMAP projection skipped because dense embeddings are unavailable or too few.")

# %% [markdown]
# Teaching point: both sparse and dense systems use vectors and cosine similarity, but their vector spaces are built from different training signals. TF-IDF knows local corpus statistics. Sentence-transformer embeddings inherit semantic structure from a pretrained encoder.

# %% [markdown]
# # Part 5. Transformer encoder intuition
#
# A transformer encoder is not a chatbot. It reads a sequence and produces contextual representations.
#
# The practical only needs three ideas:
#
# - **Subword tokenization**: rare words can be split into reusable pieces.
# - **Attention**: each token can use information from other tokens in the same sequence.
# - **Contextual embeddings**: the representation of a word depends on the sentence around it.

# %%
if RUN_TOKENIZER_DEMO:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    tokenizer_examples = [
        "The pitcher threw a curveball in the ninth inning.",
        "A graphics pipeline renders textured polygons.",
        "Reusable embeddings support semantic search.",
        "Tokenization can split unfamiliar words into subwords.",
    ]

    rows = []
    for text in tokenizer_examples:
        tokens = tokenizer.tokenize(text)
        rows.append({
            "text": text,
            "tokens": tokens,
            "token_count": len(tokens),
        })

    display(pd.DataFrame(rows))
else:
    print("Tokenizer demo skipped. Set RUN_TOKENIZER_DEMO = True to run it.")

# %%
encoder_decoder_map = pd.DataFrame([
    {
        "architecture": "Encoder",
        "core action": "Read the whole input and produce representations",
        "typical use": "classification, embeddings, retrieval, reranking",
        "example": "BERT-style encoders, sentence-transformers",
    },
    {
        "architecture": "Decoder",
        "core action": "Predict the next token repeatedly",
        "typical use": "chat, completion, code generation, tool calls",
        "example": "GPT-style LLMs",
    },
    {
        "architecture": "Encoder-decoder",
        "core action": "Read one sequence and generate another",
        "typical use": "translation, summarization, some QA systems",
        "example": "T5-style models",
    },
])

display(encoder_decoder_map)

# %% [markdown]
# # Part 6. NLP metrics map
#
# Metrics depend on the output shape. A classifier, a retrieval system, and a language model are not judged with the same number.
#
# Useful mental map:
#
# - **Classification**: did we assign the right label? Use accuracy, precision, recall, F1, and confusion matrices.
# - **Retrieval / search**: did the useful documents appear near the top? Use precision@k, recall@k, MRR, and nDCG-style metrics.
# - **Probabilistic language modeling**: did the model assign high probability to the observed text? Use surprisal, cross-entropy, and perplexity.
# - **Generation tasks**: automatic metrics can help, but human review, factuality, safety, and task success matter a lot.
#
# Surprisal, also called information content or surprise, is the penalty for assigning probability $p$ to an event that actually happened. With log base $b$:
#
# $$
# I_b(x) = -\log_b p(x)
# $$
#
# Common units:
#
# $$
# I_2(x) = -\log_2 p(x) \quad \text{bits}
# $$
#
# $$
# I_e(x) = -\ln p(x) \quad \text{nats}
# $$
#
# The log base determines the unit. The code examples below use base 2, so their surprisal and cross-entropy values are in bits. A high-probability true token has low surprisal; a low-probability true token has high surprisal.

# %%
metrics_map = pd.DataFrame([
    {
        "family": "classification",
        "metric": "accuracy",
        "intuition": "share of examples with the correct label",
        "watch out": "can hide minority-class failures",
    },
    {
        "family": "classification",
        "metric": "precision / recall / F1",
        "intuition": "quality of positive predictions and coverage of true positives",
        "watch out": "choose macro/weighted averaging deliberately",
    },
    {
        "family": "retrieval",
        "metric": "precision@k",
        "intuition": "how many of the top-k results are relevant",
        "watch out": "does not reward finding all relevant documents",
    },
    {
        "family": "retrieval",
        "metric": "recall@k",
        "intuition": "how many known relevant documents appear in the top-k",
        "watch out": "needs relevance labels or judgments",
    },
    {
        "family": "retrieval",
        "metric": "MRR",
        "intuition": "how early the first relevant result appears",
        "watch out": "ignores useful results after the first one",
    },
    {
        "family": "probabilistic LM",
        "metric": "cross-entropy",
        "intuition": "average surprisal assigned to the true next token",
        "watch out": "lower is better; tokenization affects the value",
    },
    {
        "family": "probabilistic LM",
        "metric": "perplexity",
        "intuition": "b ** cross_entropy in matching units; here 2 ** cross_entropy_bits",
        "watch out": "compare only on the same data and tokenization",
    },
])

display(metrics_map)

# %% [markdown]
# ## Classification metric example
#
# For topic classification, the confusion matrix answers **where** the model confuses labels. Precision/recall/F1 answer **how costly** those confusions are for a particular class or average.

# %%
toy_classification = pd.DataFrame({
    "actual": ["space", "space", "space", "baseball", "baseball", "baseball", "graphics", "graphics"],
    "predicted": ["space", "space", "graphics", "baseball", "space", "baseball", "graphics", "baseball"],
})

toy_labels = ["space", "baseball", "graphics"]
toy_cm = confusion_matrix(toy_classification["actual"], toy_classification["predicted"], labels=toy_labels)
display(pd.DataFrame(toy_cm, index=[f"true_{x}" for x in toy_labels], columns=[f"pred_{x}" for x in toy_labels]))
print(classification_report(
    toy_classification["actual"],
    toy_classification["predicted"],
    labels=toy_labels,
    zero_division=0,
))

# %% [markdown]
# ## Retrieval metric example
#
# For search, ranking matters. A result can contain the right answer somewhere, but if it appears too late, the user may never see it.

# %%
ranked_results = pd.DataFrame({
    "rank": [1, 2, 3, 4, 5, 6],
    "document": ["A", "B", "C", "D", "E", "F"],
    "relevant": [False, True, False, True, True, False],
})

def precision_at_k(relevance, k):
    top_k = np.asarray(relevance[:k], dtype=bool)
    return top_k.mean() if len(top_k) else 0.0


def recall_at_k(relevance, k):
    relevance = np.asarray(relevance, dtype=bool)
    total_relevant = relevance.sum()
    if total_relevant == 0:
        return 0.0
    return relevance[:k].sum() / total_relevant


def reciprocal_rank(relevance):
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1 / index
    return 0.0

relevance = ranked_results["relevant"].tolist()
retrieval_scores = pd.DataFrame([{
    "precision@3": precision_at_k(relevance, 3),
    "recall@3": recall_at_k(relevance, 3),
    "MRR": reciprocal_rank(relevance),
}])

display(ranked_results)
display(retrieval_scores.round(3))

# %% [markdown]
# ## Toy metric calculations on simple sentences
#
# The next cell uses tiny sentences so the formulas are visible. The goal is not to build a good model. The goal is to see exactly what each metric is counting.

# %%
toy_sentence_classification = pd.DataFrame({
    "sentence": [
        "the rocket entered orbit",
        "the pitcher threw a ball",
        "the image has sharp pixels",
        "the rocket launched",
        "the team won the game",
        "the image rendered slowly",
    ],
    "actual": ["space", "baseball", "graphics", "space", "baseball", "graphics"],
    "predicted": ["space", "baseball", "graphics", "baseball", "baseball", "space"],
})

toy_sentence_labels = ["space", "baseball", "graphics"]
toy_sentence_cm = confusion_matrix(
    toy_sentence_classification["actual"],
    toy_sentence_classification["predicted"],
    labels=toy_sentence_labels,
)

display(toy_sentence_classification)
display(pd.DataFrame(
    toy_sentence_cm,
    index=[f"true_{label}" for label in toy_sentence_labels],
    columns=[f"pred_{label}" for label in toy_sentence_labels],
))

correct_predictions = (
    toy_sentence_classification["actual"] == toy_sentence_classification["predicted"]
).sum()
toy_accuracy = correct_predictions / len(toy_sentence_classification)
print(f"Accuracy = correct / total = {correct_predictions} / {len(toy_sentence_classification)} = {toy_accuracy:.3f}")

space_true_positive = ((toy_sentence_classification["actual"] == "space") & (toy_sentence_classification["predicted"] == "space")).sum()
space_predicted_positive = (toy_sentence_classification["predicted"] == "space").sum()
space_actual_positive = (toy_sentence_classification["actual"] == "space").sum()
space_precision = space_true_positive / space_predicted_positive
space_recall = space_true_positive / space_actual_positive
space_f1 = 2 * space_precision * space_recall / (space_precision + space_recall)
print(f"Space precision = TP / predicted_space = {space_true_positive} / {space_predicted_positive} = {space_precision:.3f}")
print(f"Space recall = TP / actual_space = {space_true_positive} / {space_actual_positive} = {space_recall:.3f}")
print(f"Space F1 = {space_f1:.3f}")

# %% [markdown]
# ### Toy retrieval example
#
# For retrieval, labels are about the ranked list, not a single prediction. Here the query is `space mission`, and relevant documents are the ones actually about space.

# %%
toy_ranked_sentences = pd.DataFrame({
    "rank": [1, 2, 3, 4, 5],
    "sentence": [
        "the rocket reached orbit",
        "the baseball team won",
        "the satellite sent images",
        "the graphics card overheated",
        "the spacecraft landed safely",
    ],
    "relevant_to_space_mission": [True, False, True, False, True],
})

toy_relevance = toy_ranked_sentences["relevant_to_space_mission"].tolist()
toy_relevant_top_3 = sum(toy_relevance[:3])
toy_total_relevant = sum(toy_relevance)
toy_first_relevant_rank = next(
    rank for rank, is_relevant in enumerate(toy_relevance, start=1) if is_relevant
)
toy_precision_at_3 = toy_relevant_top_3 / 3
toy_recall_at_3 = toy_relevant_top_3 / toy_total_relevant
toy_mrr = 1 / toy_first_relevant_rank

display(toy_ranked_sentences)
print(f"precision@3 = relevant in top 3 / 3 = {toy_relevant_top_3} / 3 = {toy_precision_at_3:.3f}")
print(f"recall@3 = relevant in top 3 / all relevant = {toy_relevant_top_3} / {toy_total_relevant} = {toy_recall_at_3:.3f}")
print(f"MRR = 1 / rank_of_first_relevant = 1 / {toy_first_relevant_rank} = {toy_mrr:.3f}")

# %% [markdown]
# ### Toy language-model example
#
# Now calculate surprisal, cross-entropy, and perplexity on a tiny next-token prediction task. We pretend the true sentence is:
#
# ```text
# the cat sat
# ```
#
# The model assigns probabilities to the true next token at each step. Higher probability means lower surprisal.

# %%
toy_lm_steps = pd.DataFrame({
    "context": ["<start>", "the", "the cat"],
    "true_next_token": ["the", "cat", "sat"],
    "model_probability_for_true_token": [0.50, 0.25, 0.10],
})

toy_lm_steps["surprisal_bits"] = -np.log2(toy_lm_steps["model_probability_for_true_token"])
toy_cross_entropy_bits = toy_lm_steps["surprisal_bits"].mean()
toy_perplexity = 2 ** toy_cross_entropy_bits

display(toy_lm_steps.round(3))
for _, row in toy_lm_steps.iterrows():
    print(
        f"I_2({row['true_next_token']} | {row['context']}) "
        f"= -log2({row['model_probability_for_true_token']:.2f}) "
        f"= {row['surprisal_bits']:.3f} bits"
    )
print(f"Cross-entropy = average surprisal = {toy_cross_entropy_bits:.3f} bits/token")
print(f"Perplexity = 2 ** cross_entropy_bits = {toy_perplexity:.2f}")

better_toy_lm_steps = toy_lm_steps.copy()
better_toy_lm_steps["model_probability_for_true_token"] = [0.80, 0.60, 0.40]
better_toy_lm_steps["surprisal_bits"] = -np.log2(better_toy_lm_steps["model_probability_for_true_token"])
better_toy_cross_entropy = better_toy_lm_steps["surprisal_bits"].mean()
better_toy_perplexity = 2 ** better_toy_cross_entropy

display(better_toy_lm_steps.round(3))
print(f"Better model cross-entropy = {better_toy_cross_entropy:.3f} bits/token")
print(f"Better model perplexity = {better_toy_perplexity:.2f}")

# %% [markdown]
# ## Entropy, cross-entropy, surprisal, and perplexity
#
# Use one log base consistently. With base $b$, surprisal is:
#
# $$
# I_b(x) = -\log_b p(x)
# $$
#
# Bits use $b=2$:
#
# $$
# I_2(x) = -\log_2 p(x)
# $$
#
# Nats use $b=e$:
#
# $$
# I_e(x) = -\ln p(x)
# $$
#
# Entropy is expected surprisal under the true distribution $P$:
#
# $$
# H_b(P) = \mathbb{E}_{x \sim P}[-\log_b P(x)] = -\sum_x P(x)\log_b P(x)
# $$
#
# Cross-entropy is expected surprisal when examples come from $P$ but are scored by a model distribution $Q$:
#
# $$
# H_b(P, Q) = \mathbb{E}_{x \sim P}[-\log_b Q(x)] = -\sum_x P(x)\log_b Q(x)
# $$
#
# For an observed sequence of true next tokens $x_1, \ldots, x_T$, empirical next-token cross-entropy is:
#
# $$
# \hat{H}_b = -\frac{1}{T}\sum_{t=1}^{T}\log_b Q(x_t \mid x_{<t})
# $$
#
# Perplexity is exponentiated cross-entropy using the same base:
#
# $$
# \mathrm{PPL} = b^{\hat{H}_b}
# $$
#
# For bits:
#
# $$
# \mathrm{PPL} = 2^{\hat{H}_2}
# $$
#
# For nats:
#
# $$
# \mathrm{PPL} = \exp(\hat{H}_e)
# $$
#
# A perplexity of 10 roughly means the model is as uncertain as choosing among 10 equally likely tokens at each step. Lower is better, but values are only comparable under the same dataset, tokenization, and evaluation setup.

# %%
def entropy_bits(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())

label_distribution_examples = pd.DataFrame([
    {
        "distribution": "certain label",
        "probabilities": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "distribution": "one dominant label",
        "probabilities": [0.70, 0.10, 0.10, 0.10],
    },
    {
        "distribution": "balanced four labels",
        "probabilities": [0.25, 0.25, 0.25, 0.25],
    },
])
label_distribution_examples["entropy_bits"] = label_distribution_examples["probabilities"].map(entropy_bits)
display(label_distribution_examples)

next_token_predictions = pd.DataFrame({
    "position": [1, 2, 3, 4, 5],
    "true_next_token": ["the", "space", "shuttle", "entered", "orbit"],
    "model_probability_for_true_token": [0.80, 0.40, 0.25, 0.10, 0.50],
})
next_token_predictions["surprisal_bits"] = -np.log2(next_token_predictions["model_probability_for_true_token"])
cross_entropy_bits = next_token_predictions["surprisal_bits"].mean()
perplexity = 2 ** cross_entropy_bits

display(next_token_predictions.round(3))
print(f"Cross-entropy: {cross_entropy_bits:.3f} bits/token")
print(f"Perplexity: {perplexity:.2f}")

better_model_probs = np.array([0.90, 0.70, 0.60, 0.40, 0.80])
better_cross_entropy = float((-np.log2(better_model_probs)).mean())
print(f"Better toy model cross-entropy: {better_cross_entropy:.3f} bits/token")
print(f"Better toy model perplexity: {2 ** better_cross_entropy:.2f}")

# %% [markdown]
# # Part 7. NLP task map
#
# NLP is not one task. Different tasks ask for different output shapes, evaluation methods, and product risks.

# %%
task_map = pd.DataFrame([
    {
        "task": "Document classification",
        "input": "document",
        "output": "label or probabilities",
        "example metric": "accuracy, F1, confusion matrix",
    },
    {
        "task": "Semantic search / retrieval",
        "input": "query + corpus",
        "output": "ranked documents",
        "example metric": "recall@k, MRR, nDCG",
    },
    {
        "task": "Clustering",
        "input": "documents",
        "output": "cluster assignments",
        "example metric": "silhouette, manual review",
    },
    {
        "task": "Named entity recognition",
        "input": "document",
        "output": "labeled spans",
        "example metric": "span-level precision/recall/F1",
    },
    {
        "task": "Summarization",
        "input": "long document",
        "output": "short generated text",
        "example metric": "human review, factuality checks",
    },
    {
        "task": "Translation",
        "input": "source-language text",
        "output": "target-language text",
        "example metric": "human review, BLEU/COMET-style scores",
    },
    {
        "task": "Question answering",
        "input": "question + context or corpus",
        "output": "answer span or generated answer",
        "example metric": "exact match, F1, groundedness review",
    },
])

display(task_map)

# %% [markdown]
# # Part 8. Bridge to LLMs
#
# This practical taught the foundation before LLM applications:
#
# 1. raw text needs representation;
# 2. sparse features make lexical evidence visible;
# 3. TF-IDF + linear models are strong supervised baselines;
# 4. cosine similarity turns text vectors into search systems;
# 5. dense embeddings add reusable semantic representations;
# 6. transformer encoders read text and produce contextual vectors;
# 7. NLP metrics must match the output shape: labels, rankings, probabilities, or generated text;
# 8. decoder LLMs generate text token by token and deserve a separate workflow.
#
# LLMs are the next block: prompting, RAG, agents, evaluation, safety, and deployment.
#
# Suggested debrief questions:
#
# - Which top terms looked like real topic evidence, and which looked like artifacts?
# - Which errors were genuinely ambiguous?
# - Where did TF-IDF similarity work well, and where would dense embeddings help?
# - Why is a sentence embedding model useful even when it is not a chatbot?
# - Why is perplexity useful for next-token prediction but insufficient for judging a chat assistant?
# - What new engineering risks appear once the model generates text instead of assigning labels?


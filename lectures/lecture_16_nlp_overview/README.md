# Lecture 16: Natural Language Processing

This directory currently contains a practical-first draft for an introductory NLP block before LLM applications.

The material focuses on the text intuition students need before using modern language models:

- documents, labels, and leakage risks;
- corpus EDA for text: split balance, document lengths, short examples, and frequent terms;
- tokenization, stop words, n-grams, count features, and TF-IDF;
- classical supervised text classification with linear models;
- text similarity, nearest-neighbor search, and sparse vector geometry;
- Word2Vec as the bridge from sparse lexical vectors to dense word vectors;
- explicit sentence/text embedding calculations: mean pooling, weighted pooling, contextual pooling, normalization, and chunk aggregation;
- dense encoder embeddings and semantic search;
- embedding formulas, vector length, pooling choices, and PCA/UMAP visualization;
- transformer encoder intuition: subword tokens, contextual embeddings, and attention;
- NLP metrics for labels, rankings, probabilities, and generated text, including entropy and perplexity;
- a bridge from encoder embeddings to decoder LLMs.

## Core Files

- `lecture_notes.md`
- `links.yaml`

## NLP Package References

The practical uses the repo baseline stack plus the `nlp` dependency group:

- `scikit-learn`: sparse text features, TF-IDF, linear baselines, metrics.
- `datasets`: lightweight dataset objects and `.map(...)` preprocessing for corpus diagnostics.
- `gensim`: classroom-safe Word2Vec training, nearest-word inspection, and vector-space demos.
- `tokenizers`: a tiny local WordPiece tokenizer demo before pretrained tokenizers.
- `evaluate`: Hugging Face metric helpers alongside the scikit-learn report.
- `sentence-transformers`, `torch`: GPU-aware sentence embeddings, semantic search, retrieval, reranking.
- `transformers`: pretrained tokenizer demo for encoder-style models.
- `umap-learn`: nonlinear visualization of dense embedding vectors.

The lecture metadata also points students to broader ecosystem libraries: `spaCy` for production NLP pipelines, `NLTK` for classic NLP teaching/corpora, and `Gensim` for topic modeling and vector-space models.

## Practical Session

- `practical_session/`: 90-minute classroom practical covering text classification on 20 Newsgroups, corpus EDA, sparse features, TF-IDF baselines, embedding formulas, Word2Vec, sentence/text embedding calculations, embedding similarity, PCA/UMAP visualization, transformer encoders, NLP metrics, an NLP task map, and a short bridge to LLMs.
- `practical_session/nlp_overview_practical_student_90min.ipynb`: student notebook with guided TODO cells.
- `practical_session/nlp_overview_practical_student_90min.py`: generated companion script for review and diffing.

Colab note: choose **Runtime -> Change runtime type -> T4 GPU** before running the sentence-transformer embedding section. Classical TF-IDF cells are CPU-friendly.

## Draft Status

This lecture does not yet have a slide deck or separate lecture example notebooks in the repository.
The practical notebook is the canonical working artifact for now.

---

[← Previous](../lecture_15_computer_vision/README.md) | [All Lectures](../README.md)

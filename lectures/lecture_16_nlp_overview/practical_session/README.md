# Lecture 16 Practical Session

This directory contains the draft practical for an introductory Natural Language Processing lecture before LLM applications.

## Files

| File | Description |
| --- | --- |
| `nlp_overview_practical_student_90min.ipynb` | Student notebook with guided TODO cells and runnable demos |
| `nlp_overview_practical_student_90min.py` | Generated student companion script for review and diffing |

## Structure

The practical follows a text-representation storyline, with a heavier EDA opening so students first understand the corpus before they build features:

| Part | Topic | Dataset / assets |
| --- | --- | --- |
| 1 | Text as data and corpus EDA | 20 Newsgroups subset, split balance, text diagnostics, short documents, frequent terms |
| 2 | Tokenization and sparse features | Tokenizer inspection, CountVectorizer, TfidfVectorizer |
| 3 | Classical NLP baseline | Naive Bayes, logistic regression, linear SVM |
| 4 | Sparse similarity, Word2Vec, and dense embeddings | Embedding formulas, TF-IDF cosine similarity, Word2Vec skip-gram, nearest words, mean/weighted pooling, contextual pooling, chunk aggregation, SentenceTransformer, PCA/UMAP |
| 5 | Transformer encoder intuition | Tokenizer demo, sentence-transformer embeddings |
| 6 | NLP metrics map | Classification metrics, retrieval metrics, entropy, cross-entropy, perplexity |
| 7 | NLP task map | Classification, retrieval, clustering, NER, summarization, translation, QA |
| 8 | Bridge to LLMs | Encoder vs decoder vs encoder-decoder |

## Environment

Google Colab: the notebook auto-installs missing NLP packages when running in Colab.

Local:

```bash
uv sync --group nlp
uv run jupyter lab
```

The `nlp` dependency group includes the modern NLP stack used or referenced by the practical: `transformers`, `tokenizers`, `sentence-transformers`, `datasets`, `evaluate`, `gensim`, `torch`, and `umap-learn`.

Colab GPU: select **Runtime -> Change runtime type -> T4 GPU** before the dense embedding section. The notebook detects CUDA and passes the GPU device into `SentenceTransformer`.

## Notes

- Main classification cells use scikit-learn and run quickly on CPU.
- The Word2Vec cells train a small local `gensim` model and do not download pretrained vectors.
- The dense-embedding and pretrained-tokenizer demos download Hugging Face model files. They are controlled by `RUN_SENTENCE_TRANSFORMER_DEMO` and `RUN_TOKENIZER_DEMO`.
- The Hugging Face `datasets` preprocessing demo is controlled by `RUN_HF_DATASETS_DEMO`. It is enabled by default in Colab and has a pandas fallback for local/offline runs.
- The Hugging Face `evaluate` demo is controlled by `RUN_HF_EVALUATE_DEMO`. It is enabled by default in Colab and skipped locally unless the flag is set to `True`.
- The student notebook includes guided TODOs for selected corpus-EDA, vocabulary-inspection, tokenizer-inspection, vectorization, baseline-comparison, feature-inspection, error-analysis, Word2Vec document search, and nearest-neighbor cells. The metrics section is a guided demo with small numeric examples.
- LLMs are mentioned only as a bridge. Prompting, RAG, agents, evaluation, safety, and deployment belong in the next block.

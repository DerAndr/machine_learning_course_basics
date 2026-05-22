# Lecture 16 Notes: Natural Language Processing

These notes summarize the concepts used in the practical notebook.

## Text as Data

Natural language is unstructured for a machine, but an NLP workflow still needs the same supervised-learning discipline as other ML tasks:

- define documents and labels;
- split data before learning patterns from labels;
- inspect class balance, document lengths, tiny examples, and representative documents;
- inspect frequent terms before modeling to distinguish topic evidence from artifacts;
- avoid leakage from metadata, quoted text, or duplicated templates.

The practical uses 20 Newsgroups because it is small, familiar, and available through scikit-learn. We remove headers, footers, and quoted replies in the main workflow so that the classifier has to use message content rather than easy metadata shortcuts.

For NLP, EDA is not optional. Students should first check:

- how many documents appear in each split and class;
- whether some documents are empty or too short to carry evidence;
- whether one class has much longer posts than another;
- which words dominate the whole corpus and each label;
- whether examples contain metadata, quoted replies, signatures, or other formatting artifacts.

Only after that does it make sense to choose tokenization and feature extraction settings.

## Tokenization and Sparse Features

Classical NLP turns text into sparse vectors.

Common choices:

- `CountVectorizer`: counts how often each token or n-gram appears;
- `TfidfVectorizer`: downweights words that appear in many documents and upweights terms that are more specific to a document;
- stop-word removal: removes very common words when they are not helpful for the task;
- n-grams: represent short phrases such as `space shuttle` or `graphics card`.

Sparse vectors are interpretable and efficient, but they do not know that two different words can have similar meaning unless the training data and model weights make that relationship visible.

Useful package map:

- `scikit-learn`: `CountVectorizer`, `TfidfVectorizer`, linear classifiers, classical metrics;
- `datasets`: dataset objects and `.map(...)` preprocessing for corpus diagnostics;
- `tokenizers`: local WordPiece tokenizer training and inspection;
- `evaluate`: reusable classification metrics alongside scikit-learn reports;
- `sentence-transformers`: sentence/document embeddings for semantic search and reranking;
- `transformers`: pretrained subword tokenizer demo for encoder models;
- `spaCy`: production NLP pipelines, tokenization, NER, linguistic annotations;
- `NLTK`: classic teaching toolkit with corpora and linguistic algorithms;
- `Gensim`: topic modeling, vector-space models, and word/document embeddings.

## Classical Text Classification

A strong introductory baseline is TF-IDF plus a classical classifier. In practice, it is useful to compare several simple baselines before trusting one model.

These baselines are useful because they are:

- fast on CPU;
- easy to inspect through feature weights;
- competitive on many topic-classification tasks;
- honest about the ML workflow students already know: train, predict, evaluate, inspect errors.

The practical compares Naive Bayes, logistic regression, and a linear SVM on the same TF-IDF representation. Logistic regression is kept for interpretation because its class-level feature weights are easy to inspect.

Misclassified examples matter. They often reveal ambiguous topics, mixed vocabulary, short documents, or artifacts of preprocessing. The practical turns selected mistakes into short error-analysis cards: actual label, predicted label, document evidence, and a hypothesis.

## Text Similarity

Once text has become vectors, we can compare documents with cosine similarity.

Sparse TF-IDF similarity often works well when two documents share important vocabulary. It struggles when two texts are semantically close but use different words.

Dense embeddings solve a different problem. A sentence embedding model maps text into a lower-dimensional dense vector space where semantic neighbors can be close even when exact words differ.

Useful formulas:

$$
\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t)
$$

$$
\mathrm{idf}(t) = \log\frac{1 + N}{1 + \mathrm{df}(t)} + 1
$$

$$
\cos(\mathbf{a}, \mathbf{b}) =
\frac{\mathbf{a}^{\top}\mathbf{b}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{b} \rVert_2}
$$

For token embeddings $\mathbf{h}_1, \ldots, \mathbf{h}_T$:

$$
\mathbf{s}_{\mathrm{mean}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{h}_i
$$

With weights $w_i$:

$$
\mathbf{s}_{\mathrm{weighted}} = \frac{\sum_{i=1}^{T} w_i\mathbf{h}_i}{\sum_{i=1}^{T} w_i}
$$

The embedding matrix shape is:

$$
X \in \mathbb{R}^{n_{\mathrm{documents}} \times d_{\mathrm{embedding}}}
$$

Embeddings can be built at several levels:

- **word embeddings**: Word2Vec, GloVe, and fastText assign vectors to words or subword units. Classic dimensions are often 50, 100, 200, or 300;
- **contextual token embeddings**: BERT-style encoders produce one vector per token in context, so the representation of `pitcher` can differ in baseball and kitchen contexts;
- **sentence embeddings**: sentence-transformer models pool encoder outputs into one vector for a sentence or paragraph. `all-MiniLM-L6-v2` produces 384-dimensional vectors;
- **document embeddings**: long documents can be embedded by truncating, chunking into paragraphs, averaging chunk vectors, or indexing chunks separately;
- **pair embeddings / rerankers**: a cross-encoder reads query and document together, which is slower but often more accurate for final ranking.

Embedding length is model-specific. A sparse TF-IDF vector may have tens of thousands of dimensions because each vocabulary item gets a feature. A dense embedding usually has hundreds or thousands of dimensions, with most values non-zero. For cosine retrieval, vectors are often L2-normalized so that dot product and cosine similarity are equivalent.

PCA and UMAP are useful for visualizing embedding vectors, but the 2D plot is only an inspection tool. PCA is linear and highlights high-variance directions. UMAP is nonlinear and tries to preserve local neighborhoods. In both cases, apparent clusters should be treated as hypotheses to inspect, not as proof of semantic separation.

## How Sentence and Text Embeddings Are Calculated

Word vectors are not automatically sentence vectors. A sentence or text embedding requires a pooling or aggregation rule.

With static word vectors, a simple sentence embedding is the mean of known token embeddings:

$$
\mathbf{s}_{\mathrm{mean}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{e}_{w_i}
$$

This treats every known token equally. A weighted version can emphasize more informative words:

$$
\mathbf{s}_{\mathrm{weighted}} = \frac{\sum_{i=1}^{T}\alpha_i\mathbf{e}_{w_i}}{\sum_{i=1}^{T}\alpha_i}
$$

where $\alpha_i$ could be a TF-IDF weight, an attention weight, or another importance score.

With transformer encoders, the vectors being pooled are contextual hidden states $\mathbf{h}_i$. Padding tokens are excluded with an attention mask $m_i$:

$$
\mathbf{s}_{\mathrm{encoder}} = \frac{\sum_{i=1}^{T}m_i\mathbf{h}_i}{\sum_{i=1}^{T}m_i}
$$

For cosine search, the vector is often L2-normalized:

$$
\tilde{\mathbf{s}} = \frac{\mathbf{s}}{\lVert \mathbf{s} \rVert_2}
$$

Longer texts are usually chunked. Each chunk receives its own vector $\mathbf{s}_{c_j}$. You can either search over chunks directly or aggregate chunks into a document vector:

$$
\mathbf{d} = \frac{\sum_{j=1}^{M}\beta_j\mathbf{s}_{c_j}}{\sum_{j=1}^{M}\beta_j}
$$

where $\beta_j$ can be 1, chunk length, or a relevance weight. In retrieval systems, indexing chunks separately is often better than averaging a whole long document because it preserves local evidence.

## Word2Vec as the Bridge

Word2Vec is the classic bridge between sparse lexical features and modern neural embeddings. It does not create a vector for a whole document. It learns a dense vector for each word from the contexts where that word appears.

The key idea is the distributional hypothesis: words that occur in similar contexts tend to have related meanings. Word2Vec has two common training views:

- **CBOW** predicts a center word from its surrounding context words;
- **skip-gram** predicts surrounding context words from a center word.

For skip-gram, if $w_c$ is the center word and $w_o$ is an observed context word, the full-softmax view is:

$$
P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{o}^{\top}\mathbf{v}_{c})}{\sum_{j=1}^{|V|}\exp(\mathbf{u}_{j}^{\top}\mathbf{v}_{c})}
$$

Here $\mathbf{v}_{c}$ is the input vector for the center word and $\mathbf{u}_{o}$ is the output vector for the context word. Computing the denominator over the full vocabulary is expensive, so practical Word2Vec often uses negative sampling:

$$
\log \sigma(\mathbf{u}_{o}^{\top}\mathbf{v}_{c}) +
\sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}
\left[\log \sigma(-\mathbf{u}_{k}^{\top}\mathbf{v}_{c})\right]
$$

The trained embedding table has shape:

$$
E \in \mathbb{R}^{|V| \times d}
$$

In the practical, students train a small `gensim.Word2Vec` model on the 20 Newsgroups subset, inspect nearest words, and project word vectors with PCA/UMAP. To compare with document retrieval, they also average known word vectors:

$$
\mathbf{d}_{\mathrm{avg}} = \frac{1}{|T_d|}\sum_{w_i \in T_d} \mathbf{e}_{w_i}
$$

This mean-pooled document vector is useful as a baseline, but it loses word order, syntax, negation, and context. The word `pitcher` receives one static vector whether it appears in baseball text or in a kitchen sentence. This limitation motivates contextual encoders and sentence-transformer embeddings.

## Transformer Encoders

Encoder-style transformer models, such as BERT-style encoders and sentence-transformer models, are useful before generative LLMs.

Important ideas:

- subword tokenization splits unknown or rare words into reusable pieces;
- attention lets tokens exchange information with other tokens in the same sequence;
- contextual embeddings represent a token differently depending on nearby words;
- sentence embeddings compress a document or sentence into a reusable vector.

The practical uses a sentence-transformer checkpoint through the `sentence-transformers` package as an embedding extractor, not as a chat model.
In Colab, this embedding section should run on a T4 GPU runtime when available; the notebook detects CUDA and passes the device into `SentenceTransformer`.

## NLP Metrics

NLP metrics depend on the output shape.

For classification:

- accuracy is the share of correct labels;
- precision asks how many predicted positives were truly positive;
- recall asks how many true positives were found;
- F1 balances precision and recall;
- confusion matrices show which labels are confused with which other labels.

For retrieval:

- precision@k measures how many of the top `k` results are relevant;
- recall@k measures how many known relevant documents appeared in the top `k`;
- MRR rewards systems that put the first relevant document early;
- nDCG-style metrics reward relevant documents near the top of a ranked list.

For probabilistic language modeling:

- surprisal, or surprise, measures how unexpected one observed event is;
- entropy measures expected surprisal in a probability distribution;
- cross-entropy measures average surprisal assigned by the model to the true next token;
- perplexity is exponentiated cross-entropy.

Use one log base consistently. With base $b$, surprisal is:

$$
I_b(x) = -\log_b p(x)
$$

Bits use $b=2$:

$$
I_2(x) = -\log_2 p(x)
$$

Nats use $b=e$:

$$
I_e(x) = -\ln p(x)
$$

Entropy:

$$
H_b(P) = \mathbb{E}_{x \sim P}[-\log_b P(x)] = -\sum_x P(x)\log_b P(x)
$$

Cross-entropy:

$$
H_b(P, Q) = \mathbb{E}_{x \sim P}[-\log_b Q(x)] = -\sum_x P(x)\log_b Q(x)
$$

Empirical next-token cross-entropy:

$$
\hat{H}_b = -\frac{1}{T}\sum_{t=1}^{T}\log_b Q(x_t \mid x_{<t})
$$

Perplexity:

$$
\mathrm{PPL} = b^{\hat{H}_b}
$$

For bits:

$$
\mathrm{PPL} = 2^{\hat{H}_2}
$$

For nats:

$$
\mathrm{PPL} = \exp(\hat{H}_e)
$$

Perplexity can be read as a rough average branching factor. A perplexity of 10 means the model is about as uncertain as choosing among 10 equally likely next tokens at each step. Lower is better, but perplexity should only be compared on the same dataset, tokenization, and evaluation setup. It is not a complete metric for chat quality, factuality, safety, or usefulness.

The practical includes toy metric calculations on short sentences:

- classification examples count correct labels, precision, recall, and F1 by hand;
- retrieval examples compute precision@3, recall@3, and MRR from a ranked list;
- language-model examples compute token-level surprisal for a sentence such as `the cat sat`, then average it into cross-entropy and exponentiate it into perplexity.

## NLP Task Map

NLP includes many task shapes:

- classification: assign labels to documents;
- semantic search and retrieval: find texts similar to a query;
- clustering: group texts without labels;
- named entity recognition: identify spans such as people, places, organizations, and dates;
- summarization, translation, and question answering: generate or transform text.

This lecture focuses on the representation and classification foundation. Generation is reserved for the later LLM block.

## Bridge to LLMs

The final distinction is architectural:

- encoder models read a sequence and produce representations;
- decoder models predict the next token and can generate continuations;
- encoder-decoder models read one sequence and generate another.

LLMs belong in the next block because using them well involves prompting, retrieval-augmented generation, tool use, evaluation, safety, cost, latency, and deployment choices.

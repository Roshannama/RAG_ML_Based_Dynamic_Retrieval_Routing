# RAG_ML_Based_Dynamic_Retrieval_Routing
# Self-Adaptive RAG with Dynamic Retrieval & ML Routing

A local-hosted prototype of a **self-adaptive Retrieval-Augmented Generation (RAG) chatbot** that dynamically selects the most suitable retrieval strategy for each user query using a **machine-learning router**.

The system combines an LLM-based query analyzer, classical ML routing, multiple RAG retrieval strategies, retrieval enhancement techniques, user feedback, and an experience database to create a foundation for continuously improving retrieval decisions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [What Makes This Project Different](#-what-makes-this-project-different)
- [System Architecture](#-system-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Retrieval Pipelines](#-retrieval-pipelines)
- [Common Enhancement Layer](#-common-enhancement-layer)
- [Query Analyzer](#-query-analyzer)
- [ML Router](#-ml-router)
- [Why ML Routing](#-why-random-forest)
- [Experience & Feedback Learning](#-experience--feedback-learning)
- [Training Dataset Generation](#-training-dataset-generation)
- [Mathematical & ML Concepts](#-mathematical--ml-concepts)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Installation](#️-installation)
- [Environment Variables](#-environment-variables)
- [Running the Project](#️-running-the-project)
- [Training the Router](#-training-the-router)
- [Testing the Router](#-testing-the-router)
- [Data Flow](#-complete-runtime-flow)
- [Latency Considerations](#-latency-considerations)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Interview Explanation](#-interview-explanation)
- [License](#-license)

---

# 🧠 Overview

Traditional RAG systems generally follow a fixed retrieval strategy:

```text
User Query
    ↓
Embedding
    ↓
Vector Search
    ↓
Retrieved Documents
    ↓
LLM
    ↓
Answer
```

However, different questions have different retrieval requirements.

For example:

| Query                                                                | Potentially suitable strategy |
| -------------------------------------------------------------------- | ----------------------------- |
| "What is machine learning?"                                          | Dense                         |
| "What is the exact refund policy?"                                   | Hybrid                        |
| "Compare authentication and authorization."                          | Multi-query                   |
| "Explain this concept using related information from the documents." | HyDE                          |
| "What are the complete steps described across this document?"        | Parent Document               |

Instead of forcing every query through the same retrieval mechanism, this project introduces an **ML-based retrieval router**.

The resulting architecture is:

```text
                         USER QUERY
                             │
                             ▼
                    ┌─────────────────┐
                    │  Query Analyzer │
                    │      LLM        │
                    └────────┬────────┘
                             │
                             ▼
                    Structured Features
                             │
                             ▼
                    ┌─────────────────┐
                    │    ML Router    │
                    │ Random Forest   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
           Dense          Hybrid       Multi Query
              │              │              │
              └──────────────┼──────────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
                HyDE             Parent Document
                  │                     │
                  └──────────┬──────────┘
                             ▼
                  Common Enhancement Layer
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Metadata        Reranking      Compression
          Filtering                       │
              │                            ▼
              └──────────────► Reordering
                             │
                             ▼
                     Answer Generation
                          LLM
                             │
                             ▼
                           Answer
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 Feedback       Experience DB
                                      │
                                      ▼
                              Training Dataset
                                      │
                                      ▼
                              Retrain ML Router
```

---

# 🎯 Problem Statement

A fixed RAG retrieval strategy does not perform equally well for every type of query.

Dense retrieval may work well for semantic questions but struggle with exact keywords.

Keyword-based retrieval can work well for exact terminology but may fail when the user's query is expressed differently from the source document.

Some complex questions may require:

- multiple search perspectives,
- hypothetical document generation,
- broader parent-document context,
- or information from multiple documents.

Therefore, the problem addressed by this project is:

> **How can a RAG system automatically select the most appropriate retrieval strategy for a given query instead of using a single fixed retrieval method?**

The proposed solution uses:

1. An LLM Query Analyzer to understand the query.
2. A machine-learning classifier to select the retrieval pipeline.
3. Multiple retrieval strategies.
4. Common retrieval enhancement techniques.
5. User feedback and experience logging.
6. Offline evaluation to generate training labels.
7. Periodic retraining of the routing model.

---

# 🎯 Project Objectives

The major objectives are:

- Build a modular RAG chatbot.
- Analyze user queries using structured features.
- Dynamically select a retrieval strategy.
- Compare five retrieval approaches.
- Improve retrieved context using reranking and compression.
- Capture retrieval and response information.
- Collect user feedback.
- Build an experience dataset.
- Retrain the ML router using accumulated experience.
- Evaluate routing performance using classification metrics.

---

# 🚀 What Makes This Project Different?

The main idea is **adaptive retrieval**.

A traditional RAG system:

```text
Every Query
    ↓
Same Retriever
    ↓
Answer
```

This project:

```text
Query
  ↓
Analyze Query
  ↓
Understand Query Characteristics
  ↓
ML Router
  ↓
Choose Retrieval Strategy
  ↓
Retrieve
  ↓
Enhance Context
  ↓
Generate Answer
```

The important distinction is that the **LLM is not responsible for deciding the retrieval pipeline directly**.

Instead:

```text
LLM
 ↓
Query Features

ML Model
 ↓
Retrieval Pipeline
```

This makes the routing decision:

- measurable,
- trainable,
- reproducible,
- and independently evaluatable.

---

# 🏗️ System Architecture

## High-Level Architecture

```text
                         ┌───────────────┐
                         │     User      │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    FastAPI    │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Query Analyzer│
                         │     LLM       │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   ML Router   │
                         │ Random Forest │
                         └───────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
           Dense              Hybrid           Multi Query
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
                  HyDE                 Parent Document
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                      Common Enhancements
                                 │
             ┌───────────────────┼──────────────────┐
             ▼                   ▼                  ▼
       Metadata Filter        Reranker         Compression
                                                    │
                                                    ▼
                                                Reordering
                                                    │
                                                    ▼
                                             Context Builder
                                                    │
                                                    ▼
                                            Generation LLM
                                                    │
                                                    ▼
                                                 Answer
                                                    │
                            ┌───────────────────────┴────────────┐
                            ▼                                    ▼
                        Feedback                           Experience DB
                                                                 │
                                                                 ▼
                                                          Training Dataset
                                                                 │
                                                                 ▼
                                                            ML Retraining
```

---

# 🔄 End-to-End Workflow

## Step 1 — User Query

The user submits a question through the FastAPI chatbot.

Example:

```text
How does JWT authentication work and what are its advantages?
```

---

## Step 2 — Query Analyzer

The query is sent to the Query Analyzer LLM.

The analyzer converts natural language into structured features.

Example:

```json
{
  "query_type": "conceptual",
  "complexity": "medium",
  "multi_concept": true,
  "keyword_heavy": false,
  "requires_multiple_documents": true,
  "requires_exact_match": false
}
```

The router does not directly consume the natural-language query.

It consumes these structured features.

---

# 🤖 Query Analyzer

The Query Analyzer is an **LLM-based feature extraction component**.

It is not the ML router.

Its responsibility is:

```text
Natural Language
      ↓
LLM
      ↓
Structured Query Features
```

Current feature set:

| Feature                       | Description                              |
| ----------------------------- | ---------------------------------------- |
| `query_type`                  | Type of question                         |
| `complexity`                  | Low / medium / high                      |
| `multi_concept`               | Whether multiple concepts are involved   |
| `keyword_heavy`               | Whether exact terminology is important   |
| `requires_multiple_documents` | Whether multiple sources may be required |
| `requires_exact_match`        | Whether exact matching is important      |

### Query Types

The current analyzer supports:

```text
definition
factual
comparison
procedural
multi_hop
troubleshooting
```

---

# 🧮 ML Router

The ML Router receives the six extracted features.

```text
Query Analyzer
      │
      ▼
6 Features
      │
      ▼
Random Forest
      │
      ▼
Pipeline
```

The router predicts one of:

```text
dense
hybrid
multi_query
hyde
parent_document
```

The router is implemented using:

```text
RandomForestClassifier
```

with preprocessing:

```text
Categorical Features
        ↓
OneHotEncoder

Boolean Features
        ↓
Passthrough

        ↓
Random Forest
```

---

# 🌳 Why Random Forest?

Random Forest is suitable for this prototype because the routing problem is a relatively small tabular classification problem.

Input:

```text
query_type
complexity
multi_concept
keyword_heavy
requires_multiple_documents
requires_exact_match
```

Output:

```text
retrieval_pipeline
```

Random Forest provides:

- nonlinear decision boundaries,
- robustness on small datasets,
- categorical feature compatibility after encoding,
- class weighting,
- probability estimates,
- relatively simple training,
- interpretable feature importance.

---

# 🔀 Retrieval Pipelines

The project currently contains five retrieval strategies.

---

## 1. Dense Retrieval

Dense retrieval converts the query into an embedding and performs semantic similarity search.

```text
Query
 ↓
Embedding Model
 ↓
Vector
 ↓
Pinecone
 ↓
Similar Documents
```

Useful for:

- conceptual questions,
- semantic matching,
- natural-language questions.

---

## 2. Hybrid Retrieval

Hybrid retrieval combines semantic and keyword-based retrieval.

```text
             Query
             /   \
            /     \
           ▼       ▼
       Dense      BM25
       Search     Search
           \       /
            \     /
             ▼   ▼
          Combined
           Results
```

Useful when both:

- semantic similarity,
- and exact keywords

are important.

---

## 3. Multi-Query Retrieval

A single query is transformed into multiple related queries.

Example:

```text
Original:
How does JWT authentication work?

Generated:
1. How does JWT authentication work?
2. JWT authentication process
3. JWT token validation mechanism
4. JWT authentication advantages
```

Each query is retrieved separately.

The results are then combined.

This can improve recall by searching from multiple perspectives.

---

## 4. HyDE

HyDE stands for **Hypothetical Document Embeddings**.

Instead of embedding only the original question:

```text
Question
   ↓
Embedding
   ↓
Search
```

HyDE generates a hypothetical answer/document:

```text
Question
   ↓
LLM
   ↓
Hypothetical Document
   ↓
Embedding
   ↓
Vector Search
```

The generated hypothetical document can be semantically closer to the actual source documents.

---

## 5. Parent Document Retrieval

Parent Document Retrieval separates:

```text
retrieval granularity
```

from:

```text
context granularity
```

A smaller child chunk can be used for similarity search.

Once a relevant child chunk is found, its larger parent document/chunk can be returned.

```text
Large Parent Document
        │
        ├── Child Chunk 1
        ├── Child Chunk 2
        ├── Child Chunk 3
        └── Child Chunk 4
```

Search:

```text
Query
 ↓
Child Chunks
 ↓
Relevant Child
 ↓
Parent Document
```

This provides broader context to the generation model.

---

# 🛠️ Common Enhancement Layer

Regardless of which retrieval pipeline is selected, the retrieved documents pass through a common enhancement layer.

```text
Retrieved Documents
        ↓
Metadata Filtering
        ↓
Reranking
        ↓
Context Compression
        ↓
Long Context Reordering
        ↓
Final Context
```

---

## Metadata Filtering

Metadata can be used to restrict documents.

Examples:

```text
department = engineering
document_type = policy
year = 2026
product = xyz
```

The current implementation contains a placeholder for this functionality.

---

# 🔎 Reranking

Initial retrieval usually produces candidates rather than perfectly ordered results.

A reranker evaluates:

```text
Query + Document
```

and produces a relevance score.

Current reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Conceptually:

```text
Query
  +
Document
  ↓
Cross Encoder
  ↓
Relevance Score
```

Documents are then sorted according to relevance.

---

# 🗜️ Context Compression

Retrieval can return documents containing a lot of irrelevant information.

Context compression attempts to keep only the information useful for the current query.

```text
Large Document
      ↓
Relevant Sentences
      ↓
Compressed Context
```

This helps reduce:

- context size,
- token consumption,
- irrelevant information.

---

# 🔄 Long-Context Reordering

Even when relevant documents are retrieved, their ordering can affect LLM performance.

The reordering layer changes the sequence of retrieved information before generation.

```text
Retrieved Documents
       ↓
Reordering
       ↓
Optimized Context Order
       ↓
LLM
```

---

# 🧠 Answer Generation

After retrieval and enhancement:

```text
Final Context
      +
User Query
      ↓
Generation LLM
      ↓
Final Answer
```

The generation prompt instructs the model to:

- answer only from the provided context,
- avoid outside knowledge,
- state that information is unavailable when it cannot be found.

---

# 💾 Experience & Feedback Learning

The system stores information about every interaction.

An experience record can contain:

```text
query
answer
query_features
selected_pipeline
retrieved_contexts
latency
feedback
timestamp
```

Conceptually:

```text
User Query
    ↓
Query Analysis
    ↓
ML Routing
    ↓
Retrieval
    ↓
Answer
    ↓
Experience DB
```

The user can provide:

```text
👍 Good
👎 Bad
```

feedback.

---

# ⚠️ Feedback Is Not RLHF

A thumbs-up/down signal is treated as a feedback label.

It is **not automatically RLHF**.

The current learning approach is:

```text
Experience
    ↓
Feedback / Evaluation
    ↓
Dataset
    ↓
Supervised ML
    ↓
Router Retraining
```

RLHF would instead involve preference/reward optimization of a generative model.

---

# 📊 Training Dataset Generation

The router requires training examples such as:

```text
query_type
complexity
multi_concept
keyword_heavy
requires_multiple_documents
requires_exact_match
best_pipeline
```

Example:

```csv
query_type,complexity,multi_concept,keyword_heavy,requires_multiple_documents,requires_exact_match,best_pipeline
definition,low,false,false,false,false,dense
factual,medium,false,true,false,true,hybrid
comparison,medium,true,false,true,false,multi_query
conceptual,medium,true,false,true,false,hyde
multi_hop,high,true,false,true,false,parent_document
```

---

# 🏷️ Ground Truth Generation

A major design decision is that a manually assigned:

```text
suggested_pipeline
```

should **not automatically be treated as ground truth**.

Instead, for offline dataset creation:

```text
Seed Query
     ↓
Query Analyzer
     ↓
Run Pipeline 1
Run Pipeline 2
Run Pipeline 3
Run Pipeline 4
Run Pipeline 5
     ↓
Evaluate Results
     ↓
Select Best Pipeline
     ↓
best_pipeline
```

This produces a more meaningful routing dataset.

---

# 🧪 Offline Benchmark

Suppose the benchmark contains:

```text
50 queries
```

For each query:

```text
             Query
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
    Dense    Hybrid   Multi-query
      │        │        │
      └────────┼────────┘
               │
         ┌─────┴─────┐
         ▼           ▼
       HyDE      Parent Doc
         │           │
         └─────┬─────┘
               ▼
          Evaluation
               │
               ▼
         Best Pipeline
```

The resulting dataset can be used to train the router.

---

# 📐 Mathematical & ML Concepts

## Cosine Similarity

Dense retrieval commonly uses cosine similarity:

$$
\text{similarity}(A,B)
=
\frac{A \cdot B}
{\|A\|\|B\|}
$$

where:

- \(A\) = query embedding
- \(B\) = document embedding

Values closer to 1 indicate greater directional similarity.

---

## TF-IDF / BM25

Hybrid retrieval can use lexical retrieval such as BM25.

BM25 scores a document based on:

- term frequency,
- inverse document frequency,
- document length.

A simplified form is:

$$
BM25(D,Q)
=
\sum_{t \in Q}
IDF(t)
\frac{f(t,D)(k_1+1)}
{f(t,D)+k_1(1-b+b\frac{|D|}{avgdl})}
$$

where:

- \(f(t,D)\) = frequency of term \(t\) in document \(D\)
- \(|D|\) = document length
- \(avgdl\) = average document length
- \(k_1,b\) = tuning parameters.

---

# 🌲 Random Forest

Random Forest combines multiple decision trees.

For classification:

$$
\hat{y}
=
\operatorname{mode}
\{T_1(x),T_2(x),...,T_n(x)\}
$$

where:

- \(T_i\) = individual decision tree
- \(x\) = query feature vector
- \(\hat{y}\) = predicted pipeline.

---

# 📊 Router Confidence

The Random Forest can provide class probabilities.

For example:

```text
dense           0.08
hybrid          0.71
multi_query     0.12
hyde            0.05
parent_document 0.04
```

The router prediction becomes:

```text
hybrid
```

with confidence:

```text
0.71
```

The confidence can be logged in the experience database.

---

# 📈 Evaluation

The router should not be evaluated only using accuracy.

Important metrics include:

### Accuracy

$$
Accuracy =
\frac{Correct\ Predictions}
{Total\ Predictions}
$$

### Precision

$$
Precision =
\frac{TP}{TP+FP}
$$

### Recall

$$
Recall =
\frac{TP}{TP+FN}
$$

### F1 Score

$$
F1 =
2
\frac{Precision \times Recall}
{Precision + Recall}
$$

For a multi-class router, **macro F1** is particularly useful because it gives equal importance to each retrieval class.

---

# 🧪 RAG Evaluation

The retrieval system can also be evaluated independently.

Possible metrics include:

```text
Precision@K
Recall@K
MRR
Hit Rate
Context Relevance
Answer Relevance
Faithfulness
```

For example:

$$
Recall@K =
\frac{\text{Relevant documents retrieved in top K}}
{\text{Total relevant documents}}
$$

---

# ⏱️ Latency Evaluation

The system records total RAG latency.

Conceptually:

$$
T_{total}
=
T_{analysis}
+
T_{routing}
+
T_{retrieval}
+
T_{enhancement}
+
T_{generation}
$$

Where:

```text
T_analysis      → Query Analyzer LLM
T_routing       → ML Router
T_retrieval     → Selected pipeline
T_enhancement   → Reranking + compression + reordering
T_generation    → Answer LLM
```

This is important because some retrieval strategies require additional LLM calls.

---

# 📁 Project Structure

A typical project structure is:

```text
self-learning-rag/
│
├── app.py
├── rag.py
├── router.py
├── train.py
├── query_analyzer.py
│
├── dense.py
├── hybrid.py
├── multi_query.py
├── hyde.py
├── parent_doc.py
│
├── Reranker.py
├── compression.py
├── reordering.py
│
├── database.py
├── models.py
├── schema.py
│
├── router_dataset.csv
├── router_model.pkl
│
├── chat.db
│
├── templates/
│   └── index.html
│
├── .env
├── requirements.txt
└── README.md
```

---

# 🧰 Technology Stack

## Backend

```text
Python
FastAPI
SQLAlchemy
SQLite
```

## LLM

```text
OpenRouter API
NVIDIA Nemotron
```

## RAG

```text
LangChain
Pinecone
Sentence Transformers
BM25
```

## Machine Learning

```text
Scikit-learn
Random Forest
Pandas
NumPy
Joblib
```

## Retrieval Enhancement

```text
Cross Encoder Reranker
Context Compression
Long Context Reordering
Metadata Filtering
```

## Frontend

```text
HTML
CSS
Jinja2
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd self-learning-rag
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Add any additional configuration required by the retrieval modules.

---

# ▶️ Running the Project

Start the FastAPI application:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 🧠 Training the Router

The training dataset should contain:

```text
query
query_type
complexity
multi_concept
keyword_heavy
requires_multiple_documents
requires_exact_match
best_pipeline
```

Train the model:

```bash
python train.py
```

The resulting model is:

```text
router_model.pkl
```

---

# 🔌 Router Inference

The trained model is loaded by:

```text
router.py
```

The inference flow is:

```text
features
   ↓
Pandas DataFrame
   ↓
Saved sklearn Pipeline
   ↓
Random Forest
   ↓
Pipeline prediction
```

The model should receive the **same six features used during training**.

---

# 🧪 Testing the Router

Example test:

```python
from router import route_query

features = {
    "query_type": "factual",
    "complexity": "medium",
    "multi_concept": False,
    "keyword_heavy": True,
    "requires_multiple_documents": False,
    "requires_exact_match": True
}

result = route_query(features)

print(result)
```

Possible result:

```python
{
    "pipeline": "hybrid",
    "confidence": 0.82
}
```

---

# 🔄 Complete Runtime Flow

```text
                     USER
                      │
                      ▼
                 FastAPI UI
                      │
                      ▼
               User Question
                      │
                      ▼
              Query Analyzer
                   (LLM)
                      │
                      ▼
              Query Features
                      │
                      ▼
                ML Router
             Random Forest
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Dense       Hybrid     Multi Query
          │           │           │
          └───────────┼───────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
            HyDE        Parent Document
              │               │
              └───────┬───────┘
                      ▼
              Retrieved Documents
                      │
                      ▼
              Metadata Filtering
                      │
                      ▼
                  Reranking
                      │
                      ▼
               Compression
                      │
                      ▼
                Reordering
                      │
                      ▼
              Final Context
                      │
                      ▼
             Generation LLM
                      │
                      ▼
                   Answer
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         User                Experience
       Feedback                Database
                                │
                                ▼
                         Training Dataset
                                │
                                ▼
                          Router Retraining
```

---

# 🔁 Self-Learning Loop

The adaptive component of the project can be represented as:

```text
              ┌──────────────────────┐
              │      User Query      │
              └──────────┬───────────┘
                         ▼
                  Query Analyzer
                         ▼
                    ML Router
                         ▼
                 Retrieval Pipeline
                         ▼
                  Answer Generation
                         ▼
                       User
                         │
                    Feedback
                         │
                         ▼
                 Experience Database
                         │
                         ▼
                 Dataset Generation
                         │
                         ▼
                  Router Retraining
                         │
                         └──────────────► ML Router
```

This creates an iterative improvement cycle.

---

# ⚡ Latency Considerations

The system contains multiple components that can contribute to latency.

For example:

```text
Query Analyzer
     ↓
ML Router
     ↓
Multi Query / HyDE
     ↓
Retrieval
     ↓
Reranking
     ↓
Compression
     ↓
Generation
```

The ML router itself is inexpensive compared with LLM API calls.

The major latency contributors are generally:

```text
LLM calls
Network calls
Multi-query generation
HyDE generation
Reranking
```

The architectural advantage is that **only the selected retrieval pipeline is executed during normal online inference**.

The system does not normally execute all five pipelines for every user query.

---

# 🧪 Offline vs Online Architecture

## Online

For a real user query:

```text
Query
 ↓
Analyzer
 ↓
Router
 ↓
ONE selected pipeline
 ↓
Enhancement
 ↓
Generation
```

This minimizes latency.

---

## Offline

For training-data creation:

```text
Query
 ↓
Evaluate Dense
 ↓
Evaluate Hybrid
 ↓
Evaluate Multi-query
 ↓
Evaluate HyDE
 ↓
Evaluate Parent Document
 ↓
Compare
 ↓
Best Pipeline
```

This is more expensive but allows the system to learn which retrieval method works best.

---

# ⚠️ Important Design Decision

The system does **not** retrain the generation LLM every time feedback is received.

Instead:

```text
Feedback
   ↓
Experience
   ↓
Router Dataset
   ↓
ML Router
```

The generation LLM remains unchanged.

This keeps the prototype simpler and makes the adaptive component clearly measurable.

---

# 📌 Limitations

This is a prototype system, so there are several limitations.

### 1. Query Analyzer Dependency

The quality of the router depends on the quality of the features produced by the Query Analyzer.

### 2. Small Training Dataset

A small dataset can result in an unstable classifier.

### 3. Synthetic Labels

If pipeline labels are generated automatically, the quality of the evaluation method determines the quality of the training data.

### 4. Feedback Ambiguity

A thumbs-down does not necessarily mean that the retrieval pipeline was wrong.

The problem could be:

```text
Retrieval
Generation
Context
Question ambiguity
Prompt
```

Therefore, feedback should be used carefully.

### 5. API Latency

LLM-based Query Analysis, Multi-query and HyDE introduce additional latency.

### 6. Prototype Database

SQLite is suitable for local development but is not intended as the production database for a high-concurrency deployment.

---

# 🚀 Future Improvements

Possible future improvements include:

## 1. Better Router Features

Add numerical features such as:

```text
query length
number of entities
number of keywords
number of sentences
question-word type
embedding-based complexity
```

---

## 2. Better ML Models

Compare:

```text
Random Forest
XGBoost
LightGBM
Logistic Regression
SVM
Neural Network
```

---

## 3. Better Routing Objective

Instead of predicting only:

```text
best_pipeline
```

the router could predict:

```text
pipeline
+
expected quality
+
expected latency
```

Then routing could optimize:

$$
Utility =
Quality - \lambda \times Latency
$$

where \(\lambda\) controls the importance of latency.

---

## 4. Confidence-Based Routing

If:

```text
confidence < threshold
```

the system could use a safer fallback strategy.

For example:

```text
ML Router
    │
    ├── High Confidence → Selected Pipeline
    │
    └── Low Confidence → Hybrid
```

---

## 5. Online Learning

Future versions could periodically retrain the router using newly collected experiences.

```text
New Experiences
       ↓
Data Validation
       ↓
Dataset Update
       ↓
Retraining
       ↓
Model Evaluation
       ↓
Model Versioning
       ↓
Deployment
```

---

## 6. MLOps

The project can later incorporate:

```text
MLflow
DVC
Docker
CI/CD
Model Registry
Monitoring
```

to create a complete ML lifecycle.

---

# 💡 Key Technical Insight

The most important architectural separation is:

```text
             ┌────────────────────┐
             │   Query Analyzer   │
             │        LLM         │
             └─────────┬──────────┘
                       │
                       ▼
                Query Features
                       │
                       ▼
             ┌────────────────────┐
             │     ML Router      │
             │  Random Forest     │
             └─────────┬──────────┘
                       │
                       ▼
               Retrieval Strategy
```

The Query Analyzer **understands the query**.

The ML Router **makes the retrieval decision**.

The retrieval pipelines **retrieve the evidence**.

The enhancement layer **improves the evidence**.

The generation LLM **generates the final answer**.

The experience layer **captures information for future improvement**.

---

# 📊 Project Summary

| Component        | Responsibility                  |
| ---------------- | ------------------------------- |
| FastAPI          | Backend/API                     |
| Query Analyzer   | Extract query features          |
| ML Router        | Select retrieval pipeline       |
| Dense            | Semantic retrieval              |
| Hybrid           | Semantic + keyword retrieval    |
| Multi-query      | Multiple query perspectives     |
| HyDE             | Hypothetical-document retrieval |
| Parent Document  | Broader contextual retrieval    |
| Metadata Filter  | Restrict documents              |
| Reranker         | Improve document ordering       |
| Compression      | Remove irrelevant context       |
| Reordering       | Optimize context sequence       |
| Generation LLM   | Generate final answer           |
| SQLite           | Store conversations/experiences |
| Pinecone         | Vector retrieval                |
| Feedback         | Capture user signal             |
| Training Dataset | Improve router                  |
| Random Forest    | Routing classifier              |

---

# 🏁 Final Architecture

```text
                         ┌─────────────────────┐
                         │        USER         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query Analyzer    │
                         │        LLM          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query Features    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      ML Router      │
                         │   Random Forest     │
                         └──────────┬──────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
       Dense                    Hybrid                  Multi Query
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                       HyDE            Parent Document
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Metadata Filtering  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │      Reranker       │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Context Compression │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Long Context Order  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Generation LLM    │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │       ANSWER        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                     Feedback           Experience DB
                                               │
                                               ▼
                                      Training Dataset
                                               │
                                               ▼
                                        ML Retraining
                                               │
                                               └──────► Router
```

---

# 📄 License

This project is developed as an educational/research prototype for demonstrating adaptive RAG, retrieval routing, machine learning, and feedback-driven system improvement.

---

## ⭐ Core Idea

```text
Fixed RAG:
Query → One Retriever → Answer

This Project:
Query
  ↓
Understand Query
  ↓
ML Routing
  ↓
Choose Best Retriever
  ↓
Enhance Retrieved Context
  ↓
Generate Answer
  ↓
Collect Experience
  ↓
Improve Router
```

**The central research/engineering idea is:**

> **Make retrieval adaptive rather than fixed, while keeping query understanding, routing, retrieval, enhancement, generation, and learning as separate measurable components.**

# Bridging the Preference Gap between Retrievers and LLMs (BGM) 🌉

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Model](https://img.shields.io/badge/Model-Llama--3.2--1B-orange.svg)

---

## 📖 Description

This repository contains the implementation and documentation of the **Bridge the Gap Model (BGM)**, a framework designed to improve **Retrieval-Augmented Generation (RAG)** systems by addressing the *preference gap* between retrieval models and Large Language Models (LLMs).

In standard RAG pipelines, retrievers and LLMs operate independently. As a result, retrieved documents that are semantically relevant may still be suboptimal or even harmful for downstream generation.  
BGM introduces an intermediate adaptation layer that aligns retrieved information with the preferences of the LLM, improving generation accuracy and robustness.

---

## 🎯 Motivation

Traditional retrievers rank documents based on semantic similarity to the query. However, **semantic relevance does not guarantee usefulness for generation**.

This mismatch leads to the **Preference Gap**:
- Retrievers optimize for similarity
- LLMs optimize for generation quality

BGM bridges this gap by explicitly modeling document usefulness from the LLM’s perspective.

---

## 🚀 Key Features

- **Document Selection**  
  Filters out irrelevant or misleading documents that could degrade generation quality.

- **Document Re-ranking**  
  Reorders retrieved documents to prioritize the most useful information, mitigating issues such as *Lost in the Middle*.

- **No-doc Forwarding**  
  Outputs a special `NO_DOCS` token when no retrieved document is sufficiently useful, reducing hallucinations.

- **Plug-and-Play Design**  
  Can be integrated into existing RAG pipelines **without retraining** the retriever or the LLM.

---

## 🏗️ Architecture

- **Base Model**: Llama-3.2-1B (Meta)
- **Input**: User query + retrieved documents
- **Output**: Ordered and filtered document sequence (or `NO_DOCS`)
- **Model Type**: Auto-regressive sequence model

BGM operates as a lightweight intermediate module between the retriever and the LLM.

---

## 📚 Dataset

- **Dataset**: Natural Questions Open
- **Final Corpus Size**: ~21 million documents (after filtering)
- **Task**: Question Answering and generation

---

## ⚙️ Training Methodology

- **Training Strategy**: Supervised Fine-Tuning (SFT)
- **Training Library**: Hugging Face TRL
- **Data Construction**:
  - Greedy search algorithm to identify optimal document sequences
  - Optimization targets include **Exact Match (EM)** and **BLEU score**

---

## 📊 Experimental Results

Evaluation performed on **2,889 test examples** from Natural Questions Open.

| Pipeline | Metric | Score |
|--------|--------|-------|
| RAG Standard | Exact Match (EM) | 51.40 |
| **RAG + BGM** | **Exact Match (EM)** | **65.59** |

**Absolute improvement: +14.19 EM**

---

## 🛠️ Installation

### Requirements
- Python ≥ 3.10
- CUDA-compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/solidx00/Bridge_the_GAP.git
cd Bridge_the_GAP
pip install -r requirements.txt

# Cross-Subject Imagined Speech EEG-to-Text Decoder

A **cross-subject, CLIP-style contrastive learning system** for decoding *sentence-level imagined speech* from EEG, evaluated on the **ChiSCO** dataset.

This project demonstrates that **semantic alignment between EEG and text is possible across subjects**, despite extremely low inter-subject neural correlation.

---

## 🎯 Key Result

**17.0% Top-1 accuracy on 200-way EEG → text retrieval**,
**34× better than chance (0.5%)**, without subject-specific calibration.

> Achieved despite inter-subject EEG correlations of only **r = 0.13–0.17**, meaning ~87% of the signal is subject-specific noise.

---

## 📊 Performance Summary

### Retrieval Metrics (Test Set)

| Metric                     | Value       | Chance    |
| -------------------------- | ----------- | --------- |
| Top-1 Accuracy             | **17.0%**   | 0.5%      |
| Top-5 Accuracy             | **47.5%**   | 2.5%      |
| Mean Reciprocal Rank (MRR) | **0.327**   | 0.005     |
| Median Rank                | **6 / 200** | 100 / 200 |
| Separation Gap             | **0.509**   | ~0.0      |

**Training accuracy:** 34% (proof of learning)
**Test accuracy:** 17% (cross-subject generalization with limited subjects)

---

## 🧠 Why This Matters

Most EEG-based imagined speech decoders require **900+ minutes of per-user calibration**.

This work explores a **subject-invariant alternative**:

* No subject IDs
* No per-user fine-tuning
* Open-vocabulary sentence retrieval

This is a step toward **practical, calibration-light BCIs**.

---
## 🧠 Neural Embeddings & Shared Latent Space

This project is built around the concept of **neural embeddings**.

Rather than directly predicting text from EEG, both modalities are encoded into a **shared continuous embedding space** (ℝ⁵¹²):

- **EEG signals** → EEG encoder → 512-D embedding  
- **Text sentences** → BERT encoder → 512-D embedding  

In this shared space:
- **Correct EEG–text pairs lie close together**
- **Incorrect pairs are far apart**

The task is therefore framed as **cross-modal retrieval**, not classification or sequence generation.

---

### Why Neural Embeddings?

Neural embeddings enable:
- Modality-agnostic semantic representations
- Robust cross-subject generalization
- Scalable evaluation without fixed vocabularies
- Natural handling of open-set retrieval problems

This is especially important for EEG, where:
- Signals are noisy and subject-specific
- Absolute decoding is unreliable
- Relative similarity is more stable than direct prediction

---

### What the Model Actually Learns

The model does **not** learn word boundaries or phonemes.

Instead, it learns to **align distributions**:

- EEG embeddings are pulled toward their matching text embeddings
- Non-matching pairs are pushed away
- Training uses a **CLIP-style contrastive objective**

This encourages the model to capture **semantic intent**, not surface-level signal patterns.

---

### Why Retrieval-Based Evaluation?

Because outputs are embeddings, performance is measured using **ranking metrics**, not token accuracy:

- Top-K Accuracy
- Mean Reciprocal Rank (MRR)
- Median Rank
- Recall@K
- Chance-normalized accuracy
- Similarity separation gap

These metrics reveal:
- Whether failures are near-misses or catastrophic
- Whether embeddings are collapsed or well-structured
- Whether learning exceeds chance by a meaningful margin

---

### Interpreting the Results

Strong retrieval performance (e.g., **17% Top-1 over 200 candidates**) indicates that the learned **embedding geometry is meaningful**, even across subjects with extremely low EEG correlation.

This validates the core hypothesis:
> *Imagined speech EEG contains transferable semantic structure when represented as neural embeddings.*


## 🏗️ Architecture Overview

**Dual-Encoder Contrastive Learning (CLIP-inspired)**

### EEG Branch

```
Raw EEG (125 × 1650)
        ↓
Conv1D Feature Extractor
(125 → 256 → 512)
        ↓
Time Projection (512 → 512)
        ↓
Positional Encoding
        ↓
Transformer Encoder
(2 layers, 8 heads)
        ↓
Temporal Mean Pooling
        ↓
L2 Normalization
→ 512-D EEG Embedding
```

### Text Branch

```
Chinese Text
        ↓
BERT-base-chinese
        ↓
[CLS] Token
        ↓
Linear Projection (768 → 512)
        ↓
L2 Normalization
→ 512-D Text Embedding
```

### Training Objective

```
Cosine Similarity (EEG × Text)
        ↓
Bidirectional CLIP Contrastive Loss
```

EEG and text embeddings are trained to align **only when they represent the same sentence**.

---

## 🔬 Core Idea: Contrastive Alignment

For a batch of size `B`, the model learns a similarity matrix:

```
EEG_i ↔ Text_j   →   High similarity only when i == j
```

This forces the model to:

* Pull matching EEG–text pairs together
* Push non-matching pairs apart
* Learn **semantic features**, not subject-specific patterns

---

## 🧪 Diagnostic Findings

### ✅ Permutation Test

| Condition       | Accuracy  |
| --------------- | --------- |
| True labels     | **17.0%** |
| Shuffled labels | ~0.5%     |

→ Confirms genuine EEG–text learning

### ⚠️ Representation Compression

* Embedding standard deviation: **0.042**
* Expected healthy diversity: ~0.2

This indicates **low embedding diversity**, not full collapse. Improving representation richness is a key next step.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/HuzyafaK/ChiSCO-CrossSubject-Decoder.git
cd ChiSCO-CrossSubject-Decoder
pip install -r requirements.txt
```

### Dataset

Download the **ChiSCO dataset**:

* OpenNeuro: `ds005170`
* Place `.pkl` files under `data/`

### Training

```bash
python src/train.py \
  --data_glob "data/**/*.pkl" \
  --subjects S1 S2 \
  --batch_size 32 \
  --epochs 30
```

### Evaluation

```bash
python src/evaluate.py \
  --checkpoint_path chisco_clip_best_model.pt
```

---

## 🔧 Memory-Efficient Design

Designed to run on **Kaggle / single-GPU systems**:

* Lazy EEG loading (index-based)
* Streaming mean/std normalization
* Chunked gallery encoding
* No O(N²) similarity matrices

---

## 📚 Dataset Attribution

This work builds on the **ChiSCO (Chinese Imagined Speech Corpus)**:

* Zhang et al., *Scientific Data*, 2024
* 6,681 trials × 3 subjects
* EEG: 125 channels @ 500 Hz
* Sentence-level imagined speech

This project is **independent** and **not affiliated** with the original authors.

---

## 🛠️ Tech Stack

* PyTorch
* Hugging Face Transformers (BERT)
* NumPy, tqdm, pickle
* NLTK (BLEU), ROUGE

---

## 🎯 Future Work

* Diversity regularization to improve embeddings
* Attention-based temporal pooling
* Temperature scheduling
* Few-minute subject adaptation
* Self-supervised EEG pretraining

---

## 📖 Citation

```bibtex
@software{khokhar2025crosssubject,
  author = {Khokhar, Muhammad Huzyafa},
  title = {Cross-Subject Imagined Speech EEG-to-Text Decoder},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HuzyafaK/ChiSCO-CrossSubject-Decoder}
}
```

---

## 📄 License

MIT License

---

Built with 🧠 by **Muhammad Huzyafa Khokhar**
Excelleve — Non-invasive Thought-to-Speech Systems

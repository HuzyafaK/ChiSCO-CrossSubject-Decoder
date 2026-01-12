<details>
<summary><b>Click to expand README.md (copy entire content)</b></summary>
````markdown
# Cross-Subject Imagined Speech EEG-to-Text Decoder
First cross-subject contrastive learning system for sentence-level imagined speech decoding
Show Image
Show Image
Show Image

🎯 Key Achievement
Achieved 17% top-1 accuracy on 200-way EEG-to-text retrieval (34x better than chance) despite extremely low inter-subject correlation (r=0.13-0.17) on the ChiSCO imagined speech dataset.
Why this matters: Most EEG decoders require 900+ minutes of per-user calibration. This cross-subject approach eliminates that barrier, making BCIs practical for real-world deployment.
Built on ChiSCO Dataset
This work builds upon the ChiSCO (Chinese Imagined Speech Corpus) dataset created by Zhang et al. (2024) at Harbin Institute of Technology—the first dataset for sentence-level imagined speech with 6,681 trials per subject. The ChiSCO team's rigorous data collection (900+ minutes per subject, attention checks, quality controls) and their pioneering work on imagined speech paradigms made this cross-subject research possible.
Key insight from ChiSCO paper: Inter-subject correlations of 0.13-0.17 indicate that EEG signals are ~87% subject-specific—making cross-subject learning the critical challenge for practical BCIs.

📊 Performance Metrics
MetricValueBaselineTop-1 Accuracy17.0%0.5% (chance)Top-5 Accuracy47.5%2.5% (chance)Mean Reciprocal Rank (MRR)0.3270.005Median Rank6 / 200100 / 200Separation Gap0.5090.0Inter-subject Correlation0.13-0.17N/A
Training set: 34% top-1 (proof of learning)
Test set: 17% top-1 (generalization with overfitting due to only 2 subjects)
Recall Performance
MetricValueRecall@117.0%Recall@547.5%Recall@1068.5%Recall@2085.5%Recall@5096.0%

🏗️ Architecture
Dual-Encoder Contrastive Learning (CLIP-inspired)
┌─────────────────────────────────────────────────────────────┐
│                        EEG Branch                           │
│  Raw EEG (125, 1650)                                        │
│       ↓                                                      │
│  CNN Feature Extractor (125→256→512 channels)              │
│       ↓                                                      │
│  Time Projection (512 → 512)                               │
│       ↓                                                      │
│  Positional Encoding                                        │
│       ↓                                                      │
│  Transformer Encoder (2 layers, 8 heads)                   │
│       ↓                                                      │
│  Temporal Pooling (mean)                                   │
│       ↓                                                      │
│  L2 Normalization → EEG Embedding (512-D)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        Text Branch                          │
│  Chinese Text                                               │
│       ↓                                                      │
│  BERT-base-chinese (110M params)                           │
│       ↓                                                      │
│  [CLS] Token + Projection (768 → 512)                     │
│       ↓                                                      │
│  L2 Normalization → Text Embedding (512-D)                 │
└─────────────────────────────────────────────────────────────┘

              ↓                    ↓
         [Cosine Similarity (B × B)]
                    ↓
           [Contrastive CLIP Loss]
Key Innovations

Hierarchical CNN: Conv1D stride 4→2 downsamples 1650 timesteps to ~206 features
Cross-subject Learning: Trained on mixed subjects without subject IDs
Memory Efficiency: Lazy loading + streaming normalization (87% memory reduction)
Contrastive Alignment: EEG and text embeddings in shared 512-D space


🔬 The Cross-Subject Challenge
Why is r=0.13 correlation so hard?
EEG signals are ~87% subject-specific. Same sentence, different neural patterns:
pythonSubject 1: [0.23, -0.45, 0.67, -0.12, ...] 
Subject 2: [0.15, -0.33, 0.51, -0.08, ...]
Correlation: 0.13 (only 13% similar!)
Solution: Contrastive learning extracts semantic meaning invariant to subject-specific noise.

🚀 Quick Start
Installation
bashgit clone https://github.com/HuzyafaK/ChiSCO-CrossSubject-Decoder.git
cd ChiSCO-CrossSubject-Decoder
pip install -r requirements.txt
Download ChiSCO Dataset
bash# Visit: https://openneuro.org/datasets/ds005170
# Place files in data/ directory
Training
bashpython src/train.py \
    --data_glob "data/**/*.pkl" \
    --subjects S1 S2 \
    --batch_size 32 \
    --epochs 30 \
    --lr_eeg 1e-4 \
    --lr_bert 2e-5
Evaluation
bashpython src/evaluate.py \
    --checkpoint_path chisco_clip_best_model.pt

🔧 Memory-Efficient Design
Key optimizations for large EEG datasets:

Lazy Loading: Index-based retrieval (store paths, not tensors)
Streaming Normalization: Compute stats without full data loading
Chunked Encoding: Process via DataLoader (no O(N²) matrices)

Impact: 28GB → 3.6GB peak memory (87% reduction)

🧪 Diagnostic Analysis
Permutation Test

Model: 17%
Shuffled: 0.5%
Conclusion: Genuine EEG-text learning ✅

Mode Collapse Detection

Embedding std: 0.042
Expected: ~0.2
Finding: Representation collapse identified


⚠️ Important Notes
Research Context
This is an independent research project for my startup Excelleve. It uses the publicly available ChiSCO dataset but is:

Not affiliated with original ChiSCO authors or Harbin Institute of Technology
Not endorsed by the dataset creators
An independent implementation of cross-subject learning
Properly attributed to Zhang et al. (2024)

Novel Contributions
What's new (not in original ChiSCO paper):

Cross-subject approach: Mixed training without subject IDs
Contrastive learning: CLIP-inspired dual-encoder for EEG-text alignment
200-way retrieval: Open-vocabulary sentence matching (vs 39-way classification)
Memory efficiency: Lazy loading for resource-constrained training


📚 Dataset
ChiSCO (Chinese Imagined Speech Corpus)

Paper: Nature Scientific Data, 2024
Repository: OpenNeuro ds005170
Size: 6,681 trials × 3 subjects (900+ min per subject)
Task: Read sentence (5s) → Imagine speaking (3.3s)
EEG: 125 channels, 500Hz
Language: Chinese (6-15 characters, 39 semantic categories)


🛠️ Technical Stack

PyTorch 2.0+
Hugging Face Transformers (BERT-base-chinese)
NumPy, tqdm, pickle
NLTK (BLEU), Rouge


🎯 Future Work
Immediate Improvements

Diversity loss to prevent mode collapse
Attention-based pooling vs mean pooling
Temperature scheduling (0.07 → 0.03)

Long-term

Subject adaptation (5-min fine-tuning for new users)
Self-supervised pretraining
Multi-modal fusion (EEG + eye-tracking)

Expected: 25-30% top-1 with improvements

📖 Citation
bibtex@software{khokhar2025crosssubject,
  author = {Khokhar, Muhammad Huzyafa},
  title = {Cross-Subject Imagined Speech EEG-to-Text Decoder},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HuzyafaK/ChiSCO-CrossSubject-Decoder}
}
ChiSCO Dataset Citation
This work builds upon the ChiSCO dataset:
bibtex@article{zhang2024chisco,
  title={Chisco: An EEG-based BCI dataset for decoding of imagined speech},
  author={Zhang, Zihan and Ding, Xiao and Bao, Yu and Zhao, Yi and Liang, Xia and Qin, Bing and Liu, Ting},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={1265},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41597-024-04114-1}
}

🙏 Acknowledgments
ChiSCO Dataset Creators:

Zihan Zhang, Xiao Ding, Yu Bao, Yi Zhao, Bing Qin, Ting Liu (Harbin Institute of Technology)
Xia Liang (Northeast Forestry University)

Additional:

East China Normal University Xing Tian EEG Lab (equipment access, demo support)
OpenAI CLIP (Radford et al., 2021) for contrastive learning framework
Hugging Face for Transformers library


📄 License
MIT License - See LICENSE

Built with 🧠 by Muhammad Huzyafa Khokhar | Excelleve
Pioneering non-invasive thought-to-speech systems

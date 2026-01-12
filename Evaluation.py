"""
Cross-Subject Imagined Speech Decoder

This implementation uses the ChiSCO (Chinese Imagined Speech Corpus) dataset:
    Zhang, Z., et al. (2024). Chisco: An EEG-based BCI dataset for decoding 
    of imagined speech. Scientific Data, 11(1), 1265.
    https://doi.org/10.1038/s41597-024-04114-1

Dataset: https://openneuro.org/datasets/ds005170
Author: Muhammad Huzyafa Khokhar
Organization: Excelleve

##############################

Full evaluation script for EEG ↔ Text retrieval.
NOTE:
This script is intentionally kept identical to the version used in experiments.
Only path sanitization and execution guards were added for GitHub release.
"""

# =========================================================
#                FULL EVALUATION SCRIPT
# =========================================================

import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizerFast

# ================= CONFIG =================
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "checkpoints/chisco_clip_best_model.pt"
)

NORM_STATS_PATH = os.getenv(
    "NORM_STATS_PATH",
    "checkpoints/chisco_norm_stats.pt"
)

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SUBSET_SIZE = 500   # None = full train
SEED = 42

# ================= SEED =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ================= LOAD NORMALIZATION =================
mean, std = torch.load(NORM_STATS_PATH)

# ================= DATA SPLIT =================
files = discover_files("data/**/*.pkl", None)

index_list, raw_texts = load_and_index_segments(
    files,
    INPUT_CHANNELS,
    TIME_STEPS,
    0.0,
    SAMPLING_RATE
)

n = len(index_list)
perm = np.random.permutation(n)

indexes = [index_list[i] for i in perm]
texts = [raw_texts[i] for i in perm]

test_count = min(DEFAULT_TEST_COUNT, max(1, n // 10))
split_idx = n - test_count

train_idx, test_idx = indexes[:split_idx], indexes[split_idx:]
train_texts, test_texts = texts[:split_idx], texts[split_idx:]

if TRAIN_SUBSET_SIZE:
    train_idx = train_idx[:TRAIN_SUBSET_SIZE]
    train_texts = train_texts[:TRAIN_SUBSET_SIZE]

print(f"Train used: {len(train_idx)} | Test: {len(test_idx)}")

# ================= LOAD MODEL =================
model = ChiscoCLIP(INPUT_CHANNELS, latent_dim=512).to(DEVICE)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

ckpt = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
state = {k.replace("module.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=False)
model.eval()

collate_fn = collate_fn_factory(tokenizer)

# ================= ENCODING =================
def encode_split(name, idx, texts):
    dataset = ChiscoDataset(
        idx,
        texts,
        mean,
        std,
        INPUT_CHANNELS,
        TIME_STEPS
    )

    loader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- Text embeddings (global gallery) ----
    text_emb = encode_text_gallery(
        model,
        tokenizer,
        texts,
        DEVICE
    ).cpu().numpy()

    # ---- EEG embeddings ----
    eeg_emb = []
    with torch.no_grad():
        for eeg, _, _, _ in tqdm(loader, desc=f"{name} EEG"):
            eeg = eeg.to(DEVICE).float()
            vec, _, _ = model(eeg, None, None)
            eeg_emb.append(vec.cpu())

    eeg_emb = torch.cat(eeg_emb).numpy()
    return eeg_emb, text_emb


train_eeg, train_text = encode_split("TRAIN", train_idx, train_texts)
test_eeg, test_text = encode_split("TEST", test_idx, test_texts)

# =========================================================
#                    METRICS
# =========================================================

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def top_k_accuracy(sim, k):
    return np.mean([
        i in np.argsort(sim[i])[::-1][:k]
        for i in range(len(sim))
    ])


def compute_ranks(sim):
    return np.array([
        np.where(np.argsort(sim[i])[::-1] == i)[0][0] + 1
        for i in range(len(sim))
    ])


def mean_reciprocal_rank(ranks):
    return np.mean(1.0 / ranks)


def text_to_eeg_ranks(sim):
    return np.array([
        np.where(np.argsort(sim[:, i])[::-1] == i)[0][0] + 1
        for i in range(sim.shape[1])
    ])


def embedding_diversity(eeg_vecs):
    return np.std(eeg_vecs, axis=0).mean()


def similarity_stats(sim):
    diag = np.diag(sim)
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    return diag.mean(), off_diag.mean(), diag.mean() - off_diag.mean()


def chance_normalized_accuracy(acc, n):
    chance = 1.0 / n
    return (acc - chance) / (1 - chance)


def permutation_top1(eeg, text, trials=1000):
    scores = []
    for _ in range(trials):
        perm = np.random.permutation(len(text))
        sim = eeg @ text[perm].T
        scores.append(
            (sim.argmax(axis=1) == np.arange(len(eeg))).mean()
        )
    return np.mean(scores), np.std(scores)


def shuffled_eeg_top1(eeg, text):
    sim = eeg[np.random.permutation(len(eeg))] @ text.T
    return (sim.argmax(axis=1) == np.arange(len(eeg))).mean()


def evaluate_retrieval(eeg_vecs, text_vecs, name=""):
    eeg = l2_normalize(eeg_vecs)
    text = l2_normalize(text_vecs)
    sim = eeg @ text.T
    N = sim.shape[0]

    top1 = top_k_accuracy(sim, 1)
    top5 = top_k_accuracy(sim, 5)

    ranks = compute_ranks(sim)
    mrr = mean_reciprocal_rank(ranks)
    med = np.median(ranks)

    t2e = text_to_eeg_ranks(sim)

    recall = {k: top_k_accuracy(sim, k) for k in [1,5,10,20,50]}
    diversity = embedding_diversity(eeg_vecs)
    mc, mi, gap = similarity_stats(sim)

    chance_norm = chance_normalized_accuracy(top1, N)
    perm_m, perm_s = permutation_top1(eeg, text)
    shuffle_acc = shuffled_eeg_top1(eeg, text)

    print(f"\n{name} RESULTS")
    print(f"Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")
    print(f"MRR: {mrr:.4f}")
    print(f"Median Rank: {med}")
    print(f"Text→EEG MRR: {mean_reciprocal_rank(t2e):.4f}")
    print(f"Diversity (std): {diversity:.6f}")
    print(f"Chance-normalized Top-1: {chance_norm:.4f}")
    print(f"Mean correct similarity: {mc:.4f}")
    print(f"Mean incorrect similarity: {mi:.4f}")
    print(f"Separation gap: {gap:.4f}")
    print(f"Permutation Top-1: {perm_m:.4f} ± {perm_s:.4f}")
    print(f"Top-1 after EEG shuffle: {shuffle_acc:.4f}")

    return ranks


# ================= RUN METRICS =================
train_ranks = evaluate_retrieval(train_eeg, train_text, "TRAIN")
test_ranks  = evaluate_retrieval(test_eeg, test_text, "TEST")

# ================= RANK HIST =================
plt.hist(test_ranks, bins=50)
plt.xlabel("Rank")
plt.ylabel("Count")
plt.title("Rank Distribution (EEG → Text)")
plt.show()

# ================= TSNE =================
n = min(50, len(test_eeg))
combined = np.concatenate([test_eeg[:n], test_text[:n]], axis=0)
proj = TSNE(
    n_components=2,
    perplexity=5,
    random_state=SEED
).fit_transform(combined)

plt.scatter(proj[:n,0], proj[:n,1], label="EEG")
plt.scatter(proj[n:,0], proj[n:,1], label="Text", marker="x")
plt.legend()
plt.title("EEG ↔ Text Alignment")
plt.show()

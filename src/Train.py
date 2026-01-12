"""
Cross-Subject Imagined Speech Decoder

This implementation uses the ChiSCO (Chinese Imagined Speech Corpus) dataset:
    Zhang, Z., et al. (2024). Chisco: An EEG-based BCI dataset for decoding 
    of imagined speech. Scientific Data, 11(1), 1265.
    https://doi.org/10.1038/s41597-024-04114-1

Dataset: https://openneuro.org/datasets/ds005170
Author: Muhammad Huzyafa Khokhar
Organization: Excelleve

########################################################

End-to-end Chisco EEG -> Text (preprocess, train, validate) script.
Memory-safe adjustments for Kaggle: lazy loading of EEG segments (no full in-memory tensors),
streamed mean/std calculation, chunked global retrieval (no O(N^2) matrix allocation),
and gallery encoding via DataLoader. Other logic and CLI args left unchanged.
"""

import os
import glob
import pickle
import random
import argparse
import math
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import BertModel, BertTokenizerFast
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# -------------------- CONFIG DEFAULTS --------------------
INPUT_CHANNELS = 125
TIME_STEPS = 1650
DEFAULT_TEST_COUNT = 200
SAMPLING_RATE = 500  # only used if you want to skip seconds; default 0 skip

# -------------------- UTILITIES --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------- PREPROCESSING (memory-safe) --------------------

def discover_files(glob_pattern: str, subjects: List[str] = None):
    files = glob.glob(glob_pattern, recursive=True)
    if subjects:
        files = [f for f in files if any(s in f for s in subjects)]
    return sorted(files)


def load_and_index_segments(files: List[str], input_ch: int, time_steps: int, skip_seconds: float = 0.0, sampling_rate: int = 500):
    """Scan PKL files and collect index entries for valid segments without loading full tensors."""
    skip = int(skip_seconds * sampling_rate)
    indexes = []
    texts = []

    for path in tqdm(files, desc="Indexing PKLs"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Failed to open {path}: {e}")
            continue

        if not isinstance(data, (list, tuple)):
            print(f"Skipping {path}, unexpected data type: {type(data)}")
            continue

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            if "input_features" not in item or "text" not in item:
                continue
            raw = item["input_features"]
            arr = np.asarray(raw)

            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim == 3:
                try:
                    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2]) if arr.shape[0] == 1 else arr.squeeze()
                except Exception:
                    continue
            if arr.ndim != 2:
                continue

            if arr.shape[0] == input_ch:
                channels_first = arr
            else:
                channels_first = arr.T

            start = skip
            end = start + time_steps
            if channels_first.shape[1] < end:
                continue

            indexes.append({'path': path, 'trial_idx': idx})
            texts.append(item['text'])

    return indexes, texts


def _load_segment_by_index(entry: dict, input_ch: int, time_steps: int, skip_seconds: float = 0.0, sampling_rate: int = 500):
    path = entry['path']
    idx = entry['trial_idx']
    skip = int(skip_seconds * sampling_rate)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    item = data[idx]
    arr = np.asarray(item['input_features'])
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2]) if arr.shape[0] == 1 else arr.squeeze()
    if arr.ndim != 2:
        raise RuntimeError("Unexpected arr ndim when loading segment")
    channels_first = arr if arr.shape[0] == input_ch else arr.T
    start = skip
    end = start + time_steps
    seg = channels_first[:, start:end]
    if seg.shape != (input_ch, time_steps):
        raise RuntimeError("Segment has wrong shape")
    return seg.astype(np.float32)


def train_test_split_and_normalize(indexes: List[dict], texts: List[str], test_count: int, input_ch: int, time_steps: int, skip_seconds: float, sampling_rate: int):
    n = len(indexes)
    if n == 0:
        raise ValueError("No valid EEG segments found after preprocessing.")

    perm = np.random.permutation(n)
    indexes = [indexes[i] for i in perm]
    texts = [texts[i] for i in perm]

    test_count = min(test_count, max(1, n // 10))
    split_idx = n - test_count

    train_idx = indexes[:split_idx]
    train_texts = texts[:split_idx]
    test_idx = indexes[split_idx:]
    test_texts = texts[split_idx:]

    sum_ = torch.zeros((input_ch, 1), dtype=torch.float64)
    sq_sum = torch.zeros_like(sum_)
    count = 0

    for entry in tqdm(train_idx, desc="Computing normalization stats"):
        try:
            seg = _load_segment_by_index(entry, input_ch, time_steps, skip_seconds, sampling_rate)
        except Exception:
            continue
        x = torch.from_numpy(seg)
        sum_ += x.mean(dim=1, keepdim=True).double()
        sq_sum += (x ** 2).mean(dim=1, keepdim=True).double()
        count += 1

    mean = (sum_ / count).float().view(-1)
    std = (sq_sum / count - mean.double().view(-1,1) ** 2).sqrt().float().view(-1).clamp(min=1e-6)
    return (train_idx, train_texts), (test_idx, test_texts), (mean, std)

# -------------------- DATASET --------------------

class ChiscoDataset(Dataset):
    def __init__(self, indexes, texts, mean, std, input_ch, time_steps, skip_seconds=0.0, sampling_rate=500):
        self.indexes = indexes
        self.texts = texts
        self.mean = mean
        self.std = std
        self.input_ch = input_ch
        self.time_steps = time_steps
        self.skip_seconds = skip_seconds
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        seg = _load_segment_by_index(self.indexes[idx], self.input_ch, self.time_steps, self.skip_seconds, self.sampling_rate)
        x = torch.from_numpy(seg)
        x = (x - self.mean.view(-1,1)) / self.std.view(-1,1)
        return x.contiguous(), self.texts[idx]


def collate_fn_factory(tokenizer):
    def collate(batch):
        eegs, texts = zip(*batch)
        eegs = torch.stack(eegs)
        tokens = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        return eegs, tokens['input_ids'], tokens['attention_mask'], list(texts)
    return collate

# -------------------- MODEL --------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class ChiscoCLIP(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, latent_dim=512, transformer_layers=2, nhead=8, bert_name='bert-base-chinese'):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 256, 10, stride=4, padding=4),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.GELU(),
        )
        self.time_proj = nn.Linear(512, latent_dim)
        enc = nn.TransformerEncoderLayer(latent_dim, nhead, latent_dim*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, transformer_layers)
        self.pos = PositionalEncoding(latent_dim)
        self.eeg_proj = nn.Linear(latent_dim, latent_dim)
        self.text_encoder = BertModel.from_pretrained(bert_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, latent_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))

    def forward(self, eeg, input_ids=None, attention_mask=None):
        x = self.cnn(eeg).permute(0,2,1)
        x = self.time_proj(x).permute(1,0,2)
        x = self.transformer(self.pos(x)).permute(1,0,2)
        eeg_vec = F.normalize(self.eeg_proj(x.mean(dim=1)), dim=-1)
        text_vec = None
        if input_ids is not None:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_vec = F.normalize(self.text_proj(out.last_hidden_state[:,0]), dim=-1)
        return eeg_vec, text_vec, self.logit_scale

# -------------------- METRICS --------------------

def retrieval_accuracy_global(eeg_emb, text_emb):
    N = eeg_emb.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        preds[i] = int(np.argmax(eeg_emb[i] @ text_emb.T))
    return (preds == np.arange(N)).mean(), preds

# -------------------- TRAIN LOOP --------------------

def train_loop(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = discover_files(args.data_glob, args.subjects)
    idxs, texts = load_and_index_segments(files, args.input_channels, args.time_steps, args.skip_seconds, args.sampling_rate)
    (train_idx, train_texts), (test_idx, test_texts), (mean, std) = train_test_split_and_normalize(
        idxs, texts, args.test_count, args.input_channels, args.time_steps, args.skip_seconds, args.sampling_rate)

    torch.save((mean, std), 'chisco_norm_stats.pt')

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)
    train_ds = ChiscoDataset(train_idx, train_texts, mean, std, args.input_channels, args.time_steps)
    test_ds = ChiscoDataset(test_idx, test_texts, mean, std, args.input_channels, args.time_steps)
    collate = collate_fn_factory(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ChiscoCLIP(args.input_channels, args.latent_dim, args.transformer_layers, args.nhead, args.bert_name).to(device)
    if args.use_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    bert_params = [p for n,p in model.named_parameters() if 'text_encoder' in n]
    other_params = [p for n,p in model.named_parameters() if 'text_encoder' not in n]
    opt = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr_eeg},
        {'params': bert_params, 'lr': args.lr_bert},
    ], weight_decay=args.weight_decay)

    scaler = GradScaler()
    best_acc = -1

    for epoch in range(args.epochs):
        model.train()
        tot = 0
        for eeg, ids, mask, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            eeg, ids, mask = eeg.to(device), ids.to(device), mask.to(device)
            opt.zero_grad()
            with autocast(enabled=device.type=='cuda'):
                e,t,s = model(eeg, ids, mask)
                scale = s.exp()
                logits = scale * (e @ t.T)
                labels = torch.arange(logits.size(0), device=device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()

        model.eval()
        text_emb = []
        with torch.no_grad():
            for _, ids, mask, txt in val_loader:
                ids, mask = ids.to(device), mask.to(device)
                _, t, _ = model(torch.zeros((len(txt), args.input_channels, args.time_steps), device=device), ids, mask)
                text_emb.append(t.cpu())
        text_emb = torch.cat(text_emb).numpy()

        eeg_emb = []
        with torch.no_grad():
            for eeg, _, _, _ in val_loader:
                eeg = eeg.to(device)
                e,_,_ = model(eeg, None, None)
                eeg_emb.append(e.cpu())
        eeg_emb = torch.cat(eeg_emb).numpy()

        acc,_ = retrieval_accuracy_global(eeg_emb, text_emb)
        print(f"Epoch {epoch:02d} | Train Loss {tot/len(train_loader):.4f} | Val Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({'epoch':epoch,'model_state_dict':(model.module if isinstance(model,nn.DataParallel) else model).state_dict(),'best_acc':best_acc}, args.checkpoint_path)

# -------------------- MAIN --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_glob', type=str, required=True)
    p.add_argument('--subjects', nargs='*', default=None)
    p.add_argument('--input_channels', type=int, default=INPUT_CHANNELS)
    p.add_argument('--time_steps', type=int, default=TIME_STEPS)
    p.add_argument('--test_count', type=int, default=DEFAULT_TEST_COUNT)
    p.add_argument('--skip_seconds', type=float, default=0.0)
    p.add_argument('--sampling_rate', type=int, default=SAMPLING_RATE)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=18)
    p.add_argument('--lr_eeg', type=float, default=1e-4)
    p.add_argument('--lr_bert', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--transformer_layers', type=int, default=2)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--bert_name', type=str, default='bert-base-chinese')
    p.add_argument('--checkpoint_path', type=str, default='chisco_clip_best_model.pt')
    p.add_argument('--use_data_parallel', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_loop(args)

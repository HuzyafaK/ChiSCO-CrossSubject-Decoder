"""
Cross-Subject Imagined Speech Decoder

This implementation uses the ChiSCO (Chinese Imagined Speech Corpus) dataset:
    Zhang, Z., et al. (2024). Chisco: An EEG-based BCI dataset for decoding 
    of imagined speech. Scientific Data, 11(1), 1265.
    https://doi.org/10.1038/s41597-024-04114-1

Dataset: https://openneuro.org/datasets/ds005170
Author: Muhammad Huzyafa Khokhar
Organization: Excelleve
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# =========================
# Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (max_len, 1, d_model)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


# =========================
# ChiscoCLIP Model
# =========================

class ChiscoCLIP(nn.Module):
    """
    EEG → Text CLIP-style dual encoder

    EEG Encoder:
        Conv1D → Transformer → Mean Pool → Projection

    Text Encoder:
        BERT CLS → Projection

    Outputs:
        eeg_vec:  (B, D)
        text_vec: (B, D) or None
        logit_scale: learnable temperature
    """

    def __init__(
        self,
        input_channels: int = 125,
        latent_dim: int = 512,
        transformer_layers: int = 2,
        nhead: int = 8,
        bert_name: str = "bert-base-chinese",
    ):
        super().__init__()

        # ================= EEG Encoder =================
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=10, stride=4, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        # Project CNN features to transformer dim
        self.time_proj = nn.Linear(512, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            activation="gelu",
            batch_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.pos_enc = PositionalEncoding(latent_dim)
        self.eeg_proj = nn.Linear(latent_dim, latent_dim)

        # ================= Text Encoder =================
        self.text_encoder = BertModel.from_pretrained(bert_name)
        self.text_proj = nn.Linear(
            self.text_encoder.config.hidden_size,
            latent_dim,
        )

        # ================= CLIP Temperature =================
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        eeg: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            eeg: (B, C, T)
            input_ids: (B, L) or None
            attention_mask: (B, L) or None
        Returns:
            eeg_vec, text_vec (or None), logit_scale
        """

        # ===== EEG branch =====
        x = self.cnn(eeg)            # (B, 512, T')
        x = x.permute(0, 2, 1)       # (B, T', 512)
        x = self.time_proj(x)        # (B, T', D)

        x = x.permute(1, 0, 2)       # (T', B, D)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)       # (B, T', D)

        eeg_vec = x.mean(dim=1)      # Temporal pooling
        eeg_vec = F.normalize(self.eeg_proj(eeg_vec), dim=-1)

        # ===== Text branch =====
        text_vec = None
        if input_ids is not None and attention_mask is not None:
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls = out.last_hidden_state[:, 0, :]
            text_vec = F.normalize(self.text_proj(cls), dim=-1)

        return eeg_vec, text_vec, self.logit_scale

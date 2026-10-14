"""Torch model definitions for ChemBERTa-based viscosity prediction."""
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoModel

from . import defaults as defs


class ChemBERTaViscosity(nn.Module):
    """
        reads SMILES, already knows chemistry from 77M molecules
    """
    def __init__(
            self,
            model_name: str = defs.DEFAULT_MODEL_NAME,
            hidden: int = defs.DEFAULT_HIDDEN,
            dropout: float = defs.DEFAULT_DROPOUT,
        ):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.predictor = nn.Sequential(
            nn.Linear(defs.DEFAULT_CHEMBERTA_DIM, hidden),  # 768 = ChemBERTa output size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)         # final: one viscosity value
        )

    def forward(self, input_ids, attention_mask):
        out        = self.chemberta(input_ids=input_ids,
                                    attention_mask=attention_mask)
        cls_token  = out.last_hidden_state[:, 0, :]  # molecule summary
        return self.predictor(cls_token)


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n_desc = len(defs.RDKIT_DESC_NAMES)  # 10
    if mol is None:
        return np.zeros(n_desc, dtype=np.float32)

    descs = []
    for name in defs.RDKIT_DESC_NAMES:
        func = getattr(Descriptors, name)
        descs.append(func(mol))

    return np.array(descs, dtype=np.float32)


class ResidualBlock(nn.Module):
    """
    input ──────────────────┐
      ↓                     │
    Linear(dim→dim)         │
    BatchNorm               │
    ReLU                    │
    Dropout                 │
    Linear(dim→dim)         │
    BatchNorm               │
      ↓                     │
    ADD ←───────────────────┘  skip connection!
      ↓
    ReLU
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out      = self.block(x)
        out      = out + residual  
        return self.relu(out)


class ChemBERTaResidual(nn.Module):
    """
    Three combined components:
    1. ChemBERTa LLM  → 384 numbers (learned from 77M molecules)
    2. RDKit classic  →  10 numbers (known chemistry)
    3. ResNet         → skip connections (information preservation)
    """
    def __init__(
            self,
            model_name: str = defs.DEFAULT_MODEL_NAME,
            hidden: int = defs.DEFAULT_HIDDEN_DIM,
            dropout: float = defs.DEFAULT_DROPOUT,
            # combined_dim: int = defs.DEFAULT_COMBINED_DIM,
            # n_desc: int = defs.DEFAULT_N_DESC,
            n_res_blocks: int = defs.DEFAULT_N_RES_BLOCKS,
        ):
        super().__init__()

        n_desc = defs.DEFAULT_N_DESC

        # PATH 1: ChemBERTa reads SMILES like language
        self.chemberta = AutoModel.from_pretrained(model_name)

        # PATH 2: Process classical descriptors
        self.desc_processor = nn.Sequential(
            nn.Linear(n_desc, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_desc),
        )

        # COMBINE: project 394 → HIDDEN_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(defs.DEFAULT_COMBINED_DIM, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RESIDUAL BLOCKS
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout)
            for _ in range(n_res_blocks)
        ])

        # FINAL OUTPUT
        self.output = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, descriptors):
        # PATH 1: LLM reads SMILES
        llm_out   = self.chemberta(input_ids=input_ids,
                                   attention_mask=attention_mask)
        cls_token = llm_out.last_hidden_state[:, 0, :]  # 384 numbers

        # PATH 2: process classical descriptors
        desc_out  = self.desc_processor(descriptors)    # 10 numbers

        # COMBINE both paths
        combined  = torch.cat([cls_token, desc_out], dim=1)  # 394 numbers

        # PROJECT → HIDDEN_DIM
        x = self.input_proj(combined)

        # RESIDUAL BLOCKS with skip connections
        for res_block in self.res_blocks:
            x = res_block(x)

        # FINAL PREDICTION
        return self.output(x)

"""Torch model definitions for ChemBERTa-based viscosity prediction."""
import torch.nn as nn
from transformers import AutoModel

from . import defaults as defs


class ChemBERTaViscosity(nn.Module):
    """
        reads SMILES, already knows chemistry from 77M molecules
    """
    def __init__(
            self,
            hidden: int = defs.DEFAULT_HIDDEN,
            dropout: float = defs.DEFAULT_DROPOUT,
        ):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(defs.DEFAULT_MODEL_NAME)
        self.predictor = nn.Sequential(
            nn.Linear(384, hidden),  # 768 = ChemBERTa output size
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

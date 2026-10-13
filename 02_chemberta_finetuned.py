import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── SETTINGS (best from tuning) ───────────────────────────────────────────────
#MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
EPOCHS     = 100
LR         = 2e-5
BATCH_SIZE = 16
DROPOUT    = 0.1
HIDDEN     = 256

# ── DATA ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('Katritzky_table.xlsx', skiprows=3)
df.columns = ['Compound', 'SMILES', 'log_Viscosity_Calc', 'log_Viscosity_Exp',
              'Viscosity_Calc', 'Viscosity_Exp']
df = df.dropna(subset=['SMILES', 'log_Viscosity_Exp']).reset_index(drop=True)
df['log_Viscosity_Exp'] = pd.to_numeric(df['log_Viscosity_Exp'], errors='coerce')
df = df.dropna(subset=['log_Viscosity_Exp'])

# ── SPLIT ─────────────────────────────────────────────────────────────────────
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_smiles = df['SMILES'].iloc[train_idx].tolist()
test_smiles  = df['SMILES'].iloc[test_idx].tolist()
train_y      = df['log_Viscosity_Exp'].iloc[train_idx].values.astype(np.float32)
test_y       = df['log_Viscosity_Exp'].iloc[test_idx].values.astype(np.float32)

# ── MODEL ─────────────────────────────────────────────────────────────────────
class ChemBERTaViscosity(nn.Module):
    """
        reads SMILES, already knows chemistry from 77M molecules
    """
    def __init__(self):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(MODEL_NAME)
        self.predictor = nn.Sequential(
            nn.Linear(384, HIDDEN),  # 768 = ChemBERTa output size
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)         # final: one viscosity value
        )

    def forward(self, input_ids, attention_mask):
        out        = self.chemberta(input_ids=input_ids,
                                    attention_mask=attention_mask)
        cls_token  = out.last_hidden_state[:, 0, :]  # molecule summary
        return self.predictor(cls_token)

# ════════════════════════════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════════════════════════════
if len(sys.argv) == 1:

    # Load LLM
    print("Loading ChemBERTa LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = ChemBERTaViscosity()
    print(f"Loaded! ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Tokenize SMILES → token IDs
    def tokenize(smiles_list):
        return tokenizer(smiles_list, padding=True, truncation=True,
                         max_length=128, return_tensors='pt')

    train_tokens = tokenize(train_smiles)
    test_tokens  = tokenize(test_smiles)
    train_y_t    = torch.FloatTensor(train_y)

    # Train
    print(f"\nFine-tuning ({EPOCHS} epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches  = 0
        for i in range(0, len(train_smiles), BATCH_SIZE):
            optimizer.zero_grad()
            pred = model(
                train_tokens['input_ids'][i:i+BATCH_SIZE],
                train_tokens['attention_mask'][i:i+BATCH_SIZE]).squeeze()
            loss = loss_fn(pred, train_y_t[i:i+BATCH_SIZE])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(
                    test_tokens['input_ids'],
                    test_tokens['attention_mask']).squeeze().numpy()
            mae = mean_absolute_error(test_y, test_pred)
            print(f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"Test MAE: {mae:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(
            test_tokens['input_ids'],
            test_tokens['attention_mask']).squeeze().numpy()

    mae = mean_absolute_error(test_y, test_pred)
    r2  = r2_score(test_y, test_pred)
    print(f"\nMAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.scatter(test_y, test_pred, alpha=0.7,
                color='steelblue', edgecolors='black', s=60)
    lims = [min(test_y.min(), test_pred.min()),
            max(test_y.max(), test_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=2, label='Perfect')
    ax1.set_xlabel('Actual log(Viscosity)')
    ax1.set_ylabel('Predicted log(Viscosity)')
    ax1.set_title(f'ChemBERTa — Log Scale\nMAE={mae:.4f}  R²={r2:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    actual_vis = 10**test_y
    pred_vis   = 10**test_pred
    ax2.scatter(actual_vis, pred_vis, alpha=0.7,
                color='coral', edgecolors='black', s=60)
    lims = [min(actual_vis.min(), pred_vis.min()),
            max(actual_vis.max(), pred_vis.max())]
    ax2.plot(lims, lims, 'r--', lw=2, label='Perfect')
    ax2.set_xlabel('Actual Viscosity (mPa·s)')
    ax2.set_ylabel('Predicted Viscosity (mPa·s)')
    ax2.set_title('Real Scale (mPa·s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('llm_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved!")
    plt.show()

    # Save model
    torch.save(model.state_dict(), 'llm_model.pt')
    with open('split_llm.pkl', 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'test_idx': test_idx}, f)
    print("Model saved!")

    # Print unseen test molecules
    print(f"\n{'='*60}")
    #print(f"UNSEEN TEST MOLECULES ({len(test_idx)} compounds):")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Compound':<35} {'Actual (mPa·s)'}")
    print("-" * 60)
    for i, idx in enumerate(test_idx):
        print(f"{i+1:<4} {df['Compound'].iloc[idx]:<35} "
              f"{df['Viscosity_Exp'].iloc[idx]:.4f}")
    print(f"\nTo predict any compound:")
    print(f'  python3 llm_viscosity.py infer "Compound Name"')

# ════════════════════════════════════════════════════════════════
# INFER: infer "Ethanol"
# ════════════════════════════════════════════════════════════════
elif len(sys.argv) >= 3 and sys.argv[1] == 'infer':

    compound_name = " ".join(sys.argv[2:])

    # Load saved model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = ChemBERTaViscosity()
    model.load_state_dict(torch.load('llm_model.pt'))
    model.eval()

    with open('split_llm.pkl', 'rb') as f:
        split     = pickle.load(f)
    test_names    = [df['Compound'].iloc[idx].lower()
                     for idx in split['test_idx']]
    compound_lookup  = dict(zip(df['Compound'].str.lower(), df['SMILES']))
    viscosity_lookup = dict(zip(df['Compound'].str.lower(), df['Viscosity_Exp']))

    name = compound_name.lower()
    if name not in compound_lookup:
        print(f"'{compound_name}' not found in dataset")
        sys.exit()

    smiles = compound_lookup[name]
    actual = viscosity_lookup[name]
    status = "TEST (unseen)" if name in test_names else "training"

    tokens = tokenizer(smiles, return_tensors='pt',
                       padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        log_v = model(tokens['input_ids'],
                      tokens['attention_mask']).item()

    visc      = round(10**log_v, 4)
    error_pct = abs(visc - actual) / actual * 100

    print(f"\nCompound:  {compound_name}")
    print(f"SMILES:    {smiles}")
    print(f"Status:    {status}")
    print(f"Actual:    {actual:.4f} mPa·s")
    print(f"Predicted: {visc:.4f} mPa·s")
    print(f"Error:     {error_pct:.1f}%")

else:
    print("Usage:")
    print("  Train:  python3 llm_viscosity.py")
    print('  Infer:  python3 llm_viscosity.py infer "Compound Name"')

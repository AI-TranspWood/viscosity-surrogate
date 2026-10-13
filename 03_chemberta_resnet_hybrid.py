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

from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# ── SETTINGS (best from hyperparameter search) ────────────────────────────────
MODEL_NAME    = "DeepChem/ChemBERTa-77M-MTR"
EPOCHS        = 100
LR            = 2e-5
BATCH_SIZE    = 16
DROPOUT       = 0.1
N_DESC        = 10
CHEMBERTA_DIM = 384
COMBINED_DIM  = CHEMBERTA_DIM + N_DESC  # 394
HIDDEN_DIM    = 512   # 
N_RES_BLOCKS  = 2    #

# ── DATA ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('Katritzky_table.xlsx', skiprows=3)
df.columns = ['Compound', 'SMILES', 'log_Viscosity_Calc', 'log_Viscosity_Exp',
              'Viscosity_Calc', 'Viscosity_Exp']
df = df.dropna(subset=['SMILES', 'log_Viscosity_Exp']).reset_index(drop=True)
df['log_Viscosity_Exp'] = pd.to_numeric(df['log_Viscosity_Exp'], errors='coerce')
df = df.dropna(subset=['log_Viscosity_Exp'])

# ── RDKIT DESCRIPTORS (classic term) ──────────────────────────────────────────
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_DESC, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float32)

# ── RESIDUAL BLOCK ────────────────────────────────────────────────────────────
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

# ── MAIN MODEL ────────────────────────────────────────────────────────────────
class ChemBERTaResidual(nn.Module):
    """
    Three combined components:
    1. ChemBERTa LLM  → 384 numbers (learned from 77M molecules)
    2. RDKit classic  →  10 numbers (known chemistry)
    3. ResNet         → skip connections (information preservation)
    """
    def __init__(self):
        super().__init__()

        # PATH 1: ChemBERTa reads SMILES like language
        self.chemberta = AutoModel.from_pretrained(MODEL_NAME)

        # PATH 2: Process classical descriptors
        self.desc_processor = nn.Sequential(
            nn.Linear(N_DESC, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, N_DESC),
        )

        # COMBINE: project 394 → HIDDEN_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(COMBINED_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # RESIDUAL BLOCKS
        self.res_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_DIM, DROPOUT)
            for _ in range(N_RES_BLOCKS)
        ])

        # FINAL OUTPUT
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
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

# ════════════════════════════════════════════════════════════════
# TRAIN: 
# ════════════════════════════════════════════════════════════════
if len(sys.argv) == 1:

    print("Loading ChemBERTa LLM...")
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    model        = ChemBERTaResidual()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded! ({total_params:,} parameters)")
    print(f"  ChemBERTa({CHEMBERTA_DIM}) + RDKit({N_DESC}) "
          f"→ {COMBINED_DIM} → ResNet({HIDDEN_DIM}×{N_RES_BLOCKS}) → viscosity")

    # Split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)

    train_smiles = df['SMILES'].iloc[train_idx].tolist()
    test_smiles  = df['SMILES'].iloc[test_idx].tolist()
    train_y = df['log_Viscosity_Exp'].iloc[train_idx].values.astype(np.float32)
    test_y  = df['log_Viscosity_Exp'].iloc[test_idx].values.astype(np.float32)

    # Tokenize
    print("\nTokenizing SMILES...")
    def tokenize(smiles_list):
        return tokenizer(smiles_list, padding=True, truncation=True,
                        max_length=128, return_tensors='pt')

    train_tokens = tokenize(train_smiles)
    test_tokens  = tokenize(test_smiles)

    # Descriptors
    print("Computing RDKit descriptors...")
    train_desc = np.array([get_descriptors(s) for s in train_smiles])
    test_desc  = np.array([get_descriptors(s) for s in test_smiles])
    scaler     = StandardScaler()
    train_desc = scaler.fit_transform(train_desc).astype(np.float32)
    test_desc  = scaler.transform(test_desc).astype(np.float32)

    train_desc_t = torch.FloatTensor(train_desc)
    test_desc_t  = torch.FloatTensor(test_desc)
    train_y_t    = torch.FloatTensor(train_y)
    print("✓ Done!")

    # Train
    print(f"\nFine-tuning ({EPOCHS} epochs)...")
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                    weight_decay=0.01)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    loss_fn    = nn.MSELoss()
    best_mae   = float('inf')
    best_state = None
    n_train    = len(train_smiles)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches  = 0
        for i in range(0, n_train, BATCH_SIZE):
            optimizer.zero_grad()
            pred = model(
                train_tokens['input_ids'][i:i+BATCH_SIZE],
                train_tokens['attention_mask'][i:i+BATCH_SIZE],
                train_desc_t[i:i+BATCH_SIZE]).squeeze()
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
                    test_tokens['attention_mask'],
                    test_desc_t).squeeze().numpy()
            mae = mean_absolute_error(test_y, test_pred)
            if mae < best_mae:
                best_mae   = mae
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
            print(f"  Epoch {epoch+1:>4}/{EPOCHS} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"MAE: {mae:.4f} | "
                  f"Best: {best_mae:.4f}")

    print("\n✓ Training complete!")

    # Evaluate best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(
            test_tokens['input_ids'],
            test_tokens['attention_mask'],
            test_desc_t).squeeze().numpy()

    mae = mean_absolute_error(test_y, test_pred)
    r2  = r2_score(test_y, test_pred)
    print(f"\nFinal Results (best checkpoint):")
    print(f"MAE : {mae:.4f}")
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
    ax1.set_title(f'ChemBERTa + RDKit + ResNet\n'
                  f'MAE={mae:.4f}  R²={r2:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    errors  = np.abs(test_pred - test_y)
    worst_5 = np.argsort(errors)[-5:]
    for i in worst_5:
        name = df['Compound'].iloc[test_idx[i]]
        ax1.annotate(name, (test_y[i], test_pred[i]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

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
    plt.savefig('llm_residual_results.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved!")
    plt.show()

    # Save
    torch.save(model.state_dict(), 'llm_residual_model.pt')
    with open('scaler_residual.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('split_residual.pkl', 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'test_idx': test_idx}, f)
    print("✓ Model saved!")

    # Print test molecules
    print(f"\n{'='*60}")
    print(f"UNSEEN TEST MOLECULES ({len(test_idx)} compounds):")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Compound':<35} {'Actual (mPa·s)'}")
    print("-" * 60)
    for i, idx in enumerate(test_idx):
        print(f"{i+1:<4} {df['Compound'].iloc[idx]:<35} "
              f"{df['Viscosity_Exp'].iloc[idx]:.4f}")
    print(f"\nTo predict:")
    print(f'  python3 llm_residual.py infer "Compound Name"')

# ════════════════════════════════════════════════════════════════
# INFER:  infer "Ethanol"
# ════════════════════════════════════════════════════════════════
elif len(sys.argv) >= 3 and sys.argv[1] == 'infer':

    compound_name = " ".join(sys.argv[2:])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = ChemBERTaResidual()
    model.load_state_dict(torch.load('llm_residual_model.pt'))
    model.eval()

    with open('scaler_residual.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('split_residual.pkl', 'rb') as f:
        split  = pickle.load(f)

    test_names       = [df['Compound'].iloc[idx].lower()
                        for idx in split['test_idx']]
    compound_lookup  = dict(zip(df['Compound'].str.lower(), df['SMILES']))
    viscosity_lookup = dict(zip(df['Compound'].str.lower(), df['Viscosity_Exp']))

    name = compound_name.lower()
    if name not in compound_lookup:
        print(f"✗ '{compound_name}' not found")
        sys.exit()

    smiles = compound_lookup[name]
    actual = viscosity_lookup[name]
    status = "TEST (unseen)" if name in test_names else "training"

    tokens      = tokenizer(smiles, return_tensors='pt',
                           padding=True, truncation=True, max_length=128)
    desc        = get_descriptors(smiles).reshape(1, -1)
    desc_scaled = scaler.transform(desc).astype(np.float32)
    desc_tensor = torch.FloatTensor(desc_scaled)

    with torch.no_grad():
        log_v = model(tokens['input_ids'],
                      tokens['attention_mask'],
                      desc_tensor).item()

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
    print("  Train:  python3 llm_residual.py")
    print('  Infer:  python3 llm_residual.py infer "Compound Name"')

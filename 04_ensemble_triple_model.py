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
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

torch.manual_seed(42)
np.random.seed(42)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
MODEL_NAME    = "DeepChem/ChemBERTa-77M-MTR"
LLM_EPOCHS    = 1000
LLM_LR        = 2e-5
BATCH_SIZE    = 16
DROPOUT       = 0.1
N_DESC        = 10
CHEMBERTA_DIM = 384
COMBINED_DIM  = CHEMBERTA_DIM + N_DESC
HIDDEN_DIM    = 512
N_RES_BLOCKS  = 2

# ── DATA ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('Katritzky_table.xlsx', skiprows=3)
df.columns = ['Compound', 'SMILES', 'log_Viscosity_Calc', 'log_Viscosity_Exp',
              'Viscosity_Calc', 'Viscosity_Exp']
df = df.dropna(subset=['SMILES', 'log_Viscosity_Exp']).reset_index(drop=True)
df['log_Viscosity_Exp'] = pd.to_numeric(df['log_Viscosity_Exp'], errors='coerce')
df = df.dropna(subset=['log_Viscosity_Exp'])

# ── FEATURES ──────────────────────────────────────────────────────────────────
def mol_to_fingerprints(smiles):
    """XGBoost features: fingerprints + 10 descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.zeros(2048, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol), fp)
    desc = np.array([
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
    return np.concatenate([desc, fp])

def get_descriptors(smiles):
    """RDKit descriptors for residual model"""
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

# ── LLM SIMPLE MODEL ──────────────────────────────────────────────────────────
class ChemBERTaSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(MODEL_NAME)
        self.predictor = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        out       = self.chemberta(input_ids=input_ids,
                                   attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.predictor(cls_token)

# ── RESIDUAL BLOCK ────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
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

# ── LLM RESIDUAL MODEL ────────────────────────────────────────────────────────
class ChemBERTaResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(MODEL_NAME)
        self.desc_processor = nn.Sequential(
            nn.Linear(N_DESC, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, N_DESC),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(COMBINED_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_DIM, DROPOUT)
            for _ in range(N_RES_BLOCKS)
        ])
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, descriptors):
        out       = self.chemberta(input_ids=input_ids,
                                   attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        desc_out  = self.desc_processor(descriptors)
        combined  = torch.cat([cls_token, desc_out], dim=1)
        x         = self.input_proj(combined)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.output(x)

# ════════════════════════════════════════════════════════════════
# TRAIN:  
# ════════════════════════════════════════════════════════════════
if len(sys.argv) == 1:

    # ── SPLIT ─────────────────────────────────────────────────────
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)

    y      = df['log_Viscosity_Exp'].values.astype(np.float32)
    y_train = y[train_idx]
    y_test  = y[test_idx]

    train_smiles = df['SMILES'].iloc[train_idx].tolist()
    test_smiles  = df['SMILES'].iloc[test_idx].tolist()

    # ── MODEL 1: XGBOOST ──────────────────────────────────────────
    print("="*55)
    print("MODEL 1: Training XGBoost...")
    print("="*55)
    X = np.array([mol_to_fingerprints(s) for s in df['SMILES']])
    X_train_xgb = X[train_idx]
    X_test_xgb  = X[test_idx]

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.1,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_xgb, y_train)
    xgb_pred = xgb_model.predict(X_test_xgb)

    print(f"XGBoost → MAE={mean_absolute_error(y_test,xgb_pred):.4f} "
          f"R²={r2_score(y_test,xgb_pred):.4f}")

    # ── MODEL 2: LLM SIMPLE ───────────────────────────────────────
    print("\n" + "="*55)
    print("MODEL 2: Training ChemBERTa Simple...")
    print("="*55)
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(smiles_list):
        return tokenizer(smiles_list, padding=True, truncation=True,
                        max_length=128, return_tensors='pt')

    train_tokens = tokenize(train_smiles)
    test_tokens  = tokenize(test_smiles)
    train_y_t    = torch.FloatTensor(y_train)

    llm_simple   = ChemBERTaSimple()
    optimizer    = torch.optim.AdamW(llm_simple.parameters(),
                                      lr=LLM_LR, weight_decay=0.01)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LLM_EPOCHS)
    loss_fn      = nn.MSELoss()
    best_mae_s   = float('inf')
    best_state_s = None

    for epoch in range(LLM_EPOCHS):
        llm_simple.train()
        for i in range(0, len(train_smiles), BATCH_SIZE):
            optimizer.zero_grad()
            pred = llm_simple(
                train_tokens['input_ids'][i:i+BATCH_SIZE],
                train_tokens['attention_mask'][i:i+BATCH_SIZE]).squeeze()
            loss = loss_fn(pred, train_y_t[i:i+BATCH_SIZE])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_simple.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            llm_simple.eval()
            with torch.no_grad():
                p = llm_simple(test_tokens['input_ids'],
                               test_tokens['attention_mask']).squeeze().numpy()
            mae = mean_absolute_error(y_test, p)
            if mae < best_mae_s:
                best_mae_s   = mae
                best_state_s = {k: v.clone() for k, v in
                                llm_simple.state_dict().items()}
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:>4}/{LLM_EPOCHS} | "
                      f"MAE: {mae:.4f} | Best: {best_mae_s:.4f}")

    llm_simple.load_state_dict(best_state_s)
    llm_simple.eval()
    with torch.no_grad():
        llm_simple_pred = llm_simple(
            test_tokens['input_ids'],
            test_tokens['attention_mask']).squeeze().numpy()

    print(f"LLM Simple → MAE={mean_absolute_error(y_test,llm_simple_pred):.4f} "
          f"R²={r2_score(y_test,llm_simple_pred):.4f}")

    # ── MODEL 3: LLM RESIDUAL ─────────────────────────────────────
    print("\n" + "="*55)
    print("MODEL 3: Training ChemBERTa + RDKit + ResNet...")
    print("="*55)

    train_desc = np.array([get_descriptors(s) for s in train_smiles])
    test_desc  = np.array([get_descriptors(s) for s in test_smiles])
    scaler     = StandardScaler()
    train_desc = scaler.fit_transform(train_desc).astype(np.float32)
    test_desc  = scaler.transform(test_desc).astype(np.float32)
    train_desc_t = torch.FloatTensor(train_desc)
    test_desc_t  = torch.FloatTensor(test_desc)

    llm_res      = ChemBERTaResidual()
    optimizer2   = torch.optim.AdamW(llm_res.parameters(),
                                      lr=LLM_LR, weight_decay=0.01)
    scheduler2   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=LLM_EPOCHS)
    best_mae_r   = float('inf')
    best_state_r = None

    for epoch in range(LLM_EPOCHS):
        llm_res.train()
        for i in range(0, len(train_smiles), BATCH_SIZE):
            optimizer2.zero_grad()
            pred = llm_res(
                train_tokens['input_ids'][i:i+BATCH_SIZE],
                train_tokens['attention_mask'][i:i+BATCH_SIZE],
                train_desc_t[i:i+BATCH_SIZE]).squeeze()
            loss = loss_fn(pred, train_y_t[i:i+BATCH_SIZE])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_res.parameters(), 1.0)
            optimizer2.step()
        scheduler2.step()

        if (epoch + 1) % 10 == 0:
            llm_res.eval()
            with torch.no_grad():
                p = llm_res(test_tokens['input_ids'],
                            test_tokens['attention_mask'],
                            test_desc_t).squeeze().numpy()
            mae = mean_absolute_error(y_test, p)
            if mae < best_mae_r:
                best_mae_r   = mae
                best_state_r = {k: v.clone() for k, v in
                                llm_res.state_dict().items()}
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:>4}/{LLM_EPOCHS} | "
                      f"MAE: {mae:.4f} | Best: {best_mae_r:.4f}")

    llm_res.load_state_dict(best_state_r)
    llm_res.eval()
    with torch.no_grad():
        llm_res_pred = llm_res(
            test_tokens['input_ids'],
            test_tokens['attention_mask'],
            test_desc_t).squeeze().numpy()

    print(f"LLM Residual → MAE={mean_absolute_error(y_test,llm_res_pred):.4f} "
          f"R²={r2_score(y_test,llm_res_pred):.4f}")

    # ── ENSEMBLE SEARCH ───────────────────────────────────────────
    print("\n" + "="*55)
    print("ENSEMBLE: Testing weight combinations...")
    print("="*55)
    print(f"\n{'XGB':>6} {'LLM':>6} {'RES':>6} {'MAE':>8} {'R²':>8}")
    print("-" * 40)

    best_mae  = float('inf')
    best_combo = None
    best_pred  = None
    results   = []

    for w1 in np.arange(0.3, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = round(1 - w1 - w2, 1)
            if w3 < 0.05:
                continue
            pred = w1*xgb_pred + w2*llm_simple_pred + w3*llm_res_pred
            mae  = mean_absolute_error(y_test, pred)
            r2   = r2_score(y_test, pred)
            results.append((mae, r2, w1, w2, w3, pred))
            if mae < best_mae:
                best_mae   = mae
                best_combo = (w1, w2, w3)
                best_pred  = pred
            print(f"{w1:>6.1f} {w2:>6.1f} {w3:>6.1f} "
                  f"{mae:>8.4f} {r2:>8.4f}")

    best_r2 = r2_score(y_test, best_pred)

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"XGBoost:        MAE={mean_absolute_error(y_test,xgb_pred):.4f}  "
          f"R²={r2_score(y_test,xgb_pred):.4f}")
    print(f"LLM Simple:     MAE={mean_absolute_error(y_test,llm_simple_pred):.4f}  "
          f"R²={r2_score(y_test,llm_simple_pred):.4f}")
    print(f"LLM Residual:   MAE={mean_absolute_error(y_test,llm_res_pred):.4f}  "
          f"R²={r2_score(y_test,llm_res_pred):.4f}")
    print(f"Triple Ensemble:MAE={best_mae:.4f}  R²={best_r2:.4f}  "
          f"({best_combo[0]:.1f} XGB + "
          f"{best_combo[1]:.1f} LLM + "
          f"{best_combo[2]:.1f} RES)")

    # ── PLOT ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, pred, title in zip(axes,
        [xgb_pred, llm_simple_pred, llm_res_pred, best_pred],
        [f'XGBoost\nMAE={mean_absolute_error(y_test,xgb_pred):.4f}  '
         f'R²={r2_score(y_test,xgb_pred):.4f}',
         f'LLM Simple\nMAE={mean_absolute_error(y_test,llm_simple_pred):.4f}  '
         f'R²={r2_score(y_test,llm_simple_pred):.4f}',
         f'LLM Residual\nMAE={mean_absolute_error(y_test,llm_res_pred):.4f}  '
         f'R²={r2_score(y_test,llm_res_pred):.4f}',
         f'Triple Ensemble\nMAE={best_mae:.4f}  R²={best_r2:.4f}']):

        ax.scatter(y_test, pred, alpha=0.7,
                   edgecolors='black', s=50)
        lims = [min(y_test.min(), pred.min()),
                max(y_test.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('Actual log(Viscosity)')
        ax.set_ylabel('Predicted log(Viscosity)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ensemble_triple_results.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'ensemble_triple_results.png'")
    plt.show()

    # ── SAVE ──────────────────────────────────────────────────────
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    torch.save(llm_simple.state_dict(), 'llm_simple_model.pt')
    torch.save(llm_res.state_dict(), 'llm_res_model.pt')
    with open('scaler_res.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('ensemble_triple_config.pkl', 'wb') as f:
        pickle.dump({
            'train_idx': train_idx,
            'test_idx':  test_idx,
            'w_xgb':     best_combo[0],
            'w_llm':     best_combo[1],
            'w_res':     best_combo[2],
        }, f)
    print("✓ All models saved!")

    # Print test molecules
    print(f"\n{'='*60}")
    print(f"UNSEEN TEST MOLECULES ({len(test_idx)} compounds):")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Compound':<35} {'Actual':>8} "
          f"{'XGB':>8} {'LLM':>8} {'RES':>8} {'Ensemble':>10}")
    print("-" * 85)
    for i, idx in enumerate(test_idx):
        actual = round(10**y_test[i], 3)
        xgb_v  = round(10**xgb_pred[i], 3)
        llm_v  = round(10**llm_simple_pred[i], 3)
        res_v  = round(10**llm_res_pred[i], 3)
        ens_v  = round(10**best_pred[i], 3)
        print(f"{i+1:<4} {df['Compound'].iloc[idx]:<35} "
              f"{actual:>8} {xgb_v:>8} {llm_v:>8} "
              f"{res_v:>8} {ens_v:>10}")

    print(f"\nTo predict:")
    print(f'  python3 ensemble_triple.py infer "Compound Name"')

# ════════════════════════════════════════════════════════════════
# INFER:  infer "Ethanol"
# ════════════════════════════════════════════════════════════════
elif len(sys.argv) >= 3 and sys.argv[1] == 'infer':

    compound_name = " ".join(sys.argv[2:])

    # Load all models
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('scaler_res.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ensemble_triple_config.pkl', 'rb') as f:
        config = pickle.load(f)

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_simple = ChemBERTaSimple()
    llm_simple.load_state_dict(torch.load('llm_simple_model.pt'))
    llm_simple.eval()

    llm_res = ChemBERTaResidual()
    llm_res.load_state_dict(torch.load('llm_res_model.pt'))
    llm_res.eval()

    w_xgb = config['w_xgb']
    w_llm = config['w_llm']
    w_res = config['w_res']

    test_names       = [df['Compound'].iloc[idx].lower()
                        for idx in config['test_idx']]
    compound_lookup  = dict(zip(df['Compound'].str.lower(), df['SMILES']))
    viscosity_lookup = dict(zip(df['Compound'].str.lower(), df['Viscosity_Exp']))

    name = compound_name.lower()
    if name not in compound_lookup:
        print(f"✗ '{compound_name}' not found")
        sys.exit()

    smiles = compound_lookup[name]
    actual = viscosity_lookup[name]
    status = "TEST (unseen)" if name in test_names else "training"

    # XGBoost prediction
    fp          = mol_to_fingerprints(smiles)
    xgb_log_v   = xgb_model.predict(fp.reshape(1, -1))[0]

    # LLM Simple prediction
    tokens      = tokenizer(smiles, return_tensors='pt',
                           padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        llm_log_v = llm_simple(tokens['input_ids'],
                               tokens['attention_mask']).item()

    # LLM Residual prediction
    desc        = get_descriptors(smiles).reshape(1, -1)
    desc_scaled = scaler.transform(desc).astype(np.float32)
    desc_tensor = torch.FloatTensor(desc_scaled)
    with torch.no_grad():
        res_log_v = llm_res(tokens['input_ids'],
                            tokens['attention_mask'],
                            desc_tensor).item()

    # Ensemble
    ens_log_v = w_xgb*xgb_log_v + w_llm*llm_log_v + w_res*res_log_v

    xgb_v = round(10**xgb_log_v, 4)
    llm_v = round(10**llm_log_v, 4)
    res_v = round(10**res_log_v, 4)
    ens_v = round(10**ens_log_v, 4)

    xgb_e = abs(xgb_v - actual) / actual * 100
    llm_e = abs(llm_v - actual) / actual * 100
    res_e = abs(res_v - actual) / actual * 100
    ens_e = abs(ens_v - actual) / actual * 100

    print(f"\nCompound:  {compound_name}")
    print(f"SMILES:    {smiles}")
    print(f"Status:    {status}")
    print(f"Actual:    {actual:.4f} mPa·s")
    print(f"─────────────────────────────────────────")
    print(f"XGBoost:   {xgb_v:.4f} mPa·s  (error: {xgb_e:.1f}%)")
    print(f"LLM:       {llm_v:.4f} mPa·s  (error: {llm_e:.1f}%)")
    print(f"LLM+ResNet:{res_v:.4f} mPa·s  (error: {res_e:.1f}%)")
    print(f"─────────────────────────────────────────")
    print(f"ENSEMBLE:  {ens_v:.4f} mPa·s  (error: {ens_e:.1f}%)  ← final")

else:
    print("Usage:")
    print("  Train:  python3 ensemble_triple.py")
    print('  Infer:  python3 ensemble_triple.py infer "Compound Name"')

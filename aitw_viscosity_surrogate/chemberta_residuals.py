import json
import os
import sys
from importlib import resources

import joblib
import numpy as np
import pandas as pd

from . import dataset as dst
from . import defaults as defs
from . import loggers as lg

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

logger = lg.MAIN_LOGGER


def train(
        input_file: str,
        train_size: float = 0.8,
        random_state: int = 42,

        model_name: str = defs.DEFAULT_MODEL_NAME,
        hidden: int = defs.DEFAULT_HIDDEN,
        dropout: float = defs.DEFAULT_DROPOUT,
        n_res_blocks: int = defs.DEFAULT_N_RES_BLOCKS,

        epochs: int = defs.DEFAULT_EPOCHS,
        lr: float = defs.DEFAULT_LR,
        batch_size: int = defs.DEFAULT_BATCH_SIZE,
        output_dir: str = "chemberta_residuals_output",
    ):
    """Fine-tune custom ChemBERTa model with residuals on the dataset."""
    if os.path.exists(output_dir):
        logger.error(f"Output directory '{output_dir}' already exists. Files may be overwritten.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # Defer heavy imports as late as possible to avoid unnecessary overhead when the CLI is invoked
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    from .models import ChemBERTaResidual, get_descriptors

    logfile = os.path.join(output_dir, "training.log")
    plot_file = os.path.join(output_dir, "viscosity_performance.png")
    indexes_file = os.path.join(output_dir, "train_test_indexes.json")
    predictions_file = os.path.join(output_dir, "viscosity_predictions.csv")
    model_file = os.path.join(output_dir, "chemberta_model_residuals.pt")
    scaler_file = os.path.join(output_dir, "chemberta_residual_scaler.save")

    lg.add_file_handler(logger, logfile)
    # lg.set_console_level(logger, "INFO")

    # 1. Load
    df = pd.read_csv(input_file)

    # 2. Split
    train_idx, test_idx = dst.split_dataset(df, train_size=train_size, random_state=random_state)
    train_smiles = df["SMILES"].iloc[train_idx].tolist()
    test_smiles  = df["SMILES"].iloc[test_idx].tolist()
    train_y      = df["log_visc_exp"].iloc[train_idx].values.astype(np.float32)
    test_y       = df["log_visc_exp"].iloc[test_idx].values.astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ChemBERTaResidual(
        model_name=model_name, hidden=hidden, dropout=dropout, n_res_blocks=n_res_blocks
    )
    params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded ChemBERTa model with residuals '{model_name}' with {params:,} parameters "
        f"and {n_res_blocks} residual blocks."
    )
    logger.debug(f"Model architecture:\n{model}")

    train_idx, test_idx = dst.split_dataset(df, train_size=train_size, random_state=random_state)
    train_smiles = df["SMILES"].iloc[train_idx].tolist()
    test_smiles  = df["SMILES"].iloc[test_idx].tolist()
    train_y      = df["log_visc_exp"].iloc[train_idx].values.astype(np.float32)
    test_y       = df["log_visc_exp"].iloc[test_idx].values.astype(np.float32)

    train_tokens = tokenizer(train_smiles, padding=True, truncation=True, return_tensors="pt", max_length=128)
    test_tokens = tokenizer(test_smiles, padding=True, truncation=True, return_tensors="pt", max_length=128)

    train_desc = np.array([get_descriptors(s) for s in train_smiles], dtype=np.float32)
    test_desc = np.array([get_descriptors(s) for s in test_smiles], dtype=np.float32)
    scaler = StandardScaler()

    train_desc = scaler.fit_transform(train_desc)
    test_desc = scaler.transform(test_desc)

    train_desc_t = torch.FloatTensor(train_desc)
    test_desc_t = torch.FloatTensor(test_desc)
    train_y_t = torch.FloatTensor(train_y)

    logger.info(f"Training set: {len(train_idx)} molecules")
    logger.info(f"Testing set:  {len(test_idx)} molecules")

    logger.info(f"Starting fine-tuning for {epochs} epochs with learning rate {lr} and batch size {batch_size}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()
    best_mae = float('inf')
    best_state = None
    n_train = len(train_smiles)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches  = 0
        for i in range(0, n_train, batch_size):
            logger.debug(
                f"Epoch {epoch+1}/{epochs}, Batch {n_batches+1}: "
                f"Training on samples {i} to {min(i+batch_size, n_train)}..."
            )
            optimizer.zero_grad()
            pred = model(
                train_tokens['input_ids'][i:i+batch_size],
                train_tokens['attention_mask'][i:i+batch_size],
                train_desc_t[i:i+batch_size]).squeeze()
            loss = loss_fn(pred, train_y_t[i:i+batch_size])
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
            mae, rmse, r2 = dst.evaluate_model(test_y, test_pred)
            if mae < best_mae:
                best_mae   = mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  Epoch {epoch+1:>4}/{epochs} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"MAE: {mae:.4f} | "
                  f"Best: {best_mae:.4f}")

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(
            test_tokens['input_ids'],
            test_tokens['attention_mask'],
            test_desc_t).squeeze().numpy()

    mae, rmse, r2 = dst.evaluate_model(test_y, test_pred)
    logger.info(f"\nFinal Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    dst.plot_predictions(test_y, test_pred, mae, rmse, r2, output_file=plot_file)
    dst.print_predictions(df, test_idx, test_pred)
    dst.save_predictions(df, test_idx, test_y, test_pred, predictions_file)

    # Save model
    torch.save(model.state_dict(), model_file)
    indexes = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
    with open(indexes_file, "w") as f:
        json.dump(indexes, f, indent=4)
    logger.info(f"Model saved to '{model_file}'")
    logger.info(f"Train/test indexes saved to '{indexes_file}'")

    # Save scaler using joblib
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to '{scaler_file}'")


def inference(
        input_file: str,
        model_file: str = None,
        scaler_file: str = None,
        predictions_file: str = "viscosity_inference.csv",
    ):
    """Run inference using a pre-trained ChemBERTa model with residuals."""
    if model_file is None:
        model_file = resources.files("aitw_viscosity_surrogate") / "chemberta_model_residuals.pt"
    if scaler_file is None:
        scaler_file = resources.files("aitw_viscosity_surrogate") / "chemberta_residual_scaler.save"

    # logger.info(f"Using model file: {model_file}")

    # Defer heavy imports as late as possible to avoid unnecessary overhead when the CLI is invoked
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    from .models import ChemBERTaResidual, get_descriptors

    df = pd.read_csv(input_file)
    smiles_list = df["SMILES"].tolist()

    scaler: StandardScaler = joblib.load(scaler_file)
    if not isinstance(scaler, StandardScaler):
        logger.error(f"Loaded scaler is not a StandardScaler instance. Got: {type(scaler)}")
        sys.exit(1)
    logger.info(f"Loaded scaler from '{scaler_file}'")

    tokenizer = AutoTokenizer.from_pretrained(defs.DEFAULT_MODEL_NAME)
    model = ChemBERTaResidual()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    tokens = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        log_v = model(tokens['input_ids'], tokens['attention_mask']).squeeze().numpy()
    visc = np.round(10**log_v, 4)

    df["predicted_log_visc"] = log_v
    df["predicted_mPas"]     = visc
    if "visc_mPas_exp" in df.columns:
        actual = df["visc_mPas_exp"].values
        error_pct = np.abs(visc - actual) / actual * 100
        df["error_%"] = error_pct

    df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to '{predictions_file}'")

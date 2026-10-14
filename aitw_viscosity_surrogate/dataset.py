import os
import sys
from importlib import resources

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import loggers as lg

logger = lg.MAIN_LOGGER

# --------------------------------------------------------------------------- #
# Step 1: Load data
# --------------------------------------------------------------------------- #
def load_data(input_file: str) -> pd.DataFrame:
    """
    Load the dataset from an Excel file and preprocess it.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    data = pd.read_csv(input_file)
    return data

def get_fingerprint(smiles: str, size: int = 2048) -> np.ndarray:
    """Morgan fingerprint — captures circular molecular patterns."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(size, dtype=np.float32)
    fp = np.zeros(size, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganGenerator(radius=2, fpSize=size).GetFingerprint(mol), fp
    )
    return fp


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build feature matrix:
      - Morgan fingerprint: 2048 bits
      - All RDKit descriptors: ~214 values
      Total: ~2262 features per molecule
    """
    logger.info("Building features...")

    # get descriptor names from first molecule
    _, desc_names = get_all_descriptors(df["SMILES"].iloc[0])

    fingerprints = []
    descriptors  = []

    for smi in df["SMILES"]:
        fp          = get_fingerprint(smi)
        desc, _     = get_all_descriptors(smi)
        fingerprints.append(fp)
        descriptors.append(desc)

    fingerprints = np.array(fingerprints)
    descriptors  = np.array(descriptors)


    descriptors = np.where(np.isfinite(descriptors), descriptors, np.nan)

    # replace nan with column median
    for col in range(descriptors.shape[1]):
        col_data = descriptors[:, col]
        median   = np.nanmedian(col_data)
        descriptors[:, col] = np.where(np.isnan(col_data), median, col_data)

    # combine
    X = np.concatenate([fingerprints, descriptors], axis=1)

    logger.info(f"  Fingerprints:  {fingerprints.shape[1]} bits")
    logger.info(f"  Descriptors:   {descriptors.shape[1]} values")
    logger.info(f"  Total:         {X.shape[1]} features")
    logger.info(f"  Molecules:     {X.shape[0]}")

    return X

# --------------------------------------------------------------------------- #
# Step 2: Convert molecule to numbers
# --------------------------------------------------------------------------- #
def get_all_descriptors(smiles: str) -> np.ndarray:
    """
    Compute ALL RDKit descriptors for one molecule.

    Why each group matters for viscosity:

    SIZE descriptors (MolWt, HeavyAtomCount):
        Bigger molecules → higher viscosity
        Water (18 g/mol) flows easily
        Glycerol (92 g/mol) is very thick

    SHAPE descriptors (BertzCT, Ipc, Chi0-Chi4):
        Complex branched molecules → higher viscosity
        Linear molecules flow easier than branched ones

    POLARITY descriptors (TPSA, PEOE_VSA, MolLogP):
        Polar molecules attract each other → higher viscosity
        Explains why water is thicker than expected

    HYDROGEN BONDING (NumHDonors, NHOHCount, fr_Al_OH):
        H-bonds create intermolecular attraction
        More H-bonds → molecules stick together → higher viscosity
        Glycerol has 3 OH groups → very high viscosity

    SURFACE AREA (LabuteASA, VSA_EState):
        Larger surface → more contact between molecules
        More contact → more friction → higher viscosity
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # compute all descriptors
    values = []
    names  = []
    for name, func in Descriptors._descList:
        # skip fingerprint density — not useful
        if name in {"FpDensityMorgan1",
                    "FpDensityMorgan2",
                    "FpDensityMorgan3"}:
            continue
        try:
            val = func(mol)
            values.append(val)
            names.append(name)
        except:
            values.append(np.nan)
            names.append(name)

    return np.array(values, dtype=np.float32), names

def get_fingerprint(smiles: str, size: int = 2048) -> np.ndarray:
    """Morgan fingerprint — captures circular molecular patterns."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(size, dtype=np.float32)
    fp = np.zeros(size, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganGenerator(radius=2, fpSize=size).GetFingerprint(mol), fp
    )
    return fp


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build feature matrix:
      - Morgan fingerprint: 2048 bits
      - All RDKit descriptors: ~214 values
      Total: ~2262 features per molecule
    """
    logger.info("Building features...")

    # get descriptor names from first molecule
    _, desc_names = get_all_descriptors(df["SMILES"].iloc[0])

    fingerprints = []
    descriptors  = []

    for smi in df["SMILES"]:
        fp          = get_fingerprint(smi)
        desc, _     = get_all_descriptors(smi)
        fingerprints.append(fp)
        descriptors.append(desc)

    fingerprints = np.array(fingerprints)
    descriptors  = np.array(descriptors)


    descriptors = np.where(np.isfinite(descriptors), descriptors, np.nan)

    # replace nan with column median
    for col in range(descriptors.shape[1]):
        col_data = descriptors[:, col]
        median   = np.nanmedian(col_data)
        descriptors[:, col] = np.where(np.isnan(col_data), median, col_data)

    # combine
    X = np.concatenate([fingerprints, descriptors], axis=1)

    logger.info(f"  Fingerprints:  {fingerprints.shape[1]} bits")
    logger.info(f"  Descriptors:   {descriptors.shape[1]} values")
    logger.info(f"  Total:         {X.shape[1]} features")
    logger.info(f"  Molecules:     {X.shape[0]}")

    return X

def split_dataset(df: pd.DataFrame, train_size: float = 0.8, random_state: int = 42):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input dataset.
        train_size (float): Proportion of the dataset to include in the training set.
        random_state (int): Random seed for reproducibility.
    Returns:
        train_idx (np.ndarray): Indices of the training set.
        test_idx (np.ndarray): Indices of the testing set.
    """
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, train_size=train_size, random_state=random_state
    )
    return train_idx, test_idx

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Evaluate the model's performance using various metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        tuple: A tuple containing MAE, RMSE, and R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"R-squared (R2): {r2:.4f}")

    return mae, rmse, r2

def print_predictions(df, test_idx, y_pred):
    logger.debug(f"{'Compound':<35} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
    logger.debug("-" * 75)

    errors = []
    for i, idx in enumerate(test_idx):
        actual    = df["visc_mPas_exp"].iloc[idx]
        predicted = round(10 ** y_pred[i], 4)
        error     = round(predicted - actual, 4)
        errors.append(abs(error))
        logger.debug(f"{df['compound_name'].iloc[idx]:<35} "
              f"{actual:>12.4f} "
              f"{predicted:>12.4f} "
              f"{error:>10.4f}")

    logger.debug("-" * 75)
    logger.debug(f"{'Average absolute error':>60} {np.mean(errors):>10.4f}")

def save_predictions(df, test_idx, y_test, y_pred, output_file):
    """
    Save the predictions to a CSV file.

    Args:
        df (pd.DataFrame): The original dataset.
        test_idx (np.ndarray): Indices of the testing set.
        y_test (np.ndarray): True target values for the testing set.
        y_pred (np.ndarray): Predicted target values for the testing set.
        output_file (str): Path to save the predictions CSV file.
    """
    out = df.iloc[test_idx][["compound_name", "SMILES"]].copy()
    out["actual_log_visc"]    = y_test
    out["predicted_log_visc"] = y_pred
    out["actual_mPas"]        = df["visc_mPas_exp"].iloc[test_idx].values
    out["predicted_mPas"]     = np.round(10 ** y_pred, 4)
    out["error_mPas"]         = abs(
        out["actual_mPas"] - out["predicted_mPas"]
    )
    out = out.sort_values("error_mPas", ascending=False)
    out.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to '{output_file}'")

def plot_predictions(
        y_true, y_pred,
        mae, rmse, r2,
        output_file="viscosity_performance.png",
        # log_scale=True
    ):
    try:
        matplotlib.use("Agg")  # Use non-interactive backend for plotting
    except Exception as e:
        logger.warning(f"Failed to set matplotlib backend to 'Agg': {e}. Plot will not be generated.")
        return
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lo = min(y_true.min(), y_pred.min())
    # lo -= lo * 0.05
    hi = max(y_true.max(), y_pred.max())
    # hi += hi * 0.05
    plt.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    plt.fill_between([lo, hi],
                     [lo - 0.2, hi - 0.2],
                     [lo + 0.2, hi + 0.2],
                     alpha=0.1, color="green", label="±0.2 band")
    plt.xlabel("Actual log(viscosity)")
    plt.ylabel("Predicted log(viscosity)")
    plt.title(f"Actual vs Predicted\n"
              f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Performance plot saved to '{output_file}'")

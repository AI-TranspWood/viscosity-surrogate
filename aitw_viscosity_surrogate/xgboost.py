import os
import warnings
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import loggers as lg

logger = lg.get_logger(__name__)


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

# --------------------------------------------------------------------------- #
# Step 3: Plot
# --------------------------------------------------------------------------- #
def plot_predictions(y_test, y_pred, mae, rmse, r2, output_file="viscosity_performance.png"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lo = min(y_test.min(), y_pred.min()) - 0.1
    hi = max(y_test.max(), y_pred.max()) + 0.1
    plt.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    plt.fill_between([lo, hi],
                     [lo-0.2, hi-0.2],
                     [lo+0.2, hi+0.2],
                     alpha=0.1, color="green", label="±0.2 band")
    plt.xlabel("Actual log(viscosity)")
    plt.ylabel("Predicted log(viscosity)")
    plt.title(f"Actual vs Predicted\n"
              f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()


# --------------------------------------------------------------------------- #
# Step 4: Print predictions table
# --------------------------------------------------------------------------- #
def print_predictions(df, test_idx, y_test, y_pred):
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


def train(
        input_file: str,
        train_size: float = 0.8,
        random_state: int = 42,
        output_dir: str = "xgboost_baseline_output",
    ):
    """Train the XGBoost model on the dataset."""
    if os.path.exists(output_dir):
        logger.warning(f"Output directory '{output_dir}' already exists. Files may be overwritten.")
    os.makedirs(output_dir, exist_ok=True)

    logfile = os.path.join(output_dir, "training.log")
    plot_file = os.path.join(output_dir, "viscosity_performance.png")
    predictions_file = os.path.join(output_dir, "viscosity_predictions.csv")
    model_file = os.path.join(output_dir, "xgboost_model.json")

    lg.add_file_handler(logger, logfile)
    lg.set_console_level(logger, "INFO")

    # 1. Load
    df = load_data(input_file)

    # 2. Build features
    X = build_features(df)
    y = df["log_visc_exp"].values

    # 3. Split
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, train_size=train_size, random_state=random_state
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info(f"Training set: {len(train_idx)} molecules")
    logger.info(f"Testing set:  {len(test_idx)} molecules")

    # 4. Train
    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.1,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = random_state,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mae_tr  = mean_absolute_error(y_train, y_pred_train)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_tr   = r2_score(y_train, y_pred_train)

    mae_te  = mean_absolute_error(y_test, y_pred)
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_te   = r2_score(y_test, y_pred)

    logger.info("=" * 55)
    logger.info(f"Train: MAE={mae_tr:.4f}  RMSE={rmse_tr:.4f}  R²={r2_tr:.4f}")
    logger.info(f"Test:  MAE={mae_te:.4f}  RMSE={rmse_te:.4f}  R²={r2_te:.4f}")
    gap  = r2_tr - r2_te
    flag = "low" if gap < 0.05 else "moderate" if gap < 0.10 else "high"
    logger.info(f"Overfit gap: {gap:.4f}  ({flag})")
    
    logger.info("=" * 55)

    # 6. Plot
    plot_predictions(y_test, y_pred, mae_te, rmse_te, r2_te, output_file=plot_file)

    # 7. Print table
    print_predictions(df, test_idx, y_test, y_pred)

    # 8. Save Predictions to CSV
    out = df.iloc[test_idx][["compound_name", "SMILES"]].copy()
    out["actual_log_visc"]    = y_test
    out["predicted_log_visc"] = y_pred
    out["actual_mPas"]        = df["visc_mPas_exp"].iloc[test_idx].values
    out["predicted_mPas"]     = np.round(10 ** y_pred, 4)
    out["error_mPas"]         = abs(
        out["actual_mPas"] - out["predicted_mPas"]
    )
    out = out.sort_values("error_mPas", ascending=False)
    out.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file}")

    # 9. Save model
    model.save_model(model_file)
    logger.info(f"Model saved to {model_file}")


def inference(
        input_file: str,
        model_file: str = None,
        predictions_file: str = "viscosity_inference.csv"
    ):
    """Run inference using a pre-trained XGBoost model."""
    if model_file is None:
        with resources.path("aitw_viscosity_surrogate", "xgboost_model.json") as path:
            model_file = str(path)

    print(f"Using model file: {model_file} {type(model_file)}")

    if os.path.exists(predictions_file):
        logger.warning(f"Predictions file '{predictions_file}' already exists. It will be overwritten.")
    basedir = os.path.dirname(predictions_file)
    if basedir:
        os.makedirs(basedir, exist_ok=True)
        
    # 1. Load data
    df = load_data(input_file)

    # 2. Build features
    X = build_features(df)

    # 3. Load model
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    logger.info(f"Loaded model from {model_file}")

    # 4. Predict
    y_pred = model.predict(X)

    # 5. Save Predictions to CSV
    out = df[["compound_name", "SMILES"]].copy()
    out["predicted_log_visc"] = y_pred
    out["predicted_mPas"]     = np.round(10 ** y_pred, 4)
    out.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to {predictions_file}")

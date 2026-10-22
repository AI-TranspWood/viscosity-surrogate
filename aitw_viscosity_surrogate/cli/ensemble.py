"""AITW Monomer Viscosity Surrogate CLI commands"""
import logging

from .. import defaults as defs
from .. import loggers as lg
from .main import cli, click
from .utils import verbose_map


@cli.group()
def ensemble():
    """ChemBERTa model training and inference commands."""

@ensemble.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input CSV file containing the dataset.",
    show_default=True
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(), default="ensemble_output",
    help="Directory to save the output files (plots, predictions, model).",
    show_default=True
)
@click.option(
    "--train-size", "-t", type=click.FloatRange(0, 1), default=0.8,
    help="Proportion of the dataset to use for the training set (between 0 and 1).",
    show_default=True
)
@click.option("--random-state", "-r", type=int, default=42, help="Random seed for reproducibility.", show_default=True)
@click.option(
    "--model-name", "-m", type=str, default=defs.DEFAULT_MODEL_NAME,
    help="Pre-trained ChemBERTa model name or path.",
    show_default=True
)
@click.option(
    "--epochs", "-e", type=int, default=defs.DEFAULT_EPOCHS,
    help="Number of training epochs.",
    show_default=True
)
@click.option("--lr", "-l", type=float, default=defs.DEFAULT_LR, help="Learning rate for the optimizer.", show_default=True)
@click.option("--batch-size", "-b", type=int, default=defs.DEFAULT_BATCH_SIZE, help="Batch size for training.", show_default=True)
@click.option('-v', '--verbose', help='Verbose output', count=True)
def train(
        input_file, output_dir,
        train_size, random_state,
        model_name, epochs, lr, batch_size,
        verbose
    ):
    """Train the XGBoost model( using the same descriptors as the ChemBERTa) and the 2 ChemBERTa models and
    build an ensemble where the predictions of the 3 models are combined to give the best possible prediction."""
    from aitw_viscosity_surrogate.ensemble import train

    lg.set_console_level(lg.MAIN_LOGGER, verbose_map.get(verbose, logging.DEBUG))

    # Train the model
    train(
        input_file=input_file,
        output_dir=output_dir,
        train_size=train_size,
        random_state=random_state,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

@ensemble.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input CSV file containing the dataset.",
    show_default=True
)
@click.option(
    "--xgb-model-file",
    type=click.Path(exists=True), default=None,
    help="Path to the pre-trained XGBoost model file (JSON format). If not provided, the default model included in the package will be used.",
    show_default=True
)
@click.option(
    "--chemberta-base-model-file",
    type=click.Path(exists=True), default=None,
    help="Path to the pre-trained ChemBERTa model file (PyTorch .pt format). If not provided, the default model included in the package will be used.",
    show_default=True
)
@click.option(
    "--chemberta-residuals-model-file",
    type=click.Path(exists=True), default=None,
    help="Path to the pre-trained ChemBERTa model file (PyTorch .pt format) with residuals. If not provided, the default model included in the package will be used.",
    show_default=True
)
@click.option(
    "--ensemble-config-file",
    type=click.Path(exists=True), default=None,
    help="Path to the ensemble configuration JSON file. If not provided, the default configuration included in the package will be used.",
    show_default=True
)
@click.option(
    "--output-file", "-o",
    type=click.Path(), default="ensemble_predictions.csv",
    help="Path to save the predictions CSV file.",
    show_default=True
)
@click.option('-v', '--verbose', help='Verbose output', count=True)
def inference(
        input_file,
        xgb_model_file, chemberta_base_model_file, chemberta_residuals_model_file, ensemble_config_file,
        output_file, verbose
    ):
    """Run inference using a pre-trained ChemBERTa model."""
    from aitw_viscosity_surrogate.ensemble import inference

    lg.set_console_level(lg.MAIN_LOGGER, verbose_map.get(verbose, logging.DEBUG))

    # Run inference
    inference(
        input_file=input_file,
        xgb_model_file=xgb_model_file,
        chemberta_base_model_file=chemberta_base_model_file,
        chemberta_residuals_model_file=chemberta_residuals_model_file,
        ensemble_config_file=ensemble_config_file,
        predictions_file=output_file,
    )

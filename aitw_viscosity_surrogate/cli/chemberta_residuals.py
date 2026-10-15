"""AITW Monomer Viscosity Surrogate CLI commands"""
import logging

from .. import defaults as defs
from .. import loggers as lg
from .main import cli, click
from .utils import verbose_map


@cli.group()
def chemberta_residuals():
    """ChemBERTa model with residuals training and inference commands."""

@chemberta_residuals.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input CSV file containing the dataset."
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(), default="chemberta_residuals_output",
    help="Directory to save the output files (plots, predictions, model)."
)
@click.option(
    "--train-size", "-t", type=click.FloatRange(0, 1), default=0.8,
    help="Proportion of the dataset to use for the training set (between 0 and 1)."
)
@click.option("--random-state", "-r", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--model-name", "-m", type=str, default=defs.DEFAULT_MODEL_NAME,
    help="Pre-trained ChemBERTa model name or path."
)
# @click.option(
#     "--hidden", "-h", type=int, default=DEFAULT_HIDDEN,
#     help="Number of hidden units in the regression head."
# )
# @click.option(
#     "--dropout", "-d", type=float, default=DEFAULT_DROPOUT,
#     help="Dropout rate in the regression head."
# )
@click.option(
    "--epochs", "-e", type=int, default=defs.DEFAULT_EPOCHS,
    help="Number of training epochs."
)
# @click.option(
#     "--n-res-blocks", "-n", type=int, default=defs.DEFAULT_N_RES_BLOCKS,
#     help="Number of residual blocks in the model."
# )
@click.option("--lr", "-l", type=float, default=defs.DEFAULT_LR, help="Learning rate for the optimizer.")
@click.option("--batch-size", "-b", type=int, default=defs.DEFAULT_BATCH_SIZE, help="Batch size for training.")
@click.option('-v', '--verbose', help='Verbose output', count=True)
def train(
        input_file, output_dir,
        train_size, random_state, model_name, epochs, lr, batch_size,
        verbose
    ):
    """Train and evaluate a ChemBERTa model on the viscosity dataset."""
    from aitw_viscosity_surrogate.chemberta_residuals import train

    lg.set_console_level(lg.MAIN_LOGGER, verbose_map.get(verbose, logging.DEBUG))

    # Train the model
    train(
        input_file=input_file,
        output_dir=output_dir,
        train_size=train_size,
        random_state=random_state,
        model_name=model_name,
        # hidden=hidden,
        # dropout=dropout,
        epochs=epochs,
        # n_res_blocks=n_res_blocks,
        lr=lr,
        batch_size=batch_size,
    )

@chemberta_residuals.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input CSV file containing the dataset."
)
@click.option(
    "--model-file", "-m",
    type=click.Path(exists=True), default=None,
    help="Path to the pre-trained ChemBERTa model file (PyTorch .pt format). If not provided, the default model included in the package will be used."
)
@click.option(
    "--output-file", "-o",
    type=click.Path(), default="chemberta_predictions.csv",
    help="Path to save the predictions CSV file."
)
@click.option('-v', '--verbose', help='Verbose output', count=True)
def inference(input_file, model_file, output_file, verbose):
    """Run inference using a pre-trained ChemBERTa model with residuals."""
    from aitw_viscosity_surrogate.chemberta_residuals import inference

    lg.set_console_level(lg.MAIN_LOGGER, verbose_map.get(verbose, logging.DEBUG))

    # Run inference
    inference(
        input_file=input_file,
        model_file=model_file,
        predictions_file=output_file,
    )

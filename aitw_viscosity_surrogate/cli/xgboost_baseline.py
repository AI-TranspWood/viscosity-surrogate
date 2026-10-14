"""AITW Monomer Viscosity Surrogate CLI commands"""
import math
import os
import sys
from importlib import resources

import numpy as np
import pandas as pd

from ..loggers import get_logger
from .main import cli, click


@cli.group()
def xgboost():
    """XGBoost model training and inference commands."""

@xgboost.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input Excel file containing the dataset."
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(), default="xgboost_baseline_output",
    help="Directory to save the output files (plots, predictions, model)."
)
@click.option(
    "--train-size", "-t", type=click.FloatRange(0, 1), default=0.8,
    help="Proportion of the dataset to use for the training set (between 0 and 1)."
)
@click.option("--random-state", "-r", type=int, default=42, help="Random seed for reproducibility.")
def train(input_file, output_dir, train_size, random_state):
    """Train and evaluate an XGBoost model on the viscosity dataset."""
    from ..xgboost import train

    # Train the model
    train(
        input_file=input_file,
        output_dir=output_dir,
        train_size=train_size,
        random_state=random_state,
    )

@xgboost.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True), required=True, help="Path to the input Excel file containing the dataset."
)
@click.option(
    "--model-file", "-m",
    type=click.Path(exists=True), default=None,
    help="Path to the pre-trained XGBoost model file (JSON format). If not provided, the default model included in the package will be used."
)
@click.option(
    "--predictions-file", "-p",
    type=click.Path(), default="viscosity_predictions.csv",
    help="Path to save the predictions CSV file."
)
def inference(input_file, model_file, predictions_file):
    """Run inference using a pre-trained XGBoost model."""
    from ..xgboost import inference

    # Run inference
    inference(
        input_file=input_file,
        model_file=model_file,
        predictions_file=predictions_file,
    )

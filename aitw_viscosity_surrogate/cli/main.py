"""AITW Monomer Viscosity Surrogate CLI commands"""
import os

import click as original_click
import rich
import rich_click as click
from rich.traceback import install
from trogon import tui


@tui()
@click.group()
@click.version_option(package_name="aitw-viscosity-surrogate", prog_name="AITW Viscosity Surrogate")
def cli():
    install(
        show_locals=os.getenv('AITW_VISCOSITY_SURROGATE_DEBUG', '0').lower() in ('1', 'true', 'yes'),
        suppress=[rich, click, original_click],
    )

__all__ = [
    'cli',
]

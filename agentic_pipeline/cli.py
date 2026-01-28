"""Agentic Pipeline CLI."""

import click
from rich.console import Console

console = Console()


@click.group()
def main():
    """Agentic Pipeline - AI-powered book ingestion."""
    pass


@main.command()
def version():
    """Show version."""
    from . import __version__
    console.print(f"agentic-pipeline v{__version__}")


if __name__ == "__main__":
    main()

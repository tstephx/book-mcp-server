"""Agentic Pipeline CLI."""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

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


@main.command()
def init():
    """Initialize the database with agentic pipeline tables."""
    from .db.config import get_db_path
    from .db.migrations import run_migrations

    db_path = get_db_path()
    console.print(f"[blue]Initializing database at {db_path}[/blue]")

    run_migrations(db_path)

    console.print("[green]Database initialized successfully![/green]")


@main.command()
def pending():
    """List books pending approval."""
    from .db.config import get_db_path
    from .approval.queue import ApprovalQueue

    db_path = get_db_path()
    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    if result["pending_count"] == 0:
        console.print("[yellow]No books pending approval[/yellow]")
        return

    console.print(f"\n[bold]Pending Approval: {result['pending_count']} books[/bold]")
    console.print(f"  High confidence (≥90%): {result['stats']['high_confidence']}")
    console.print(f"  Needs attention (<80%): {result['stats']['needs_attention']}")
    console.print()

    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Confidence")
    table.add_column("Source")

    for book in result["books"]:
        conf = book["confidence"]
        conf_style = "green" if conf >= 0.9 else "yellow" if conf >= 0.8 else "red"
        table.add_row(
            book["id"][:8] + "...",
            book["book_type"],
            f"[{conf_style}]{conf:.0%}[/{conf_style}]",
            Path(book["source_path"]).name[:40],
        )

    console.print(table)


@main.command()
@click.argument("pipeline_id")
def approve(pipeline_id: str):
    """Approve a pending book."""
    from .db.config import get_db_path
    from .approval.actions import approve_book

    db_path = get_db_path()
    result = approve_book(db_path, pipeline_id, actor="human:cli")

    if result["success"]:
        console.print(f"[green]Approved: {pipeline_id}[/green]")
    else:
        console.print(f"[red]Failed: {result.get('error')}[/red]")


@main.command()
@click.argument("pipeline_id")
@click.option("--reason", "-r", required=True, help="Reason for rejection")
@click.option("--retry", is_flag=True, help="Queue for retry instead of permanent rejection")
def reject(pipeline_id: str, reason: str, retry: bool):
    """Reject a pending book."""
    from .db.config import get_db_path
    from .approval.actions import reject_book

    db_path = get_db_path()
    result = reject_book(db_path, pipeline_id, reason, actor="human:cli", retry=retry)

    if result["success"]:
        action = "Queued for retry" if retry else "Rejected"
        console.print(f"[yellow]{action}: {pipeline_id}[/yellow]")
    else:
        console.print(f"[red]Failed: {result.get('error')}[/red]")


@main.command()
def strategies():
    """List available processing strategies."""
    from .pipeline.strategy import StrategySelector

    selector = StrategySelector()
    names = selector.list_strategies()

    console.print("\n[bold]Available Strategies:[/bold]\n")
    for name in sorted(names):
        strategy = selector.load_strategy(name)
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    Book type: {strategy['book_type']}")
        console.print(f"    Version: {strategy.get('version', 1)}")
        console.print()


if __name__ == "__main__":
    main()

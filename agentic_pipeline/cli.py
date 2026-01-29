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


@main.command()
@click.option("--text", "-t", required=True, help="Text to classify (or path to file)")
@click.option("--provider", "-p", default="openai", help="Provider to use (openai, anthropic)")
def classify(text: str, provider: str):
    """Classify book text and show the result."""
    import hashlib
    from .db.config import get_db_path
    from .agents.classifier import ClassifierAgent
    from .agents.providers.openai_provider import OpenAIProvider
    from .agents.providers.anthropic_provider import AnthropicProvider

    # Check if text is a file path
    text_path = Path(text)
    if text_path.exists():
        text = text_path.read_text()
        console.print(f"[dim]Read {len(text)} chars from {text_path}[/dim]")

    # Generate hash from text
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Select provider
    if provider == "anthropic":
        primary = AnthropicProvider()
    else:
        primary = OpenAIProvider()

    db_path = get_db_path()
    agent = ClassifierAgent(db_path, primary=primary)

    console.print(f"[blue]Classifying with {provider}...[/blue]")

    result = agent.classify(text, content_hash=content_hash)

    console.print(f"\n[bold]Classification Result:[/bold]")
    console.print(f"  Type: [cyan]{result.book_type.value}[/cyan]")

    conf = result.confidence
    conf_style = "green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red"
    console.print(f"  Confidence: [{conf_style}]{conf:.0%}[/{conf_style}]")

    if result.suggested_tags:
        console.print(f"  Tags: {', '.join(result.suggested_tags)}")

    console.print(f"  Reasoning: [dim]{result.reasoning}[/dim]")


@main.command()
@click.argument("book_path", type=click.Path(exists=True))
def process(book_path: str):
    """Process a single book through the pipeline."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    console.print(f"[blue]Processing: {book_path}[/blue]")

    result = orchestrator.process_one(book_path)

    if result.get("skipped"):
        console.print(f"[yellow]Skipped: {result['reason']}[/yellow]")
        return

    state = result.get("state", "unknown")
    if state == "complete":
        console.print(f"[green]Complete![/green]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "pending_approval":
        console.print(f"[yellow]Pending approval[/yellow]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "needs_retry":
        console.print(f"[red]Failed - queued for retry[/red]")
        console.print(f"  Error: {result.get('error')}")
    else:
        console.print(f"[dim]State: {state}[/dim]")

    console.print(f"  Pipeline ID: {result.get('pipeline_id')}")


@main.command()
def worker():
    """Run the queue worker (processes books continuously)."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    console.print("[blue]Starting worker... Press Ctrl+C to stop gracefully.[/blue]")
    orchestrator.run_worker()
    console.print("[green]Worker stopped.[/green]")


@main.command()
@click.option("--max-attempts", "-m", default=3, help="Max retry attempts before rejection")
def retry(max_attempts: int):
    """Retry books in NEEDS_RETRY state."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    config.max_retry_attempts = max_attempts
    orchestrator = Orchestrator(config)

    console.print(f"[blue]Retrying failed books (max {max_attempts} attempts)...[/blue]")

    results = orchestrator.retry_failed()

    if not results:
        console.print("[yellow]No books to retry[/yellow]")
        return

    for result in results:
        pid = result.get("pipeline_id", "?")[:8]
        state = result.get("state")
        if state == "complete":
            console.print(f"  [green]{pid}...: Complete[/green]")
        elif state == "rejected":
            console.print(f"  [red]{pid}...: Rejected ({result.get('reason')})[/red]")
        else:
            console.print(f"  [yellow]{pid}...: {state}[/yellow]")


@main.command()
@click.argument("pipeline_id")
def status(pipeline_id: str):
    """Show status of a pipeline."""
    from .db.config import get_db_path
    from .db.pipelines import PipelineRepository
    import json

    db_path = get_db_path()
    repo = PipelineRepository(db_path)

    pipeline = repo.get(pipeline_id)

    if not pipeline:
        console.print(f"[red]Pipeline not found: {pipeline_id}[/red]")
        return

    console.print(f"\n[bold]Pipeline: {pipeline_id}[/bold]")
    console.print(f"  State: [cyan]{pipeline['state']}[/cyan]")
    console.print(f"  Source: {pipeline['source_path']}")

    if pipeline.get("book_profile"):
        profile = json.loads(pipeline["book_profile"]) if isinstance(pipeline["book_profile"], str) else pipeline["book_profile"]
        console.print(f"  Type: {profile.get('book_type')}")
        conf = profile.get("confidence", 0)
        conf_style = "green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red"
        console.print(f"  Confidence: [{conf_style}]{conf:.0%}[/{conf_style}]")

    console.print(f"  Retries: {pipeline.get('retry_count', 0)}")
    console.print(f"  Created: {pipeline.get('created_at')}")

    if pipeline.get("approved_by"):
        console.print(f"  Approved by: {pipeline['approved_by']}")


# Phase 4: Production Hardening Commands

@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def health(as_json: bool):
    """Show pipeline health status."""
    from .db.config import get_db_path
    from .health import HealthMonitor, StuckDetector
    import json as json_module

    db_path = get_db_path()
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    report["stuck"] = detector.detect()

    if as_json:
        console.print(json_module.dumps(report, indent=2))
        return

    console.print("\n[bold]Pipeline Health[/bold]")
    console.print("-" * 35)
    console.print(f"  Active:     {report['active']} (processing now)")

    stuck_count = len(report['stuck'])
    if stuck_count > 0:
        console.print(f"  Stuck:      [red]{stuck_count} [!][/red]")
    else:
        console.print(f"  Stuck:      {stuck_count}")

    console.print(f"  Queued:     {report['queued']} (waiting)")
    console.print(f"  Completed:  {report['completed_24h']} (last 24h)")
    console.print(f"  Failed:     {report['failed']} (needs_retry)")
    console.print("-" * 35)

    if report['alerts']:
        console.print("\n[yellow]Alerts:[/yellow]")
        for alert in report['alerts']:
            console.print(f"  [{alert['severity']}] {alert['message']}")


@main.command()
@click.option("--recover", is_flag=True, help="Auto-recover stuck pipelines")
def stuck(recover: bool):
    """List stuck pipelines."""
    from .db.config import get_db_path
    from .health import StuckDetector

    db_path = get_db_path()
    detector = StuckDetector(db_path)
    stuck_list = detector.detect()

    if not stuck_list:
        console.print("[green]No stuck pipelines[/green]")
        return

    console.print(f"\n[yellow]Found {len(stuck_list)} stuck pipeline(s):[/yellow]\n")

    for item in stuck_list:
        console.print(f"  {item['id'][:8]}... [{item['state']}]")
        console.print(f"    Stuck for {item['stuck_minutes']} min (expected: {item['expected_minutes']} min)")
        console.print(f"    Source: {Path(item['source_path']).name}")


@main.command("batch-approve")
@click.option("--min-confidence", type=float, help="Minimum confidence threshold")
@click.option("--book-type", help="Filter by book type")
@click.option("--max-count", default=50, help="Maximum books to approve")
@click.option("--execute", is_flag=True, help="Actually execute (otherwise dry-run)")
def batch_approve(min_confidence: float, book_type: str, max_count: int, execute: bool):
    """Approve books matching filters."""
    from .db.config import get_db_path
    from .batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        min_confidence=min_confidence,
        book_type=book_type,
        max_count=max_count,
    )

    result = ops.approve(filter, actor="human:cli", execute=execute)

    if execute:
        console.print(f"[green]Approved {result['approved']} books[/green]")
    else:
        console.print(f"[yellow]Would approve {result['would_approve']} books (dry-run)[/yellow]")
        for book in result['books'][:10]:
            console.print(f"  {book['id'][:8]}... {Path(book['source_path']).name}")
        if len(result['books']) > 10:
            console.print(f"  ... and {len(result['books']) - 10} more")


@main.command("batch-reject")
@click.option("--book-type", help="Filter by book type")
@click.option("--max-confidence", type=float, help="Maximum confidence threshold")
@click.option("--reason", required=True, help="Rejection reason")
@click.option("--max-count", default=50, help="Maximum books to reject")
@click.option("--execute", is_flag=True, help="Actually execute (otherwise dry-run)")
def batch_reject(book_type: str, max_confidence: float, reason: str, max_count: int, execute: bool):
    """Reject books matching filters."""
    from .db.config import get_db_path
    from .batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        book_type=book_type,
        max_confidence=max_confidence,
        max_count=max_count,
    )

    result = ops.reject(filter, reason=reason, actor="human:cli", execute=execute)

    if execute:
        console.print(f"[yellow]Rejected {result['rejected']} books[/yellow]")
    else:
        console.print(f"[yellow]Would reject {result['would_reject']} books (dry-run)[/yellow]")


@main.command()
@click.option("--last", default=50, help="Number of recent entries")
@click.option("--actor", help="Filter by actor")
@click.option("--action", help="Filter by action type")
@click.option("--book-id", help="Filter by book ID")
def audit(last: int, actor: str, action: str, book_id: str):
    """Query the audit trail."""
    from .db.config import get_db_path
    from .audit import AuditTrail

    db_path = get_db_path()
    trail = AuditTrail(db_path)

    entries = trail.query(
        book_id=book_id,
        actor=actor,
        action=action,
        limit=last,
    )

    if not entries:
        console.print("[yellow]No audit entries found[/yellow]")
        return

    console.print(f"\n[bold]Audit Trail ({len(entries)} entries)[/bold]\n")

    table = Table()
    table.add_column("Time", style="dim")
    table.add_column("Action")
    table.add_column("Actor")
    table.add_column("Book")

    for entry in entries:
        table.add_row(
            entry["performed_at"][:19] if entry.get("performed_at") else "?",
            entry["action"],
            entry["actor"],
            entry["book_id"][:16] + "..." if len(entry["book_id"]) > 16 else entry["book_id"],
        )

    console.print(table)


# Phase 5: Autonomy Commands

@main.group()
def autonomy():
    """Manage autonomy settings."""
    pass


@autonomy.command("status")
def autonomy_status():
    """Show current autonomy mode and thresholds."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig, MetricsCollector

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)

    mode = config.get_mode()
    escape_active = config.is_escape_hatch_active()
    metrics = collector.get_metrics(days=30)

    console.print(f"\n[bold]Autonomy Status[/bold]")
    console.print("-" * 35)

    if escape_active:
        console.print(f"  Mode: [red]ESCAPE HATCH ACTIVE[/red]")
    else:
        console.print(f"  Mode: [cyan]{mode}[/cyan]")

    console.print(f"\n[bold]Last 30 Days:[/bold]")
    console.print(f"  Total processed: {metrics['total_processed']}")
    console.print(f"  Auto-approved:   {metrics['auto_approved']}")
    console.print(f"  Human approved:  {metrics['human_approved']}")
    console.print(f"  Human rejected:  {metrics['human_rejected']}")


@autonomy.command("enable")
@click.argument("mode", type=click.Choice(["partial", "confident"]))
def autonomy_enable(mode: str):
    """Enable an autonomy mode."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if config.is_escape_hatch_active():
        console.print("[red]Cannot enable autonomy while escape hatch is active.[/red]")
        console.print("Run: agentic-pipeline autonomy resume")
        return

    config.set_mode(mode)
    console.print(f"[green]Autonomy mode set to: {mode}[/green]")


@autonomy.command("disable")
def autonomy_disable():
    """Disable autonomy (revert to supervised)."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.set_mode("supervised")
    console.print("[yellow]Autonomy disabled. All books require human review.[/yellow]")


@autonomy.command("resume")
def autonomy_resume():
    """Resume autonomy after escape hatch."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if not config.is_escape_hatch_active():
        console.print("[yellow]Escape hatch is not active.[/yellow]")
        return

    config.deactivate_escape_hatch()
    console.print("[green]Escape hatch deactivated. Autonomy resumed.[/green]")


@main.command("escape-hatch")
@click.argument("reason")
def escape_hatch(reason: str):
    """Activate escape hatch - immediately revert to supervised mode."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.activate_escape_hatch(reason)

    console.print("\n[red bold]⚠️  ESCAPE HATCH ACTIVATED[/red bold]")
    console.print("\nAll autonomy disabled. Reverting to supervised mode.")
    console.print(f"Reason: {reason}")
    console.print("\nTo resume: agentic-pipeline autonomy resume")


@main.command("spot-check")
@click.option("--list", "list_pending", is_flag=True, help="List pending spot-checks")
def spot_check(list_pending: bool):
    """Start or manage spot-check reviews."""
    from .db.config import get_db_path
    from .autonomy import SpotCheckManager

    db_path = get_db_path()
    manager = SpotCheckManager(db_path)

    if list_pending:
        pending = manager.select_for_review()
        if not pending:
            console.print("[green]No spot-checks pending.[/green]")
            return

        console.print(f"\n[bold]Pending Spot-Checks ({len(pending)} books)[/bold]\n")
        for book in pending[:10]:
            console.print(f"  {book['book_id'][:8]}... [{book['original_book_type']}] {book['original_confidence']:.0%}")
    else:
        console.print("Use --list to see pending spot-checks")
        console.print("Interactive spot-check not yet implemented")


if __name__ == "__main__":
    main()

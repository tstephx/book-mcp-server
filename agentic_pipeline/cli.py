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
    """Approve a pending book and generate embeddings."""
    from .db.config import get_db_path
    from .approval.actions import approve_book

    db_path = get_db_path()
    result = approve_book(db_path, pipeline_id, actor="human:cli")

    if result["success"]:
        state = result.get("state", "approved")
        if state == "complete":
            chapters = result.get("chapters_embedded", 0)
            console.print(f"[green]Approved & embedded: {pipeline_id} ({chapters} chapters)[/green]")
        elif state == "needs_retry":
            console.print(f"[yellow]Approved but embedding failed: {pipeline_id}[/yellow]")
            console.print(f"  Error: {result.get('embedding_error')}")
        else:
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

    console.print("\n[bold]Classification Result:[/bold]")
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
        console.print("[green]Complete![/green]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "pending_approval":
        console.print("[yellow]Pending approval[/yellow]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "needs_retry":
        console.print("[red]Failed - queued for retry[/red]")
        console.print(f"  Error: {result.get('error')}")
    else:
        console.print(f"[dim]State: {state}[/dim]")

    console.print(f"  Pipeline ID: {result.get('pipeline_id')}")


@main.command()
@click.option("--watch-dir", type=click.Path(exists=True), default=None,
              help="Directory to watch for new book files (.epub, .pdf)")
@click.option("--processed-dir", type=click.Path(), default=None,
              help="Directory to move processed book files into")
def worker(watch_dir, processed_dir):
    """Run the queue worker (processes books continuously)."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    if watch_dir:
        config.watch_dir = Path(watch_dir)
    if processed_dir:
        config.processed_dir = Path(processed_dir)

    orchestrator = Orchestrator(config)

    if config.watch_dir:
        console.print(f"[blue]Watching: {config.watch_dir}[/blue]")
    if config.processed_dir:
        console.print(f"[blue]Archive to: {config.processed_dir}[/blue]")
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
@click.argument("book_id")
def reingest(book_id: str):
    """Reprocess a book through the full pipeline.

    Archives the existing pipeline record and creates a new one.
    The book will go through classification, processing, validation,
    and approval again.
    """
    from .db.config import get_db_path
    from .db.pipelines import PipelineRepository
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    db_path = get_db_path()
    repo = PipelineRepository(db_path)

    # Check the book exists
    record = repo.get(book_id)
    if not record:
        console.print(f"[red]Pipeline record not found: {book_id}[/red]")
        console.print("[dim]Use 'agentic-pipeline backfill' first if this is a legacy book[/dim]")
        return

    source_path = record["source_path"]
    if not source_path or not Path(source_path).exists():
        console.print(f"[red]Source file not found: {source_path}[/red]")
        console.print("[dim]The original book file is needed for reingestion[/dim]")
        return

    console.print(f"[blue]Reingesting: {source_path}[/blue]")
    console.print(f"[dim]Archiving old record: {book_id}[/dim]")

    new_pid = repo.prepare_reingest(book_id)
    console.print(f"[dim]New pipeline: {new_pid}[/dim]")

    # Process through the pipeline
    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)
    result = orchestrator._process_book(new_pid, source_path, record["content_hash"])

    state = result.get("state", "unknown")
    if state == "complete":
        console.print("[green]Reingestion complete![/green]")
    elif state == "pending_approval":
        console.print("[yellow]Reingestion done - pending approval[/yellow]")
    else:
        console.print(f"[red]Reingestion ended in state: {state}[/red]")
        if result.get("error"):
            console.print(f"  Error: {result['error']}")


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
        embedded = result.get('embedded', 0)
        failures = result.get('embedding_failures', [])
        console.print(f"[green]Approved {result['approved']} books, embedded {embedded}[/green]")
        if failures:
            console.print(f"[yellow]  {len(failures)} embedding failure(s):[/yellow]")
            for f in failures[:5]:
                console.print(f"    {f['id'][:8]}... {f.get('error', 'unknown')}")
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


@main.command("library-status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def library_status(as_json: bool):
    """Show library status dashboard."""
    from .db.config import get_db_path
    from .library import LibraryStatus
    import json as json_module

    db_path = get_db_path()
    monitor = LibraryStatus(db_path)
    report = monitor.get_status()

    if as_json:
        console.print(json_module.dumps(report, indent=2))
        return

    overview = report["overview"]

    console.print("\n[bold]Library Status[/bold]")
    console.print("-" * 40)
    console.print(f"  Books:          {overview['total_books']}")
    console.print(f"  Chapters:       {overview['total_chapters']}")
    console.print(f"  Total words:    {overview['total_words']:,}")
    console.print(f"  Embedded:       {overview['embedded_chapters']}/{overview['total_chapters']} ({overview['embedding_coverage_pct']}%)")
    console.print()
    console.print(f"  [green]Ready:          {overview['books_fully_ready']}[/green]")
    console.print(f"  [yellow]Partial:        {overview['books_partially_ready']}[/yellow]")
    console.print(f"  [red]No embeddings:  {overview['books_not_embedded']}[/red]")

    if report["books"]:
        console.print()
        table = Table()
        table.add_column("Title", max_width=40)
        table.add_column("Author", max_width=20)
        table.add_column("Chapters", justify="right")
        table.add_column("Embedded", justify="right")
        table.add_column("Status")
        table.add_column("Source", style="dim")

        for book in report["books"]:
            status = book["status"]
            if status == "ready":
                status_display = "[green]ready[/green]"
            elif status == "partial":
                status_display = f"[yellow]partial ({book['embedding_pct']}%)[/yellow]"
            else:
                status_display = "[red]no embeddings[/red]"

            table.add_row(
                book["title"][:40],
                (book["author"] or "")[:20],
                str(book["chapters"]),
                str(book["embedded_chapters"]),
                status_display,
                book["source"],
            )

        console.print(table)

    pipeline = report["pipeline_summary"]
    if pipeline["total_pipelines"] > 0:
        console.print(f"\n[bold]Pipeline[/bold]: {pipeline['total_pipelines']} total")
        for state, count in sorted(pipeline["by_state"].items()):
            console.print(f"  {state}: {count}")


@main.command()
@click.option("--dry-run", is_flag=True, help="Preview without changes")
@click.option("--execute", is_flag=True, help="Actually create pipeline records")
def backfill(dry_run: bool, execute: bool):
    """Register legacy library books in the pipeline.

    Creates pipeline records for books that were ingested via the raw CLI
    and have no audit trail. Safe and non-destructive.
    """
    from .db.config import get_db_path
    from .backfill import BackfillManager

    if not dry_run and not execute:
        console.print("[yellow]Use --dry-run to preview or --execute to backfill[/yellow]")
        return

    db_path = get_db_path()
    manager = BackfillManager(db_path)

    if dry_run:
        result = manager.run(dry_run=True)
        count = result["would_backfill"]

        if count == 0:
            console.print("[green]All library books are tracked in the pipeline.[/green]")
            return

        console.print(f"\n[bold]Would backfill {count} books:[/bold]\n")

        table = Table()
        table.add_column("Title", max_width=40)
        table.add_column("Author", max_width=20)
        table.add_column("Chapters", justify="right")
        table.add_column("Embedded", justify="right")
        table.add_column("Quality")

        for book in result["books"]:
            quality = book["quality"]
            if quality == "good":
                q_display = "[green]good[/green]"
            elif quality == "missing_embeddings":
                q_display = "[yellow]missing embeddings[/yellow]"
            else:
                q_display = "[red]no chapters[/red]"

            table.add_row(
                (book["title"] or "?")[:40],
                (book["author"] or "")[:20],
                str(book["chapter_count"]),
                str(book["embedded_count"]),
                q_display,
            )

        console.print(table)
        console.print(f"\n[dim]Run with --execute to backfill[/dim]")
    else:
        result = manager.run(dry_run=False)
        console.print(f"[green]Backfilled {result['backfilled']} books[/green]")
        if result["skipped"] > 0:
            console.print(f"[yellow]Skipped {result['skipped']} (hash collision)[/yellow]")

        # Report quality issues
        issues = [b for b in result["books"] if b["quality"] != "good"]
        if issues:
            console.print(f"\n[yellow]{len(issues)} books with quality issues:[/yellow]")
            for b in issues[:10]:
                console.print(f"  {b['title'][:40]} - {b['quality']}")


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

    console.print("\n[bold]Autonomy Status[/bold]")
    console.print("-" * 35)

    if escape_active:
        console.print("  Mode: [red]ESCAPE HATCH ACTIVE[/red]")
    else:
        console.print(f"  Mode: [cyan]{mode}[/cyan]")

    console.print("\n[bold]Last 30 Days:[/bold]")
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


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate(as_json: bool):
    """Check library books for quality issues."""
    from .db.config import get_db_path
    from .backfill import LibraryValidator
    import json as json_module

    db_path = get_db_path()
    validator = LibraryValidator(db_path)
    issues = validator.validate()

    if as_json:
        console.print(json_module.dumps(issues, indent=2))
        return

    if not issues:
        console.print("[green]All books pass quality checks.[/green]")
        return

    console.print(f"\n[yellow]Found {len(issues)} issue(s):[/yellow]\n")

    table = Table()
    table.add_column("Title", max_width=35)
    table.add_column("Issue")
    table.add_column("Detail")

    for issue in issues:
        issue_type = issue["issue"]
        if issue_type == "no_chapters":
            style = "red"
        elif issue_type == "missing_embeddings":
            style = "yellow"
        else:
            style = "dim"

        table.add_row(
            (issue["title"] or "?")[:35],
            f"[{style}]{issue_type}[/{style}]",
            issue["detail"],
        )

    console.print(table)


if __name__ == "__main__":
    main()

# Claude Project Context

This is the **Agentic Book Processing Pipeline** - an AI-powered system that automatically processes, classifies, and ingests books into a searchable knowledge library.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Initialize database
agentic-pipeline init

# Check pipeline health
agentic-pipeline health

# Check autonomy status
agentic-pipeline autonomy status
```

## Project Structure

```
agentic_pipeline/
├── agents/                 # AI-powered components
│   ├── classifier.py       # Book type classification
│   ├── validator.py        # Quality validation
│   └── providers/          # LLM providers (OpenAI, Anthropic)
├── pipeline/               # State machine & orchestration
│   ├── states.py           # Pipeline states enum
│   ├── strategy.py         # Strategy selection
│   └── transitions.py      # State transitions
├── approval/               # Approval queue & actions
│   ├── queue.py            # ApprovalQueue
│   └── actions.py          # approve/reject/rollback
├── autonomy/               # Phase 5: Graduated trust
│   ├── config.py           # AutonomyConfig (modes, escape hatch)
│   ├── metrics.py          # MetricsCollector
│   ├── calibration.py      # CalibrationEngine (thresholds)
│   └── spot_check.py       # SpotCheckManager
├── health/                 # Phase 4: Production monitoring
│   ├── monitor.py          # HealthMonitor
│   └── stuck_detector.py   # StuckDetector
├── batch/                  # Phase 4: Bulk operations
│   ├── filters.py          # BatchFilter
│   └── operations.py       # BatchOperations
├── audit/                  # Phase 4: Audit trail
│   └── trail.py            # AuditTrail
├── db/                     # Database layer
│   ├── migrations.py       # Schema definitions
│   ├── pipelines.py        # PipelineRepository
│   └── config.py           # DB path configuration
├── cli.py                  # Click CLI commands
├── mcp_server.py           # MCP tools for Claude
├── orchestrator.py         # Main orchestrator
└── config.py               # OrchestratorConfig
```

## Key Concepts

### Pipeline States
Books flow through: `QUEUED` → `HASHING` → `CLASSIFYING` → `PROCESSING` → `VALIDATING` → `PENDING_APPROVAL` → `COMPLETE`

### Autonomy Modes
- **supervised** - All books require human approval (default)
- **partial** - Auto-approve high-confidence known types
- **confident** - Per-type calibrated thresholds

### Escape Hatch
One command reverts to fully supervised mode:
```bash
agentic-pipeline escape-hatch "reason"
```

## Common Tasks

### Adding a New Feature
1. Write tests first in `tests/`
2. Implement in appropriate module
3. Add CLI command if user-facing
4. Add MCP tool if Claude should use it
5. Run `pytest tests/ -v` to verify

### Database Changes
1. Add migration to `agentic_pipeline/db/migrations.py` in `MIGRATIONS` list
2. Write test in `tests/test_*_migrations.py`
3. Existing DBs auto-migrate on `run_migrations()`

### CLI Commands
Commands are in `agentic_pipeline/cli.py` using Click:
```python
@main.command()
@click.option("--flag", is_flag=True)
def my_command(flag: bool):
    """Command description."""
    pass
```

### MCP Tools
Tools are in `agentic_pipeline/mcp_server.py`:
```python
def my_tool(arg: str) -> dict:
    """Tool description for Claude."""
    return {"result": "value"}
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific phase
pytest tests/test_phase5*.py -v

# Single file
pytest tests/test_autonomy_config.py -v

# With coverage
pytest tests/ --cov=agentic_pipeline
```

## Environment

- Python 3.12+
- Virtual env at `.venv/`
- Database at `~/.agentic-pipeline/pipeline.db` (or `AGENTIC_PIPELINE_DB` env var)

## Architecture Decisions

1. **SQLite** - Single-file database, no server needed
2. **Click CLI** - Standard Python CLI framework
3. **Rich** - Terminal formatting and tables
4. **TDD** - Tests written before implementation
5. **Immutable Audit** - All decisions logged permanently

## Documentation

- `README.md` - User-facing overview
- `DESIGN.md` - Technical architecture
- `docs/PHASE4-PRODUCTION-HARDENING-COMPLETE.md` - Phase 4 features
- `docs/PHASE5-CONFIDENT-AUTONOMY-COMPLETE.md` - Phase 5 features
- `docs/plans/` - Design documents and implementation plans

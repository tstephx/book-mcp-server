"""Database migrations for agentic pipeline tables."""

import sqlite3
from pathlib import Path


MIGRATIONS = [
    # Pipeline tracking
    """
    CREATE TABLE IF NOT EXISTS processing_pipelines (
        id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        state TEXT NOT NULL,
        book_profile JSON,
        strategy_config JSON,
        validation_result JSON,
        processing_result JSON,
        retry_count INTEGER DEFAULT 0,
        max_retries INTEGER DEFAULT 2,
        error_log JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        completed_at TIMESTAMP,
        timeout_at TIMESTAMP,
        last_heartbeat TIMESTAMP,
        priority INTEGER DEFAULT 5,
        approved_by TEXT,
        approval_confidence REAL,
        UNIQUE(content_hash)
    )
    """,

    # State history
    """
    CREATE TABLE IF NOT EXISTS pipeline_state_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_id TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        duration_ms INTEGER,
        agent_output JSON,
        error_details JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pipeline_id) REFERENCES processing_pipelines(id)
    )
    """,

    # Strategy configurations
    """
    CREATE TABLE IF NOT EXISTS processing_strategies (
        name TEXT PRIMARY KEY,
        book_type TEXT NOT NULL,
        config JSON NOT NULL,
        version INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT TRUE
    )
    """,

    # Pipeline config
    """
    CREATE TABLE IF NOT EXISTS pipeline_config (
        key TEXT PRIMARY KEY,
        value JSON NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Audit trail
    """
    CREATE TABLE IF NOT EXISTS approval_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id TEXT NOT NULL,
        pipeline_id TEXT,
        action TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT,
        before_state JSON,
        after_state JSON,
        adjustments JSON,
        filter_used JSON,
        confidence_at_decision REAL,
        autonomy_mode TEXT,
        session_id TEXT,
        performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Retention policies
    """
    CREATE TABLE IF NOT EXISTS audit_retention (
        audit_type TEXT PRIMARY KEY,
        retain_days INTEGER NOT NULL,
        last_cleanup TIMESTAMP
    )
    """,

    # Autonomy metrics
    """
    CREATE TABLE IF NOT EXISTS autonomy_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        period_start DATE NOT NULL,
        period_end DATE NOT NULL,
        total_processed INTEGER DEFAULT 0,
        auto_approved INTEGER DEFAULT 0,
        human_approved INTEGER DEFAULT 0,
        human_rejected INTEGER DEFAULT 0,
        human_adjusted INTEGER DEFAULT 0,
        avg_confidence_auto_approved REAL,
        avg_confidence_human_approved REAL,
        avg_confidence_human_rejected REAL,
        auto_approved_later_rolled_back INTEGER DEFAULT 0,
        human_approved_later_rolled_back INTEGER DEFAULT 0,
        metrics_by_type JSON,
        confidence_buckets JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(period_start, period_end)
    )
    """,

    # Autonomy feedback
    """
    CREATE TABLE IF NOT EXISTS autonomy_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id TEXT NOT NULL,
        pipeline_id TEXT,
        original_decision TEXT NOT NULL,
        original_confidence REAL,
        original_book_type TEXT,
        human_decision TEXT NOT NULL,
        human_adjustments JSON,
        feedback_category TEXT,
        feedback_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Autonomy config (singleton)
    """
    CREATE TABLE IF NOT EXISTS autonomy_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        current_mode TEXT DEFAULT 'supervised',
        auto_approve_threshold REAL DEFAULT 0.95,
        auto_retry_threshold REAL DEFAULT 0.70,
        require_known_book_type BOOLEAN DEFAULT TRUE,
        require_zero_issues BOOLEAN DEFAULT TRUE,
        max_auto_approvals_per_day INTEGER DEFAULT 50,
        spot_check_percentage REAL DEFAULT 0.10,
        escape_hatch_active BOOLEAN DEFAULT FALSE,
        escape_hatch_activated_at TIMESTAMP,
        escape_hatch_reason TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Health metrics cache (Phase 4)
    """
    CREATE TABLE IF NOT EXISTS health_metrics (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        active_count INTEGER NOT NULL DEFAULT 0,
        queued_count INTEGER NOT NULL DEFAULT 0,
        stuck_count INTEGER NOT NULL DEFAULT 0,
        completed_24h INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        avg_processing_seconds REAL,
        queue_by_priority JSON,
        stuck_pipelines JSON,
        alerts JSON,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # State duration stats for stuck detection (Phase 4)
    """
    CREATE TABLE IF NOT EXISTS state_duration_stats (
        state TEXT PRIMARY KEY,
        sample_count INTEGER NOT NULL DEFAULT 0,
        median_seconds REAL NOT NULL DEFAULT 0,
        p95_seconds REAL NOT NULL DEFAULT 0,
        max_seconds REAL NOT NULL DEFAULT 0,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Per-type thresholds - Phase 5
    """
    CREATE TABLE IF NOT EXISTS autonomy_thresholds (
        book_type TEXT PRIMARY KEY,
        auto_approve_threshold REAL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        measured_accuracy REAL,
        last_calculated TIMESTAMP,
        calibration_data JSON,
        manual_override REAL,
        override_reason TEXT
    )
    """,

    # Spot-check tracking - Phase 5
    """
    CREATE TABLE IF NOT EXISTS spot_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id TEXT NOT NULL,
        pipeline_id TEXT,
        original_classification TEXT,
        original_confidence REAL,
        auto_approved_at TIMESTAMP,
        classification_correct BOOLEAN,
        quality_acceptable BOOLEAN,
        reviewer TEXT,
        notes TEXT,
        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pipelines_state ON processing_pipelines(state)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_hash ON processing_pipelines(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_priority ON processing_pipelines(priority, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_priority_queue ON processing_pipelines(state, priority, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_book ON approval_audit(book_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action ON approval_audit(action, performed_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_actor ON approval_audit(actor)",
    "CREATE INDEX IF NOT EXISTS idx_audit_session ON approval_audit(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_category ON autonomy_feedback(feedback_category)",
]

DEFAULT_RETENTION = [
    ("approved", 365),
    ("rejected", 90),
    ("rollback", -1),  # -1 = forever
    ("adjusted", 365),
]


def run_migrations(db_path: Path) -> None:
    """Run all migrations to set up agentic pipeline tables."""
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()

    # Enable WAL mode for concurrent read/write access
    cursor.execute("PRAGMA journal_mode = WAL")

    # Run table creation
    for migration in MIGRATIONS:
        cursor.execute(migration)

    # Run index creation
    for index in INDEXES:
        cursor.execute(index)

    # Add processing_result column to existing DBs (safe if already exists)
    try:
        cursor.execute(
            "ALTER TABLE processing_pipelines ADD COLUMN processing_result JSON"
        )
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Insert default autonomy config if not exists
    cursor.execute(
        "INSERT OR IGNORE INTO autonomy_config (id) VALUES (1)"
    )

    # Insert default retention policies
    for audit_type, retain_days in DEFAULT_RETENTION:
        cursor.execute(
            "INSERT OR IGNORE INTO audit_retention (audit_type, retain_days) VALUES (?, ?)",
            (audit_type, retain_days)
        )

    conn.commit()
    conn.close()

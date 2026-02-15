"""Project Planning Tools - Implementation Plans, BRDs, and Project Artifacts

Generates PM-ready project artifacts using the book library as a knowledge source.
Designed for translating technical projects into actionable business documents.

Key Features:
- Implementation plans with phases, milestones, dependencies
- Customizable templates by project type
- Integration with learning_path for reading recommendations
- Business-friendly framing with risk awareness
- Actionable prompts and next steps

Follows MCP best practices:
- Single responsibility per tool
- Composable (tools can use each other's output)
- Clean separation from other tools
- Comprehensive error handling
"""

import logging
import re
from typing import Optional, Literal
from datetime import datetime, timedelta
from ..database import get_db_connection, execute_query, execute_single
from ..utils.context_managers import embedding_model_context
from ..utils.vector_store import find_top_k
from ..utils.file_utils import read_chapter_content
from ..utils.excerpt_utils import extract_relevant_excerpt
from ..utils.cache import get_cache
from ..utils.chunk_loader import load_chunk_embeddings, best_chunk_per_chapter
import numpy as np
import io

logger = logging.getLogger(__name__)


# =============================================================================
# IMPLEMENTATION PLAN TEMPLATES BY PROJECT TYPE
# =============================================================================

IMPLEMENTATION_TEMPLATES = {
    "vps": {
        "name": "VPS / Server Infrastructure",
        "phases": [
            {
                "name": "Planning & Procurement",
                "duration_days": 3,
                "objectives": [
                    "Define server requirements (CPU, RAM, storage)",
                    "Select hosting provider and plan",
                    "Plan domain and DNS strategy",
                    "Document network architecture"
                ],
                "deliverables": ["Requirements document", "Provider selection rationale", "Architecture diagram"],
                "decisions": [
                    {"question": "Which hosting provider?", "options": ["Hetzner", "DigitalOcean", "Linode", "Vultr"], "considerations": "Price, location, features, support"},
                    {"question": "Server size?", "options": ["Small (2GB)", "Medium (4GB)", "Large (8GB+)"], "considerations": "Current needs vs growth, cost"}
                ],
                "risks": ["Underestimating resource needs", "Vendor lock-in"],
                "search_terms": ["server requirements planning", "VPS selection criteria"]
            },
            {
                "name": "Server Setup & Security",
                "duration_days": 5,
                "objectives": [
                    "Provision server and initial access",
                    "Harden SSH configuration",
                    "Configure firewall (UFW/iptables)",
                    "Set up fail2ban and intrusion detection",
                    "Create non-root user with sudo"
                ],
                "deliverables": ["Hardened server", "Security configuration docs", "Access credentials (secured)"],
                "decisions": [
                    {"question": "SSH key vs password?", "options": ["Key-only (recommended)", "Key + password"], "considerations": "Security vs convenience"},
                    {"question": "Firewall approach?", "options": ["UFW (simpler)", "iptables (flexible)", "nftables (modern)"], "considerations": "Complexity, control needed"}
                ],
                "risks": ["Lockout from misconfigured SSH", "Exposed services", "Weak credentials"],
                "gates": ["Can SSH with key only", "Firewall active", "fail2ban running"],
                "search_terms": ["SSH hardening", "Linux firewall configuration", "server security"]
            },
            {
                "name": "Web Server & SSL",
                "duration_days": 4,
                "objectives": [
                    "Install and configure web server (nginx/Apache)",
                    "Set up reverse proxy for applications",
                    "Configure SSL/TLS certificates",
                    "Implement security headers"
                ],
                "deliverables": ["Working web server", "SSL certificates", "Reverse proxy config"],
                "decisions": [
                    {"question": "Web server?", "options": ["nginx (recommended)", "Apache", "Caddy"], "considerations": "Performance, config style, features"},
                    {"question": "SSL provider?", "options": ["Let's Encrypt (free)", "Paid certificate"], "considerations": "Cost, validation level, wildcard needs"}
                ],
                "risks": ["SSL misconfiguration", "Downtime during setup", "Certificate expiration"],
                "gates": ["HTTPS working", "A+ SSL Labs rating", "Reverse proxy functional"],
                "search_terms": ["nginx configuration", "SSL TLS setup", "reverse proxy"]
            },
            {
                "name": "Application Deployment",
                "duration_days": 7,
                "objectives": [
                    "Set up Docker and Docker Compose",
                    "Deploy containerized applications",
                    "Configure persistent storage (volumes)",
                    "Set up application-specific configs"
                ],
                "deliverables": ["Running applications", "Docker Compose files", "Volume backup strategy"],
                "decisions": [
                    {"question": "Container orchestration?", "options": ["Docker Compose (simpler)", "Docker Swarm", "Kubernetes (complex)"], "considerations": "Scale needs, complexity tolerance"},
                    {"question": "Image management?", "options": ["Docker Hub", "Self-hosted registry", "GitHub Container Registry"], "considerations": "Privacy, cost, convenience"}
                ],
                "risks": ["Data loss from volume issues", "Resource exhaustion", "Container security"],
                "gates": ["All apps accessible", "Data persists across restarts", "Resource usage acceptable"],
                "search_terms": ["Docker deployment", "Docker Compose", "container orchestration"]
            },
            {
                "name": "Monitoring & Automation",
                "duration_days": 4,
                "objectives": [
                    "Set up monitoring and alerting",
                    "Configure automated backups",
                    "Implement log aggregation",
                    "Create maintenance runbooks"
                ],
                "deliverables": ["Monitoring dashboard", "Backup automation", "Runbooks"],
                "decisions": [
                    {"question": "Monitoring solution?", "options": ["Uptime Kuma (simple)", "Prometheus + Grafana", "Netdata"], "considerations": "Complexity, features, resource usage"},
                    {"question": "Backup destination?", "options": ["Local + offsite", "Cloud storage (S3/B2)", "Second server"], "considerations": "Cost, reliability, restore speed"}
                ],
                "risks": ["Silent failures", "Backup corruption", "Alert fatigue"],
                "gates": ["Alerts working", "Backup tested with restore", "Logs accessible"],
                "search_terms": ["server monitoring", "backup automation", "infrastructure monitoring"]
            },
            {
                "name": "Documentation & Handoff",
                "duration_days": 2,
                "objectives": [
                    "Document all configurations",
                    "Create troubleshooting guides",
                    "Establish maintenance schedule",
                    "Knowledge transfer (if applicable)"
                ],
                "deliverables": ["Complete documentation", "Troubleshooting guide", "Maintenance calendar"],
                "decisions": [],
                "risks": ["Knowledge loss", "Undocumented changes"],
                "gates": ["Docs reviewed", "Can rebuild from docs"],
                "search_terms": ["infrastructure documentation", "runbook creation"]
            }
        ],
        "success_criteria": [
            "All applications accessible via HTTPS",
            "Security scan passes with no critical issues",
            "Backups verified with successful restore test",
            "Monitoring alerts confirmed working",
            "Documentation complete and reviewed"
        ],
        "rollback_strategy": "Maintain server snapshot before major changes. Document rollback procedures for each phase. Keep previous configs in version control."
    },
    
    "web_app": {
        "name": "Web Application",
        "phases": [
            {
                "name": "Architecture & Design",
                "duration_days": 5,
                "objectives": [
                    "Define system architecture",
                    "Design data models",
                    "Plan API endpoints",
                    "Create wireframes/mockups"
                ],
                "deliverables": ["Architecture document", "Data model diagram", "API specification", "Wireframes"],
                "decisions": [
                    {"question": "Architecture pattern?", "options": ["Monolith", "Microservices", "Modular monolith"], "considerations": "Team size, complexity, scale needs"},
                    {"question": "Database?", "options": ["PostgreSQL", "MySQL", "MongoDB", "SQLite"], "considerations": "Data structure, scale, features"}
                ],
                "risks": ["Over-engineering", "Under-specifying requirements", "Missing edge cases"],
                "search_terms": ["software architecture", "API design", "data modeling"]
            },
            {
                "name": "Backend Development",
                "duration_days": 10,
                "objectives": [
                    "Set up development environment",
                    "Implement core API endpoints",
                    "Build authentication system",
                    "Create database migrations"
                ],
                "deliverables": ["Working API", "Auth system", "Database schema", "API documentation"],
                "decisions": [
                    {"question": "Framework?", "options": ["FastAPI", "Django", "Flask", "Express"], "considerations": "Team experience, features needed, performance"},
                    {"question": "Auth approach?", "options": ["JWT", "Session-based", "OAuth"], "considerations": "Use case, security requirements"}
                ],
                "risks": ["Security vulnerabilities", "Performance issues", "Technical debt"],
                "gates": ["All endpoints tested", "Auth working", "No critical vulnerabilities"],
                "search_terms": ["API development", "authentication", "backend architecture"]
            },
            {
                "name": "Frontend Development",
                "duration_days": 10,
                "objectives": [
                    "Set up frontend framework",
                    "Implement UI components",
                    "Connect to backend API",
                    "Add responsive design"
                ],
                "deliverables": ["Working UI", "Component library", "Responsive layouts"],
                "decisions": [
                    {"question": "Framework?", "options": ["React", "Vue", "Svelte", "HTMX"], "considerations": "Team skills, complexity, bundle size"},
                    {"question": "Styling?", "options": ["Tailwind", "CSS Modules", "Styled Components"], "considerations": "Maintainability, design system"}
                ],
                "risks": ["Poor UX", "Performance issues", "Browser compatibility"],
                "gates": ["Core flows working", "Mobile responsive", "Accessibility basics"],
                "search_terms": ["frontend development", "React patterns", "responsive design"]
            },
            {
                "name": "Testing & QA",
                "duration_days": 5,
                "objectives": [
                    "Write unit tests",
                    "Create integration tests",
                    "Perform security testing",
                    "User acceptance testing"
                ],
                "deliverables": ["Test suite", "Security report", "UAT sign-off"],
                "decisions": [
                    {"question": "Test coverage target?", "options": ["80%+", "Critical paths only", "TDD approach"], "considerations": "Time, risk tolerance, maintenance"}
                ],
                "risks": ["Undiscovered bugs", "Security vulnerabilities", "Poor test coverage"],
                "gates": ["Tests passing", "No critical bugs", "Security scan clean"],
                "search_terms": ["software testing", "test automation", "security testing"]
            },
            {
                "name": "Deployment & Launch",
                "duration_days": 5,
                "objectives": [
                    "Set up CI/CD pipeline",
                    "Configure production environment",
                    "Deploy application",
                    "Monitor launch"
                ],
                "deliverables": ["CI/CD pipeline", "Production deployment", "Monitoring setup"],
                "decisions": [
                    {"question": "Deployment platform?", "options": ["VPS", "PaaS (Heroku/Railway)", "Serverless", "Kubernetes"], "considerations": "Cost, control, complexity"},
                    {"question": "CI/CD tool?", "options": ["GitHub Actions", "GitLab CI", "Jenkins"], "considerations": "Integration, features, cost"}
                ],
                "risks": ["Deployment failures", "Performance issues at scale", "Data migration errors"],
                "gates": ["Successful deploy", "Monitoring active", "Rollback tested"],
                "search_terms": ["CI/CD", "deployment", "DevOps"]
            }
        ],
        "success_criteria": [
            "All user stories implemented and tested",
            "Performance meets defined SLAs",
            "Security scan passes",
            "Successful production deployment",
            "Monitoring and alerting active"
        ],
        "rollback_strategy": "Blue-green deployment or canary releases. Database migrations must be reversible. Feature flags for gradual rollout."
    },
    
    "data_pipeline": {
        "name": "Data Pipeline / Analytics",
        "phases": [
            {
                "name": "Requirements & Data Discovery",
                "duration_days": 4,
                "objectives": [
                    "Identify data sources",
                    "Document data schemas",
                    "Define transformation requirements",
                    "Establish data quality criteria"
                ],
                "deliverables": ["Data source inventory", "Schema documentation", "Quality criteria"],
                "decisions": [
                    {"question": "Pipeline type?", "options": ["Batch", "Streaming", "Hybrid"], "considerations": "Latency requirements, data volume, complexity"}
                ],
                "risks": ["Unknown data quality issues", "Missing data sources", "Schema changes"],
                "search_terms": ["data pipeline design", "ETL requirements", "data discovery"]
            },
            {
                "name": "Infrastructure Setup",
                "duration_days": 3,
                "objectives": [
                    "Set up development environment",
                    "Configure data storage",
                    "Set up orchestration tool",
                    "Establish version control"
                ],
                "deliverables": ["Dev environment", "Storage configured", "Orchestration ready"],
                "decisions": [
                    {"question": "Orchestration?", "options": ["Airflow", "Prefect", "Dagster", "Cron"], "considerations": "Complexity, features, team experience"},
                    {"question": "Storage?", "options": ["Data warehouse", "Data lake", "Hybrid lakehouse"], "considerations": "Query patterns, cost, scale"}
                ],
                "risks": ["Wrong tool selection", "Infrastructure costs", "Complexity overhead"],
                "search_terms": ["data infrastructure", "pipeline orchestration", "data storage"]
            },
            {
                "name": "ETL Development",
                "duration_days": 8,
                "objectives": [
                    "Build extraction connectors",
                    "Implement transformations",
                    "Create loading processes",
                    "Add data validation"
                ],
                "deliverables": ["Working ETL pipeline", "Transformation logic", "Validation rules"],
                "decisions": [
                    {"question": "Processing framework?", "options": ["Pandas", "Spark", "dbt", "SQL"], "considerations": "Data size, complexity, team skills"}
                ],
                "risks": ["Data loss", "Transformation errors", "Performance issues"],
                "gates": ["Pipeline runs end-to-end", "Data quality checks pass", "Performance acceptable"],
                "search_terms": ["ETL development", "data transformation", "pandas processing"]
            },
            {
                "name": "Testing & Validation",
                "duration_days": 4,
                "objectives": [
                    "Test data quality",
                    "Validate transformations",
                    "Performance testing",
                    "Edge case handling"
                ],
                "deliverables": ["Test suite", "Validation report", "Performance benchmarks"],
                "decisions": [],
                "risks": ["Undetected data quality issues", "Silent failures"],
                "gates": ["All tests passing", "Data reconciliation complete"],
                "search_terms": ["data testing", "data quality", "pipeline testing"]
            },
            {
                "name": "Deployment & Monitoring",
                "duration_days": 3,
                "objectives": [
                    "Deploy to production",
                    "Set up scheduling",
                    "Configure monitoring and alerts",
                    "Document runbooks"
                ],
                "deliverables": ["Production pipeline", "Monitoring dashboard", "Runbooks"],
                "decisions": [
                    {"question": "Scheduling frequency?", "options": ["Hourly", "Daily", "Real-time"], "considerations": "Freshness needs, cost, complexity"}
                ],
                "risks": ["Pipeline failures", "Data freshness issues", "Cost overruns"],
                "gates": ["Pipeline running on schedule", "Alerts working", "Costs within budget"],
                "search_terms": ["pipeline monitoring", "data observability", "production deployment"]
            }
        ],
        "success_criteria": [
            "Pipeline runs reliably on schedule",
            "Data quality meets defined criteria",
            "Processing time within SLA",
            "Monitoring and alerting active",
            "Documentation complete"
        ],
        "rollback_strategy": "Maintain previous pipeline version. Implement data versioning. Have manual backfill procedures ready."
    },
    
    "automation": {
        "name": "Automation / Scripting",
        "phases": [
            {
                "name": "Process Analysis",
                "duration_days": 2,
                "objectives": [
                    "Document current manual process",
                    "Identify automation opportunities",
                    "Define success metrics",
                    "Assess technical feasibility"
                ],
                "deliverables": ["Process documentation", "Automation scope", "Success metrics"],
                "decisions": [
                    {"question": "Automation scope?", "options": ["Full automation", "Partial with human review", "Assisted automation"], "considerations": "Risk tolerance, complexity, value"}
                ],
                "risks": ["Automating bad processes", "Missing edge cases", "Over-automation"],
                "search_terms": ["process automation", "workflow analysis"]
            },
            {
                "name": "Script Development",
                "duration_days": 5,
                "objectives": [
                    "Develop automation scripts",
                    "Implement error handling",
                    "Add logging and reporting",
                    "Create configuration management"
                ],
                "deliverables": ["Working scripts", "Configuration files", "Logging system"],
                "decisions": [
                    {"question": "Language?", "options": ["Python", "Bash", "PowerShell"], "considerations": "Platform, complexity, maintainability"},
                    {"question": "Config management?", "options": ["Environment variables", "Config files", "Secrets manager"], "considerations": "Security, flexibility"}
                ],
                "risks": ["Script failures", "Security issues", "Maintenance burden"],
                "gates": ["Scripts run successfully", "Error handling tested", "Logs working"],
                "search_terms": ["Python automation", "scripting best practices", "error handling"]
            },
            {
                "name": "Testing & Validation",
                "duration_days": 3,
                "objectives": [
                    "Test happy path scenarios",
                    "Test error conditions",
                    "Validate outputs",
                    "Performance testing"
                ],
                "deliverables": ["Test results", "Validation report"],
                "decisions": [],
                "risks": ["Unhandled edge cases", "Data corruption"],
                "gates": ["All tests passing", "Output validated"],
                "search_terms": ["automation testing", "script validation"]
            },
            {
                "name": "Deployment & Scheduling",
                "duration_days": 2,
                "objectives": [
                    "Deploy scripts to production",
                    "Set up scheduling (cron/task scheduler)",
                    "Configure monitoring",
                    "Create runbooks"
                ],
                "deliverables": ["Deployed automation", "Schedule configured", "Runbooks"],
                "decisions": [
                    {"question": "Scheduling?", "options": ["Cron", "systemd timers", "Cloud scheduler"], "considerations": "Platform, reliability, monitoring"}
                ],
                "risks": ["Silent failures", "Schedule conflicts", "Resource contention"],
                "gates": ["Running on schedule", "Alerts configured"],
                "search_terms": ["cron scheduling", "automation deployment"]
            }
        ],
        "success_criteria": [
            "Automation runs reliably",
            "Time savings measured and significant",
            "Error rate acceptable",
            "Monitoring active",
            "Documentation complete"
        ],
        "rollback_strategy": "Keep manual process documented. Implement kill switch. Version control all scripts."
    },
    
    "mcp_server": {
        "name": "MCP Server Development",
        "phases": [
            {
                "name": "Design & Planning",
                "duration_days": 3,
                "objectives": [
                    "Define tools and resources",
                    "Design data model",
                    "Plan error handling strategy",
                    "Document API contracts"
                ],
                "deliverables": ["Tool specifications", "Data model", "API documentation"],
                "decisions": [
                    {"question": "Language?", "options": ["Python (FastMCP)", "TypeScript (MCP SDK)"], "considerations": "Team experience, ecosystem, performance"},
                    {"question": "Data storage?", "options": ["SQLite", "PostgreSQL", "File-based"], "considerations": "Complexity, portability, scale"}
                ],
                "risks": ["Scope creep", "Poor tool design", "Missing use cases"],
                "search_terms": ["MCP server design", "API design patterns"]
            },
            {
                "name": "Core Development",
                "duration_days": 7,
                "objectives": [
                    "Set up project structure",
                    "Implement core tools",
                    "Build data layer",
                    "Add error handling and logging"
                ],
                "deliverables": ["Working MCP server", "Core tools", "Data layer"],
                "decisions": [
                    {"question": "Architecture?", "options": ["Modular (recommended)", "Monolithic"], "considerations": "Maintainability, testability"}
                ],
                "risks": ["Tool reliability issues", "Performance problems", "Security vulnerabilities"],
                "gates": ["Core tools working", "Error handling robust", "Logging active"],
                "search_terms": ["MCP development", "Python async", "error handling"]
            },
            {
                "name": "Advanced Features",
                "duration_days": 5,
                "objectives": [
                    "Add resources for RAG",
                    "Implement caching",
                    "Add semantic search (if applicable)",
                    "Performance optimization"
                ],
                "deliverables": ["Resources", "Caching layer", "Optimized performance"],
                "decisions": [
                    {"question": "Caching strategy?", "options": ["In-memory", "Redis", "File-based"], "considerations": "Persistence needs, complexity"}
                ],
                "risks": ["Cache invalidation issues", "Memory leaks"],
                "gates": ["Resources working", "Performance acceptable"],
                "search_terms": ["MCP resources", "caching strategies", "semantic search"]
            },
            {
                "name": "Testing & Documentation",
                "duration_days": 3,
                "objectives": [
                    "Write unit tests",
                    "Integration testing with Claude",
                    "Create user documentation",
                    "Document configuration"
                ],
                "deliverables": ["Test suite", "User docs", "Config docs"],
                "decisions": [],
                "risks": ["Undiscovered bugs", "Poor documentation"],
                "gates": ["Tests passing", "Docs complete", "Claude integration verified"],
                "search_terms": ["testing MCP servers", "documentation best practices"]
            },
            {
                "name": "Deployment & Distribution",
                "duration_days": 2,
                "objectives": [
                    "Package for distribution",
                    "Create installation instructions",
                    "Set up for Claude Desktop",
                    "Create troubleshooting guide"
                ],
                "deliverables": ["Packaged server", "Installation guide", "Troubleshooting docs"],
                "decisions": [
                    {"question": "Distribution?", "options": ["Local only", "npm/PyPI package", "Docker image"], "considerations": "Audience, ease of use"}
                ],
                "risks": ["Installation issues", "Environment conflicts"],
                "gates": ["Clean install works", "Claude Desktop integration verified"],
                "search_terms": ["Python packaging", "MCP deployment"]
            }
        ],
        "success_criteria": [
            "All tools working correctly",
            "Integration with Claude verified",
            "Documentation complete",
            "Error handling robust",
            "Performance acceptable"
        ],
        "rollback_strategy": "Version control all changes. Keep previous working version. Document breaking changes."
    }
}

# Generic template for unrecognized project types
GENERIC_TEMPLATE = {
    "name": "Technical Project",
    "phases": [
        {
            "name": "Planning & Design",
            "duration_days": 5,
            "objectives": ["Define requirements", "Create architecture", "Plan milestones"],
            "deliverables": ["Requirements doc", "Architecture diagram", "Project plan"],
            "decisions": [],
            "risks": ["Unclear requirements", "Scope creep"],
            "search_terms": ["project planning", "technical architecture"]
        },
        {
            "name": "Development",
            "duration_days": 10,
            "objectives": ["Build core functionality", "Implement features", "Code review"],
            "deliverables": ["Working software", "Code documentation"],
            "decisions": [],
            "risks": ["Technical debt", "Bugs"],
            "search_terms": ["software development", "coding best practices"]
        },
        {
            "name": "Testing",
            "duration_days": 5,
            "objectives": ["Unit testing", "Integration testing", "User testing"],
            "deliverables": ["Test suite", "Bug fixes"],
            "decisions": [],
            "risks": ["Undiscovered bugs", "Poor coverage"],
            "search_terms": ["software testing", "quality assurance"]
        },
        {
            "name": "Deployment",
            "duration_days": 3,
            "objectives": ["Deploy to production", "Monitor", "Document"],
            "deliverables": ["Live system", "Documentation"],
            "decisions": [],
            "risks": ["Deployment issues", "Performance problems"],
            "search_terms": ["deployment", "DevOps"]
        }
    ],
    "success_criteria": ["Project objectives met", "Quality standards achieved", "Documentation complete"],
    "rollback_strategy": "Maintain backups. Document rollback procedures. Version control everything."
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_project_type(goal: str) -> tuple[str, dict]:
    """Detect project type and return appropriate template"""
    goal_lower = goal.lower()
    
    type_keywords = {
        "vps": ["vps", "server", "hosting", "hetzner", "digitalocean", "linode", 
                "virtual private", "self-host", "deploy server", "linux server"],
        "web_app": ["web app", "website", "frontend", "backend", "full stack",
                    "react", "vue", "django", "flask", "fastapi", "api"],
        "data_pipeline": ["data pipeline", "etl", "data processing", "analytics",
                          "data analysis", "data warehouse", "reporting", "pandas"],
        "automation": ["automate", "automation", "script", "bot", "workflow",
                       "scheduled", "cron", "batch"],
        "mcp_server": ["mcp server", "mcp tool", "model context protocol",
                       "claude tool", "ai tool"]
    }
    
    best_match = None
    best_score = 0
    
    for project_type, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in goal_lower)
        if score > best_score:
            best_score = score
            best_match = project_type
    
    if best_match and best_score > 0:
        return best_match, IMPLEMENTATION_TEMPLATES[best_match]
    
    return "generic", GENERIC_TEMPLATE


def _search_for_best_practices(search_terms: list, limit: int = 3) -> list:
    """Search library for best practices using chunk-level search"""
    results = []

    try:
        embeddings_matrix, chunk_metadata = load_chunk_embeddings()
        if embeddings_matrix is None:
            return []

        with embedding_model_context() as generator:
            seen = set()
            for term in search_terms[:3]:
                query_embedding = generator.generate(term)
                top_results = find_top_k(
                    query_embedding, embeddings_matrix,
                    k=limit * 3, min_similarity=0.3
                )

                chunk_results = []
                for idx, similarity in top_results:
                    meta = chunk_metadata[idx]
                    chunk_results.append({**meta, 'similarity': similarity})

                chapter_results = best_chunk_per_chapter(chunk_results)

                for r in chapter_results[:limit]:
                    key = (r['book_id'], r['chapter_number'])
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            'id': r['chapter_id'],
                            'book_id': r['book_id'],
                            'book_title': r['book_title'],
                            'chapter_title': r['chapter_title'],
                            'chapter_number': r['chapter_number'],
                            'file_path': r.get('file_path', ''),
                            'similarity': r['similarity']
                        })

            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit * 2]

    except Exception as e:
        logger.error(f"Best practices search error: {e}", exc_info=True)
        return []


def _calculate_timeline(phases: list, start_date: datetime = None) -> list:
    """Calculate timeline with start/end dates for each phase"""
    if start_date is None:
        start_date = datetime.now()
    
    timeline = []
    current_date = start_date
    
    for phase in phases:
        duration = phase.get('duration_days', 5)
        end_date = current_date + timedelta(days=duration)
        
        timeline.append({
            "phase": phase['name'],
            "start": current_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "duration_days": duration
        })
        
        current_date = end_date + timedelta(days=1)  # 1 day buffer
    
    return timeline


def _generate_prompts_for_phase(phase: dict, project_type: str) -> list:
    """Generate actionable prompts for a phase"""
    prompts = []
    
    # Action prompts from objectives
    for obj in phase.get('objectives', []):
        prompts.append({
            "type": "action",
            "prompt": f"Help me {obj.lower()}",
            "context": phase['name']
        })
    
    # Decision prompts
    for decision in phase.get('decisions', []):
        prompts.append({
            "type": "decision",
            "prompt": f"Compare options for: {decision['question']} Options: {', '.join(decision['options'])}. Consider: {decision['considerations']}",
            "context": phase['name']
        })
    
    # Research prompts from search terms
    for term in phase.get('search_terms', []):
        prompts.append({
            "type": "research",
            "prompt": f"Search my book library for best practices on: {term}",
            "context": phase['name']
        })
    
    # Risk mitigation prompts
    for risk in phase.get('risks', []):
        prompts.append({
            "type": "risk",
            "prompt": f"How do I mitigate this risk: {risk}",
            "context": phase['name']
        })
    
    return prompts


def _build_implementation_markdown(
    goal: str,
    project_type: str,
    template: dict,
    timeline: list,
    best_practices: dict,
    all_prompts: list,
    team_size: int,
    include_prompts: bool
) -> str:
    """Build comprehensive implementation plan markdown"""
    lines = []
    
    # Header
    lines.append(f"# Implementation Plan: {template['name']}")
    lines.append(f"## {goal}")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Executive Summary
    total_days = sum(t['duration_days'] for t in timeline)
    adjusted_days = total_days if team_size == 1 else max(total_days // team_size, total_days // 2)
    
    lines.append("## 📋 Executive Summary")
    lines.append("")
    lines.append(f"**Project:** {goal}")
    lines.append(f"**Type:** {template['name']}")
    lines.append(f"**Phases:** {len(template['phases'])}")
    lines.append(f"**Duration:** {total_days} days (solo) / ~{adjusted_days} days (team of {team_size})")
    lines.append(f"**Start Date:** {timeline[0]['start']}")
    lines.append(f"**Target Completion:** {timeline[-1]['end']}")
    lines.append("")
    
    # Success Criteria
    lines.append("### Success Criteria")
    lines.append("")
    for criteria in template.get('success_criteria', []):
        lines.append(f"- [ ] {criteria}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Timeline Overview
    lines.append("## 📅 Timeline Overview")
    lines.append("")
    lines.append("```")
    for t in timeline:
        bar_length = min(t['duration_days'], 20)
        bar = "█" * bar_length
        lines.append(f"{t['phase'][:25]:<25} {t['start']} → {t['end']} ({t['duration_days']}d) {bar}")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed Phases
    for i, phase in enumerate(template['phases'], 1):
        phase_timeline = timeline[i-1]
        phase_practices = best_practices.get(phase['name'], [])
        
        lines.append(f"## Phase {i}: {phase['name']}")
        lines.append("")
        lines.append(f"**Duration:** {phase_timeline['duration_days']} days")
        lines.append(f"**Timeline:** {phase_timeline['start']} → {phase_timeline['end']}")
        lines.append("")
        
        # Objectives
        lines.append("### 🎯 Objectives")
        lines.append("")
        for obj in phase.get('objectives', []):
            lines.append(f"- [ ] {obj}")
        lines.append("")
        
        # Deliverables
        if phase.get('deliverables'):
            lines.append("### 📦 Deliverables")
            lines.append("")
            for d in phase['deliverables']:
                lines.append(f"- {d}")
            lines.append("")
        
        # Decisions
        if phase.get('decisions'):
            lines.append("### 🤔 Decisions Required")
            lines.append("")
            for decision in phase['decisions']:
                lines.append(f"**{decision['question']}**")
                lines.append(f"- Options: {', '.join(decision['options'])}")
                lines.append(f"- Consider: {decision['considerations']}")
                lines.append("")
        
        # Risks
        if phase.get('risks'):
            lines.append("### ⚠️ Risks")
            lines.append("")
            for risk in phase['risks']:
                lines.append(f"- {risk}")
            lines.append("")
        
        # Gates
        if phase.get('gates'):
            lines.append("### ✅ Phase Gates (Exit Criteria)")
            lines.append("")
            for gate in phase['gates']:
                lines.append(f"- [ ] {gate}")
            lines.append("")
        
        # Best Practices from Library
        if phase_practices:
            lines.append("### 📚 Recommended Reading")
            lines.append("")
            for bp in phase_practices[:3]:
                lines.append(f"- **{bp['book_title']}** — Ch. {bp['chapter_number']}: {bp['chapter_title']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Rollback Strategy
    lines.append("## 🔄 Rollback Strategy")
    lines.append("")
    lines.append(template.get('rollback_strategy', 'Document rollback procedures for each phase.'))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Prompts Section
    if include_prompts and all_prompts:
        lines.append("## 💬 Implementation Prompts")
        lines.append("")
        lines.append("*Copy these prompts to get help with specific tasks:*")
        lines.append("")
        
        # Group by phase
        current_phase = None
        for prompt in all_prompts:
            if prompt['context'] != current_phase:
                current_phase = prompt['context']
                lines.append(f"### {current_phase}")
                lines.append("")
            
            type_emoji = {
                "action": "🔨",
                "decision": "🤔",
                "research": "🔍",
                "risk": "⚠️"
            }.get(prompt['type'], "📝")
            
            lines.append(f"{type_emoji} **{prompt['type'].title()}:** `{prompt['prompt']}`")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("## 📝 Notes")
    lines.append("")
    lines.append("- Adjust timeline based on actual progress")
    lines.append("- Review and update risks weekly")
    lines.append("- Document decisions and rationale")
    lines.append("- Update this plan as requirements change")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated from book library with {len([bp for bps in best_practices.values() for bp in bps])} relevant chapters identified*")
    
    return "\n".join(lines)


# =============================================================================
# MAIN TOOL REGISTRATION
# =============================================================================

def register_project_planning_tools(mcp):
    """Register project planning tools with MCP server"""
    
    @mcp.tool()
    def generate_implementation_plan(
        goal: str,
        team_size: int = 1,
        start_date: str = "",
        include_prompts: bool = True,
        save_to_file: bool = False,
        output_path: str = ""
    ) -> dict:
        """Generate a project-based learning path from your book library
        
        Analyzes your goal, searches your book library for relevant content,
        and creates a phased learning guide with reading recommendations,
        time estimates, and implementation checklists.
        
        Works with both specific and ambiguous goals:
        - "Build a VPS on Hetzner to host my portfolio and apps"
        - "Create a data pipeline for analyzing CSV files"
        - "Learn enough Docker to deploy my Python app"
        - "Build an MCP server for my notes"
        
        Args:
            goal: What you want to build or achieve. Be as specific or vague as you like.
            depth: Level of detail in the guide:
                   - "quick": Essential reading only, minimal phases
                   - "comprehensive": Full learning path with all phases (default)
                   - "deep": Exhaustive coverage with advanced topics
            include_concepts: If True (default), includes concept briefs with business
                   analogies for key topics in each phase using teach_concept logic.
            save_to_file: If True, saves the guide to a markdown file
            output_path: Custom path for saved file (optional)
                   
        Returns:
            Dictionary with:
            - goal: Your stated goal
            - project_type: Detected project category
            - phases: Learning phases with topics
            - time_estimate: Hours for learning and implementation
            - books_found: Number of relevant books
            - chapters_found: Number of relevant chapters
            - reading_list: Top chapters to read
            - concept_briefs: Key concepts with explanations (if include_concepts=True)
            - guide: Full markdown learning guide
            - file_path: Path if saved to file
            
        Examples:
            generate_learning_path("Build a VPS to host my portfolio")
            generate_learning_path("Create a Python data pipeline", depth="quick")
            generate_learning_path("Learn Docker for deployment", include_concepts=True)
        """
        try:
            logger.info(f"Generating implementation plan for: {goal}")
            
            # Parse start date
            if start_date:
                try:
                    parsed_start = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    parsed_start = datetime.now()
            else:
                parsed_start = datetime.now()
            
            # Detect project type
            project_type, template = _detect_project_type(goal)
            logger.info(f"Detected project type: {project_type}")
            
            # Calculate timeline
            timeline = _calculate_timeline(template['phases'], parsed_start)
            
            # Search for best practices per phase
            best_practices = {}
            for phase in template['phases']:
                search_terms = phase.get('search_terms', [])
                if search_terms:
                    practices = _search_for_best_practices(search_terms, limit=3)
                    best_practices[phase['name']] = practices
            
            # Generate prompts for all phases
            all_prompts = []
            if include_prompts:
                for phase in template['phases']:
                    phase_prompts = _generate_prompts_for_phase(phase, project_type)
                    all_prompts.extend(phase_prompts)
            
            # Build markdown
            plan_markdown = _build_implementation_markdown(
                goal, project_type, template, timeline,
                best_practices, all_prompts, team_size, include_prompts
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                if not output_path:
                    safe_name = re.sub(r'[^\w\s-]', '', goal.lower())
                    safe_name = re.sub(r'[\s]+', '-', safe_name)[:50]
                    output_path = f"implementation-plan-{safe_name}.md"
                
                try:
                    with open(output_path, 'w') as f:
                        f.write(plan_markdown)
                    file_path = output_path
                    logger.info(f"Saved implementation plan to: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not save file: {e}")
            
            # Calculate totals
            total_days = sum(t['duration_days'] for t in timeline)
            total_chapters = sum(len(bps) for bps in best_practices.values())
            
            return {
                "goal": goal,
                "project_type": template['name'],
                "detected_type": project_type,
                "timeline": {
                    "total_days": total_days,
                    "start_date": timeline[0]['start'],
                    "end_date": timeline[-1]['end'],
                    "phases": timeline
                },
                "phases": [
                    {
                        "name": p['name'],
                        "duration_days": p.get('duration_days', 5),
                        "objectives": p.get('objectives', []),
                        "deliverables": p.get('deliverables', []),
                        "decisions": len(p.get('decisions', [])),
                        "risks": len(p.get('risks', []))
                    }
                    for p in template['phases']
                ],
                "success_criteria": template.get('success_criteria', []),
                "best_practices_found": total_chapters,
                "prompts_generated": len(all_prompts) if include_prompts else 0,
                "prompts": all_prompts[:20] if include_prompts else [],  # Limit for readability
                "plan": plan_markdown,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"generate_implementation_plan error: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    @mcp.tool()
    def list_implementation_templates() -> dict:
        """List available implementation plan templates
        
        Shows the built-in project templates with their phases, typical
        durations, and what each template covers.
        
        Returns:
            Dictionary with:
            - templates: List of templates with phases and durations
            - usage_tip: How to use with generate_implementation_plan
            
        Examples:
            list_implementation_templates()
        """
        templates = []
        
        for template_id, config in IMPLEMENTATION_TEMPLATES.items():
            total_days = sum(p.get('duration_days', 5) for p in config['phases'])
            templates.append({
                "id": template_id,
                "name": config['name'],
                "phases": [p['name'] for p in config['phases']],
                "total_days": total_days,
                "success_criteria_count": len(config.get('success_criteria', []))
            })
        
        return {
            "templates": templates,
            "usage_tip": "Use generate_implementation_plan('your goal') - the system auto-detects the best template",
            "customization": "Templates are customized based on your goal. Specific keywords trigger specific templates."
        }
    
    
    @mcp.tool()
    def get_phase_prompts(
        goal: str,
        phase_name: str = "",
        prompt_type: str = "all"
    ) -> dict:
        """Get actionable prompts for a specific phase or entire project
        
        Generates prompts you can use to get help with specific tasks,
        make decisions, research topics, or mitigate risks.
        
        Args:
            goal: Your project goal (used to detect project type)
            phase_name: Specific phase to get prompts for (empty = all phases)
            prompt_type: Filter prompts by type:
                        - "all": All prompt types (default)
                        - "action": Task execution prompts
                        - "decision": Decision-making prompts
                        - "research": Research/learning prompts
                        - "risk": Risk mitigation prompts
        
        Returns:
            Dictionary with prompts organized by phase
            
        Examples:
            get_phase_prompts("Build a VPS")
            get_phase_prompts("Build a VPS", phase_name="Security")
            get_phase_prompts("Build a VPS", prompt_type="decision")
        """
        try:
            project_type, template = _detect_project_type(goal)
            
            all_prompts = {}
            
            for phase in template['phases']:
                if phase_name and phase['name'].lower() != phase_name.lower():
                    continue
                
                phase_prompts = _generate_prompts_for_phase(phase, project_type)
                
                if prompt_type != "all":
                    phase_prompts = [p for p in phase_prompts if p['type'] == prompt_type]
                
                if phase_prompts:
                    all_prompts[phase['name']] = phase_prompts
            
            return {
                "goal": goal,
                "project_type": template['name'],
                "filter": {
                    "phase": phase_name or "all",
                    "type": prompt_type
                },
                "prompts": all_prompts,
                "total_prompts": sum(len(p) for p in all_prompts.values())
            }
            
        except Exception as e:
            logger.error(f"get_phase_prompts error: {e}", exc_info=True)
            return {"error": str(e)}

    
    # =========================================================================
    # BRD (BUSINESS REQUIREMENTS DOCUMENT) TOOL
    # =========================================================================
    
    @mcp.tool()
    def generate_brd(
        goal: str,
        template_style: str = "standard",
        business_context: str = "",
        include_technical: bool = True,
        save_to_file: bool = False,
        output_path: str = ""
    ) -> dict:
        """Generate a Business Requirements Document (BRD) for a project
        
        Creates a PM-ready BRD that translates technical projects into
        business language. Includes problem statement, scope, requirements,
        success metrics, risks, and stakeholder analysis.
        
        Args:
            goal: What you want to build or achieve
            template_style: Document format:
                           - "standard": Full BRD with all sections (default)
                           - "lean": Condensed 1-2 page version
                           - "enterprise": Formal with governance sections
            business_context: Additional context (team, budget, constraints)
            include_technical: Include technical requirements section (default: True)
            save_to_file: If True, saves the BRD to a markdown file
            output_path: Custom path for saved file (optional)
                   
        Returns:
            Dictionary with:
            - goal: Your stated goal
            - project_type: Detected project category
            - brd: Full BRD document (markdown)
            - sections: Individual sections for reference
            - file_path: Path if saved to file
            
        Examples:
            generate_brd("Build a VPS to host my portfolio")
            generate_brd("Create a data pipeline", template_style="lean")
            generate_brd("Build internal tool", template_style="enterprise", 
                        business_context="Team of 3, Q2 deadline, $5k budget")
        """
        try:
            logger.info(f"Generating BRD for: {goal}")
            
            # Validate template style
            valid_styles = ["standard", "lean", "enterprise"]
            if template_style not in valid_styles:
                return {
                    "error": f"Invalid template_style '{template_style}'. Use: {', '.join(valid_styles)}",
                    "valid_styles": valid_styles
                }
            
            # Detect project type
            project_type, impl_template = _detect_project_type(goal)
            brd_config = _get_brd_template(project_type)
            
            # Search library for relevant requirements/best practices
            best_practices = _search_for_brd_content(
                brd_config.get('search_terms', [goal]),
                limit=5
            )
            
            # Build BRD sections
            sections = _build_brd_sections(
                goal=goal,
                project_type=project_type,
                brd_config=brd_config,
                impl_template=impl_template,
                business_context=business_context,
                include_technical=include_technical,
                best_practices=best_practices,
                template_style=template_style
            )
            
            # Build full document
            brd_markdown = _build_brd_markdown(
                goal=goal,
                project_type=project_type,
                brd_config=brd_config,
                sections=sections,
                template_style=template_style
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                if not output_path:
                    safe_name = re.sub(r'[^\w\s-]', '', goal.lower())
                    safe_name = re.sub(r'[\s]+', '-', safe_name)[:50]
                    output_path = f"brd-{safe_name}.md"
                
                try:
                    with open(output_path, 'w') as f:
                        f.write(brd_markdown)
                    file_path = output_path
                    logger.info(f"Saved BRD to: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not save file: {e}")
            
            return {
                "goal": goal,
                "project_type": brd_config['name'],
                "detected_type": project_type,
                "template_style": template_style,
                "sections": sections,
                "best_practices_found": len(best_practices),
                "brd": brd_markdown,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"generate_brd error: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    # =========================================================================
    # WIREFRAME / ARCHITECTURE BRIEF TOOL
    # =========================================================================
    
    @mcp.tool()
    def generate_wireframe_brief(
        goal: str,
        audience: str = "stakeholder",
        include_diagram: bool = True,
        custom_components: list = None,
        save_to_file: bool = False,
        output_path: str = ""
    ) -> dict:
        """Generate an architecture wireframe/brief for a project
        
        Creates a visual and textual overview of system architecture suitable
        for different audiences. Includes component diagrams, data flows,
        technology decisions, and integration points.
        
        Args:
            goal: What you want to build or achieve
            audience: Target audience for the brief:
                     - "executive": High-level overview, minimal technical detail
                     - "stakeholder": Balanced view with key technical info (default)
                     - "technical": Full detail with all technology decisions
            include_diagram: Include Mermaid diagram (default: True)
            custom_components: Override default components with custom list
            save_to_file: If True, saves the brief to a markdown file
            output_path: Custom path for saved file (optional)
                   
        Returns:
            Dictionary with:
            - goal: Your stated goal
            - project_type: Detected project category
            - components: List of system components
            - data_flows: How data moves through the system
            - integrations: External system integrations
            - mermaid_diagram: Diagram code (if include_diagram=True)
            - brief: Full architecture brief (markdown)
            - file_path: Path if saved to file
            
        Examples:
            generate_wireframe_brief("Build a VPS on Hetzner")
            generate_wireframe_brief("Build a web app", audience="executive")
            generate_wireframe_brief("Build a data pipeline", audience="technical")
        """
        try:
            logger.info(f"Generating wireframe brief for: {goal}")
            
            # Validate audience
            valid_audiences = ["executive", "stakeholder", "technical"]
            if audience not in valid_audiences:
                return {
                    "error": f"Invalid audience '{audience}'. Use: {', '.join(valid_audiences)}",
                    "valid_audiences": valid_audiences
                }
            
            # Detect project type
            project_type, impl_template = _detect_project_type(goal)
            arch_template = _get_architecture_template(project_type)
            
            # Search library for architecture best practices
            search_terms = ["architecture", "system design"] + list(arch_template.get('components', [{}])[0].get('technologies', []))[:2]
            best_practices = _search_for_architecture_content(search_terms, limit=5)
            
            # Build the brief markdown
            brief_markdown = _build_wireframe_markdown(
                goal=goal,
                project_type=project_type,
                arch_template=arch_template,
                audience=audience,
                best_practices=best_practices,
                custom_components=custom_components or []
            )
            
            # Adjust content for audience
            brief_markdown = _adjust_for_audience(brief_markdown, audience)
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                if not output_path:
                    safe_name = re.sub(r'[^\w\s-]', '', goal.lower())
                    safe_name = re.sub(r'[\s]+', '-', safe_name)[:50]
                    output_path = f"architecture-{safe_name}.md"
                
                try:
                    with open(output_path, 'w') as f:
                        f.write(brief_markdown)
                    file_path = output_path
                    logger.info(f"Saved architecture brief to: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not save file: {e}")
            
            return {
                "goal": goal,
                "project_type": arch_template['name'],
                "detected_type": project_type,
                "audience": audience,
                "components": arch_template['components'],
                "data_flows": arch_template['data_flows'],
                "integrations": arch_template['integrations'],
                "technology_decisions": arch_template.get('technology_decisions', []),
                "mermaid_diagram": arch_template['mermaid_diagram'] if include_diagram else None,
                "best_practices_found": len(best_practices),
                "brief": brief_markdown,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"generate_wireframe_brief error: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    @mcp.tool()
    def list_architecture_templates() -> dict:
        """List available architecture templates
        
        Shows the built-in architecture templates with their components
        and technology stacks.
        
        Returns:
            Dictionary with available templates and their details
            
        Examples:
            list_architecture_templates()
        """
        try:
            templates = []
            for tid, config in ARCHITECTURE_TEMPLATES.items():
                templates.append({
                    "id": tid,
                    "name": config['name'],
                    "description": config['description'],
                    "component_count": len(config['components']),
                    "components": [c['name'] for c in config['components']],
                    "integration_count": len(config.get('integrations', [])),
                    "has_diagram": bool(config.get('mermaid_diagram'))
                })
            
            return {
                "templates": templates,
                "usage_tip": "Use generate_wireframe_brief('your goal') - the system auto-detects the best template",
                "audiences": ["executive", "stakeholder", "technical"]
            }
            
        except Exception as e:
            logger.error(f"list_architecture_templates error: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    # =========================================================================
    # ANALYZE PROJECT - ORCHESTRATOR TOOL
    # =========================================================================
    
    @mcp.tool()
    def analyze_project(
        goal: str,
        mode: str = "overview",
        artifacts: list = None,
        save_artifacts: bool = False,
        output_dir: str = ""
    ) -> dict:
        """Analyze a project goal and optionally generate all planning artifacts
        
        This is the entry-point tool that can:
        1. Analyze your goal and recommend next steps (overview mode)
        2. Generate essential artifacts quickly (quick mode)
        3. Generate comprehensive project documentation (full mode)
        
        Args:
            goal: What you want to build or achieve
            mode: Analysis depth:
                 - "overview": Quick analysis with recommendations (default)
                 - "quick": Generate BRD (lean) + Implementation Plan
                 - "full": Generate all artifacts (learning path, BRD, architecture, plan)
            artifacts: Custom list of artifacts to generate. Options:
                      ["learning_path", "brd", "architecture", "implementation_plan"]
                      If None, uses mode defaults.
            save_artifacts: If True, saves all generated artifacts to files
            output_dir: Directory for saved files (default: current directory)
                   
        Returns:
            Dictionary with:
            - goal: Your stated goal
            - project_type: Detected project category
            - complexity: Estimated complexity (simple/moderate/complex)
            - analysis: Project analysis summary
            - recommendations: Suggested next steps
            - artifacts: Generated artifacts (if mode != "overview")
            - file_paths: Saved file paths (if save_artifacts=True)
            
        Examples:
            analyze_project("Build a VPS on Hetzner")  # Quick overview
            analyze_project("Build a web app", mode="quick")  # Essential docs
            analyze_project("Build data pipeline", mode="full", save_artifacts=True)
        """
        try:
            logger.info(f"Analyzing project: {goal} (mode={mode})")
            
            # Validate mode
            valid_modes = ["overview", "quick", "full"]
            if mode not in valid_modes:
                return {
                    "error": f"Invalid mode '{mode}'. Use: {', '.join(valid_modes)}",
                    "valid_modes": valid_modes
                }
            
            # Detect project type and get templates
            project_type, impl_template = _detect_project_type(goal)
            arch_template = _get_architecture_template(project_type)
            brd_template = _get_brd_template(project_type)
            
            # Analyze complexity
            complexity = _assess_complexity(impl_template, arch_template)
            
            # Build analysis
            analysis = _build_project_analysis(
                goal=goal,
                project_type=project_type,
                impl_template=impl_template,
                arch_template=arch_template,
                brd_template=brd_template,
                complexity=complexity
            )
            
            # Build recommendations
            recommendations = _build_recommendations(
                goal=goal,
                project_type=project_type,
                complexity=complexity,
                mode=mode
            )
            
            result = {
                "goal": goal,
                "project_type": impl_template['name'],
                "detected_type": project_type,
                "complexity": complexity,
                "analysis": analysis,
                "recommendations": recommendations
            }
            
            # Generate artifacts if requested
            if mode != "overview":
                # Determine which artifacts to generate
                if artifacts:
                    artifact_list = artifacts
                elif mode == "quick":
                    artifact_list = ["brd", "implementation_plan"]
                else:  # full
                    artifact_list = ["learning_path", "brd", "architecture", "implementation_plan"]
                
                generated = _generate_artifacts(
                    goal=goal,
                    project_type=project_type,
                    artifact_list=artifact_list,
                    impl_template=impl_template,
                    arch_template=arch_template,
                    brd_template=brd_template,
                    save_artifacts=save_artifacts,
                    output_dir=output_dir
                )
                
                result["artifacts"] = generated["artifacts"]
                result["artifact_summary"] = generated["summary"]
                if save_artifacts:
                    result["file_paths"] = generated.get("file_paths", [])
            
            # Add overview markdown
            result["overview"] = _build_analysis_markdown(result, mode)
            
            return result
            
        except Exception as e:
            logger.error(f"analyze_project error: {e}", exc_info=True)
            return {"error": str(e)}


# =============================================================================
# BRD TEMPLATES BY PROJECT TYPE
# =============================================================================

BRD_TEMPLATES = {
    "vps": {
        "name": "VPS / Server Infrastructure",
        "problem_statement": "Need reliable, secure hosting infrastructure for web applications and services that provides full control, cost efficiency, and scalability.",
        "business_objectives": [
            "Establish reliable hosting for web applications",
            "Reduce dependency on expensive managed services",
            "Gain full control over server configuration and security",
            "Enable rapid deployment of new applications",
            "Ensure high availability and performance"
        ],
        "scope_in": [
            "Server provisioning and initial setup",
            "Security hardening (SSH, firewall, intrusion detection)",
            "Web server configuration (nginx/Apache)",
            "SSL/TLS certificate management",
            "Container deployment infrastructure",
            "Monitoring and alerting setup",
            "Backup and disaster recovery",
            "Documentation and runbooks"
        ],
        "scope_out": [
            "Application development (separate effort)",
            "Content creation",
            "Domain registration (external process)",
            "24/7 on-call support staffing",
            "Multi-region redundancy (future phase)"
        ],
        "stakeholders": [
            {"role": "Project Owner", "responsibility": "Overall project direction and decisions", "involvement": "High"},
            {"role": "Developer(s)", "responsibility": "Application deployment and integration", "involvement": "Medium"},
            {"role": "End Users", "responsibility": "Consume hosted applications", "involvement": "Low (indirect)"}
        ],
        "functional_requirements": [
            {"id": "FR-1", "requirement": "Server shall be accessible via SSH with key-based authentication only", "priority": "Must Have"},
            {"id": "FR-2", "requirement": "Web server shall serve HTTPS traffic with valid SSL certificates", "priority": "Must Have"},
            {"id": "FR-3", "requirement": "Firewall shall block all non-essential incoming ports", "priority": "Must Have"},
            {"id": "FR-4", "requirement": "System shall support containerized application deployment", "priority": "Should Have"},
            {"id": "FR-5", "requirement": "Monitoring shall alert on resource exhaustion or service failures", "priority": "Should Have"},
            {"id": "FR-6", "requirement": "Automated backups shall run daily with 7-day retention", "priority": "Should Have"},
            {"id": "FR-7", "requirement": "System shall support multiple domains/subdomains", "priority": "Could Have"}
        ],
        "non_functional_requirements": [
            {"id": "NFR-1", "requirement": "99.9% uptime target (excluding planned maintenance)", "category": "Availability"},
            {"id": "NFR-2", "requirement": "Page load time < 3 seconds for static content", "category": "Performance"},
            {"id": "NFR-3", "requirement": "Security scan shall show no critical vulnerabilities", "category": "Security"},
            {"id": "NFR-4", "requirement": "System shall handle 1000 concurrent connections", "category": "Scalability"},
            {"id": "NFR-5", "requirement": "Backup restoration time < 4 hours", "category": "Recoverability"}
        ],
        "success_metrics": [
            {"metric": "Uptime", "target": "99.9%", "measurement": "Monitoring tool uptime reports"},
            {"metric": "Security Score", "target": "A+ SSL Labs rating", "measurement": "SSL Labs scan"},
            {"metric": "Response Time", "target": "< 200ms TTFB", "measurement": "Synthetic monitoring"},
            {"metric": "Deployment Time", "target": "< 15 minutes for new app", "measurement": "Time tracking"},
            {"metric": "Cost Efficiency", "target": "< $50/month for base infra", "measurement": "Hosting invoices"}
        ],
        "assumptions": [
            "Stable internet connectivity for management access",
            "Single geographic region is sufficient initially",
            "Traffic levels will be moderate (< 10K daily visitors)",
            "No compliance requirements (HIPAA, PCI, etc.)",
            "Owner has basic Linux command line familiarity"
        ],
        "constraints": [
            "Budget limited to hosting costs + domain fees",
            "Single person managing infrastructure",
            "No dedicated operations team",
            "Must use industry-standard, well-documented tools"
        ],
        "dependencies": [
            {"dependency": "Domain registrar", "type": "External", "risk": "Low"},
            {"dependency": "Hosting provider (Hetzner/DO/etc.)", "type": "External", "risk": "Low"},
            {"dependency": "Let's Encrypt for SSL", "type": "External", "risk": "Low"},
            {"dependency": "Docker Hub for container images", "type": "External", "risk": "Low"}
        ],
        "risks": [
            {"risk": "Security breach due to misconfiguration", "probability": "Medium", "impact": "High", "mitigation": "Follow security hardening checklist, regular audits"},
            {"risk": "Data loss from hardware failure", "probability": "Low", "impact": "High", "mitigation": "Automated backups with offsite storage"},
            {"risk": "Service downtime from updates gone wrong", "probability": "Medium", "impact": "Medium", "mitigation": "Test updates in staging, maintain rollback procedures"},
            {"risk": "Cost overrun from resource usage", "probability": "Low", "impact": "Low", "mitigation": "Set up billing alerts, monitor resource usage"},
            {"risk": "Knowledge gap causing delays", "probability": "Medium", "impact": "Medium", "mitigation": "Follow documentation, use book library resources"}
        ],
        "search_terms": ["server requirements", "VPS security", "infrastructure planning", "web hosting"]
    },
    
    "web_app": {
        "name": "Web Application",
        "problem_statement": "Need a custom web application to solve a specific business problem that cannot be adequately addressed by existing off-the-shelf solutions.",
        "business_objectives": [
            "Deliver functionality that solves the core business problem",
            "Provide intuitive user experience for target audience",
            "Enable data collection and business insights",
            "Support future growth and feature expansion",
            "Maintain security and data privacy"
        ],
        "scope_in": [
            "User interface design and development",
            "Backend API development",
            "Database design and implementation",
            "User authentication and authorization",
            "Core feature implementation",
            "Testing and quality assurance",
            "Deployment pipeline setup",
            "Basic documentation"
        ],
        "scope_out": [
            "Mobile native apps (web-responsive only)",
            "Third-party integrations beyond core requirements",
            "Advanced analytics and reporting (v2)",
            "Multi-language/i18n support (future)",
            "Marketing and user acquisition"
        ],
        "stakeholders": [
            {"role": "Product Owner", "responsibility": "Define requirements, prioritize features", "involvement": "High"},
            {"role": "Developers", "responsibility": "Build and maintain application", "involvement": "High"},
            {"role": "End Users", "responsibility": "Use application, provide feedback", "involvement": "Medium"},
            {"role": "Operations", "responsibility": "Monitor and maintain production", "involvement": "Medium"}
        ],
        "functional_requirements": [
            {"id": "FR-1", "requirement": "Users shall be able to register and authenticate securely", "priority": "Must Have"},
            {"id": "FR-2", "requirement": "System shall provide core business functionality", "priority": "Must Have"},
            {"id": "FR-3", "requirement": "Users shall be able to manage their profile and preferences", "priority": "Should Have"},
            {"id": "FR-4", "requirement": "System shall send notifications for key events", "priority": "Should Have"},
            {"id": "FR-5", "requirement": "Admins shall be able to manage users and content", "priority": "Should Have"},
            {"id": "FR-6", "requirement": "System shall provide data export functionality", "priority": "Could Have"}
        ],
        "non_functional_requirements": [
            {"id": "NFR-1", "requirement": "Page load time < 3 seconds on 3G connection", "category": "Performance"},
            {"id": "NFR-2", "requirement": "Support 500 concurrent users", "category": "Scalability"},
            {"id": "NFR-3", "requirement": "99.5% uptime during business hours", "category": "Availability"},
            {"id": "NFR-4", "requirement": "OWASP Top 10 compliance", "category": "Security"},
            {"id": "NFR-5", "requirement": "WCAG 2.1 AA accessibility compliance", "category": "Accessibility"},
            {"id": "NFR-6", "requirement": "Mobile-responsive design", "category": "Usability"}
        ],
        "success_metrics": [
            {"metric": "User Adoption", "target": "100 active users in first month", "measurement": "Analytics"},
            {"metric": "Task Completion Rate", "target": "> 80%", "measurement": "User analytics"},
            {"metric": "Error Rate", "target": "< 1% of requests", "measurement": "Error monitoring"},
            {"metric": "User Satisfaction", "target": "NPS > 30", "measurement": "User surveys"},
            {"metric": "Time to Market", "target": "MVP in 8 weeks", "measurement": "Project tracking"}
        ],
        "assumptions": [
            "Target users have modern browsers with JavaScript enabled",
            "Users have reliable internet connectivity",
            "Initial user base is English-speaking",
            "MVP features are well-defined and stable",
            "Development resources are available as planned"
        ],
        "constraints": [
            "Fixed timeline for MVP launch",
            "Limited development resources",
            "Must use approved technology stack",
            "Budget constraints for third-party services"
        ],
        "dependencies": [
            {"dependency": "Cloud hosting platform", "type": "External", "risk": "Low"},
            {"dependency": "Email service provider", "type": "External", "risk": "Low"},
            {"dependency": "Payment processor (if applicable)", "type": "External", "risk": "Medium"},
            {"dependency": "Design assets and branding", "type": "Internal", "risk": "Medium"}
        ],
        "risks": [
            {"risk": "Scope creep delays delivery", "probability": "High", "impact": "High", "mitigation": "Strict scope control, clear MVP definition"},
            {"risk": "Security vulnerabilities", "probability": "Medium", "impact": "High", "mitigation": "Security testing, code review, OWASP guidelines"},
            {"risk": "Poor user adoption", "probability": "Medium", "impact": "High", "mitigation": "User research, beta testing, feedback loops"},
            {"risk": "Technical debt accumulation", "probability": "Medium", "impact": "Medium", "mitigation": "Code reviews, refactoring sprints"},
            {"risk": "Key person dependency", "probability": "Medium", "impact": "Medium", "mitigation": "Documentation, knowledge sharing"}
        ],
        "search_terms": ["web development requirements", "application architecture", "API design", "user authentication"]
    },
    
    "data_pipeline": {
        "name": "Data Pipeline / Analytics",
        "problem_statement": "Need automated, reliable data processing to transform raw data into actionable insights, reducing manual effort and enabling data-driven decisions.",
        "business_objectives": [
            "Automate manual data processing tasks",
            "Ensure data quality and consistency",
            "Enable timely access to business insights",
            "Reduce time from data collection to analysis",
            "Create foundation for advanced analytics"
        ],
        "scope_in": [
            "Data source identification and connection",
            "ETL/ELT pipeline development",
            "Data quality validation",
            "Data transformation logic",
            "Storage and warehouse setup",
            "Basic reporting and visualization",
            "Pipeline monitoring and alerting",
            "Documentation of data flows"
        ],
        "scope_out": [
            "Machine learning model development",
            "Real-time streaming (batch only for v1)",
            "Self-service BI tools",
            "Data governance framework",
            "Historical data migration (beyond 1 year)"
        ],
        "stakeholders": [
            {"role": "Data Owner", "responsibility": "Define requirements, validate outputs", "involvement": "High"},
            {"role": "Data Engineer", "responsibility": "Build and maintain pipelines", "involvement": "High"},
            {"role": "Business Analysts", "responsibility": "Consume data, provide feedback", "involvement": "Medium"},
            {"role": "IT/Operations", "responsibility": "Infrastructure support", "involvement": "Low"}
        ],
        "functional_requirements": [
            {"id": "FR-1", "requirement": "Pipeline shall extract data from defined sources on schedule", "priority": "Must Have"},
            {"id": "FR-2", "requirement": "System shall validate data quality against defined rules", "priority": "Must Have"},
            {"id": "FR-3", "requirement": "Transformations shall produce consistent, documented outputs", "priority": "Must Have"},
            {"id": "FR-4", "requirement": "Failed jobs shall generate alerts and support retry", "priority": "Should Have"},
            {"id": "FR-5", "requirement": "System shall maintain data lineage metadata", "priority": "Should Have"},
            {"id": "FR-6", "requirement": "Users shall be able to query processed data via SQL", "priority": "Should Have"}
        ],
        "non_functional_requirements": [
            {"id": "NFR-1", "requirement": "Daily batch processing complete within 4-hour window", "category": "Performance"},
            {"id": "NFR-2", "requirement": "Handle 10GB daily data volume", "category": "Scalability"},
            {"id": "NFR-3", "requirement": "99% pipeline success rate", "category": "Reliability"},
            {"id": "NFR-4", "requirement": "Data freshness < 24 hours", "category": "Timeliness"},
            {"id": "NFR-5", "requirement": "Query response < 30 seconds for standard reports", "category": "Performance"}
        ],
        "success_metrics": [
            {"metric": "Pipeline Reliability", "target": "99% success rate", "measurement": "Job monitoring"},
            {"metric": "Data Freshness", "target": "< 6 hours lag", "measurement": "Timestamp comparison"},
            {"metric": "Processing Time", "target": "Complete by 6 AM daily", "measurement": "Job completion logs"},
            {"metric": "Data Quality Score", "target": "> 95% pass rate", "measurement": "Validation reports"},
            {"metric": "Time Saved", "target": "10+ hours/week vs manual", "measurement": "Before/after comparison"}
        ],
        "assumptions": [
            "Source data formats are stable and documented",
            "Source systems can handle extraction load",
            "Business rules for transformations are defined",
            "Storage costs are within acceptable range",
            "Network connectivity to sources is reliable"
        ],
        "constraints": [
            "Processing must complete before business hours",
            "Cannot impact source system performance",
            "Must use approved data tools/platforms",
            "Limited to structured data sources initially"
        ],
        "dependencies": [
            {"dependency": "Source system access credentials", "type": "Internal", "risk": "Medium"},
            {"dependency": "Data warehouse/storage platform", "type": "External", "risk": "Low"},
            {"dependency": "Orchestration tool (Airflow/etc.)", "type": "External", "risk": "Low"},
            {"dependency": "Business rules documentation", "type": "Internal", "risk": "High"}
        ],
        "risks": [
            {"risk": "Source schema changes break pipeline", "probability": "High", "impact": "High", "mitigation": "Schema validation, change notifications"},
            {"risk": "Data quality issues in source", "probability": "High", "impact": "Medium", "mitigation": "Robust validation, data profiling"},
            {"risk": "Pipeline failures during critical periods", "probability": "Medium", "impact": "High", "mitigation": "Monitoring, retry logic, manual fallback"},
            {"risk": "Storage costs exceed budget", "probability": "Medium", "impact": "Medium", "mitigation": "Data retention policies, cost monitoring"},
            {"risk": "Undocumented business logic", "probability": "Medium", "impact": "Medium", "mitigation": "Stakeholder interviews, logic documentation"}
        ],
        "search_terms": ["ETL requirements", "data pipeline design", "data quality", "data warehouse"]
    },
    
    "automation": {
        "name": "Automation / Scripting",
        "problem_statement": "Repetitive manual tasks are consuming valuable time and introducing human error. Automation will improve efficiency, accuracy, and free up resources for higher-value work.",
        "business_objectives": [
            "Reduce time spent on repetitive tasks",
            "Eliminate human error in routine processes",
            "Improve consistency and reliability",
            "Enable scaling without proportional effort increase",
            "Free up resources for strategic work"
        ],
        "scope_in": [
            "Process analysis and documentation",
            "Script development for identified tasks",
            "Error handling and logging",
            "Scheduling and triggering setup",
            "Testing and validation",
            "Basic monitoring",
            "User documentation"
        ],
        "scope_out": [
            "Full workflow management system",
            "User interface development",
            "Integration with systems not in scope",
            "Business process re-engineering",
            "24/7 support and monitoring"
        ],
        "stakeholders": [
            {"role": "Process Owner", "responsibility": "Define requirements, validate automation", "involvement": "High"},
            {"role": "Developer", "responsibility": "Build and maintain scripts", "involvement": "High"},
            {"role": "End Users", "responsibility": "Benefit from automation, report issues", "involvement": "Low"}
        ],
        "functional_requirements": [
            {"id": "FR-1", "requirement": "Automation shall perform identified tasks without manual intervention", "priority": "Must Have"},
            {"id": "FR-2", "requirement": "System shall log all actions for audit trail", "priority": "Must Have"},
            {"id": "FR-3", "requirement": "Failed executions shall generate alerts", "priority": "Should Have"},
            {"id": "FR-4", "requirement": "System shall support scheduled and on-demand execution", "priority": "Should Have"},
            {"id": "FR-5", "requirement": "Configuration shall be externalised (not hardcoded)", "priority": "Should Have"}
        ],
        "non_functional_requirements": [
            {"id": "NFR-1", "requirement": "Automation shall complete within acceptable time window", "category": "Performance"},
            {"id": "NFR-2", "requirement": "99% execution success rate", "category": "Reliability"},
            {"id": "NFR-3", "requirement": "Scripts shall be maintainable by team members", "category": "Maintainability"},
            {"id": "NFR-4", "requirement": "Credentials shall be securely stored", "category": "Security"}
        ],
        "success_metrics": [
            {"metric": "Time Saved", "target": "X hours/week", "measurement": "Before/after time tracking"},
            {"metric": "Error Reduction", "target": "90% fewer errors", "measurement": "Error log comparison"},
            {"metric": "Execution Success Rate", "target": "> 99%", "measurement": "Job monitoring"},
            {"metric": "ROI", "target": "Break even within 3 months", "measurement": "Time savings vs development cost"}
        ],
        "assumptions": [
            "Current process is well-understood and documented",
            "Required system access is available",
            "Process is stable (not changing frequently)",
            "Edge cases are identified and documented"
        ],
        "constraints": [
            "Limited development time available",
            "Must work within existing infrastructure",
            "Cannot require additional software licenses",
            "Must be maintainable by non-experts"
        ],
        "dependencies": [
            {"dependency": "Access to source systems", "type": "Internal", "risk": "Medium"},
            {"dependency": "Scheduling infrastructure (cron/etc.)", "type": "Internal", "risk": "Low"},
            {"dependency": "Process documentation", "type": "Internal", "risk": "Medium"}
        ],
        "risks": [
            {"risk": "Process changes break automation", "probability": "Medium", "impact": "High", "mitigation": "Change notification process, flexible design"},
            {"risk": "Silent failures go unnoticed", "probability": "Medium", "impact": "High", "mitigation": "Comprehensive logging and alerting"},
            {"risk": "Automation creates new bottlenecks", "probability": "Low", "impact": "Medium", "mitigation": "Performance testing, monitoring"},
            {"risk": "Key person dependency for maintenance", "probability": "Medium", "impact": "Medium", "mitigation": "Documentation, code comments, knowledge sharing"}
        ],
        "search_terms": ["automation requirements", "scripting best practices", "process automation"]
    },
    
    "mcp_server": {
        "name": "MCP Server Development",
        "problem_statement": "Need to extend AI assistant capabilities with custom tools that integrate with specific data sources, APIs, or workflows not available out-of-the-box.",
        "business_objectives": [
            "Extend AI assistant with custom functionality",
            "Enable natural language access to internal tools/data",
            "Improve productivity through AI-powered workflows",
            "Create reusable integrations for team use",
            "Demonstrate AI integration capabilities"
        ],
        "scope_in": [
            "MCP server design and architecture",
            "Core tool implementation",
            "Data layer and storage integration",
            "Error handling and logging",
            "Testing with Claude Desktop",
            "Documentation and usage guide",
            "Deployment configuration"
        ],
        "scope_out": [
            "Custom AI model training",
            "Web interface for MCP server",
            "Multi-user authentication",
            "High-availability deployment",
            "Third-party distribution"
        ],
        "stakeholders": [
            {"role": "Developer", "responsibility": "Build and maintain MCP server", "involvement": "High"},
            {"role": "End Users", "responsibility": "Use tools via Claude", "involvement": "Medium"},
            {"role": "AI/ML Team (if applicable)", "responsibility": "Guidance on integration", "involvement": "Low"}
        ],
        "functional_requirements": [
            {"id": "FR-1", "requirement": "MCP server shall register and expose defined tools", "priority": "Must Have"},
            {"id": "FR-2", "requirement": "Tools shall return structured responses to Claude", "priority": "Must Have"},
            {"id": "FR-3", "requirement": "Server shall handle errors gracefully without crashing", "priority": "Must Have"},
            {"id": "FR-4", "requirement": "Resources shall provide contextual data for RAG", "priority": "Should Have"},
            {"id": "FR-5", "requirement": "Tools shall validate input parameters", "priority": "Should Have"},
            {"id": "FR-6", "requirement": "Server shall support configuration via environment variables", "priority": "Should Have"}
        ],
        "non_functional_requirements": [
            {"id": "NFR-1", "requirement": "Tool response time < 5 seconds for typical operations", "category": "Performance"},
            {"id": "NFR-2", "requirement": "Server shall start within 10 seconds", "category": "Performance"},
            {"id": "NFR-3", "requirement": "Modular architecture for easy extension", "category": "Maintainability"},
            {"id": "NFR-4", "requirement": "Comprehensive logging for debugging", "category": "Operability"}
        ],
        "success_metrics": [
            {"metric": "Tool Reliability", "target": "99% success rate", "measurement": "Error logs"},
            {"metric": "Response Time", "target": "< 3 seconds average", "measurement": "Performance logs"},
            {"metric": "User Adoption", "target": "Daily active use", "measurement": "Usage tracking"},
            {"metric": "Error Rate", "target": "< 5% of calls", "measurement": "Error monitoring"}
        ],
        "assumptions": [
            "Claude Desktop is available for testing",
            "Required APIs/data sources are accessible",
            "Python or TypeScript expertise is available",
            "MCP protocol is stable and documented"
        ],
        "constraints": [
            "Must follow MCP protocol specification",
            "Limited to local or approved cloud deployment",
            "Must not expose sensitive data inappropriately"
        ],
        "dependencies": [
            {"dependency": "MCP SDK (Python/TypeScript)", "type": "External", "risk": "Low"},
            {"dependency": "Claude Desktop for testing", "type": "External", "risk": "Low"},
            {"dependency": "Data sources/APIs to integrate", "type": "Internal", "risk": "Medium"}
        ],
        "risks": [
            {"risk": "MCP protocol changes break server", "probability": "Low", "impact": "High", "mitigation": "Pin SDK versions, monitor changelog"},
            {"risk": "Poor tool design limits usefulness", "probability": "Medium", "impact": "Medium", "mitigation": "User feedback, iterative design"},
            {"risk": "Performance issues with large data", "probability": "Medium", "impact": "Medium", "mitigation": "Caching, pagination, optimization"},
            {"risk": "Security vulnerabilities in integrations", "probability": "Low", "impact": "High", "mitigation": "Input validation, secure credential storage"}
        ],
        "search_terms": ["MCP development", "API design", "tool development", "Claude integration"]
    }
}

# Generic BRD template
GENERIC_BRD_TEMPLATE = {
    "name": "Technical Project",
    "problem_statement": "A technical solution is needed to address a specific business need or opportunity.",
    "business_objectives": [
        "Solve the identified problem effectively",
        "Deliver value within acceptable timeframe and budget",
        "Create maintainable and scalable solution",
        "Enable future enhancements"
    ],
    "scope_in": ["Core functionality development", "Testing and validation", "Documentation"],
    "scope_out": ["Future enhancements", "Unrelated integrations"],
    "stakeholders": [
        {"role": "Project Owner", "responsibility": "Direction and decisions", "involvement": "High"},
        {"role": "Developer", "responsibility": "Implementation", "involvement": "High"}
    ],
    "functional_requirements": [
        {"id": "FR-1", "requirement": "System shall deliver core functionality", "priority": "Must Have"}
    ],
    "non_functional_requirements": [
        {"id": "NFR-1", "requirement": "System shall be reliable and performant", "category": "General"}
    ],
    "success_metrics": [
        {"metric": "Functionality", "target": "All requirements met", "measurement": "Testing"}
    ],
    "assumptions": ["Requirements are understood", "Resources are available"],
    "constraints": ["Timeline", "Budget", "Resources"],
    "dependencies": [],
    "risks": [
        {"risk": "Requirements change", "probability": "Medium", "impact": "Medium", "mitigation": "Change control process"}
    ],
    "search_terms": ["project requirements", "software development"]
}


def _get_brd_template(project_type: str) -> dict:
    """Get BRD template for project type"""
    return BRD_TEMPLATES.get(project_type, GENERIC_BRD_TEMPLATE)


def _search_for_brd_content(search_terms: list, limit: int = 5) -> list:
    """Search library for content relevant to BRD sections"""
    # Reuse the existing search function
    return _search_for_best_practices(search_terms, limit)


def _build_brd_sections(
    goal: str,
    project_type: str,
    brd_config: dict,
    impl_template: dict,
    business_context: str,
    include_technical: bool,
    best_practices: list,
    template_style: str
) -> dict:
    """Build individual BRD sections"""
    
    sections = {
        "executive_summary": _build_executive_summary(goal, brd_config, business_context),
        "problem_statement": brd_config.get('problem_statement', ''),
        "business_objectives": brd_config.get('business_objectives', []),
        "scope": {
            "in_scope": brd_config.get('scope_in', []),
            "out_of_scope": brd_config.get('scope_out', [])
        },
        "stakeholders": brd_config.get('stakeholders', []),
        "requirements": {
            "functional": brd_config.get('functional_requirements', []),
            "non_functional": brd_config.get('non_functional_requirements', []) if include_technical else []
        },
        "success_metrics": brd_config.get('success_metrics', []),
        "assumptions": brd_config.get('assumptions', []),
        "constraints": brd_config.get('constraints', []),
        "dependencies": brd_config.get('dependencies', []),
        "risks": brd_config.get('risks', []),
        "timeline_summary": _build_timeline_summary(impl_template),
        "best_practices": best_practices
    }
    
    # Add enterprise sections if needed
    if template_style == "enterprise":
        sections["governance"] = {
            "approval_authority": "Project Sponsor / Steering Committee",
            "change_control": "All scope changes require formal change request",
            "review_schedule": "Weekly status, Monthly steering committee",
            "escalation_path": "PM → Project Sponsor → Steering Committee"
        }
        sections["compliance"] = {
            "regulatory": "N/A (update if applicable)",
            "security": "Standard security review required",
            "data_privacy": "Data handling per company policy"
        }
    
    return sections


def _build_executive_summary(goal: str, brd_config: dict, business_context: str) -> str:
    """Build executive summary paragraph"""
    summary = f"This document outlines the business requirements for: {goal}. "
    summary += f"{brd_config.get('problem_statement', '')} "
    
    if business_context:
        summary += f"Additional context: {business_context}. "
    
    objectives = brd_config.get('business_objectives', [])[:3]
    if objectives:
        summary += f"Key objectives include: {', '.join(objectives[:3]).lower()}."
    
    return summary


def _build_timeline_summary(impl_template: dict) -> dict:
    """Extract timeline summary from implementation template"""
    phases = impl_template.get('phases', [])
    total_days = sum(p.get('duration_days', 5) for p in phases)
    
    return {
        "total_days": total_days,
        "total_weeks": round(total_days / 5),
        "phases": [
            {"name": p['name'], "duration_days": p.get('duration_days', 5)}
            for p in phases
        ]
    }


def _build_brd_markdown(
    goal: str,
    project_type: str,
    brd_config: dict,
    sections: dict,
    template_style: str
) -> str:
    """Build full BRD markdown document"""
    lines = []
    
    # Header
    lines.append(f"# Business Requirements Document")
    lines.append(f"## {goal}")
    lines.append("")
    lines.append(f"**Document Type:** BRD ({template_style.title()} Template)")
    lines.append(f"**Project Type:** {brd_config['name']}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Status:** Draft")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Table of Contents (for standard and enterprise)
    if template_style in ["standard", "enterprise"]:
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Executive Summary](#executive-summary)")
        lines.append("2. [Problem Statement](#problem-statement)")
        lines.append("3. [Business Objectives](#business-objectives)")
        lines.append("4. [Scope](#scope)")
        lines.append("5. [Stakeholders](#stakeholders)")
        lines.append("6. [Requirements](#requirements)")
        lines.append("7. [Success Metrics](#success-metrics)")
        lines.append("8. [Assumptions & Constraints](#assumptions--constraints)")
        lines.append("9. [Dependencies](#dependencies)")
        lines.append("10. [Risks & Mitigations](#risks--mitigations)")
        lines.append("11. [Timeline](#timeline)")
        if template_style == "enterprise":
            lines.append("12. [Governance](#governance)")
            lines.append("13. [Compliance](#compliance)")
        lines.append("14. [Appendix: Reference Materials](#appendix-reference-materials)")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(sections['executive_summary'])
    lines.append("")
    
    if template_style == "lean":
        # Lean version - condensed format
        lines.append("---")
        lines.append("")
        lines.append("## Key Points")
        lines.append("")
        lines.append("### Objectives")
        for obj in sections['business_objectives'][:3]:
            lines.append(f"- {obj}")
        lines.append("")
        lines.append("### Scope")
        lines.append("**In:** " + ", ".join(sections['scope']['in_scope'][:4]))
        lines.append("")
        lines.append("**Out:** " + ", ".join(sections['scope']['out_of_scope'][:3]))
        lines.append("")
        lines.append("### Must-Have Requirements")
        must_haves = [r for r in sections['requirements']['functional'] if r.get('priority') == 'Must Have']
        for req in must_haves[:5]:
            lines.append(f"- {req['requirement']}")
        lines.append("")
        lines.append("### Key Risks")
        high_risks = [r for r in sections['risks'] if r.get('impact') == 'High'][:3]
        for risk in high_risks:
            lines.append(f"- **{risk['risk']}** — {risk['mitigation']}")
        lines.append("")
        lines.append("### Timeline")
        timeline = sections['timeline_summary']
        lines.append(f"**Estimated Duration:** {timeline['total_days']} days (~{timeline['total_weeks']} weeks)")
        lines.append("")
        
    else:
        # Standard and Enterprise - full format
        lines.append("---")
        lines.append("")
        
        # Problem Statement
        lines.append("## 2. Problem Statement")
        lines.append("")
        lines.append(sections['problem_statement'])
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Business Objectives
        lines.append("## 3. Business Objectives")
        lines.append("")
        for i, obj in enumerate(sections['business_objectives'], 1):
            lines.append(f"{i}. {obj}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Scope
        lines.append("## 4. Scope")
        lines.append("")
        lines.append("### In Scope")
        lines.append("")
        for item in sections['scope']['in_scope']:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Out of Scope")
        lines.append("")
        for item in sections['scope']['out_of_scope']:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Stakeholders
        lines.append("## 5. Stakeholders")
        lines.append("")
        lines.append("| Role | Responsibility | Involvement |")
        lines.append("|------|----------------|-------------|")
        for s in sections['stakeholders']:
            lines.append(f"| {s['role']} | {s['responsibility']} | {s['involvement']} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Requirements
        lines.append("## 6. Requirements")
        lines.append("")
        lines.append("### 6.1 Functional Requirements")
        lines.append("")
        lines.append("| ID | Requirement | Priority |")
        lines.append("|----|-------------|----------|")
        for req in sections['requirements']['functional']:
            lines.append(f"| {req['id']} | {req['requirement']} | {req['priority']} |")
        lines.append("")
        
        if sections['requirements']['non_functional']:
            lines.append("### 6.2 Non-Functional Requirements")
            lines.append("")
            lines.append("| ID | Requirement | Category |")
            lines.append("|----|-------------|----------|")
            for req in sections['requirements']['non_functional']:
                lines.append(f"| {req['id']} | {req['requirement']} | {req['category']} |")
            lines.append("")
        lines.append("---")
        lines.append("")
        
        # Success Metrics
        lines.append("## 7. Success Metrics")
        lines.append("")
        lines.append("| Metric | Target | Measurement Method |")
        lines.append("|--------|--------|-------------------|")
        for m in sections['success_metrics']:
            lines.append(f"| {m['metric']} | {m['target']} | {m['measurement']} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Assumptions & Constraints
        lines.append("## 8. Assumptions & Constraints")
        lines.append("")
        lines.append("### Assumptions")
        lines.append("")
        for a in sections['assumptions']:
            lines.append(f"- {a}")
        lines.append("")
        lines.append("### Constraints")
        lines.append("")
        for c in sections['constraints']:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Dependencies
        lines.append("## 9. Dependencies")
        lines.append("")
        if sections['dependencies']:
            lines.append("| Dependency | Type | Risk Level |")
            lines.append("|------------|------|------------|")
            for d in sections['dependencies']:
                lines.append(f"| {d['dependency']} | {d['type']} | {d['risk']} |")
        else:
            lines.append("*No significant dependencies identified.*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Risks
        lines.append("## 10. Risks & Mitigations")
        lines.append("")
        lines.append("| Risk | Probability | Impact | Mitigation |")
        lines.append("|------|-------------|--------|------------|")
        for r in sections['risks']:
            lines.append(f"| {r['risk']} | {r['probability']} | {r['impact']} | {r['mitigation']} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Timeline
        lines.append("## 11. Timeline")
        lines.append("")
        timeline = sections['timeline_summary']
        lines.append(f"**Estimated Total Duration:** {timeline['total_days']} days (~{timeline['total_weeks']} weeks)")
        lines.append("")
        lines.append("### Phase Breakdown")
        lines.append("")
        lines.append("| Phase | Duration |")
        lines.append("|-------|----------|")
        for p in timeline['phases']:
            lines.append(f"| {p['name']} | {p['duration_days']} days |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Enterprise-only sections
        if template_style == "enterprise":
            lines.append("## 12. Governance")
            lines.append("")
            gov = sections.get('governance', {})
            lines.append(f"**Approval Authority:** {gov.get('approval_authority', 'TBD')}")
            lines.append("")
            lines.append(f"**Change Control:** {gov.get('change_control', 'TBD')}")
            lines.append("")
            lines.append(f"**Review Schedule:** {gov.get('review_schedule', 'TBD')}")
            lines.append("")
            lines.append(f"**Escalation Path:** {gov.get('escalation_path', 'TBD')}")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            lines.append("## 13. Compliance")
            lines.append("")
            comp = sections.get('compliance', {})
            lines.append(f"**Regulatory:** {comp.get('regulatory', 'N/A')}")
            lines.append("")
            lines.append(f"**Security:** {comp.get('security', 'Standard review')}")
            lines.append("")
            lines.append(f"**Data Privacy:** {comp.get('data_privacy', 'Per company policy')}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Appendix
        appendix_num = 14 if template_style == "enterprise" else 12
        lines.append(f"## {appendix_num}. Appendix: Reference Materials")
        lines.append("")
        if sections.get('best_practices'):
            lines.append("### Recommended Reading from Library")
            lines.append("")
            for bp in sections['best_practices'][:5]:
                lines.append(f"- **{bp['book_title']}** — Ch. {bp['chapter_number']}: {bp['chapter_title']}")
            lines.append("")
        lines.append("### Related Documents")
        lines.append("")
        lines.append("- Implementation Plan (generate with `generate_implementation_plan()`)")
        lines.append("- Technical Architecture (if applicable)")
        lines.append("- User Stories / Acceptance Criteria (if applicable)")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## Document Control")
    lines.append("")
    lines.append("| Version | Date | Author | Changes |")
    lines.append("|---------|------|--------|---------|")
    lines.append(f"| 1.0 | {datetime.now().strftime('%Y-%m-%d')} | Generated | Initial draft |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This BRD was generated using the book-library MCP server. Review and customize before formal use.*")
    
    return "\n".join(lines)



# =============================================================================
# WIREFRAME / ARCHITECTURE BRIEF TEMPLATES
# =============================================================================

ARCHITECTURE_TEMPLATES = {
    "vps": {
        "name": "VPS / Server Infrastructure",
        "description": "Self-managed virtual private server for hosting web applications and services",
        "components": [
            {
                "name": "VPS Instance",
                "type": "Infrastructure",
                "description": "Virtual private server from cloud provider (Hetzner, DigitalOcean, etc.)",
                "responsibilities": ["Host all services", "Provide compute resources", "Network connectivity"],
                "technologies": ["Ubuntu/Debian Linux", "KVM virtualization"]
            },
            {
                "name": "Reverse Proxy",
                "type": "Network",
                "description": "Routes incoming traffic to appropriate backend services",
                "responsibilities": ["SSL termination", "Load distribution", "Request routing", "Security headers"],
                "technologies": ["nginx", "Caddy", "Traefik"]
            },
            {
                "name": "Container Runtime",
                "type": "Platform",
                "description": "Runs containerized applications in isolated environments",
                "responsibilities": ["Container orchestration", "Resource isolation", "Network management"],
                "technologies": ["Docker", "Docker Compose", "Podman"]
            },
            {
                "name": "Application Containers",
                "type": "Application",
                "description": "Containerized web applications and services",
                "responsibilities": ["Business logic", "API endpoints", "User interfaces"],
                "technologies": ["Application-specific (Python, Node.js, etc.)"]
            },
            {
                "name": "Database",
                "type": "Data",
                "description": "Persistent data storage for applications",
                "responsibilities": ["Data persistence", "Query processing", "Backup source"],
                "technologies": ["PostgreSQL", "MySQL", "SQLite", "Redis"]
            },
            {
                "name": "Monitoring Stack",
                "type": "Operations",
                "description": "System and application monitoring with alerting",
                "responsibilities": ["Health checks", "Metrics collection", "Alerting"],
                "technologies": ["Uptime Kuma", "Prometheus", "Grafana", "Netdata"]
            },
            {
                "name": "Backup System",
                "type": "Operations",
                "description": "Automated backup and disaster recovery",
                "responsibilities": ["Scheduled backups", "Offsite replication", "Restore capability"],
                "technologies": ["restic", "borgbackup", "rclone"]
            }
        ],
        "data_flows": [
            {"from": "Internet", "to": "Reverse Proxy", "data": "HTTPS requests", "protocol": "HTTPS/443"},
            {"from": "Reverse Proxy", "to": "Application Containers", "data": "Proxied requests", "protocol": "HTTP"},
            {"from": "Application Containers", "to": "Database", "data": "Queries/Data", "protocol": "TCP"},
            {"from": "Monitoring Stack", "to": "All Components", "data": "Health checks", "protocol": "HTTP/TCP"},
            {"from": "Backup System", "to": "Database", "data": "Backup data", "protocol": "Local/S3"}
        ],
        "integrations": [
            {"name": "DNS Provider", "purpose": "Domain resolution", "type": "External"},
            {"name": "Let's Encrypt", "purpose": "SSL certificates", "type": "External"},
            {"name": "Cloud Storage (S3/B2)", "purpose": "Offsite backups", "type": "External"},
            {"name": "Email/Slack", "purpose": "Alert notifications", "type": "External"}
        ],
        "mermaid_diagram": """graph TB
    subgraph Internet
        Users[Users/Clients]
    end
    
    subgraph VPS["VPS Instance"]
        subgraph Network["Network Layer"]
            FW[Firewall/UFW]
            RP[Reverse Proxy<br/>nginx]
        end
        
        subgraph Platform["Container Platform"]
            Docker[Docker Engine]
            subgraph Apps["Applications"]
                App1[Web App 1]
                App2[Web App 2]
                App3[API Service]
            end
        end
        
        subgraph Data["Data Layer"]
            DB[(Database<br/>PostgreSQL)]
            Cache[(Cache<br/>Redis)]
        end
        
        subgraph Ops["Operations"]
            Mon[Monitoring<br/>Uptime Kuma]
            Backup[Backup<br/>restic]
        end
    end
    
    subgraph External["External Services"]
        DNS[DNS Provider]
        SSL[Let's Encrypt]
        S3[Cloud Storage]
    end
    
    Users -->|HTTPS| FW
    FW --> RP
    RP --> App1
    RP --> App2
    RP --> App3
    App1 --> DB
    App2 --> DB
    App3 --> Cache
    Mon -.->|monitors| Apps
    Backup -->|backs up| DB
    Backup -->|stores| S3
    SSL -.->|certs| RP
    DNS -.->|resolves| FW""",
        "technology_decisions": [
            {
                "decision": "Linux Distribution",
                "chosen": "Ubuntu 22.04 LTS",
                "rationale": "Long-term support, extensive documentation, wide community",
                "alternatives": ["Debian (more stable)", "Rocky Linux (RHEL-compatible)"]
            },
            {
                "decision": "Web Server / Reverse Proxy",
                "chosen": "nginx",
                "rationale": "Industry standard, excellent performance, well-documented",
                "alternatives": ["Caddy (easier SSL)", "Traefik (container-native)"]
            },
            {
                "decision": "Container Platform",
                "chosen": "Docker + Docker Compose",
                "rationale": "Simple for single-server, good tooling, standard approach",
                "alternatives": ["Podman (rootless)", "Kubernetes (overkill for single server)"]
            },
            {
                "decision": "Database",
                "chosen": "PostgreSQL",
                "rationale": "Robust, feature-rich, excellent for most applications",
                "alternatives": ["MySQL (more common)", "SQLite (simpler, single-file)"]
            }
        ]
    },
    
    "web_app": {
        "name": "Web Application",
        "description": "Full-stack web application with frontend, backend API, and database",
        "components": [
            {
                "name": "Frontend",
                "type": "Presentation",
                "description": "User interface layer handling user interactions",
                "responsibilities": ["Render UI", "Handle user input", "API communication", "State management"],
                "technologies": ["React", "Vue", "Svelte", "HTMX"]
            },
            {
                "name": "API Gateway / Backend",
                "type": "Application",
                "description": "Server-side application handling business logic",
                "responsibilities": ["Request validation", "Business logic", "Data access", "Authentication"],
                "technologies": ["FastAPI", "Django", "Express", "Flask"]
            },
            {
                "name": "Authentication Service",
                "type": "Security",
                "description": "Handles user identity and access control",
                "responsibilities": ["User registration", "Login/logout", "Token management", "Authorization"],
                "technologies": ["JWT", "OAuth2", "Session-based"]
            },
            {
                "name": "Database",
                "type": "Data",
                "description": "Persistent storage for application data",
                "responsibilities": ["Data persistence", "Query optimization", "Data integrity"],
                "technologies": ["PostgreSQL", "MySQL", "MongoDB"]
            },
            {
                "name": "Cache Layer",
                "type": "Performance",
                "description": "In-memory caching for frequently accessed data",
                "responsibilities": ["Response caching", "Session storage", "Rate limiting"],
                "technologies": ["Redis", "Memcached"]
            },
            {
                "name": "File Storage",
                "type": "Data",
                "description": "Storage for user uploads and static assets",
                "responsibilities": ["File uploads", "Static assets", "CDN integration"],
                "technologies": ["S3", "MinIO", "Local filesystem"]
            }
        ],
        "data_flows": [
            {"from": "User Browser", "to": "Frontend", "data": "User interactions", "protocol": "HTTPS"},
            {"from": "Frontend", "to": "API Gateway", "data": "API requests", "protocol": "REST/GraphQL"},
            {"from": "API Gateway", "to": "Auth Service", "data": "Auth tokens", "protocol": "Internal"},
            {"from": "API Gateway", "to": "Database", "data": "Queries", "protocol": "TCP"},
            {"from": "API Gateway", "to": "Cache", "data": "Cached data", "protocol": "TCP"},
            {"from": "API Gateway", "to": "File Storage", "data": "Files", "protocol": "S3/HTTP"}
        ],
        "integrations": [
            {"name": "Email Service", "purpose": "Transactional emails", "type": "External"},
            {"name": "Payment Processor", "purpose": "Payment handling", "type": "External"},
            {"name": "Analytics", "purpose": "Usage tracking", "type": "External"},
            {"name": "CDN", "purpose": "Static asset delivery", "type": "External"}
        ],
        "mermaid_diagram": """graph TB
    subgraph Client["Client Layer"]
        Browser[Web Browser]
        Mobile[Mobile App]
    end
    
    subgraph Frontend["Frontend Layer"]
        UI[React/Vue App]
        Static[Static Assets]
    end
    
    subgraph Backend["Backend Layer"]
        API[API Server<br/>FastAPI/Django]
        Auth[Auth Service<br/>JWT/OAuth]
        Workers[Background Workers<br/>Celery]
    end
    
    subgraph Data["Data Layer"]
        DB[(Primary DB<br/>PostgreSQL)]
        Cache[(Cache<br/>Redis)]
        Files[File Storage<br/>S3]
    end
    
    subgraph External["External Services"]
        Email[Email Service]
        Payment[Payment Gateway]
        Analytics[Analytics]
    end
    
    Browser --> UI
    Mobile --> API
    UI --> API
    UI --> Static
    API --> Auth
    API --> DB
    API --> Cache
    API --> Files
    API --> Workers
    Workers --> DB
    Workers --> Email
    API --> Payment
    UI --> Analytics""",
        "technology_decisions": [
            {
                "decision": "Frontend Framework",
                "chosen": "React",
                "rationale": "Large ecosystem, good tooling, team familiarity",
                "alternatives": ["Vue (simpler)", "Svelte (faster)", "HTMX (minimal JS)"]
            },
            {
                "decision": "Backend Framework",
                "chosen": "FastAPI",
                "rationale": "Modern Python, automatic docs, async support, type hints",
                "alternatives": ["Django (batteries included)", "Flask (minimal)", "Express (Node.js)"]
            },
            {
                "decision": "Database",
                "chosen": "PostgreSQL",
                "rationale": "Reliable, feature-rich, excellent Python support",
                "alternatives": ["MySQL", "MongoDB (document store)"]
            },
            {
                "decision": "Authentication",
                "chosen": "JWT with refresh tokens",
                "rationale": "Stateless, scalable, mobile-friendly",
                "alternatives": ["Session-based (simpler)", "OAuth2 (third-party)"]
            }
        ]
    },
    
    "data_pipeline": {
        "name": "Data Pipeline / Analytics",
        "description": "ETL/ELT pipeline for processing and analyzing data",
        "components": [
            {
                "name": "Data Sources",
                "type": "Input",
                "description": "Origin systems providing raw data",
                "responsibilities": ["Provide source data", "API access", "Change notifications"],
                "technologies": ["APIs", "Databases", "Files", "Streams"]
            },
            {
                "name": "Ingestion Layer",
                "type": "Extract",
                "description": "Extracts data from source systems",
                "responsibilities": ["Connect to sources", "Extract data", "Handle pagination", "Error recovery"],
                "technologies": ["Python scripts", "Airbyte", "Fivetran"]
            },
            {
                "name": "Staging Area",
                "type": "Storage",
                "description": "Temporary storage for raw extracted data",
                "responsibilities": ["Store raw data", "Data versioning", "Audit trail"],
                "technologies": ["S3", "Data lake", "Staging tables"]
            },
            {
                "name": "Transformation Engine",
                "type": "Transform",
                "description": "Processes and transforms raw data",
                "responsibilities": ["Data cleaning", "Business logic", "Aggregations", "Joins"],
                "technologies": ["dbt", "pandas", "Spark", "SQL"]
            },
            {
                "name": "Data Warehouse",
                "type": "Storage",
                "description": "Optimized storage for analytics queries",
                "responsibilities": ["Store processed data", "Fast queries", "Historical data"],
                "technologies": ["PostgreSQL", "Snowflake", "BigQuery", "DuckDB"]
            },
            {
                "name": "Orchestrator",
                "type": "Control",
                "description": "Schedules and monitors pipeline execution",
                "responsibilities": ["Job scheduling", "Dependency management", "Monitoring", "Alerting"],
                "technologies": ["Airflow", "Prefect", "Dagster", "cron"]
            },
            {
                "name": "Data Quality",
                "type": "Validation",
                "description": "Validates data meets quality standards",
                "responsibilities": ["Schema validation", "Business rules", "Anomaly detection"],
                "technologies": ["Great Expectations", "dbt tests", "Custom validators"]
            }
        ],
        "data_flows": [
            {"from": "Data Sources", "to": "Ingestion Layer", "data": "Raw data", "protocol": "API/JDBC"},
            {"from": "Ingestion Layer", "to": "Staging Area", "data": "Extracted data", "protocol": "S3/SQL"},
            {"from": "Staging Area", "to": "Transformation Engine", "data": "Raw data", "protocol": "SQL"},
            {"from": "Transformation Engine", "to": "Data Warehouse", "data": "Transformed data", "protocol": "SQL"},
            {"from": "Data Quality", "to": "All Stages", "data": "Validation results", "protocol": "Internal"},
            {"from": "Orchestrator", "to": "All Components", "data": "Triggers/Status", "protocol": "Internal"}
        ],
        "integrations": [
            {"name": "Source Systems", "purpose": "Data extraction", "type": "Internal"},
            {"name": "BI Tools", "purpose": "Visualization", "type": "External"},
            {"name": "Alerting (Slack/Email)", "purpose": "Notifications", "type": "External"},
            {"name": "Data Catalog", "purpose": "Metadata management", "type": "Internal"}
        ],
        "mermaid_diagram": """graph LR
    subgraph Sources["Data Sources"]
        API[APIs]
        DB1[(Source DBs)]
        Files[Files/CSV]
    end
    
    subgraph Extract["Extract"]
        Ingest[Ingestion<br/>Python/Airbyte]
    end
    
    subgraph Stage["Staging"]
        Raw[(Raw Layer<br/>S3/Staging)]
    end
    
    subgraph Transform["Transform"]
        DBT[Transformation<br/>dbt/pandas]
        DQ[Data Quality<br/>Great Expectations]
    end
    
    subgraph Load["Load"]
        DW[(Data Warehouse<br/>PostgreSQL)]
        Marts[(Data Marts)]
    end
    
    subgraph Consume["Consumption"]
        BI[BI Tools<br/>Metabase]
        Reports[Reports]
        ML[ML Models]
    end
    
    subgraph Control["Orchestration"]
        Orch[Orchestrator<br/>Airflow]
        Mon[Monitoring]
    end
    
    API --> Ingest
    DB1 --> Ingest
    Files --> Ingest
    Ingest --> Raw
    Raw --> DBT
    DBT --> DQ
    DQ --> DW
    DW --> Marts
    Marts --> BI
    Marts --> Reports
    Marts --> ML
    Orch -.->|triggers| Ingest
    Orch -.->|triggers| DBT
    Mon -.->|monitors| DW""",
        "technology_decisions": [
            {
                "decision": "Orchestration",
                "chosen": "Airflow",
                "rationale": "Industry standard, flexible, good monitoring",
                "alternatives": ["Prefect (modern)", "Dagster (data-aware)", "cron (simple)"]
            },
            {
                "decision": "Transformation",
                "chosen": "dbt + pandas",
                "rationale": "SQL-based transforms with Python for complex logic",
                "alternatives": ["Spark (big data)", "Pure SQL", "Python only"]
            },
            {
                "decision": "Data Warehouse",
                "chosen": "PostgreSQL",
                "rationale": "Cost-effective, good performance for moderate scale",
                "alternatives": ["Snowflake (scale)", "BigQuery (serverless)", "DuckDB (embedded)"]
            },
            {
                "decision": "Data Quality",
                "chosen": "Great Expectations",
                "rationale": "Comprehensive validation, good documentation",
                "alternatives": ["dbt tests (simpler)", "Custom scripts"]
            }
        ]
    },
    
    "automation": {
        "name": "Automation / Scripting",
        "description": "Automated workflow for repetitive tasks",
        "components": [
            {
                "name": "Trigger/Scheduler",
                "type": "Control",
                "description": "Initiates automation execution",
                "responsibilities": ["Schedule execution", "Event triggers", "Manual invocation"],
                "technologies": ["cron", "systemd timers", "Cloud Scheduler"]
            },
            {
                "name": "Script Engine",
                "type": "Processing",
                "description": "Core automation logic",
                "responsibilities": ["Execute tasks", "Error handling", "Logging"],
                "technologies": ["Python", "Bash", "PowerShell"]
            },
            {
                "name": "Configuration",
                "type": "Control",
                "description": "External configuration for flexibility",
                "responsibilities": ["Store settings", "Environment variables", "Secrets"],
                "technologies": ["YAML/JSON config", "Environment variables", "Secrets manager"]
            },
            {
                "name": "Data Inputs",
                "type": "Input",
                "description": "Source data for automation",
                "responsibilities": ["Provide input data", "File access", "API responses"],
                "technologies": ["Files", "APIs", "Databases"]
            },
            {
                "name": "Data Outputs",
                "type": "Output",
                "description": "Results and artifacts from automation",
                "responsibilities": ["Store results", "Generate reports", "Send notifications"],
                "technologies": ["Files", "APIs", "Email", "Databases"]
            },
            {
                "name": "Logging/Monitoring",
                "type": "Operations",
                "description": "Track execution and errors",
                "responsibilities": ["Log actions", "Track errors", "Alert on failures"],
                "technologies": ["File logs", "Syslog", "Email alerts"]
            }
        ],
        "data_flows": [
            {"from": "Trigger/Scheduler", "to": "Script Engine", "data": "Execution command", "protocol": "OS"},
            {"from": "Configuration", "to": "Script Engine", "data": "Settings", "protocol": "File/Env"},
            {"from": "Data Inputs", "to": "Script Engine", "data": "Input data", "protocol": "File/API"},
            {"from": "Script Engine", "to": "Data Outputs", "data": "Results", "protocol": "File/API"},
            {"from": "Script Engine", "to": "Logging/Monitoring", "data": "Logs", "protocol": "File/Syslog"}
        ],
        "integrations": [
            {"name": "Source Systems", "purpose": "Input data", "type": "Internal/External"},
            {"name": "Target Systems", "purpose": "Output destination", "type": "Internal/External"},
            {"name": "Notification Service", "purpose": "Alerts and reports", "type": "External"}
        ],
        "mermaid_diagram": """graph TB
    subgraph Trigger["Trigger Layer"]
        Cron[Scheduler<br/>cron/systemd]
        Manual[Manual<br/>CLI]
        Event[Event Trigger<br/>webhook]
    end
    
    subgraph Config["Configuration"]
        Conf[Config Files<br/>YAML/JSON]
        Env[Environment<br/>Variables]
        Secrets[Secrets<br/>Manager]
    end
    
    subgraph Engine["Script Engine"]
        Main[Main Script<br/>Python]
        Libs[Libraries<br/>requests, pandas]
        Error[Error Handler]
    end
    
    subgraph IO["Input/Output"]
        Input[Input Data<br/>Files/APIs]
        Output[Output Data<br/>Files/APIs]
    end
    
    subgraph Monitor["Monitoring"]
        Logs[Log Files]
        Alerts[Alerts<br/>Email/Slack]
    end
    
    Cron --> Main
    Manual --> Main
    Event --> Main
    Conf --> Main
    Env --> Main
    Secrets --> Main
    Input --> Main
    Main --> Libs
    Main --> Output
    Main --> Error
    Error --> Alerts
    Main --> Logs""",
        "technology_decisions": [
            {
                "decision": "Scripting Language",
                "chosen": "Python",
                "rationale": "Versatile, great libraries, readable, maintainable",
                "alternatives": ["Bash (system tasks)", "PowerShell (Windows)"]
            },
            {
                "decision": "Scheduling",
                "chosen": "cron + systemd",
                "rationale": "Built-in, reliable, no external dependencies",
                "alternatives": ["Cloud Scheduler", "Airflow (complex workflows)"]
            },
            {
                "decision": "Configuration",
                "chosen": "YAML + environment variables",
                "rationale": "Readable, flexible, secure for secrets",
                "alternatives": ["JSON", "TOML", "Config classes"]
            }
        ]
    },
    
    "mcp_server": {
        "name": "MCP Server Development",
        "description": "Model Context Protocol server for extending AI assistant capabilities",
        "components": [
            {
                "name": "MCP Server",
                "type": "Core",
                "description": "Main server handling MCP protocol communication",
                "responsibilities": ["Protocol handling", "Tool registration", "Request routing"],
                "technologies": ["FastMCP (Python)", "MCP SDK (TypeScript)"]
            },
            {
                "name": "Tool Modules",
                "type": "Application",
                "description": "Individual tools exposed to the AI assistant",
                "responsibilities": ["Implement tool logic", "Input validation", "Response formatting"],
                "technologies": ["Python functions", "TypeScript handlers"]
            },
            {
                "name": "Resource Providers",
                "type": "Data",
                "description": "Resources that provide context data (RAG pattern)",
                "responsibilities": ["Provide context", "Dynamic content", "Data retrieval"],
                "technologies": ["Python generators", "Database queries"]
            },
            {
                "name": "Data Layer",
                "type": "Data",
                "description": "Persistent storage for server data",
                "responsibilities": ["Store data", "Query processing", "Caching"],
                "technologies": ["SQLite", "PostgreSQL", "File system"]
            },
            {
                "name": "Utilities",
                "type": "Support",
                "description": "Shared utilities and helpers",
                "responsibilities": ["Logging", "Validation", "Embeddings", "Caching"],
                "technologies": ["Python modules", "sentence-transformers"]
            },
            {
                "name": "Configuration",
                "type": "Control",
                "description": "Server configuration and settings",
                "responsibilities": ["Environment config", "Feature flags", "Paths"],
                "technologies": ["Environment variables", "Config files"]
            }
        ],
        "data_flows": [
            {"from": "Claude Desktop", "to": "MCP Server", "data": "Tool calls", "protocol": "stdio/MCP"},
            {"from": "MCP Server", "to": "Tool Modules", "data": "Requests", "protocol": "Internal"},
            {"from": "Tool Modules", "to": "Data Layer", "data": "Queries", "protocol": "SQL/File"},
            {"from": "Resource Providers", "to": "MCP Server", "data": "Context data", "protocol": "Internal"},
            {"from": "MCP Server", "to": "Claude Desktop", "data": "Responses", "protocol": "stdio/MCP"}
        ],
        "integrations": [
            {"name": "Claude Desktop", "purpose": "AI assistant interface", "type": "External"},
            {"name": "External APIs", "purpose": "Third-party data", "type": "External"},
            {"name": "Local Files", "purpose": "Document access", "type": "Internal"}
        ],
        "mermaid_diagram": """graph TB
    subgraph Claude["Claude Desktop"]
        AI[Claude AI]
    end
    
    subgraph Server["MCP Server"]
        Core[Server Core<br/>FastMCP]
        
        subgraph Tools["Tool Modules"]
            T1[Tool 1]
            T2[Tool 2]
            T3[Tool 3]
        end
        
        subgraph Resources["Resources"]
            R1[Resource 1]
            R2[Resource 2]
        end
        
        subgraph Utils["Utilities"]
            Log[Logging]
            Val[Validators]
            Emb[Embeddings]
            Cache[Cache]
        end
    end
    
    subgraph Data["Data Layer"]
        DB[(SQLite DB)]
        Files[Files/Docs]
        Index[Vector Index]
    end
    
    subgraph External["External"]
        APIs[External APIs]
    end
    
    AI <-->|MCP Protocol| Core
    Core --> T1
    Core --> T2
    Core --> T3
    Core --> R1
    Core --> R2
    T1 --> DB
    T2 --> Files
    T3 --> APIs
    R1 --> DB
    R2 --> Index
    T1 --> Emb
    T1 --> Cache
    T1 --> Log""",
        "technology_decisions": [
            {
                "decision": "Language/Framework",
                "chosen": "Python with FastMCP",
                "rationale": "Pythonic API, good async support, familiar syntax",
                "alternatives": ["TypeScript MCP SDK (type safety)", "Low-level Python"]
            },
            {
                "decision": "Data Storage",
                "chosen": "SQLite",
                "rationale": "Simple, portable, no server needed, good for local use",
                "alternatives": ["PostgreSQL (scale)", "File-based (simplest)"]
            },
            {
                "decision": "Embeddings",
                "chosen": "sentence-transformers (all-MiniLM-L6-v2)",
                "rationale": "Good balance of speed and quality, local execution",
                "alternatives": ["OpenAI embeddings (better quality, API cost)", "Larger models"]
            },
            {
                "decision": "Architecture",
                "chosen": "Modular with separate tool files",
                "rationale": "Maintainable, testable, easy to extend",
                "alternatives": ["Monolithic (simpler)", "Plugin system (complex)"]
            }
        ]
    }
}

# Generic architecture template
GENERIC_ARCHITECTURE = {
    "name": "Technical System",
    "description": "General technical system architecture",
    "components": [
        {"name": "Frontend/Interface", "type": "Presentation", "description": "User interface", "responsibilities": ["User interaction"], "technologies": ["Web/CLI/API"]},
        {"name": "Application Logic", "type": "Application", "description": "Core business logic", "responsibilities": ["Processing"], "technologies": ["Language-specific"]},
        {"name": "Data Storage", "type": "Data", "description": "Persistent storage", "responsibilities": ["Data persistence"], "technologies": ["Database/Files"]}
    ],
    "data_flows": [
        {"from": "Interface", "to": "Application", "data": "Requests", "protocol": "Internal"},
        {"from": "Application", "to": "Data Storage", "data": "Data", "protocol": "Internal"}
    ],
    "integrations": [],
    "mermaid_diagram": """graph TB
    User[User] --> Interface[Interface]
    Interface --> App[Application Logic]
    App --> Data[(Data Storage)]""",
    "technology_decisions": []
}


def _get_architecture_template(project_type: str) -> dict:
    """Get architecture template for project type"""
    return ARCHITECTURE_TEMPLATES.get(project_type, GENERIC_ARCHITECTURE)


def _build_component_table(components: list) -> str:
    """Build markdown table of components"""
    lines = []
    lines.append("| Component | Type | Responsibilities | Technologies |")
    lines.append("|-----------|------|------------------|--------------|")
    for c in components:
        responsibilities = ", ".join(c.get('responsibilities', [])[:3])
        technologies = ", ".join(c.get('technologies', [])[:3])
        lines.append(f"| **{c['name']}** | {c['type']} | {responsibilities} | {technologies} |")
    return "\n".join(lines)


def _build_data_flow_table(flows: list) -> str:
    """Build markdown table of data flows"""
    lines = []
    lines.append("| From | To | Data | Protocol |")
    lines.append("|------|-----|------|----------|")
    for f in flows:
        lines.append(f"| {f['from']} | {f['to']} | {f['data']} | {f['protocol']} |")
    return "\n".join(lines)


def _build_integration_table(integrations: list) -> str:
    """Build markdown table of integrations"""
    if not integrations:
        return "*No external integrations required.*"
    
    lines = []
    lines.append("| Integration | Purpose | Type |")
    lines.append("|-------------|---------|------|")
    for i in integrations:
        lines.append(f"| {i['name']} | {i['purpose']} | {i['type']} |")
    return "\n".join(lines)


def _build_decisions_section(decisions: list) -> str:
    """Build technology decisions section"""
    if not decisions:
        return "*Technology decisions to be made during implementation.*"
    
    lines = []
    for d in decisions:
        lines.append(f"### {d['decision']}")
        lines.append("")
        lines.append(f"**Chosen:** {d['chosen']}")
        lines.append("")
        lines.append(f"**Rationale:** {d['rationale']}")
        lines.append("")
        if d.get('alternatives'):
            lines.append(f"**Alternatives Considered:** {', '.join(d['alternatives'])}")
            lines.append("")
    return "\n".join(lines)


def _adjust_for_audience(content: str, audience: str) -> str:
    """Adjust content detail level for audience"""
    if audience == "executive":
        # Remove technical details
        lines = content.split("\n")
        filtered = []
        skip_section = False
        for line in lines:
            if "Technology Decisions" in line or "Data Flows" in line:
                skip_section = True
            elif line.startswith("## ") or line.startswith("# "):
                skip_section = False
            
            if not skip_section:
                filtered.append(line)
        return "\n".join(filtered)
    return content


def _build_wireframe_markdown(
    goal: str,
    project_type: str,
    arch_template: dict,
    audience: str,
    best_practices: list,
    custom_components: list
) -> str:
    """Build wireframe/architecture brief markdown"""
    lines = []
    
    # Header
    lines.append(f"# Architecture Brief: {arch_template['name']}")
    lines.append(f"## {goal}")
    lines.append("")
    lines.append(f"**Audience:** {audience.title()}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overview
    lines.append("## 📋 Overview")
    lines.append("")
    lines.append(arch_template['description'])
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # System Diagram
    lines.append("## 🏗️ System Architecture")
    lines.append("")
    lines.append("```mermaid")
    lines.append(arch_template['mermaid_diagram'])
    lines.append("```")
    lines.append("")
    
    # Text fallback for non-Mermaid renderers
    lines.append("<details>")
    lines.append("<summary>View as text diagram</summary>")
    lines.append("")
    lines.append("```")
    lines.append(_generate_ascii_diagram(arch_template))
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Components
    components = custom_components if custom_components else arch_template['components']
    lines.append("## 🧩 Components")
    lines.append("")
    lines.append(_build_component_table(components))
    lines.append("")
    
    # Component Details (not for executive)
    if audience != "executive":
        lines.append("### Component Details")
        lines.append("")
        for c in components:
            lines.append(f"#### {c['name']}")
            lines.append("")
            lines.append(f"**Type:** {c['type']}")
            lines.append("")
            lines.append(f"**Description:** {c['description']}")
            lines.append("")
            lines.append("**Responsibilities:**")
            for r in c.get('responsibilities', []):
                lines.append(f"- {r}")
            lines.append("")
            lines.append(f"**Technologies:** {', '.join(c.get('technologies', []))}")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Data Flows (not for executive)
    if audience != "executive":
        lines.append("## 🔄 Data Flows")
        lines.append("")
        lines.append(_build_data_flow_table(arch_template['data_flows']))
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Integrations
    lines.append("## 🔌 External Integrations")
    lines.append("")
    lines.append(_build_integration_table(arch_template['integrations']))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Technology Decisions (technical audience only)
    if audience == "technical":
        lines.append("## 🔧 Technology Decisions")
        lines.append("")
        lines.append(_build_decisions_section(arch_template['technology_decisions']))
        lines.append("---")
        lines.append("")
    
    # Key Considerations
    lines.append("## 💡 Key Considerations")
    lines.append("")
    lines.append("### Scalability")
    lines.append("")
    lines.append("- Current design supports single-instance deployment")
    lines.append("- Horizontal scaling possible by adding load balancer")
    lines.append("- Database may need sharding/replication at scale")
    lines.append("")
    lines.append("### Security")
    lines.append("")
    lines.append("- All external traffic over HTTPS")
    lines.append("- Authentication required for sensitive operations")
    lines.append("- Secrets managed via environment variables or secrets manager")
    lines.append("")
    lines.append("### Reliability")
    lines.append("")
    lines.append("- Monitoring and alerting recommended for all components")
    lines.append("- Automated backups for data layer")
    lines.append("- Graceful degradation where possible")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Reference Materials
    if best_practices:
        lines.append("## 📚 Reference Materials")
        lines.append("")
        lines.append("From your book library:")
        lines.append("")
        for bp in best_practices[:5]:
            lines.append(f"- **{bp['book_title']}** — Ch. {bp['chapter_number']}: {bp['chapter_title']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("## 📝 Notes")
    lines.append("")
    lines.append("- This is a high-level architecture brief for planning purposes")
    lines.append("- Detailed design decisions should be documented separately")
    lines.append("- Review with technical team before implementation")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated using book-library MCP server. Review and customize for your specific needs.*")
    
    return "\n".join(lines)


def _generate_ascii_diagram(arch_template: dict) -> str:
    """Generate simple ASCII representation of architecture"""
    components = arch_template.get('components', [])
    if not components:
        return "No components defined"
    
    lines = []
    lines.append("ARCHITECTURE OVERVIEW")
    lines.append("=" * 50)
    lines.append("")
    
    # Group by type
    types = {}
    for c in components:
        t = c.get('type', 'Other')
        if t not in types:
            types[t] = []
        types[t].append(c['name'])
    
    for type_name, component_names in types.items():
        lines.append(f"[{type_name}]")
        for name in component_names:
            lines.append(f"  └── {name}")
        lines.append("")
    
    lines.append("DATA FLOW")
    lines.append("-" * 30)
    for flow in arch_template.get('data_flows', [])[:5]:
        lines.append(f"  {flow['from']} → {flow['to']}")
    
    return "\n".join(lines)


def _search_for_architecture_content(search_terms: list, limit: int = 5) -> list:
    """Search library for architecture-related content"""
    return _search_for_best_practices(search_terms, limit)



# =============================================================================
# ANALYZE PROJECT HELPER FUNCTIONS
# =============================================================================

def _assess_complexity(impl_template: dict, arch_template: dict) -> dict:
    """Assess project complexity based on templates"""
    
    # Count factors
    num_phases = len(impl_template.get('phases', []))
    total_days = sum(p.get('duration_days', 5) for p in impl_template.get('phases', []))
    num_components = len(arch_template.get('components', []))
    num_integrations = len(arch_template.get('integrations', []))
    num_decisions = len(arch_template.get('technology_decisions', []))
    
    # Calculate complexity score
    score = 0
    score += num_phases * 2
    score += total_days // 5
    score += num_components * 1.5
    score += num_integrations * 2
    score += num_decisions
    
    # Determine level
    if score < 20:
        level = "simple"
        description = "Straightforward project with well-defined scope"
    elif score < 40:
        level = "moderate"
        description = "Standard complexity with some technical decisions"
    else:
        level = "complex"
        description = "Multi-faceted project requiring careful planning"
    
    return {
        "level": level,
        "score": round(score, 1),
        "description": description,
        "factors": {
            "phases": num_phases,
            "estimated_days": total_days,
            "components": num_components,
            "integrations": num_integrations,
            "decisions": num_decisions
        }
    }


def _build_project_analysis(
    goal: str,
    project_type: str,
    impl_template: dict,
    arch_template: dict,
    brd_template: dict,
    complexity: dict
) -> dict:
    """Build comprehensive project analysis"""
    
    # Extract key info
    phases = impl_template.get('phases', [])
    components = arch_template.get('components', [])
    objectives = brd_template.get('business_objectives', [])
    risks = brd_template.get('risks', [])
    
    # Key decisions needed
    decisions = []
    for phase in phases:
        for dec in phase.get('decisions', []):
            decisions.append({
                "phase": phase['name'],
                "decision": dec.get('name', dec) if isinstance(dec, dict) else dec,
                "options": dec.get('options', []) if isinstance(dec, dict) else []
            })
    
    # High-priority risks
    high_risks = [r for r in risks if r.get('impact') == 'High'][:3]
    
    return {
        "summary": f"This is a {complexity['level']} {impl_template['name']} project with {len(phases)} phases over approximately {complexity['factors']['estimated_days']} days.",
        "key_objectives": objectives[:3],
        "main_components": [c['name'] for c in components[:5]],
        "key_decisions": decisions[:5],
        "top_risks": [{"risk": r['risk'], "mitigation": r['mitigation']} for r in high_risks],
        "estimated_effort": {
            "days": complexity['factors']['estimated_days'],
            "weeks": round(complexity['factors']['estimated_days'] / 5, 1),
            "phases": len(phases)
        }
    }


def _build_recommendations(
    goal: str,
    project_type: str,
    complexity: dict,
    mode: str
) -> list:
    """Build actionable recommendations"""
    
    recommendations = []
    
    # Always recommend learning path for complex projects
    if complexity['level'] in ['moderate', 'complex']:
        recommendations.append({
            "priority": "High",
            "action": "Review relevant chapters from your book library",
            "tool": "generate_learning_path",
            "reason": f"Your library likely has content on {project_type.replace('_', ' ')} best practices"
        })
    
    # BRD recommendation
    recommendations.append({
        "priority": "High",
        "action": "Document business requirements before starting",
        "tool": "generate_brd",
        "reason": "Clear requirements prevent scope creep and align stakeholders"
    })
    
    # Architecture recommendation for technical projects
    if project_type in ['vps', 'web_app', 'data_pipeline', 'mcp_server']:
        recommendations.append({
            "priority": "Medium",
            "action": "Review architecture before implementation",
            "tool": "generate_wireframe_brief",
            "reason": "Understanding component relationships reduces rework"
        })
    
    # Implementation plan
    recommendations.append({
        "priority": "High",
        "action": "Create phased implementation plan",
        "tool": "generate_implementation_plan",
        "reason": "Breaks complex work into manageable phases with clear gates"
    })
    
    # Phase prompts for complex projects
    if complexity['level'] == 'complex':
        recommendations.append({
            "priority": "Medium",
            "action": "Get detailed prompts for each phase",
            "tool": "get_phase_prompts",
            "reason": "Actionable prompts help maintain momentum"
        })
    
    return recommendations


def _generate_artifacts(
    goal: str,
    project_type: str,
    artifact_list: list,
    impl_template: dict,
    arch_template: dict,
    brd_template: dict,
    save_artifacts: bool,
    output_dir: str
) -> dict:
    """Generate requested artifacts"""
    
    artifacts = {}
    file_paths = []
    summary = []
    
    # Setup output directory
    if save_artifacts and output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate each requested artifact
    if "learning_path" in artifact_list:
        # Build learning path data
        phases = impl_template.get('phases', [])
        lp_phases = []
        for i, phase in enumerate(phases, 1):
            lp_phases.append({
                "phase": i,
                "name": phase['name'],
                "topics": phase.get('objectives', [])[:3],
                "duration_days": phase.get('duration_days', 5)
            })
        
        artifacts["learning_path"] = {
            "goal": goal,
            "project_type": impl_template['name'],
            "phases": lp_phases,
            "note": "Use generate_learning_path() for full version with library search"
        }
        summary.append(f"Learning Path: {len(lp_phases)} phases outlined")
    
    if "brd" in artifact_list:
        # Build BRD summary
        brd_data = {
            "goal": goal,
            "project_type": brd_template['name'],
            "problem_statement": brd_template.get('problem_statement', ''),
            "objectives": brd_template.get('business_objectives', []),
            "scope_in": brd_template.get('scope_in', [])[:5],
            "scope_out": brd_template.get('scope_out', [])[:3],
            "requirements": {
                "functional": len(brd_template.get('functional_requirements', [])),
                "non_functional": len(brd_template.get('non_functional_requirements', []))
            },
            "risks": len(brd_template.get('risks', [])),
            "note": "Use generate_brd() for full document"
        }
        
        if save_artifacts:
            # Generate full BRD for saving
            sections = _build_brd_sections(
                goal=goal,
                project_type=project_type,
                brd_config=brd_template,
                impl_template=impl_template,
                business_context="",
                include_technical=True,
                best_practices=[],
                template_style="standard"
            )
            brd_md = _build_brd_markdown(goal, project_type, brd_template, sections, "standard")
            
            path = f"{output_dir}/brd.md" if output_dir else "brd.md"
            with open(path, 'w') as f:
                f.write(brd_md)
            file_paths.append(path)
            brd_data["file_path"] = path
        
        artifacts["brd"] = brd_data
        summary.append(f"BRD: {brd_data['requirements']['functional']} functional requirements")
    
    if "architecture" in artifact_list:
        # Build architecture summary
        arch_data = {
            "goal": goal,
            "project_type": arch_template['name'],
            "components": [{"name": c['name'], "type": c['type']} for c in arch_template['components']],
            "integrations": arch_template.get('integrations', []),
            "data_flows": len(arch_template.get('data_flows', [])),
            "has_diagram": bool(arch_template.get('mermaid_diagram')),
            "note": "Use generate_wireframe_brief() for full document with diagram"
        }
        
        if save_artifacts:
            # Generate full architecture brief for saving
            arch_md = _build_wireframe_markdown(
                goal=goal,
                project_type=project_type,
                arch_template=arch_template,
                audience="stakeholder",
                best_practices=[],
                custom_components=[]
            )
            
            path = f"{output_dir}/architecture.md" if output_dir else "architecture.md"
            with open(path, 'w') as f:
                f.write(arch_md)
            file_paths.append(path)
            arch_data["file_path"] = path
        
        artifacts["architecture"] = arch_data
        summary.append(f"Architecture: {len(arch_data['components'])} components")
    
    if "implementation_plan" in artifact_list:
        # Build implementation plan summary
        phases = impl_template.get('phases', [])
        total_days = sum(p.get('duration_days', 5) for p in phases)
        
        plan_data = {
            "goal": goal,
            "project_type": impl_template['name'],
            "phases": [{"name": p['name'], "days": p.get('duration_days', 5)} for p in phases],
            "total_days": total_days,
            "total_weeks": round(total_days / 5, 1),
            "note": "Use generate_implementation_plan() for full document with prompts"
        }
        
        if save_artifacts:
            # Generate full implementation plan for saving
            timeline = _calculate_timeline(phases, None)
            all_prompts = []
            for phase in phases:
                phase_prompts = _generate_prompts_for_phase(phase, project_type)
                all_prompts.extend(phase_prompts)
            
            plan_md = _build_implementation_markdown(
                goal=goal,
                project_type=project_type,
                template=impl_template,
                timeline=timeline,
                best_practices={},
                all_prompts=all_prompts,
                team_size=1,
                include_prompts=True
            )
            
            path = f"{output_dir}/implementation-plan.md" if output_dir else "implementation-plan.md"
            with open(path, 'w') as f:
                f.write(plan_md)
            file_paths.append(path)
            plan_data["file_path"] = path
        
        artifacts["implementation_plan"] = plan_data
        summary.append(f"Implementation Plan: {len(phases)} phases, {total_days} days")
    
    return {
        "artifacts": artifacts,
        "summary": summary,
        "file_paths": file_paths if save_artifacts else None
    }


def _build_analysis_markdown(result: dict, mode: str) -> str:
    """Build analysis overview markdown"""
    lines = []
    
    # Header
    lines.append(f"# Project Analysis")
    lines.append(f"## {result['goal']}")
    lines.append("")
    lines.append(f"**Project Type:** {result['project_type']}")
    lines.append(f"**Complexity:** {result['complexity']['level'].title()} ({result['complexity']['description']})")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Analysis Summary
    analysis = result.get('analysis', {})
    lines.append("## 📋 Summary")
    lines.append("")
    lines.append(analysis.get('summary', ''))
    lines.append("")
    
    # Estimated Effort
    effort = analysis.get('estimated_effort', {})
    lines.append("### Estimated Effort")
    lines.append("")
    lines.append(f"- **Duration:** {effort.get('days', 0)} days (~{effort.get('weeks', 0)} weeks)")
    lines.append(f"- **Phases:** {effort.get('phases', 0)}")
    lines.append("")
    
    # Key Objectives
    objectives = analysis.get('key_objectives', [])
    if objectives:
        lines.append("### Key Objectives")
        lines.append("")
        for obj in objectives:
            lines.append(f"- {obj}")
        lines.append("")
    
    # Main Components
    components = analysis.get('main_components', [])
    if components:
        lines.append("### Main Components")
        lines.append("")
        for comp in components:
            lines.append(f"- {comp}")
        lines.append("")
    
    # Top Risks
    risks = analysis.get('top_risks', [])
    if risks:
        lines.append("### Top Risks")
        lines.append("")
        for risk in risks:
            lines.append(f"- **{risk['risk']}** — {risk['mitigation']}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        lines.append("## 🎯 Recommendations")
        lines.append("")
        for rec in recommendations:
            priority_emoji = "🔴" if rec['priority'] == 'High' else "🟡"
            lines.append(f"### {priority_emoji} {rec['action']}")
            lines.append("")
            lines.append(f"**Tool:** `{rec['tool']}()`")
            lines.append("")
            lines.append(f"**Why:** {rec['reason']}")
            lines.append("")
    
    # Artifacts (if generated)
    if mode != "overview" and result.get('artifact_summary'):
        lines.append("---")
        lines.append("")
        lines.append("## 📦 Generated Artifacts")
        lines.append("")
        for item in result.get('artifact_summary', []):
            lines.append(f"- ✅ {item}")
        lines.append("")
        
        if result.get('file_paths'):
            lines.append("### Saved Files")
            lines.append("")
            for path in result['file_paths']:
                lines.append(f"- `{path}`")
            lines.append("")
    
    # Next Steps
    lines.append("---")
    lines.append("")
    lines.append("## 🚀 Quick Start")
    lines.append("")
    if mode == "overview":
        lines.append("```python")
        lines.append(f"# Generate essential documents")
        lines.append(f'analyze_project("{result["goal"]}", mode="quick")')
        lines.append("")
        lines.append(f"# Or generate everything")
        lines.append(f'analyze_project("{result["goal"]}", mode="full", save_artifacts=True)')
        lines.append("```")
    else:
        lines.append("Your project artifacts have been generated. Next steps:")
        lines.append("")
        lines.append("1. Review the BRD for requirements alignment")
        lines.append("2. Share architecture brief with stakeholders")
        lines.append("3. Follow the implementation plan phases")
        lines.append("4. Use `get_phase_prompts()` for detailed guidance")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Generated using book-library MCP server project planning tools.*")
    
    return "\n".join(lines)

# Project Planning Tools: Quick Reference

A practical guide to choosing and using the right planning tool.

---

## When to Use What

| Situation | Tool | Command |
|-----------|------|---------|
| "I have an idea, where do I start?" | `analyze_project` | `analyze_project("your idea")` |
| "I need to explain this to stakeholders" | `generate_brd` | `generate_brd("goal", template_style="standard")` |
| "What's the architecture look like?" | `generate_wireframe_brief` | `generate_wireframe_brief("goal")` |
| "How long will this take?" | `generate_implementation_plan` | `generate_implementation_plan("goal")` |
| "What should I do next?" | `get_phase_prompts` | `get_phase_prompts("goal")` |
| "I need all the docs" | `analyze_project` | `analyze_project("goal", mode="full", save_artifacts=True)` |

---

## Decision Tree

```
START: "I want to build something"
           │
           ▼
   ┌───────────────────┐
   │  analyze_project  │  ← Always start here
   │    (overview)     │
   └───────────────────┘
           │
           ▼
   What do you need?
           │
     ┌─────┼─────┬─────────────┐
     ▼     ▼     ▼             ▼
   Learn  Plan  Document    All of it
     │     │     │             │
     ▼     ▼     ▼             ▼
  generate  generate   generate    analyze_project
  _learning _implement _brd or     (mode="full")
  _path     _plan      _wireframe
```

---

## Common Workflows

### 1. Quick Project Assessment

**Use when:** You have an idea and want to understand scope/complexity

```python
analyze_project("Build a VPS on Hetzner for my portfolio")
```

**You get:**
- Project type (VPS, Web App, Data Pipeline, etc.)
- Complexity score (simple/moderate/complex)
- Estimated duration and phases
- Key risks
- Recommended next steps

---

### 2. Stakeholder Presentation

**Use when:** You need to explain a project to non-technical stakeholders

```python
# Business requirements doc
generate_brd("Build a customer portal", template_style="standard")

# High-level architecture
generate_wireframe_brief("Build a customer portal", audience="executive")
```

**You get:**
- Problem statement and objectives
- Scope (what's in/out)
- Success metrics
- Simple architecture diagram
- Risk summary

---

### 3. Sprint Planning

**Use when:** You're ready to start building and need a roadmap

```python
generate_implementation_plan(
    "Build a VPS on Hetzner",
    team_size=1,
    start_date="2025-01-15"
)
```

**You get:**
- Phase-by-phase breakdown
- Start/end dates
- Deliverables per phase
- Gate criteria
- Actionable prompts

---

### 4. Daily Work

**Use when:** You need specific next actions

```python
# All prompts for current phase
get_phase_prompts("Build a VPS", phase_name="Server Setup & Security")

# Just decisions you need to make
get_phase_prompts("Build a VPS", prompt_type="decision")
```

**You get:**
- Action prompts: "Configure UFW firewall rules"
- Decision prompts: "Compare nginx vs Caddy for reverse proxy"
- Research prompts: "Search library for SSH hardening"
- Risk prompts: "How to prevent SSH lockout"

---

### 5. Complete Documentation Package

**Use when:** You need formal project documentation

```python
analyze_project(
    "Build a data pipeline for sales analytics",
    mode="full",
    save_artifacts=True,
    output_dir="./project-docs"
)
```

**You get (saved to files):**
- `brd.md` — Business Requirements Document
- `architecture.md` — System architecture with diagram
- `implementation-plan.md` — Phased timeline with prompts

---

## Tool Comparison

### analyze_project modes

| Mode | Time | Output | Best For |
|------|------|--------|----------|
| `overview` | 2 sec | Analysis only | Quick assessment |
| `quick` | 5 sec | BRD + Plan | Essential docs |
| `full` | 10 sec | All 4 artifacts | Complete package |

### generate_brd styles

| Style | Pages | Best For |
|-------|-------|----------|
| `lean` | 1-2 | Quick reviews, small projects |
| `standard` | 4-6 | Most projects |
| `enterprise` | 6-8 | Formal approval processes |

### generate_wireframe_brief audiences

| Audience | Detail | Best For |
|----------|--------|----------|
| `executive` | Minimal | C-suite, investors |
| `stakeholder` | Balanced | PMs, business leads |
| `technical` | Full | Engineers, architects |

---

## Prompt Types Explained

| Type | Purpose | Example Output |
|------|---------|----------------|
| `action` | Execute a task | "Help me configure nginx as reverse proxy" |
| `decision` | Make a choice | "Compare PostgreSQL vs MySQL for this use case" |
| `research` | Find information | "Search library for Docker security best practices" |
| `risk` | Mitigate problems | "What backup strategy prevents data loss?" |

---

## Project Types

The tools auto-detect your project type from the goal:

| Keywords | Detected Type |
|----------|---------------|
| VPS, server, Hetzner, hosting, deploy | VPS / Server Infrastructure |
| web app, website, portal, dashboard | Web Application |
| pipeline, ETL, analytics, data warehouse | Data Pipeline |
| script, automate, cron, batch | Automation / Scripting |
| MCP, Claude, tool, assistant | MCP Server Development |

---

## Tips

### Be Specific in Goals

```python
# ❌ Too vague
analyze_project("Build something")

# ✅ Specific
analyze_project("Build a VPS on Hetzner to host my portfolio site and 2 side projects")
```

### Add Business Context

```python
generate_brd(
    "Build a customer portal",
    business_context="Q2 deadline, $30k budget, team of 3, must integrate with Salesforce"
)
```

### Iterate on Phases

```python
# Start with overview
analyze_project("Build a VPS")

# Deep dive on current phase
get_phase_prompts("Build a VPS", phase_name="Server Setup & Security")

# Get specific help
# → Use the prompts with Claude to execute tasks
```

### Save for Later

```python
# Save individual artifacts
generate_brd("goal", save_to_file=True)
generate_wireframe_brief("goal", save_to_file=True)
generate_implementation_plan("goal", save_to_file=True)

# Or save everything at once
analyze_project("goal", mode="full", save_artifacts=True, output_dir="./docs")
```

---

## Quick Copy-Paste

### New Project

```python
analyze_project("YOUR GOAL HERE")
```

### Essential Docs

```python
analyze_project("YOUR GOAL HERE", mode="quick")
```

### Full Documentation

```python
analyze_project("YOUR GOAL HERE", mode="full", save_artifacts=True)
```

### Just the BRD

```python
generate_brd("YOUR GOAL HERE")
```

### Just the Architecture

```python
generate_wireframe_brief("YOUR GOAL HERE")
```

### Just the Timeline

```python
generate_implementation_plan("YOUR GOAL HERE")
```

### Next Actions

```python
get_phase_prompts("YOUR GOAL HERE")
```

---

## Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT PLANNING TOOLS                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  START HERE:                                                │
│    analyze_project("goal")         → Quick assessment       │
│    analyze_project("goal", mode="full", save_artifacts=True)│
│                                    → Everything + files     │
│                                                             │
│  INDIVIDUAL TOOLS:                                          │
│    generate_brd("goal")            → Requirements doc       │
│    generate_wireframe_brief("goal")→ Architecture diagram   │
│    generate_implementation_plan("goal") → Timeline          │
│    get_phase_prompts("goal")       → Next actions           │
│                                                             │
│  OPTIONS:                                                   │
│    template_style = "lean" | "standard" | "enterprise"      │
│    audience = "executive" | "stakeholder" | "technical"     │
│    mode = "overview" | "quick" | "full"                     │
│    prompt_type = "action" | "decision" | "research" | "risk"│
│                                                             │
│  SAVE FILES:                                                │
│    save_to_file=True  OR  save_artifacts=True               │
│    output_dir="./my-project"                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

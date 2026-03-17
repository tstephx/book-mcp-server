---
status: active
tags: []
type: project
created: '2026-02-23'
modified: '2026-02-23'
---

# Project Learning Path Generator

## Overview

The **Project Learning Path Generator** transforms your book library into a project advisor. Instead of just searching for topics, you can describe what you want to build, and it will:

1. **Analyze your goal** — Detect the type of project
2. **Search semantically** — Find all relevant content across your library
3. **Organize into phases** — Create a logical learning sequence
4. **Estimate time** — Provide realistic learning and implementation hours
5. **Generate a guide** — Create a comprehensive markdown document

## Tools

### `generate_learning_path`

Main tool for creating project-based learning guides.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `goal` | string | required | What you want to build or achieve |
| `depth` | string | "comprehensive" | Level of detail: "quick", "comprehensive", "deep" |
| `include_concepts` | bool | true | Include concept briefs section in the guide |
| `save_to_file` | bool | false | Save the guide to a markdown file |
| `output_path` | string | "" | Custom path for saved file |

**Returns:**
```python
{
    "goal": "Your stated goal",
    "project_type": "Detected project category (human-readable name)",
    "detected_type": "internal_key",  # e.g. "vps", "web_app", "data_pipeline"
    "phases": [
        {"name": "Phase Name", "topics": [...], "chapters_found": 5}
    ],
    "time_estimate": {
        "learn_hours": 40,
        "implement_hours": 20,
        "total_hours": 60
    },
    "books_found": 8,
    "chapters_found": 45,
    "reading_list": [...],  # Top 10 chapters
    "concept_briefs": [...],  # Key concepts explained (if include_concepts=True)
    "guide": "# Full markdown guide...",
    "file_path": "/path/if/saved.md"
}
```

**Examples:**

```python
# Basic usage
generate_learning_path("Build a VPS on Hetzner to host my portfolio")

# Quick overview
generate_learning_path("Learn Docker basics", depth="quick")

# Save to file
generate_learning_path(
    "Create a data pipeline for CSV analysis",
    save_to_file=True,
    output_path="/path/to/my-learning-guide.md"
)
```

### `list_project_templates`

View available project templates and example goals.

**Returns:**
```python
{
    "templates": [
        {
            "id": "vps",
            "name": "VPS / Server Infrastructure",
            "phases": ["Foundation", "Security", "Web Infrastructure", ...],
            "example_goals": ["Build a VPS on Hetzner...", ...]
        },
        ...
    ],
    "custom_goals": ["Build a VPS on Hetzner...", ...],  # Alias of example_goals
    "usage_tip": "Use generate_learning_path('your goal') - auto-detects template"
}
```

## Supported Project Types

| Type | Description | Phases |
|------|-------------|--------|
| **VPS** | Server infrastructure, hosting | Foundation → Security → Web Infrastructure → Containerization → Deploy → Automation |
| **Web App** | Full-stack web applications | Architecture → Backend → Frontend → DevOps |
| **Data Pipeline** | ETL, analytics, reporting | Data Foundations → Data Processing → Storage → Analysis → Automation |
| **ML Project** | Machine learning, AI, LLMs | ML Foundations → Data Preparation → Modeling → Deep Learning → Deployment |
| **Automation** | Scripts, bots, workflows | Scripting Basics → File Operations → API Integration → Infrastructure → Scheduling |
| **MCP Server** | Claude tools, MCP development | MCP Fundamentals → Backend Development → Data Layer → Integration → Deployment |

## How It Works

1. **Goal Analysis**: Parses your goal for keywords to detect project type
2. **Semantic Search**: Uses embeddings to find relevant chapters across all books
3. **Phase Mapping**: Groups content into logical learning phases
4. **Time Estimation**: Calculates hours based on content volume
5. **Guide Generation**: Creates structured markdown with:
   - Overview and time estimates
   - Books/chapters found
   - Phase-by-phase learning path
   - Recommended reading per phase
   - Implementation checklists

## Example Output

```markdown
# Project Learning Guide: VPS / Server Infrastructure
## Build a VPS on Hetzner to host my portfolio and apps

---

## 🎯 Project Overview

**Goal:** Build a VPS on Hetzner to host my portfolio and apps

**Estimated Learning Time:** 45 hours
**Estimated Implementation Time:** 22 hours
**Total Time:** 67 hours

## 📚 Library Resources

Found **52** relevant chapters across **9** books:

| **Set Up and Manage Your Virtual Private Server** | 7 chapters |
| **Mastering Linux Security and Hardening** | 8 chapters |
| **NGINX HTTP Server** | 4 chapters |
...

## 🗺️ Learning Path

PHASE 1: FOUNDATION (8 hrs)
├── Topics: linux basics, command line, file system
└── Chapters: 12 relevant

PHASE 2: SECURITY (10 hrs)
├── Topics: SSH, firewall, user management, fail2ban
└── Chapters: 15 relevant
...
```

## Integration with Existing Tools

The project learning tools complement existing tools:

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `generate_learning_path` | Project-based learning | "I want to build X" |
| `teach_concept` | Single concept teaching | "Explain Docker to me" |
| `semantic_search` | Find specific content | "Find chapters on SSH" |
| `get_topic_coverage` | Topic analysis | "How well is Docker covered?" |

## Best Practices

1. **Be Specific**: "Build a VPS on Hetzner for Python apps" > "Learn servers"
2. **Start Comprehensive**: Use default depth, narrow with "quick" if overwhelmed
3. **Save Your Guides**: Use `save_to_file=True` for reference
4. **Follow the Phases**: The order is designed for dependency learning
5. **Check the Reading List**: Top chapters are ranked by relevance

## Technical Details

- Uses same embedding model as `semantic_search` (OpenAI `text-embedding-3-large`)
- Searches with multiple terms per project type for comprehensive coverage
- Deduplicates chapters across search terms
- Groups results by phase using topic matching
- Time estimates based on content volume and complexity

---

*File: `src/tools/project_learning_tools.py`*
*Registered in: `src/server.py`*

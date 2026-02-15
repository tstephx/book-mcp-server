"""Project Learning Path Generator

Generates comprehensive, phased learning guides for project goals using
the book library as a knowledge source. Designed for hands-on learning
where users want to build something real.

Key Features:
- Semantic search to discover relevant content
- Dependency-ordered learning phases
- Implementation checklists
- Time estimates
- Book/chapter recommendations

Follows MCP best practices:
- Single responsibility (project learning paths only)
- Clean separation from other tools
- Comprehensive error handling
"""

import logging
import re
from typing import Optional, Literal
from datetime import datetime
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
# PROJECT TEMPLATES & KNOWLEDGE BASE
# =============================================================================

# Maps common project types to required knowledge domains
PROJECT_DOMAINS = {
    "vps": {
        "name": "VPS / Server Infrastructure",
        "search_terms": [
            "linux server administration VPS setup",
            "SSH security hardening firewall",
            "nginx web server reverse proxy",
            "Docker containers deployment",
            "SSL TLS certificates HTTPS",
            "systemd services daemon",
            "Ansible automation configuration",
            "backup monitoring logging"
        ],
        "phases": [
            {"name": "Foundation", "topics": ["linux basics", "command line", "file system"]},
            {"name": "Security", "topics": ["SSH", "firewall", "user management", "fail2ban"]},
            {"name": "Web Infrastructure", "topics": ["nginx", "SSL", "reverse proxy", "domains"]},
            {"name": "Containerization", "topics": ["Docker", "Docker Compose", "volumes"]},
            {"name": "Deployment", "topics": ["application hosting", "databases", "services"]},
            {"name": "Automation", "topics": ["Ansible", "backups", "monitoring"]}
        ]
    },
    "web_app": {
        "name": "Web Application",
        "search_terms": [
            "web development frontend backend",
            "API REST design patterns",
            "database SQL data modeling",
            "authentication security",
            "Docker containerization",
            "CI/CD deployment pipeline"
        ],
        "phases": [
            {"name": "Architecture", "topics": ["design patterns", "API design", "data modeling"]},
            {"name": "Backend", "topics": ["API development", "database", "authentication"]},
            {"name": "Frontend", "topics": ["UI development", "state management"]},
            {"name": "DevOps", "topics": ["Docker", "CI/CD", "deployment"]}
        ]
    },
    "data_pipeline": {
        "name": "Data Pipeline / Analytics",
        "search_terms": [
            "data processing pipeline ETL",
            "Python pandas data cleaning",
            "database SQL queries",
            "data visualization analysis",
            "machine learning models",
            "automation scheduling"
        ],
        "phases": [
            {"name": "Data Foundations", "topics": ["Python", "pandas", "data structures"]},
            {"name": "Data Processing", "topics": ["cleaning", "transformation", "ETL"]},
            {"name": "Storage", "topics": ["databases", "SQL", "data modeling"]},
            {"name": "Analysis", "topics": ["visualization", "statistics", "reporting"]},
            {"name": "Automation", "topics": ["scheduling", "pipelines", "monitoring"]}
        ]
    },
    "ml_project": {
        "name": "Machine Learning Project",
        "search_terms": [
            "machine learning fundamentals",
            "deep learning neural networks",
            "data preprocessing features",
            "model training evaluation",
            "LLM language models",
            "deployment inference serving"
        ],
        "phases": [
            {"name": "ML Foundations", "topics": ["statistics", "Python", "numpy"]},
            {"name": "Data Preparation", "topics": ["cleaning", "feature engineering"]},
            {"name": "Modeling", "topics": ["algorithms", "training", "evaluation"]},
            {"name": "Deep Learning", "topics": ["neural networks", "frameworks"]},
            {"name": "Deployment", "topics": ["serving", "inference", "monitoring"]}
        ]
    },
    "automation": {
        "name": "Automation / Scripting",
        "search_terms": [
            "Python automation scripting",
            "bash shell scripting",
            "Ansible infrastructure automation",
            "API integration webhooks",
            "scheduling cron jobs"
        ],
        "phases": [
            {"name": "Scripting Basics", "topics": ["Python", "bash", "command line"]},
            {"name": "File Operations", "topics": ["file handling", "text processing"]},
            {"name": "API Integration", "topics": ["REST APIs", "web scraping"]},
            {"name": "Infrastructure", "topics": ["Ansible", "configuration management"]},
            {"name": "Scheduling", "topics": ["cron", "task queues", "monitoring"]}
        ]
    },
    "mcp_server": {
        "name": "MCP Server Development",
        "search_terms": [
            "MCP Model Context Protocol server",
            "Python API development",
            "TypeScript Node.js development",
            "database SQLite integration",
            "API design patterns"
        ],
        "phases": [
            {"name": "MCP Fundamentals", "topics": ["MCP protocol", "tools", "resources"]},
            {"name": "Backend Development", "topics": ["Python/TypeScript", "async"]},
            {"name": "Data Layer", "topics": ["database", "caching", "storage"]},
            {"name": "Integration", "topics": ["API design", "error handling"]},
            {"name": "Deployment", "topics": ["packaging", "distribution"]}
        ]
    }
}

# Time estimates per topic complexity
TIME_ESTIMATES = {
    "foundational": {"learn": 4, "implement": 2},  # hours
    "intermediate": {"learn": 6, "implement": 4},
    "advanced": {"learn": 10, "implement": 8}
}

# =============================================================================
# KEY CONCEPTS PER PROJECT PHASE
# =============================================================================

# Maps project phases to key concepts that should be taught
PHASE_CONCEPTS = {
    "vps": {
        "Foundation": ["linux", "bash", "file permissions"],
        "Security": ["SSH", "firewall", "fail2ban"],
        "Web Infrastructure": ["nginx", "reverse proxy", "SSL certificates"],
        "Containerization": ["docker", "docker compose", "volumes"],
        "Deployment": ["systemd", "process management"],
        "Automation": ["ansible", "cron", "backup strategies"]
    },
    "web_app": {
        "Architecture": ["API", "REST", "microservices"],
        "Backend": ["database", "authentication", "ORM"],
        "Frontend": ["components", "state management"],
        "DevOps": ["docker", "CI/CD", "deployment"]
    },
    "data_pipeline": {
        "Data Foundations": ["pandas", "data structures"],
        "Data Processing": ["ETL", "data cleaning"],
        "Storage": ["SQL", "database design"],
        "Analysis": ["visualization", "statistics"],
        "Automation": ["scheduling", "pipelines"]
    },
    "ml_project": {
        "ML Foundations": ["statistics", "numpy", "linear algebra"],
        "Data Preparation": ["feature engineering", "normalization"],
        "Modeling": ["classification", "regression", "evaluation metrics"],
        "Deep Learning": ["neural networks", "backpropagation"],
        "Deployment": ["model serving", "inference"]
    },
    "automation": {
        "Scripting Basics": ["python", "bash"],
        "File Operations": ["file handling", "text processing"],
        "API Integration": ["REST API", "webhooks"],
        "Infrastructure": ["ansible", "configuration management"],
        "Scheduling": ["cron", "task queues"]
    },
    "mcp_server": {
        "MCP Fundamentals": ["MCP protocol", "tools", "resources"],
        "Backend Development": ["async programming", "Python"],
        "Data Layer": ["SQLite", "caching"],
        "Integration": ["API design", "error handling"],
        "Deployment": ["packaging", "distribution"]
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_project_type(goal: str) -> tuple[str, dict]:
    """Detect project type from goal description"""
    goal_lower = goal.lower()
    
    # Keyword matching for project types
    type_keywords = {
        "vps": ["vps", "server", "hosting", "hetzner", "digitalocean", "linode", 
                "virtual private", "self-host", "deploy server"],
        "web_app": ["web app", "website", "frontend", "backend", "full stack",
                    "react", "vue", "django", "flask", "fastapi"],
        "data_pipeline": ["data pipeline", "etl", "data processing", "analytics",
                          "data analysis", "data warehouse", "reporting"],
        "ml_project": ["machine learning", "ml", "deep learning", "neural network",
                       "model training", "llm", "ai model", "prediction"],
        "automation": ["automate", "automation", "script", "bot", "workflow",
                       "scheduled", "cron", "batch"],
        "mcp_server": ["mcp server", "mcp tool", "model context protocol",
                       "claude tool", "ai tool"]
    }
    
    # Find best match
    best_match = None
    best_score = 0
    
    for project_type, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in goal_lower)
        if score > best_score:
            best_score = score
            best_match = project_type
    
    if best_match and best_score > 0:
        return best_match, PROJECT_DOMAINS[best_match]
    
    # Default to generic project
    return "generic", {
        "name": "Technical Project",
        "search_terms": [goal],
        "phases": [
            {"name": "Research", "topics": ["requirements", "architecture"]},
            {"name": "Foundation", "topics": ["core concepts", "tools"]},
            {"name": "Implementation", "topics": ["development", "testing"]},
            {"name": "Deployment", "topics": ["deployment", "monitoring"]}
        ]
    }


def _search_library_for_topics(search_terms: list, limit_per_term: int = 5) -> list:
    """Search library for all topics using chunk-level search, deduplicate results"""
    all_results = []
    seen_chapters = set()

    try:
        embeddings_matrix, chunk_metadata = load_chunk_embeddings()
        if embeddings_matrix is None:
            return []

        with embedding_model_context() as generator:
            for term in search_terms:
                query_embedding = generator.generate(term)
                top_results = find_top_k(
                    query_embedding, embeddings_matrix,
                    k=limit_per_term * 3, min_similarity=0.25
                )

                chunk_results = []
                for idx, similarity in top_results:
                    meta = chunk_metadata[idx]
                    chunk_results.append({**meta, 'similarity': similarity})

                # Aggregate to chapter level
                chapter_results = best_chunk_per_chapter(chunk_results)

                for r in chapter_results[:limit_per_term]:
                    chapter_key = (r['book_id'], r['chapter_number'])
                    if chapter_key not in seen_chapters:
                        seen_chapters.add(chapter_key)
                        all_results.append({
                            'id': r['chapter_id'],
                            'book_id': r['book_id'],
                            'book_title': r['book_title'],
                            'author': r.get('author', ''),
                            'chapter_title': r['chapter_title'],
                            'chapter_number': r['chapter_number'],
                            'file_path': r.get('file_path', ''),
                            'word_count': r.get('word_count', 0),
                            'search_term': term,
                            'similarity': r['similarity'],
                            'excerpt': r.get('excerpt', '')
                        })

            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_results

    except Exception as e:
        logger.error(f"Library search error: {e}", exc_info=True)
        return []


def _group_results_by_phase(results: list, phases: list) -> dict:
    """Group search results into project phases"""
    phase_results = {phase['name']: [] for phase in phases}
    phase_results['Other'] = []
    
    for result in results:
        matched = False
        result_text = f"{result['chapter_title']} {result['search_term']}".lower()
        
        for phase in phases:
            phase_topics = " ".join(phase['topics']).lower()
            if any(topic.lower() in result_text for topic in phase['topics']):
                phase_results[phase['name']].append(result)
                matched = True
                break
        
        if not matched:
            phase_results['Other'].append(result)
    
    return phase_results


def _get_phase_concepts(project_type: str, phase_name: str) -> list:
    """Get key concepts for a specific phase"""
    if project_type in PHASE_CONCEPTS:
        return PHASE_CONCEPTS[project_type].get(phase_name, [])
    return []


def _generate_concept_briefs(concepts: list, project_type: str) -> dict:
    """Generate brief explanations for concepts using teach_concept logic
    
    Returns dict mapping concept -> brief explanation
    """
    from .learning_tools import get_analogy, _find_relevant_sources, _generate_quick_summary
    
    briefs = {}
    
    for concept in concepts[:5]:  # Limit to 5 concepts per phase
        try:
            # Get analogy if available
            analogy_data = get_analogy(concept)
            
            if analogy_data:
                briefs[concept] = {
                    "brief": analogy_data.get("analogy", "")[:300],
                    "pm_context": analogy_data.get("pm_context", ""),
                    "has_analogy": True
                }
            else:
                # Search library for brief
                sources = _find_relevant_sources(concept, limit=2, min_similarity=0.3)
                if sources:
                    summary = _generate_quick_summary(concept, sources)
                    briefs[concept] = {
                        "brief": summary[:300] if summary else f"A key concept in {project_type} projects.",
                        "pm_context": "",
                        "has_analogy": False,
                        "source": sources[0]['book_title'] if sources else None
                    }
                else:
                    briefs[concept] = {
                        "brief": f"A key concept for this phase. Use teach_concept('{concept}') to learn more.",
                        "pm_context": "",
                        "has_analogy": False
                    }
        except Exception as e:
            logger.debug(f"Could not generate brief for {concept}: {e}")
            briefs[concept] = {
                "brief": f"Use teach_concept('{concept}') to learn about this.",
                "pm_context": "",
                "has_analogy": False
            }
    
    return briefs


def _estimate_time(phases: list, results_by_phase: dict) -> dict:
    """Estimate learning and implementation time"""
    total_learn = 0
    total_implement = 0
    phase_times = {}
    
    for phase in phases:
        phase_name = phase['name']
        result_count = len(results_by_phase.get(phase_name, []))
        
        # More results = more content to cover
        base_hours = 4 + (result_count * 1.5)
        impl_hours = 2 + (result_count * 0.75)
        
        phase_times[phase_name] = {
            "learn_hours": round(base_hours),
            "implement_hours": round(impl_hours)
        }
        total_learn += base_hours
        total_implement += impl_hours
    
    return {
        "phases": phase_times,
        "total_learn_hours": round(total_learn),
        "total_implement_hours": round(total_implement),
        "total_hours": round(total_learn + total_implement)
    }


def _generate_checklist(phases: list, results_by_phase: dict) -> list:
    """Generate implementation checklist"""
    checklist = []
    
    for phase in phases:
        phase_name = phase['name']
        phase_items = {
            "phase": phase_name,
            "items": []
        }
        
        # Add topic-based items
        for topic in phase['topics']:
            phase_items["items"].append({
                "task": f"Learn {topic}",
                "type": "learn"
            })
        
        # Add implementation items based on results
        results = results_by_phase.get(phase_name, [])
        if results:
            phase_items["items"].append({
                "task": f"Review relevant chapters ({len(results)} found)",
                "type": "read"
            })
        
        phase_items["items"].append({
            "task": f"Implement {phase_name.lower()} components",
            "type": "implement"
        })
        
        checklist.append(phase_items)
    
    return checklist


def _build_markdown_guide(
    goal: str,
    project_type: str,
    project_config: dict,
    results: list,
    results_by_phase: dict,
    time_estimates: dict,
    checklist: list,
    concept_briefs: dict = None
) -> str:
    """Build comprehensive markdown learning guide
    
    Args:
        concept_briefs: Optional dict mapping phase_name -> {concept: brief_data}
    """
    lines = []
    
    # Header
    lines.append(f"# Project Learning Guide: {project_config['name']}")
    lines.append(f"## {goal}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overview
    lines.append("## 🎯 Project Overview")
    lines.append("")
    lines.append(f"**Goal:** {goal}")
    lines.append("")
    lines.append(f"**Estimated Learning Time:** {time_estimates['total_learn_hours']} hours")
    lines.append(f"**Estimated Implementation Time:** {time_estimates['total_implement_hours']} hours")
    lines.append(f"**Total Time:** {time_estimates['total_hours']} hours")
    lines.append("")
    
    # Books found
    books_used = {}
    for r in results:
        if r['book_title'] not in books_used:
            books_used[r['book_title']] = {
                'author': r.get('author', ''),
                'chapters': []
            }
        books_used[r['book_title']]['chapters'].append(r['chapter_title'])
    
    lines.append("## 📚 Library Resources")
    lines.append("")
    lines.append(f"Found **{len(results)}** relevant chapters across **{len(books_used)}** books:")
    lines.append("")
    
    for book, info in sorted(books_used.items(), key=lambda x: -len(x[1]['chapters'])):
        author_str = f" by {info['author']}" if info['author'] else ""
        lines.append(f"| **{book}**{author_str} | {len(info['chapters'])} chapters |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Learning Path
    lines.append("## 🗺️ Learning Path")
    lines.append("")
    lines.append("```")
    for i, phase in enumerate(project_config['phases'], 1):
        phase_name = phase['name']
        topics = ", ".join(phase['topics'])
        time = time_estimates['phases'].get(phase_name, {})
        lines.append(f"PHASE {i}: {phase_name.upper()} ({time.get('learn_hours', 4)} hrs)")
        lines.append(f"├── Topics: {topics}")
        lines.append(f"└── Chapters: {len(results_by_phase.get(phase_name, []))} relevant")
        lines.append("")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed phases
    for i, phase in enumerate(project_config['phases'], 1):
        phase_name = phase['name']
        phase_results = results_by_phase.get(phase_name, [])
        time = time_estimates['phases'].get(phase_name, {})
        
        lines.append(f"## Phase {i}: {phase_name}")
        lines.append("")
        lines.append(f"**Learning Time:** ~{time.get('learn_hours', 4)} hours")
        lines.append(f"**Topics:** {', '.join(phase['topics'])}")
        lines.append("")
        
        # Add concept briefs if available
        if concept_briefs and phase_name in concept_briefs:
            phase_concepts = concept_briefs[phase_name]
            if phase_concepts:
                lines.append("### 💡 Key Concepts")
                lines.append("")
                for concept, brief_data in phase_concepts.items():
                    lines.append(f"**{concept.title()}**")
                    if brief_data.get('brief'):
                        lines.append(f"> {brief_data['brief']}")
                    if brief_data.get('pm_context'):
                        lines.append(f">")
                        lines.append(f"> *PM Context: {brief_data['pm_context']}*")
                    if brief_data.get('source'):
                        lines.append(f">")
                        lines.append(f"> *Source: {brief_data['source']}*")
                    lines.append("")
        
        if phase_results:
            lines.append("### 📖 Recommended Reading")
            lines.append("")
            for r in phase_results[:5]:  # Top 5 per phase
                lines.append(f"**{r['book_title']}** — Chapter {r['chapter_number']}: {r['chapter_title']}")
                lines.append(f"- Relevance: {r['similarity']:.0%}")
                if r.get('excerpt'):
                    excerpt = r['excerpt'][:200].replace('\n', ' ')
                    lines.append(f"- Preview: *{excerpt}...*")
                lines.append("")
        
        lines.append("### ✅ Checklist")
        lines.append("")
        for topic in phase['topics']:
            lines.append(f"- [ ] Learn: {topic}")
        if phase_results:
            lines.append(f"- [ ] Read: Review {len(phase_results)} relevant chapters")
        lines.append(f"- [ ] Implement: {phase_name} components")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Full reading list
    lines.append("## 📋 Complete Reading List")
    lines.append("")
    lines.append("All relevant chapters, ordered by relevance:")
    lines.append("")
    
    for i, r in enumerate(results[:20], 1):  # Top 20
        lines.append(f"{i}. **{r['book_title']}** — Ch. {r['chapter_number']}: {r['chapter_title']} ({r['similarity']:.0%})")
    
    if len(results) > 20:
        lines.append(f"\n*...and {len(results) - 20} more chapters*")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Implementation checklist
    lines.append("## 📝 Implementation Checklist")
    lines.append("")
    
    for phase_check in checklist:
        lines.append(f"### {phase_check['phase']}")
        for item in phase_check['items']:
            lines.append(f"- [ ] {item['task']}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Footer
    lines.append("*Generated from your book library using semantic search*")
    lines.append(f"*Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    return "\n".join(lines)


# =============================================================================
# MAIN TOOL REGISTRATION
# =============================================================================

def register_project_learning_tools(mcp):
    """Register project learning path tools with MCP server"""
    
    @mcp.tool()
    def generate_learning_path(
        goal: str,
        depth: str = "comprehensive",
        include_concepts: bool = True,
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
            # Validate depth
            valid_depths = ["quick", "comprehensive", "deep"]
            if depth not in valid_depths:
                return {
                    "error": f"Invalid depth '{depth}'. Use: {', '.join(valid_depths)}",
                    "valid_depths": valid_depths
                }
            
            logger.info(f"Generating learning path for: {goal}")
            
            # Detect project type
            project_type, project_config = _detect_project_type(goal)
            logger.info(f"Detected project type: {project_type}")
            
            # Adjust search scope based on depth
            limit_per_term = {"quick": 3, "comprehensive": 5, "deep": 8}[depth]
            
            # Expand search terms with the goal itself
            search_terms = [goal] + project_config['search_terms']
            
            # Search library
            results = _search_library_for_topics(search_terms, limit_per_term)
            
            if not results:
                return {
                    "goal": goal,
                    "project_type": project_type,
                    "message": "No relevant content found in your library for this goal.",
                    "suggestion": "Try rephrasing your goal or check your library with list_books()"
                }
            
            # Group by phase
            results_by_phase = _group_results_by_phase(results, project_config['phases'])
            
            # Estimate time
            time_estimates = _estimate_time(project_config['phases'], results_by_phase)
            
            # Generate checklist
            checklist = _generate_checklist(project_config['phases'], results_by_phase)
            
            # Generate concept briefs if requested
            concept_briefs = None
            if include_concepts:
                logger.info("Generating concept briefs for phases...")
                concept_briefs = {}
                for phase in project_config['phases']:
                    phase_name = phase['name']
                    phase_concepts = _get_phase_concepts(project_type, phase_name)
                    if phase_concepts:
                        concept_briefs[phase_name] = _generate_concept_briefs(
                            phase_concepts, project_type
                        )
            
            # Build guide
            guide = _build_markdown_guide(
                goal, project_type, project_config,
                results, results_by_phase, time_estimates, checklist,
                concept_briefs=concept_briefs
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                if not output_path:
                    # Generate filename from goal
                    safe_name = re.sub(r'[^\w\s-]', '', goal.lower())
                    safe_name = re.sub(r'[\s]+', '-', safe_name)[:50]
                    output_path = f"learning-path-{safe_name}.md"
                
                try:
                    with open(output_path, 'w') as f:
                        f.write(guide)
                    file_path = output_path
                    logger.info(f"Saved learning path to: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not save file: {e}")
            
            # Get unique books
            books_found = len(set(r['book_title'] for r in results))
            
            # Build reading list summary
            reading_list = [
                {
                    "book": r['book_title'],
                    "chapter": r['chapter_number'],
                    "title": r['chapter_title'],
                    "relevance": f"{r['similarity']:.0%}"
                }
                for r in results[:10]
            ]
            
            return {
                "goal": goal,
                "project_type": project_config['name'],
                "detected_type": project_type,
                "phases": [
                    {
                        "name": p['name'],
                        "topics": p['topics'],
                        "chapters_found": len(results_by_phase.get(p['name'], []))
                    }
                    for p in project_config['phases']
                ],
                "time_estimate": {
                    "learn_hours": time_estimates['total_learn_hours'],
                    "implement_hours": time_estimates['total_implement_hours'],
                    "total_hours": time_estimates['total_hours']
                },
                "books_found": books_found,
                "chapters_found": len(results),
                "reading_list": reading_list,
                "concept_briefs": concept_briefs,
                "guide": guide,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"generate_learning_path error: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    @mcp.tool()
    def list_project_templates() -> dict:
        """List available project templates and their learning phases
        
        Shows the built-in project templates that the learning path generator
        recognizes. Use these as inspiration or guidance for your goals.
        
        Returns:
            Dictionary with:
            - templates: List of project templates with phases
            - usage_tip: How to use with generate_learning_path
            
        Examples:
            list_project_templates()
        """
        templates = []
        
        for template_id, config in PROJECT_DOMAINS.items():
            templates.append({
                "id": template_id,
                "name": config['name'],
                "phases": [p['name'] for p in config['phases']],
                "example_goals": _get_example_goals(template_id)
            })
        
        return {
            "templates": templates,
            "usage_tip": "Use generate_learning_path('your goal here') - the system will auto-detect the best template",
            "custom_goals": "You can also use completely custom goals - the system will search your library semantically"
        }


def _get_example_goals(template_id: str) -> list:
    """Get example goals for a template"""
    examples = {
        "vps": [
            "Build a VPS on Hetzner to host my portfolio",
            "Set up a self-hosted server for my side projects",
            "Deploy my Python apps to a cloud server"
        ],
        "web_app": [
            "Build a full-stack web application",
            "Create a REST API with authentication",
            "Develop a React frontend with Python backend"
        ],
        "data_pipeline": [
            "Create an ETL pipeline for CSV data",
            "Build a data analytics dashboard",
            "Automate data processing and reporting"
        ],
        "ml_project": [
            "Train a machine learning model",
            "Build an LLM-powered application",
            "Create a prediction system"
        ],
        "automation": [
            "Automate my daily workflows with Python",
            "Build a web scraper and data collector",
            "Create scheduled automation scripts"
        ],
        "mcp_server": [
            "Build an MCP server for my notes",
            "Create custom Claude tools",
            "Develop an AI-integrated application"
        ]
    }
    return examples.get(template_id, ["Custom project goal"])

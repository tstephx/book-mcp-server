"""Learning tools for PM/MBA persona - concept teaching with business analogies

Designed for:
- 35-year-old Program Manager with MBA
- Learning style: frameworks, stakeholder thinking, business impact
- Need: Fast concept acquisition without wading through implementation details

Follows MCP best practices:
- Single responsibility (teaching/learning only)
- Clean separation from other tools
- Comprehensive error handling
"""

import logging
from typing import Optional, Literal
from ..database import get_db_connection, execute_query
from ..utils.context_managers import embedding_model_context
from ..utils.vector_store import find_top_k
from ..utils.file_utils import read_chapter_content
from ..utils.excerpt_utils import extract_relevant_excerpt
from ..utils.cache import get_cache
import numpy as np
import io

logger = logging.getLogger(__name__)

# =============================================================================
# TEACHING CONFIGURATION
# =============================================================================

DEPTH_CONFIGS = {
    "executive": {
        "description": "2-minute read. One-paragraph summary + business impact.",
        "source_limit": 3,
        "min_similarity": 0.35,
        "output_sections": ["summary", "business_impact", "vocabulary"],
        "detail_level": "minimal"
    },
    "working": {
        "description": "10-minute read. Enough to discuss intelligently with engineers.",
        "source_limit": 5,
        "min_similarity": 0.30,
        "output_sections": ["summary", "what_it_is", "business_impact", "vocabulary", 
                          "decisions", "prerequisites", "related"],
        "detail_level": "moderate"
    },
    "practitioner": {
        "description": "30-minute read. Enough to evaluate decisions or do basic work.",
        "source_limit": 8,
        "min_similarity": 0.25,
        "output_sections": ["summary", "what_it_is", "how_it_works", "business_impact",
                          "vocabulary", "decisions", "tradeoffs", "examples", 
                          "prerequisites", "related", "deep_reading"],
        "detail_level": "comprehensive"
    }
}

OUTPUT_FORMATS = {
    "markdown": "Structured markdown document for saving/reference",
    "conversational": "Natural explanation for dialogue"
}


# =============================================================================
# BUSINESS/ORGANIZATIONAL ANALOGIES
# =============================================================================

CONCEPT_ANALOGIES = {
    # Version Control / Git
    "git": {
        "analogy": "Git is like having a corporate document management system with unlimited 'undo' and parallel workstreams. Imagine if every change to a proposal created a timestamped backup, and multiple teams could work on different versions simultaneously without overwriting each other.",
        "pm_context": "Think of branches like parallel project workstreams - you can run experiments without affecting the main deliverable."
    },
    "branch": {
        "analogy": "A branch is like creating a sandbox copy of a project for experimentation. It's as if you could duplicate your entire project plan, try risky changes, and only merge back the ideas that work.",
        "pm_context": "Like spinning up a pilot program that doesn't affect production operations until proven successful."
    },
    "merge": {
        "analogy": "Merging is like reconciling two versions of a project plan after different teams worked on them independently. The system helps identify conflicts where both teams changed the same thing.",
        "pm_context": "Similar to consolidating feedback from multiple stakeholders into one final document."
    },
    "commit": {
        "analogy": "A commit is like signing off on a milestone checkpoint. It creates a permanent record of the project state at that moment, with notes about what changed and why.",
        "pm_context": "Think of it as documenting a decision - you're creating an audit trail with rationale."
    },
    "pull request": {
        "analogy": "A pull request is like a formal change proposal with built-in review workflow. You're saying 'I'd like to incorporate these changes - please review before approving.'",
        "pm_context": "It's your approval gate - like a stage-gate review before changes go live."
    },
    
    # Containers / Docker
    "docker": {
        "analogy": "Docker is like shipping containers for software. Just as standardized containers revolutionized global shipping (any cargo, any ship, any port), Docker standardizes how software runs (any app, any server, any cloud).",
        "pm_context": "Eliminates 'it works on my machine' problems. Your team ships predictable, consistent deliverables."
    },
    "container": {
        "analogy": "A container is like a pre-configured, self-sufficient office in a box. It includes everything needed to do the work - the desk, computer, software, even the specific versions of tools - isolated from other 'offices.'",
        "pm_context": "Reduces deployment risk and environment dependencies. Easier to scale teams and projects."
    },
    
    # Architecture
    "api": {
        "analogy": "An API is like a standardized interface in an organization. Just as your company has defined processes for how departments request resources from each other, an API defines how software systems make requests.",
        "pm_context": "APIs create contracts between teams. When defined well, teams can work independently."
    },
    "microservices": {
        "analogy": "Microservices is like organizing a company into specialized, autonomous teams rather than one monolithic department. Each team owns their deliverable end-to-end.",
        "pm_context": "Enables parallel workstreams and independent releases. Tradeoff: more coordination overhead."
    },
}

def get_analogy(concept: str) -> dict:
    """Get business/organizational analogy for a concept"""
    concept_lower = concept.lower()
    
    # Direct match
    if concept_lower in CONCEPT_ANALOGIES:
        return CONCEPT_ANALOGIES[concept_lower]
    
    # Partial match
    for key, value in CONCEPT_ANALOGIES.items():
        if key in concept_lower or concept_lower in key:
            return value
    
    return None


# =============================================================================
# MAIN TOOL REGISTRATION
# =============================================================================

def register_learning_tools(mcp):
    """Register learning/teaching tools with MCP server"""
    
    @mcp.tool()
    def teach_concept(
        concept: str,
        depth: str = "working",
        output_format: str = "markdown"
    ) -> dict:
        """Teach a technical concept with business/organizational analogies
        
        Designed for a Program Manager with MBA background who needs to learn
        technical concepts quickly and discuss them intelligently with engineers.
        
        Uses your book library as the knowledge source, translating technical
        jargon into business/PM-friendly language.
        
        Args:
            concept: The concept to learn (e.g., "git branching", "docker containers")
            depth: Learning depth:
                   - "executive": 2-min read. Summary + business impact only.
                   - "working": 10-min read. Enough to discuss with engineers. (default)
                   - "practitioner": 30-min read. Enough to evaluate decisions.
            output_format: How to present the information:
                   - "markdown": Structured document for saving/reference (default)
                   - "conversational": Natural explanation for dialogue
                   
        Returns:
            Dictionary with teaching content including:
            - concept: What was taught
            - depth: Level of detail
            - analogy: Business/organizational analogy
            - content: The teaching content (markdown or conversational)
            - sources: Book chapters used
            - related_concepts: What to learn next
            
        Examples:
            teach_concept("git", depth="executive")
            teach_concept("docker containers", depth="working")
            teach_concept("API design", depth="practitioner", output_format="conversational")
        """
        try:
            # Validate depth
            if depth not in DEPTH_CONFIGS:
                return {
                    "error": f"Invalid depth '{depth}'. Use: executive, working, or practitioner",
                    "valid_depths": list(DEPTH_CONFIGS.keys())
                }
            
            # Validate output format
            if output_format not in OUTPUT_FORMATS:
                return {
                    "error": f"Invalid output_format '{output_format}'. Use: markdown or conversational",
                    "valid_formats": list(OUTPUT_FORMATS.keys())
                }
            
            config = DEPTH_CONFIGS[depth]
            logger.info(f"Teaching '{concept}' at depth={depth}, format={output_format}")
            
            # Get business analogy if available
            analogy_data = get_analogy(concept)
            
            # Search library for relevant content
            sources = _find_relevant_sources(
                concept, 
                limit=config["source_limit"],
                min_similarity=config["min_similarity"]
            )
            
            if not sources:
                return {
                    "concept": concept,
                    "depth": depth,
                    "message": f"No content found about '{concept}' in your library.",
                    "suggestion": "Try a more specific term or check your library contents with list_books()"
                }
            
            # Build teaching content
            if output_format == "markdown":
                content = _build_markdown_content(concept, depth, config, analogy_data, sources)
            else:
                content = _build_conversational_content(concept, depth, config, analogy_data, sources)
            
            # Extract related concepts from sources
            related = _extract_related_concepts(concept, sources)
            
            return {
                "concept": concept,
                "depth": depth,
                "depth_description": config["description"],
                "analogy": analogy_data.get("analogy") if analogy_data else None,
                "pm_context": analogy_data.get("pm_context") if analogy_data else None,
                "content": content,
                "sources": [
                    {
                        "book": s["book_title"],
                        "chapter": s["chapter_title"],
                        "chapter_number": s["chapter_number"],
                        "relevance": f"{s['similarity']:.0%}"
                    }
                    for s in sources
                ],
                "related_concepts": related,
                "next_steps": f"To go deeper, try: teach_concept('{concept}', depth='practitioner')" if depth != "practitioner" else None
            }
            
        except Exception as e:
            logger.error(f"teach_concept error: {e}", exc_info=True)
            return {"error": str(e)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_relevant_sources(query: str, limit: int = 5, min_similarity: float = 0.30) -> list:
    """Find relevant sources from the library using semantic search"""
    try:
        # Get cached embeddings or load from database
        cache = get_cache()
        cached = cache.get_embeddings()
        
        with embedding_model_context() as generator:
            query_embedding = generator.generate(query)
            
            if cached:
                embeddings_matrix, chapter_metadata = cached
            else:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT
                            c.id, c.chapter_number, c.title as chapter_title,
                            c.embedding, c.file_path, b.title as book_title
                        FROM chapters c
                        JOIN books b ON c.book_id = b.id
                        WHERE c.embedding IS NOT NULL
                    """)
                    rows = cursor.fetchall()
                
                if not rows:
                    return []
                
                chapter_embeddings = []
                chapter_metadata = []
                
                for row in rows:
                    embedding = np.load(io.BytesIO(row['embedding']))
                    chapter_embeddings.append(embedding)
                    chapter_metadata.append({
                        'id': row['id'],
                        'book_title': row['book_title'],
                        'chapter_title': row['chapter_title'],
                        'chapter_number': row['chapter_number'],
                        'file_path': row['file_path']
                    })
                
                embeddings_matrix = np.vstack(chapter_embeddings)
                cache.set_embeddings(embeddings_matrix, chapter_metadata)
            
            # Find top matches
            top_results = find_top_k(
                query_embedding, embeddings_matrix,
                k=limit, min_similarity=min_similarity
            )
            
            sources = []
            for idx, similarity in top_results:
                metadata = chapter_metadata[idx]
                
                # Get excerpt
                try:
                    content = read_chapter_content(metadata['file_path'])
                    excerpt = extract_relevant_excerpt(
                        query_embedding, content, generator, max_chars=600
                    )
                except:
                    excerpt = ""
                
                sources.append({
                    **metadata,
                    'similarity': similarity,
                    'excerpt': excerpt
                })
            
            return sources
            
    except Exception as e:
        logger.error(f"Source search error: {e}", exc_info=True)
        return []


def _build_markdown_content(concept: str, depth: str, config: dict, 
                            analogy_data: dict, sources: list) -> str:
    """Build structured markdown teaching content"""
    lines = []
    
    # Header
    lines.append(f"# {concept.title()} — {depth.title()} Knowledge")
    lines.append("")
    lines.append(f"*{config['description']}*")
    lines.append("")
    
    # The 60-Second Version (Business Analogy)
    lines.append("## The 60-Second Version")
    lines.append("")
    if analogy_data:
        lines.append(analogy_data["analogy"])
        lines.append("")
        lines.append(f"**PM Context:** {analogy_data['pm_context']}")
    else:
        lines.append(_generate_quick_summary(concept, sources))
    lines.append("")
    
    # What It Actually Is (if not executive)
    if depth != "executive" and sources:
        lines.append("## What It Actually Is")
        lines.append("")
        lines.append(_synthesize_definition(concept, sources))
        lines.append("")
    
    # Why It Matters (Business Impact)
    lines.append("## Why It Matters (Business Impact)")
    lines.append("")
    lines.append(_generate_business_impact(concept, sources))
    lines.append("")
    
    # Key Vocabulary (for working and practitioner)
    if depth in ["working", "practitioner"]:
        vocab = _extract_vocabulary(concept, sources)
        if vocab:
            lines.append("## Key Vocabulary")
            lines.append("")
            lines.append("*Terms you need to sound credible in technical discussions:*")
            lines.append("")
            for term, definition in vocab.items():
                lines.append(f"- **{term}**: {definition}")
            lines.append("")
    
    # Decisions You'll Encounter (for working and practitioner)
    if depth in ["working", "practitioner"]:
        lines.append("## Common Decisions You'll Encounter")
        lines.append("")
        lines.append(_generate_decisions(concept, sources))
        lines.append("")
    
    # Tradeoffs (practitioner only)
    if depth == "practitioner":
        lines.append("## Tradeoffs to Consider")
        lines.append("")
        lines.append(_generate_tradeoffs(concept, sources))
        lines.append("")
    
    # Source Chapters
    lines.append("## Source Chapters")
    lines.append("")
    lines.append("*Deeper reading from your library:*")
    lines.append("")
    for s in sources[:5]:
        lines.append(f"- **{s['book_title']}** — Chapter {s['chapter_number']}: {s['chapter_title']} ({s['similarity']:.0%} relevant)")
    lines.append("")
    
    return "\n".join(lines)


def _build_conversational_content(concept: str, depth: str, config: dict,
                                   analogy_data: dict, sources: list) -> str:
    """Build natural conversational teaching content"""
    parts = []
    
    # Opening
    if analogy_data:
        parts.append(f"Let me explain {concept} in a way that'll click for you as a PM.")
        parts.append("")
        parts.append(analogy_data["analogy"])
        parts.append("")
        parts.append(analogy_data["pm_context"])
    else:
        parts.append(f"Alright, let's break down {concept}.")
        parts.append("")
        parts.append(_generate_quick_summary(concept, sources))
    
    # Add depth-appropriate detail
    if depth != "executive":
        parts.append("")
        parts.append("Here's what's actually happening under the hood:")
        parts.append("")
        parts.append(_synthesize_definition(concept, sources))
    
    # Business impact
    parts.append("")
    parts.append("From a business perspective, this matters because:")
    parts.append("")
    parts.append(_generate_business_impact(concept, sources))
    
    # Key terms for working/practitioner
    if depth in ["working", "practitioner"]:
        vocab = _extract_vocabulary(concept, sources)
        if vocab:
            parts.append("")
            parts.append("A few terms you'll hear engineers throw around:")
            parts.append("")
            for term, definition in list(vocab.items())[:4]:
                parts.append(f"• {term} — {definition}")
    
    # Decisions for practitioner
    if depth == "practitioner":
        parts.append("")
        parts.append("When this comes up in planning or review meetings, you might need to weigh in on:")
        parts.append("")
        parts.append(_generate_decisions(concept, sources))
    
    # Close with sources
    parts.append("")
    parts.append(f"This is based on what I found in your library, primarily from '{sources[0]['book_title']}' if you want to dig deeper.")
    
    return "\n".join(parts)


def _generate_quick_summary(concept: str, sources: list) -> str:
    """Generate a quick summary from source excerpts"""
    if not sources:
        return f"{concept} is a technical concept used in software development."
    
    # Use the most relevant excerpt as the basis
    best_excerpt = sources[0].get('excerpt', '')
    if best_excerpt:
        # Return a cleaned version of the first paragraph
        paragraphs = best_excerpt.split('\n\n')
        return paragraphs[0][:500] if paragraphs else best_excerpt[:500]
    
    return f"Based on your library, {concept} appears in discussions about {sources[0]['book_title']}."


def _synthesize_definition(concept: str, sources: list) -> str:
    """Synthesize a definition from multiple sources"""
    if not sources:
        return ""
    
    excerpts = [s.get('excerpt', '')[:400] for s in sources[:3] if s.get('excerpt')]
    if excerpts:
        return "\n\n".join(excerpts)
    return ""


def _generate_business_impact(concept: str, sources: list) -> str:
    """Generate business impact statement"""
    # Default business impacts by common concepts
    impact_templates = {
        "git": "Reduces coordination overhead between teams, enables parallel development, and provides complete audit trail for compliance.",
        "docker": "Standardizes deployment environments, reduces 'works on my machine' issues, and enables faster scaling.",
        "container": "Reduces infrastructure costs, speeds up deployment cycles, and improves resource utilization.",
        "api": "Enables team independence, reduces integration complexity, and creates clear contracts between systems.",
        "branch": "Enables risk-free experimentation and parallel workstreams without disrupting production.",
        "merge": "Critical for integrating work from distributed teams. Poor merge practices cause delays.",
        "kubernetes": "Automates scaling and recovery, reducing operational burden and improving reliability.",
    }
    
    concept_lower = concept.lower()
    for key, impact in impact_templates.items():
        if key in concept_lower:
            return impact
    
    # Default template
    return f"Understanding {concept} helps you make informed decisions about technical approaches, evaluate vendor solutions, and communicate effectively with engineering teams."


def _extract_vocabulary(concept: str, sources: list) -> dict:
    """Extract key vocabulary terms from sources"""
    # Static vocabulary for common concepts
    vocab_database = {
        "git": {
            "Repository (repo)": "The project folder with all version history",
            "Commit": "A saved checkpoint with a message explaining changes",
            "Branch": "A parallel version for isolated development",
            "Merge": "Combining changes from one branch into another",
            "Pull Request (PR)": "A formal request to merge with review process",
            "Clone": "Creating a local copy of a remote repository",
            "Push/Pull": "Uploading to or downloading from a shared repository"
        },
        "docker": {
            "Image": "A template/blueprint for creating containers",
            "Container": "A running instance of an image",
            "Dockerfile": "Instructions for building an image",
            "Registry": "Where images are stored and shared",
            "Volume": "Persistent storage that survives container restarts",
            "Compose": "Tool for defining multi-container applications"
        },
        "api": {
            "Endpoint": "A specific URL where requests are sent",
            "REST": "A common architectural style for web APIs",
            "Request/Response": "The ask and answer pattern",
            "Authentication": "Verifying who's making the request",
            "Rate Limiting": "Controlling how many requests are allowed"
        }
    }
    
    concept_lower = concept.lower()
    for key, vocab in vocab_database.items():
        if key in concept_lower:
            return vocab
    
    return {}


def _generate_decisions(concept: str, sources: list) -> str:
    """Generate common decisions a PM might face"""
    decision_templates = {
        "git": """- **Branching strategy**: How should the team organize branches? (GitFlow, trunk-based, feature branches)
- **Code review requirements**: How many approvers? What's the SLA?
- **Release cadence**: How often do we merge to production?""",
        
        "docker": """- **When to containerize**: Is the migration effort worth it for this project?
- **Build vs buy**: Use managed container services or self-host?
- **Image management**: How do we version and store container images?""",
        
        "api": """- **API versioning**: How do we evolve without breaking consumers?
- **Documentation requirements**: What's mandatory before launch?
- **SLA commitments**: What uptime and latency do we guarantee?"""
    }
    
    concept_lower = concept.lower()
    for key, decisions in decision_templates.items():
        if key in concept_lower:
            return decisions
    
    return f"- When to invest in {concept} vs alternatives\n- Build vs buy decisions\n- Team training and adoption timeline"


def _generate_tradeoffs(concept: str, sources: list) -> str:
    """Generate tradeoff analysis for practitioner level"""
    return f"""When evaluating {concept}, consider these tradeoffs:

**Complexity vs Flexibility**: More powerful solutions often require more expertise to implement and maintain.

**Speed vs Stability**: Moving fast can introduce risk; more process means slower but safer changes.

**Team Capability vs Tooling**: The best tool is worthless if the team can't use it effectively.

Review the source chapters for specific tradeoffs mentioned by the authors."""


def _extract_related_concepts(concept: str, sources: list) -> list:
    """Extract related concepts to learn next"""
    related_map = {
        "git": ["branching strategies", "code review", "CI/CD", "GitHub Actions"],
        "docker": ["container orchestration", "Kubernetes", "microservices", "container security"],
        "api": ["REST vs GraphQL", "API gateway", "authentication", "rate limiting"],
        "branch": ["merge strategies", "pull requests", "code review"],
        "container": ["Docker", "container networking", "volumes", "orchestration"]
    }
    
    concept_lower = concept.lower()
    for key, related in related_map.items():
        if key in concept_lower:
            return related
    
    return ["architecture patterns", "best practices", "team workflows"]


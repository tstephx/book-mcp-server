#!/usr/bin/env python3
"""Test semantic search functionality"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Need to set PYTHONPATH for module imports to work
import os
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

# Import using absolute imports from src
import utils.embeddings as embeddings_module
import utils.vector_store as vector_store_module
import database as db_module
import numpy as np
import io

def test_semantic_search():
    """Test semantic search end-to-end"""
    print("🧪 Testing Semantic Search Functionality")
    print("=" * 60)
    
    # Test 1: Embedding generation
    print("\n1. Testing embedding generation...")
    generator = embeddings_module.EmbeddingGenerator()
    test_query = "docker container networking"
    query_embedding = generator.generate(test_query)
    print(f"   ✅ Generated embedding: shape={query_embedding.shape}, dim={generator.dimension}")
    
    # Test 2: Fetch embeddings from database
    print("\n2. Testing database embeddings...")
    chapters = db_module.execute_query("""
        SELECT id, chapter_number, title, embedding
        FROM chapters
        WHERE embedding IS NOT NULL
        LIMIT 5
    """)
    print(f"   ✅ Found {len(chapters)} chapters with embeddings")
    
    # Test 3: Similarity calculation
    print("\n3. Testing similarity calculation...")
    for chapter in chapters[:3]:
        embedding_blob = chapter['embedding']
        chapter_embedding = np.load(io.BytesIO(embedding_blob))
        similarity = vector_store_module.cosine_similarity(query_embedding, chapter_embedding)
        print(f"   • Chapter {chapter['chapter_number']}: {chapter['title'][:50]}")
        print(f"     Similarity: {similarity:.3f}")
    
    # Test 4: Top-K search
    print("\n4. Testing top-K search...")
    all_chapters = db_module.execute_query("""
        SELECT id, chapter_number, title, embedding
        FROM chapters
        WHERE embedding IS NOT NULL
    """)
    
    embeddings_list = []
    for chapter in all_chapters:
        embedding_blob = chapter['embedding']
        embedding = np.load(io.BytesIO(embedding_blob))
        embeddings_list.append(embedding)
    
    embeddings_matrix = np.vstack(embeddings_list)
    top_results = vector_store_module.find_top_k(query_embedding, embeddings_matrix, k=5, min_similarity=0.0)
    
    print(f"   ✅ Top 5 results for '{test_query}':")
    for idx, similarity in top_results:
        chapter = all_chapters[idx]
        print(f"   • [{similarity:.3f}] Chapter {chapter['chapter_number']}: {chapter['title']}")
    
    print("\n" + "=" * 60)
    print("✅ All semantic search tests passed!")
    print("\n📝 Next steps:")
    print("1. Restart Claude Desktop")
    print("2. Try: semantic_search('docker networking', limit=5)")
    print("3. Try resource: book://semantic-context/container%20orchestration")

if __name__ == "__main__":
    try:
        test_semantic_search()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

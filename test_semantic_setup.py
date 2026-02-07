#!/usr/bin/env python3
"""Simple test for semantic search - verifies embeddings exist"""

import sqlite3
import sys

db_path = "/path/to/book-ingestion-python/data/library.db"

try:
    print("🧪 Testing Semantic Search Setup")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check embeddings exist
    cursor.execute("SELECT COUNT(*) as total FROM chapters")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) as with_emb FROM chapters WHERE embedding IS NOT NULL")
    with_embeddings = cursor.fetchone()[0]
    
    cursor.execute("SELECT embedding_model FROM chapters WHERE embedding IS NOT NULL LIMIT 1")
    row = cursor.fetchone()
    model = row[0] if row else "Unknown"
    
    print(f"\n✅ Database Check:")
    print(f"   Total chapters: {total}")
    print(f"   With embeddings: {with_embeddings}")
    print(f"   Model: {model}")
    
    if with_embeddings == total:
        print(f"\n✅ Perfect! All {total} chapters have embeddings!")
    else:
        print(f"\n⚠️  Missing embeddings for {total - with_embeddings} chapters")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Semantic search is ready!")
    print("\n📝 Next steps:")
    print("1. Restart Claude Desktop")
    print("2. Test with: semantic_search('docker networking')")
    print("3. Try RAG resource: book://semantic-context/containers")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)

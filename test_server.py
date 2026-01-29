#!/usr/bin/env python3
"""
Test script to verify MCP server works
Production-ready version with comprehensive testing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration():
    """Test configuration loading"""
    print("1. Testing configuration...")
    try:
        from src.config import Config
        Config.validate()
        print("   ✅ Configuration valid")
        if Config.DEBUG:
            print(Config.display())
        return True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def test_database():
    """Test database connection"""
    print("\n2. Testing database connection...")
    try:
        from src.database import check_database_health
        health = check_database_health()
        
        if health["status"] == "healthy":
            print(f"   ✅ Database healthy")
            print(f"      Books: {health['books']}")
            print(f"      Chapters: {health['chapters']}")
            print(f"      Total words: {health['total_words']:,}")
            return True
        else:
            print(f"   ❌ Database unhealthy: {health.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False

def test_tools():
    """Test that tools can be registered"""
    print("\n3. Testing tool registration...")
    try:
        from mcp.server.fastmcp import FastMCP
        from src.tools.book_tools import register_book_tools
        from src.tools.chapter_tools import register_chapter_tools
        from src.tools.search_tools import register_search_tools
        
        mcp = FastMCP("test-server")
        register_book_tools(mcp)
        register_chapter_tools(mcp)
        register_search_tools(mcp)
        
        print("   ✅ All tools registered successfully")
        return True
    except Exception as e:
        print(f"   ❌ Tool registration error: {e}")
        return False

def test_validation():
    """Test validation functions"""
    print("\n4. Testing input validation...")
    try:
        from src.utils.validators import (
            validate_book_id,
            validate_chapter_number,
            validate_search_query,
            ValidationError
        )
        
        # Test valid inputs
        validate_book_id("1f1bdfbf-209b-4936-9721-c6f3038eef91")
        validate_chapter_number(5)
        validate_search_query("docker")
        
        # Test invalid inputs
        try:
            validate_book_id("invalid")
            print("   ❌ Should have caught invalid book ID")
            return False
        except ValidationError:
            pass  # Expected
        
        try:
            validate_chapter_number(0)
            print("   ❌ Should have caught invalid chapter number")
            return False
        except ValidationError:
            pass  # Expected
        
        print("   ✅ Validation working correctly")
        return True
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Book Library MCP Server")
    print("="*50)
    print()
    
    tests = [
        test_configuration(),
        test_database(),
        test_tools(),
        test_validation()
    ]
    
    print("\n" + "="*50)
    
    if all(tests):
        print("✅ All tests passed!")
        print("\n📝 Next steps:")
        print("1. Configure Claude Desktop (see README.md)")
        print("2. Restart Claude Desktop")
        print("3. Ask: 'What books do I have?'")
        return 0
    else:
        print("❌ Some tests failed")
        print("\nCheck the errors above and:")
        print("1. Verify database path is correct")
        print("2. Ensure book-ingestion-python project exists")
        print("3. Check that books have been processed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

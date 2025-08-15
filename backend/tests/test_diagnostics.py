"""
Diagnostic tests for RAG system - tests against real system components
to identify actual issues causing 'query failed' errors
"""
import os
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import config
    from vector_store import VectorStore, SearchResults
    from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
    from ai_generator import AIGenerator
    from rag_system import RAGSystem
    import chromadb
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    print("This diagnostic will show which components can't be loaded")


def test_environment_setup():
    """Test basic environment and configuration"""
    print("\n=== ENVIRONMENT SETUP DIAGNOSTICS ===")
    
    try:
        # Test .env loading
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not found in environment")
            print("  Check that .env file exists and contains ANTHROPIC_API_KEY")
        elif api_key.strip() == "":
            print("✗ ANTHROPIC_API_KEY is empty")
        else:
            print(f"✓ ANTHROPIC_API_KEY found (length: {len(api_key)})")
        
        # Test config loading
        from config import config
        print(f"✓ Config loaded successfully")
        print(f"  - Model: {config.ANTHROPIC_MODEL}")
        print(f"  - Embedding model: {config.EMBEDDING_MODEL}")
        print(f"  - ChromaDB path: {config.CHROMA_PATH}")
        print(f"  - Max results: {config.MAX_RESULTS}")
        
        return True
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False


def test_chromadb_connectivity():
    """Test ChromaDB database connectivity and data"""
    print("\n=== CHROMADB CONNECTIVITY DIAGNOSTICS ===")
    
    try:
        from vector_store import VectorStore
        
        # Test ChromaDB client creation
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        print("✓ VectorStore initialized successfully")
        
        # Test collections exist
        try:
            course_catalog_count = vector_store.course_catalog.count()
            course_content_count = vector_store.course_content.count()
            
            print(f"✓ ChromaDB collections accessible")
            print(f"  - Course catalog entries: {course_catalog_count}")
            print(f"  - Course content entries: {course_content_count}")
            
            if course_catalog_count == 0:
                print("⚠ WARNING: No courses in catalog - data may not be loaded")
            
            if course_content_count == 0:
                print("⚠ WARNING: No course content - data may not be loaded")
            
            # Test course titles
            course_titles = vector_store.get_existing_course_titles()
            print(f"  - Available courses: {course_titles}")
            
            return course_catalog_count > 0 and course_content_count > 0
            
        except Exception as e:
            print(f"✗ ChromaDB collection access failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ VectorStore initialization failed: {e}")
        return False


def test_vector_search_functionality():
    """Test vector search operations"""
    print("\n=== VECTOR SEARCH DIAGNOSTICS ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Test basic search
        search_results = vector_store.search("test query")
        
        if search_results.error:
            print(f"✗ Search returned error: {search_results.error}")
            return False
        
        print(f"✓ Basic search successful")
        print(f"  - Results count: {len(search_results.documents)}")
        print(f"  - Has metadata: {len(search_results.metadata) > 0}")
        
        if search_results.is_empty():
            print("⚠ WARNING: Search returned no results - may indicate data issues")
        else:
            # Show sample result
            print(f"  - Sample result: {search_results.documents[0][:100]}...")
            print(f"  - Sample metadata: {search_results.metadata[0] if search_results.metadata else 'None'}")
        
        # Test course name resolution
        try:
            resolved = vector_store._resolve_course_name("test")
            print(f"  - Course name resolution test: {resolved}")
        except Exception as e:
            print(f"⚠ Course name resolution error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Vector search test failed: {e}")
        return False


def test_course_search_tool():
    """Test CourseSearchTool functionality"""
    print("\n=== COURSE SEARCH TOOL DIAGNOSTICS ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        
        # Test tool definition
        tool_def = search_tool.get_tool_definition()
        print("✓ Tool definition generated successfully")
        print(f"  - Tool name: {tool_def['name']}")
        print(f"  - Required params: {tool_def['input_schema']['required']}")
        
        # Test tool execution
        result = search_tool.execute("test query")
        print("✓ Tool execution completed")
        print(f"  - Result type: {type(result)}")
        print(f"  - Result preview: {result[:100]}...")
        
        if "error" in result.lower() or "failed" in result.lower():
            print("⚠ WARNING: Tool result contains error indicators")
        
        # Test source tracking
        print(f"  - Sources tracked: {len(search_tool.last_sources)}")
        
        return True
        
    except Exception as e:
        print(f"✗ CourseSearchTool test failed: {e}")
        return False


def test_course_outline_tool():
    """Test CourseOutlineTool functionality"""
    print("\n=== COURSE OUTLINE TOOL DIAGNOSTICS ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        outline_tool = CourseOutlineTool(vector_store)
        
        # Test tool definition
        tool_def = outline_tool.get_tool_definition()
        print("✓ Tool definition generated successfully")
        print(f"  - Tool name: {tool_def['name']}")
        
        # Test with available course (if any)
        course_titles = vector_store.get_existing_course_titles()
        if course_titles:
            test_course = course_titles[0]
            result = outline_tool.execute(test_course)
            print(f"✓ Tool execution completed for '{test_course}'")
            print(f"  - Result preview: {result[:200]}...")
        else:
            print("⚠ No courses available to test outline tool")
            result = outline_tool.execute("nonexistent course")
            print(f"  - No-course result: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ CourseOutlineTool test failed: {e}")
        return False


def test_ai_generator_setup():
    """Test AI Generator initialization and basic functionality"""
    print("\n=== AI GENERATOR DIAGNOSTICS ===")
    
    try:
        # Test initialization
        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        print("✓ AIGenerator initialized successfully")
        
        # Test system prompt
        print(f"✓ System prompt loaded (length: {len(AIGenerator.SYSTEM_PROMPT)})")
        
        # Check for tool references in prompt
        if "search_course_content" in AIGenerator.SYSTEM_PROMPT:
            print("✓ System prompt contains search tool reference")
        else:
            print("⚠ System prompt missing search tool reference")
            
        if "get_course_outline" in AIGenerator.SYSTEM_PROMPT:
            print("✓ System prompt contains outline tool reference")
        else:
            print("⚠ System prompt missing outline tool reference")
        
        # Test without API call (just setup)
        print("✓ AI Generator setup diagnostics passed")
        print("  NOTE: Not testing actual API calls to avoid costs/rate limits")
        
        return True
        
    except Exception as e:
        print(f"✗ AI Generator test failed: {e}")
        return False


def test_tool_manager_integration():
    """Test tool manager and tool registration"""
    print("\n=== TOOL MANAGER DIAGNOSTICS ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Create tool manager and tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        outline_tool = CourseOutlineTool(vector_store)
        
        # Register tools
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        print("✓ Tools registered successfully")
        
        # Test tool definitions
        definitions = tool_manager.get_tool_definitions()
        print(f"✓ Tool definitions retrieved: {len(definitions)} tools")
        
        for definition in definitions:
            print(f"  - {definition['name']}: {definition['description'][:50]}...")
        
        # Test tool execution
        result = tool_manager.execute_tool("search_course_content", query="test")
        print("✓ Tool execution test completed")
        print(f"  - Search result preview: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Tool Manager test failed: {e}")
        return False


def test_rag_system_integration():
    """Test full RAG system integration"""
    print("\n=== RAG SYSTEM INTEGRATION DIAGNOSTICS ===")
    
    try:
        # Test RAG system initialization
        rag_system = RAGSystem(config)
        print("✓ RAG System initialized successfully")
        
        # Test tool registration
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        print(f"✓ Tools registered in RAG system: {len(tool_definitions)}")
        
        # Test analytics
        analytics = rag_system.get_course_analytics()
        print(f"✓ Course analytics: {analytics['total_courses']} courses")
        
        # Test query processing (without AI call)
        print("✓ RAG System integration test passed")
        print("  NOTE: Not testing full query to avoid API costs")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG System integration test failed: {e}")
        return False


def test_data_loading_status():
    """Test if course data has been properly loaded"""
    print("\n=== DATA LOADING STATUS DIAGNOSTICS ===")
    
    try:
        # Check docs folder
        docs_path = Path("../docs")  # Assuming docs folder is at project root
        if docs_path.exists():
            doc_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.docx")) + list(docs_path.glob("*.txt"))
            print(f"✓ Docs folder found with {len(doc_files)} document files")
            for doc_file in doc_files[:5]:  # Show first 5
                print(f"  - {doc_file.name}")
        else:
            print("⚠ No docs folder found - data may not be loaded")
        
        # Check vector store data
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        course_count = vector_store.get_course_count()
        
        if course_count == 0:
            print("✗ CRITICAL: No courses loaded in vector store")
            print("  This is likely the cause of 'query failed' errors")
            print("  Need to run document processing to load course data")
            return False
        else:
            print(f"✓ {course_count} courses loaded in vector store")
            return True
        
    except Exception as e:
        print(f"✗ Data loading status check failed: {e}")
        return False


def run_all_diagnostics():
    """Run all diagnostic tests and provide summary"""
    print("=" * 60)
    print("RAG SYSTEM DIAGNOSTIC SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("ChromaDB Connectivity", test_chromadb_connectivity),
        ("Vector Search", test_vector_search_functionality),
        ("Course Search Tool", test_course_search_tool),
        ("Course Outline Tool", test_course_outline_tool),
        ("AI Generator Setup", test_ai_generator_setup),
        ("Tool Manager", test_tool_manager_integration),
        ("RAG System Integration", test_rag_system_integration),
        ("Data Loading Status", test_data_loading_status),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if not results.get("Data Loading Status", False):
        print("\n🔍 LIKELY ROOT CAUSE IDENTIFIED:")
        print("   No course data loaded in vector store")
        print("   This would cause 'query failed' errors")
        print("   SOLUTION: Run document processing to load course data")
    
    if not results.get("Environment Setup", False):
        print("\n🔍 CONFIGURATION ISSUE:")
        print("   Environment/API key setup problems")
        print("   SOLUTION: Check .env file and API key")


if __name__ == "__main__":
    run_all_diagnostics()
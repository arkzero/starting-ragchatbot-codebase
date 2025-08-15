"""
Test actual query processing to identify the real 'query failed' issue
"""
import sys
import os
import traceback

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_actual_rag_query():
    """Test actual RAG system query to identify real failure point"""
    print("=" * 60)
    print("ACTUAL QUERY TEST - IDENTIFYING REAL FAILURE")
    print("=" * 60)
    
    try:
        from config import config
        from rag_system import RAGSystem
        
        print("✓ Importing RAG system components...")
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        print("✓ RAG system initialized")
        
        # Test simple query
        print("\n--- Testing simple content query ---")
        try:
            response, sources = rag_system.query("What is machine learning?")
            print(f"✓ Query completed successfully!")
            print(f"Response length: {len(response)}")
            print(f"Sources count: {len(sources)}")
            print(f"Response preview: {response[:200]}...")
            
            if sources:
                print("Sources:")
                for i, source in enumerate(sources[:3]):
                    print(f"  {i+1}. {source}")
            
            return True
            
        except Exception as e:
            print(f"✗ Query failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            print("Full traceback:")
            traceback.print_exc()
            return False
    
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        traceback.print_exc()
        return False


def test_tool_execution_directly():
    """Test tool execution directly without AI calls"""
    print("\n" + "=" * 60)
    print("DIRECT TOOL EXECUTION TEST")
    print("=" * 60)
    
    try:
        from config import config
        from vector_store import VectorStore
        from search_tools import CourseSearchTool, ToolManager
        
        # Initialize components
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        
        print("✓ Tools initialized")
        
        # Test direct tool execution
        print("\n--- Testing direct tool execution ---")
        result = tool_manager.execute_tool("search_course_content", query="machine learning")
        
        print(f"✓ Tool execution completed")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:300]}...")
        
        # Check for error indicators
        if "error" in result.lower():
            print("⚠ WARNING: Result contains 'error'")
        if "failed" in result.lower():
            print("⚠ WARNING: Result contains 'failed'")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct tool test failed: {e}")
        traceback.print_exc()
        return False


def test_ai_generator_with_tools():
    """Test AI generator with tools (but catch API errors gracefully)"""
    print("\n" + "=" * 60)
    print("AI GENERATOR WITH TOOLS TEST")
    print("=" * 60)
    
    try:
        from config import config
        from ai_generator import AIGenerator
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
        from vector_store import VectorStore
        
        # Initialize components
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Set up tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        outline_tool = CourseOutlineTool(vector_store)
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        print("✓ AI Generator and tools initialized")
        
        # Test tool definitions
        tools = tool_manager.get_tool_definitions()
        print(f"✓ Tool definitions ready: {len(tools)} tools")
        
        # Test AI generation with tools (this will make actual API call)
        print("\n--- Testing AI generation with tools ---")
        print("WARNING: This will make an actual API call")
        
        try:
            response = ai_generator.generate_response(
                query="What topics are covered in machine learning courses?",
                tools=tools,
                tool_manager=tool_manager
            )
            
            print(f"✓ AI generation successful!")
            print(f"Response length: {len(response)}")
            print(f"Response preview: {response[:300]}...")
            
            return True
            
        except Exception as api_error:
            print(f"✗ AI API call failed: {api_error}")
            print(f"Error type: {type(api_error).__name__}")
            
            # Check specific error types
            if "api_key" in str(api_error).lower():
                print("💡 ISSUE: API key problem")
            elif "rate" in str(api_error).lower():
                print("💡 ISSUE: Rate limiting")
            elif "permission" in str(api_error).lower():
                print("💡 ISSUE: Permission denied")
            else:
                print("💡 ISSUE: Unexpected API error")
            
            return False
    
    except Exception as e:
        print(f"✗ AI Generator setup failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all actual query tests"""
    print("Testing actual query processing to identify 'query failed' root cause...")
    
    tests = [
        ("Direct Tool Execution", test_tool_execution_directly),
        ("AI Generator with Tools", test_ai_generator_with_tools),
        ("Full RAG Query", test_actual_rag_query),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("ACTUAL QUERY TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    # Identify the failure point
    if results.get("Direct Tool Execution", False):
        print("\n🔍 ANALYSIS:")
        print("✓ Tools work correctly in isolation")
        
        if not results.get("AI Generator with Tools", False):
            print("✗ Problem is in AI Generator/API integration")
            print("💡 LIKELY CAUSE: Anthropic API issues (auth, rate limits, etc.)")
        elif not results.get("Full RAG Query", False):
            print("✗ Problem is in RAG system integration")
            print("💡 LIKELY CAUSE: Tool manager or session handling issues")
    else:
        print("\n🔍 ANALYSIS:")
        print("✗ Basic tool execution is failing")
        print("💡 LIKELY CAUSE: Vector store or tool implementation issues")


if __name__ == "__main__":
    main()
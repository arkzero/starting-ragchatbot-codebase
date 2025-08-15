"""
Test the actual FastAPI endpoint that the frontend calls
"""
import sys
import os
import traceback

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_api_endpoint_simulation():
    """Simulate the exact API endpoint call that would cause 'query failed'"""
    print("=" * 60)
    print("API ENDPOINT SIMULATION TEST")
    print("=" * 60)
    
    try:
        from app import rag_system, QueryRequest, QueryResponse
        
        print("✓ Imported app components")
        
        # Simulate the exact flow in the API endpoint
        print("\n--- Simulating API endpoint logic ---")
        
        # Create a test request (same as what frontend would send)
        request = QueryRequest(query="What is machine learning?", session_id=None)
        print(f"✓ Created request: {request.query}")
        
        # Simulate the API endpoint logic step by step
        try:
            # Step 1: Create session (lines 70-72 in app.py)
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            print(f"✓ Session created: {session_id}")
            
            # Step 2: Process query using RAG system (line 75 in app.py)
            answer, sources = rag_system.query(request.query, session_id)
            print(f"✓ RAG query completed")
            print(f"  - Answer length: {len(answer)}")
            print(f"  - Sources count: {len(sources)}")
            print(f"  - Answer preview: {answer[:100]}...")
            
            # Step 3: Convert sources to Source objects (lines 77-84 in app.py)
            from app import Source
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(Source(text=source["text"], url=source.get("url")))
                else:
                    # Handle legacy string sources
                    source_objects.append(Source(text=str(source)))
            print(f"✓ Sources converted: {len(source_objects)} source objects")
            
            # Step 4: Create response (lines 86-90 in app.py)
            response = QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
            print(f"✓ Response created successfully")
            print(f"  - Response type: {type(response)}")
            print(f"  - Response answer length: {len(response.answer)}")
            print(f"  - Response sources: {len(response.sources)}")
            
            return True
            
        except Exception as api_error:
            print(f"✗ API endpoint simulation failed at step: {api_error}")
            print(f"Error type: {type(api_error).__name__}")
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


def test_app_initialization():
    """Test if the app initializes correctly"""
    print("\n" + "=" * 60)
    print("APP INITIALIZATION TEST")
    print("=" * 60)
    
    try:
        # Import the app module (this will run initialization code)
        import app
        
        print("✓ App module imported successfully")
        
        # Check if rag_system exists and is initialized
        if hasattr(app, 'rag_system'):
            print("✓ RAG system found in app")
            
            # Test rag_system functionality
            analytics = app.rag_system.get_course_analytics()
            print(f"✓ RAG system functional: {analytics['total_courses']} courses")
            
        else:
            print("✗ RAG system not found in app")
            return False
        
        # Check if FastAPI app exists
        if hasattr(app, 'app'):
            print("✓ FastAPI app found")
        else:
            print("✗ FastAPI app not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ App initialization failed: {e}")
        traceback.print_exc()
        return False


def test_pydantic_models():
    """Test if Pydantic models work correctly"""
    print("\n" + "=" * 60)
    print("PYDANTIC MODELS TEST")
    print("=" * 60)
    
    try:
        from app import QueryRequest, QueryResponse, Source
        
        # Test QueryRequest
        request = QueryRequest(query="test query", session_id=None)
        print(f"✓ QueryRequest created: {request}")
        
        # Test Source
        source = Source(text="test source", url="https://example.com")
        print(f"✓ Source created: {source}")
        
        # Test QueryResponse
        response = QueryResponse(
            answer="test answer",
            sources=[source],
            session_id="test_session"
        )
        print(f"✓ QueryResponse created: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pydantic models test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling_scenarios():
    """Test various error scenarios that might cause 'query failed'"""
    print("\n" + "=" * 60)
    print("ERROR SCENARIOS TEST")
    print("=" * 60)
    
    try:
        from app import rag_system, QueryRequest
        
        # Test with empty query
        print("--- Testing empty query ---")
        try:
            answer, sources = rag_system.query("", None)
            print(f"✓ Empty query handled: {len(answer)} chars")
        except Exception as e:
            print(f"✗ Empty query failed: {e}")
        
        # Test with very long query
        print("--- Testing very long query ---")
        try:
            long_query = "test " * 1000  # 4000+ characters
            answer, sources = rag_system.query(long_query, None)
            print(f"✓ Long query handled: {len(answer)} chars")
        except Exception as e:
            print(f"✗ Long query failed: {e}")
        
        # Test with special characters
        print("--- Testing special characters ---")
        try:
            special_query = "What about émojis 🤖 and spéciàl cháracters?"
            answer, sources = rag_system.query(special_query, None)
            print(f"✓ Special characters handled: {len(answer)} chars")
        except Exception as e:
            print(f"✗ Special characters failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error scenarios test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all API-level tests"""
    print("Testing API endpoint layer to identify 'query failed' root cause...")
    
    tests = [
        ("App Initialization", test_app_initialization),
        ("Pydantic Models", test_pydantic_models),
        ("API Endpoint Simulation", test_api_endpoint_simulation),
        ("Error Scenarios", test_error_handling_scenarios),
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
    print("API ENDPOINT TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    if all(results.values()):
        print("\n🔍 CONCLUSION:")
        print("✅ ALL API-level tests pass!")
        print("🤔 The 'query failed' error may be:")
        print("   1. A frontend-specific issue")
        print("   2. A deployment/environment issue")
        print("   3. A timing/concurrency issue")
        print("   4. An intermittent API rate limiting issue")
        print("\n💡 RECOMMENDATION:")
        print("   - Check frontend error handling")
        print("   - Check network connectivity")
        print("   - Check for rate limiting")
        print("   - Test with actual frontend requests")
    else:
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\n🔍 IDENTIFIED ISSUES:")
        print(f"❌ Failed tests: {', '.join(failed_tests)}")


if __name__ == "__main__":
    main()
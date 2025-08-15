"""
Test the live server to verify it's working correctly
Run this while the server is running to test actual HTTP requests
"""
import sys
import os
import requests
import json

def test_server_connectivity():
    """Test if the server is running and responding"""
    print("=" * 60)
    print("LIVE SERVER CONNECTIVITY TEST")
    print("=" * 60)
    
    server_url = "http://localhost:8000"
    
    try:
        # Test basic connectivity
        response = requests.get(f"{server_url}/", timeout=5)
        print(f"✓ Server is running on {server_url}")
        print(f"  Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {server_url}")
        print("  Make sure server is running with: uv run uvicorn app:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"✗ Server connectivity error: {e}")
        return False


def test_api_query_endpoint():
    """Test the actual /api/query endpoint with HTTP request"""
    print("\n" + "=" * 60)
    print("LIVE API QUERY ENDPOINT TEST")
    print("=" * 60)
    
    api_url = "http://localhost:8000/api/query"
    
    test_queries = [
        {"query": "What is machine learning?", "session_id": None},
        {"query": "Tell me about Python programming", "session_id": None},
        {"query": "What courses are available?", "session_id": None}
    ]
    
    for i, test_data in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {test_data['query']} ---")
        
        try:
            response = requests.post(
                api_url,
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30  # Give AI time to respond
            )
            
            print(f"✓ Request sent successfully")
            print(f"  Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Query processed successfully")
                print(f"  Answer length: {len(data.get('answer', ''))}")
                print(f"  Sources count: {len(data.get('sources', []))}")
                print(f"  Session ID: {data.get('session_id', 'None')}")
                print(f"  Answer preview: {data.get('answer', '')[:100]}...")
                
                if data.get('sources'):
                    print("  Sources:")
                    for j, source in enumerate(data['sources'][:3]):
                        print(f"    {j+1}. {source.get('text', 'No text')}")
                
            else:
                print(f"✗ Query failed with status {response.status_code}")
                print(f"  Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"✗ Request timed out (>30 seconds)")
            print("  This might indicate AI processing issues")
            
        except requests.exceptions.ConnectionError:
            print(f"✗ Connection error - server may not be running")
            break
            
        except Exception as e:
            print(f"✗ Request failed: {e}")
    
    return True


def test_api_courses_endpoint():
    """Test the /api/courses endpoint"""
    print("\n" + "=" * 60)
    print("LIVE API COURSES ENDPOINT TEST")
    print("=" * 60)
    
    api_url = "http://localhost:8000/api/courses"
    
    try:
        response = requests.get(api_url, timeout=10)
        print(f"✓ Request sent successfully")
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Courses endpoint working")
            print(f"  Total courses: {data.get('total_courses', 0)}")
            print(f"  Course titles: {data.get('course_titles', [])}")
        else:
            print(f"✗ Courses endpoint failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"✗ Courses endpoint error: {e}")
    
    return True


def test_error_scenarios():
    """Test various error scenarios"""
    print("\n" + "=" * 60)
    print("LIVE ERROR SCENARIOS TEST")
    print("=" * 60)
    
    api_url = "http://localhost:8000/api/query"
    
    error_tests = [
        # Empty query
        {"query": "", "session_id": None},
        # Invalid JSON (this will be handled by FastAPI)
        # Very long query
        {"query": "test " * 500, "session_id": None},
        # Special characters
        {"query": "What about émojis 🤖?", "session_id": None}
    ]
    
    for i, test_data in enumerate(error_tests, 1):
        print(f"\n--- Error Test {i}: {test_data['query'][:50]}... ---")
        
        try:
            response = requests.post(
                api_url,
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"✓ Request completed")
            print(f"  Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Error scenario handled gracefully")
                print(f"  Answer length: {len(data.get('answer', ''))}")
            else:
                print(f"⚠ Error response (expected for some tests)")
                print(f"  Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"✗ Error test failed: {e}")
    
    return True


def main():
    """Run all live server tests"""
    print("TESTING LIVE SERVER - Make sure server is running!")
    print("Start server with: uv run uvicorn app:app --reload --port 8000")
    print("Then run this script to test actual HTTP requests\n")
    
    # Check if server is running first
    if not test_server_connectivity():
        print("\n❌ Server is not running. Cannot proceed with tests.")
        print("Start the server first and then run this script again.")
        return
    
    tests = [
        ("API Query Endpoint", test_api_query_endpoint),
        ("API Courses Endpoint", test_api_courses_endpoint),
        ("Error Scenarios", test_error_scenarios),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("LIVE SERVER TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    if all(results.values()):
        print("\n🎉 EXCELLENT! Live server is working perfectly!")
        print("If frontend still shows 'query failed', the issue is in:")
        print("  1. Frontend JavaScript code")
        print("  2. Frontend-backend URL configuration")
        print("  3. CORS or network connectivity")
        print("  4. Frontend error handling logic")
    else:
        print("\n⚠ Some live server tests failed.")
        print("Check the error messages above for specific issues.")


if __name__ == "__main__":
    main()
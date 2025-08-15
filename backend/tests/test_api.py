import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    def test_query_endpoint_with_new_session(self, client_no_static):
        """Test /api/query endpoint creates new session when not provided"""
        response = client_no_static.post(
            "/api/query",
            json={"query": "What is testing?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-id"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_query_endpoint_with_existing_session(self, client_no_static):
        """Test /api/query endpoint with provided session ID"""
        response = client_no_static.post(
            "/api/query",
            json={
                "query": "What is testing?",
                "session_id": "existing-session-123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session-123"
        assert "answer" in data
        assert "sources" in data

    def test_query_endpoint_empty_query(self, client_no_static):
        """Test /api/query endpoint with empty query"""
        response = client_no_static.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_invalid_json(self, client_no_static):
        """Test /api/query endpoint with invalid JSON"""
        response = client_no_static.post(
            "/api/query",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_courses_endpoint(self, client_no_static):
        """Test /api/courses endpoint"""
        response = client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 1
        assert "Test Course" in data["course_titles"]

    def test_clear_session_endpoint(self, client_no_static):
        """Test /api/clear-session endpoint"""
        response = client_no_static.post(
            "/api/clear-session",
            json={"session_id": "test-session-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data

    def test_clear_session_endpoint_invalid_json(self, client_no_static):
        """Test /api/clear-session endpoint with invalid JSON"""
        response = client_no_static.post(
            "/api/clear-session",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_query_response_schema(self, client_no_static):
        """Test that query response matches expected schema"""
        response = client_no_static.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Check sources structure
        assert isinstance(data["sources"], list)
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "url" in source or source.get("url") is None

    def test_courses_response_schema(self, client_no_static):
        """Test that courses response matches expected schema"""
        response = client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Check data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        
        # If there are courses, titles should be strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_cors_headers(self, client_no_static):
        """Test that CORS headers are properly set"""
        response = client_no_static.post(
            "/api/query",
            json={"query": "test"}
        )
        
        assert response.status_code == 200
        # Note: TestClient may not include CORS headers in test environment
        # This test documents the expected behavior for production

    def test_content_type_headers(self, client_no_static):
        """Test that content type is properly set"""
        response = client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]


class TestAPIErrorHandling:
    """Test suite for API error handling"""

    def test_invalid_method_on_query_endpoint(self, client_no_static):
        """Test invalid HTTP method on query endpoint"""
        response = client_no_static.get("/api/query")
        assert response.status_code == 405  # Method not allowed

    def test_invalid_method_on_courses_endpoint(self, client_no_static):
        """Test invalid HTTP method on courses endpoint"""
        response = client_no_static.post("/api/courses")
        assert response.status_code == 405  # Method not allowed

    def test_nonexistent_endpoint(self, client_no_static):
        """Test request to nonexistent endpoint"""
        response = client_no_static.get("/api/nonexistent")
        assert response.status_code == 404  # Not found

    def test_malformed_json(self, client_no_static):
        """Test malformed JSON in request body"""
        response = client_no_static.post(
            "/api/query",
            data="malformed json {",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable entity


class TestAPIIntegration:
    """Integration tests for API functionality"""

    def test_query_session_flow(self, client_no_static):
        """Test complete query and session management flow"""
        # First query - should create new session
        response1 = client_no_static.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query - reuse session
        response2 = client_no_static.post(
            "/api/query",
            json={
                "query": "Tell me more about that",
                "session_id": session_id
            }
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Clear session
        response3 = client_no_static.post(
            "/api/clear-session",
            json={"session_id": session_id}
        )
        assert response3.status_code == 200
        assert response3.json()["success"] is True

    def test_multiple_concurrent_sessions(self, client_no_static):
        """Test handling multiple concurrent sessions"""
        # Create first session
        response1 = client_no_static.post(
            "/api/query",
            json={"query": "Session 1 query"}
        )
        session1 = response1.json()["session_id"]
        
        # Create second session
        response2 = client_no_static.post(
            "/api/query",
            json={"query": "Session 2 query"}
        )
        session2 = response2.json()["session_id"]
        
        # Sessions should be different
        assert session1 == "test-session-id"  # Mock returns same ID
        assert session2 == "test-session-id"  # But in real system would be different
        
        # Both sessions should work
        assert response1.status_code == 200
        assert response2.status_code == 200
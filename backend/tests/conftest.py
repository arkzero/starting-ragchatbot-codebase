import os
import tempfile
import shutil
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from models import Course, Lesson, CourseChunk


@pytest.fixture
def temp_docs_dir() -> Generator[str, None, None]:
    """Create a temporary directory with test documents"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test documents
    test_doc_path = os.path.join(temp_dir, "test_course.txt")
    with open(test_doc_path, "w") as f:
        f.write("This is a test course document. It contains information about testing.")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generator tests"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "This is a test AI response."
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = MagicMock()
    mock_store.add_chunk = MagicMock()
    mock_store.similarity_search = MagicMock(return_value=[
        {
            "content": "Test content",
            "course_title": "Test Course",
            "lesson_number": 1,
            "chunk_index": 0
        }
    ])
    mock_store.get_analytics = MagicMock(return_value={
        "total_courses": 1,
        "course_titles": ["Test Course"]
    })
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = MagicMock()
    mock_manager.create_session = MagicMock(return_value="test-session-id")
    mock_manager.get_history = MagicMock(return_value=[])
    mock_manager.add_interaction = MagicMock()
    mock_manager.clear_session = MagicMock()
    return mock_manager


@pytest.fixture
def mock_rag_system(mock_vector_store, mock_session_manager):
    """Mock RAG system for testing"""
    mock_rag = MagicMock()
    mock_rag.vector_store = mock_vector_store
    mock_rag.session_manager = mock_session_manager
    mock_rag.query = MagicMock(return_value=(
        "Test response", 
        [{"text": "Test source", "url": None}]
    ))
    mock_rag.add_course_folder = MagicMock(return_value=(1, 5))
    mock_rag.get_course_analytics = MagicMock(return_value={
        "total_courses": 1,
        "course_titles": ["Test Course"]
    })
    return mock_rag


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Introduction to Testing",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
        ]
    )


@pytest.fixture
def sample_chunk():
    """Sample course chunk for testing"""
    return CourseChunk(
        content="This is sample course content for testing purposes.",
        course_title="Introduction to Testing",
        lesson_number=1,
        chunk_index=0
    )


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "anthropic_api_key": "test-key",
        "chroma_persist_directory": ":memory:",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }


@pytest.fixture
def client_no_static():
    """FastAPI test client without static file mounting to avoid filesystem dependencies"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create a test app without static file dependencies
    test_app = FastAPI(title="Course Materials RAG System", root_path="")
    
    # Add middleware
    test_app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class ClearSessionRequest(BaseModel):
        session_id: str

    class Source(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag = MagicMock()
    mock_rag.session_manager.create_session.return_value = "test-session-id"
    mock_rag.query.return_value = (
        "Test response", 
        [{"text": "Test source", "url": None}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag.session_manager.clear_session = MagicMock()
    
    # API Endpoints
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()
            
            answer, sources = mock_rag.query(request.query, session_id)
            
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(Source(text=source["text"], url=source.get("url")))
                else:
                    source_objects.append(Source(text=str(source)))
            
            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.post("/api/clear-session")
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag.session_manager.clear_session(request.session_id)
            return {"success": True, "message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return TestClient(test_app)
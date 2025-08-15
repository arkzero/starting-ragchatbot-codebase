"""
Tests for CourseSearchTool functionality
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool execute method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_successful_search_with_results(self):
        """Test successful search that returns results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is course content about Python", "More content about functions"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute search
        result = self.search_tool.execute("Python basics")
        
        # Verify result formatting
        assert "[Python Basics - Lesson 1]" in result
        assert "[Python Basics - Lesson 2]" in result
        assert "This is course content about Python" in result
        assert "More content about functions" in result
        
        # Verify sources were tracked
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert self.search_tool.last_sources[0]["url"] == "https://example.com/lesson1"
    
    def test_search_with_error(self):
        """Test search that returns an error"""
        # Mock error result
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="ChromaDB connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("test query")
        
        # Verify error is returned
        assert result == "ChromaDB connection failed"
        
    def test_empty_search_results(self):
        """Test search with no results"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("nonexistent content")
        
        # Verify empty result message
        assert result == "No relevant content found."
    
    def test_empty_search_with_course_filter(self):
        """Test empty search with course name filter"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with course filter
        result = self.search_tool.execute("test query", course_name="Nonexistent Course")
        
        # Verify filtered empty result message
        assert result == "No relevant content found in course 'Nonexistent Course'."
    
    def test_empty_search_with_lesson_filter(self):
        """Test empty search with lesson number filter"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with lesson filter
        result = self.search_tool.execute("test query", lesson_number=5)
        
        # Verify filtered empty result message
        assert result == "No relevant content found in lesson 5."
    
    def test_search_with_both_filters(self):
        """Test search with both course and lesson filters"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with both filters
        result = self.search_tool.execute("test query", course_name="Test Course", lesson_number=3)
        
        # Verify filtered empty result message
        assert result == "No relevant content found in course 'Test Course' in lesson 3."
    
    def test_search_with_missing_metadata(self):
        """Test search with incomplete metadata"""
        # Mock results with missing metadata
        mock_results = SearchResults(
            documents=["Content without proper metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Execute search
        result = self.search_tool.execute("test query")
        
        # Verify handling of missing metadata
        assert "[unknown]" in result
        assert "Content without proper metadata" in result
        
        # Verify source tracking with unknown course
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0]["text"] == "unknown"
        assert "url" not in self.search_tool.last_sources[0]
    
    def test_search_passes_correct_parameters(self):
        """Test that search parameters are passed correctly to vector store"""
        # Mock successful results
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with all parameters
        self.search_tool.execute("test query", course_name="My Course", lesson_number=2)
        
        # Verify parameters were passed to vector store
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="My Course", 
            lesson_number=2
        )
    
    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        # Verify required fields
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        # Verify schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]
        
        # Verify properties
        props = schema["properties"]
        assert "query" in props
        assert "course_name" in props
        assert "lesson_number" in props
        assert props["query"]["type"] == "string"
        assert props["course_name"]["type"] == "string"
        assert props["lesson_number"]["type"] == "integer"


def test_course_search_tool_integration():
    """Integration test to verify CourseSearchTool can be instantiated with real imports"""
    try:
        from vector_store import VectorStore
        from search_tools import CourseSearchTool
        
        # Create mock vector store for instantiation test
        mock_store = Mock(spec=VectorStore)
        tool = CourseSearchTool(mock_store)
        
        # Verify basic functionality
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'get_tool_definition')
        assert hasattr(tool, 'last_sources')
        
        print("✓ CourseSearchTool integration test passed")
        
    except ImportError as e:
        print(f"✗ Import error in CourseSearchTool integration test: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error in CourseSearchTool integration test: {e}")
        raise


if __name__ == "__main__":
    # Run integration test when script is executed directly
    test_course_search_tool_integration()
    print("CourseSearchTool integration test completed")
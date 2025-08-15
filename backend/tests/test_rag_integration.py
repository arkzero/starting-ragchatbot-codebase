"""
Tests for RAG System integration and end-to-end functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from vector_store import SearchResults


class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "./test_chroma_db"


class TestRAGSystemIntegration:
    """Test cases for RAG System integration"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def setup_method(self, mock_session, mock_ai, mock_vector, mock_doc):
        """Set up test fixtures with mocked dependencies"""
        self.mock_config = MockConfig()
        
        # Set up mocks
        self.mock_vector_store = Mock()
        self.mock_ai_generator = Mock()
        self.mock_session_manager = Mock()
        self.mock_doc_processor = Mock()
        
        # Configure mock returns
        mock_vector.return_value = self.mock_vector_store
        mock_ai.return_value = self.mock_ai_generator
        mock_session.return_value = self.mock_session_manager
        mock_doc.return_value = self.mock_doc_processor
        
        # Create RAG system
        self.rag_system = RAGSystem(self.mock_config)
    
    def test_tool_registration(self):
        """Test that tools are properly registered"""
        # Verify tool manager exists
        assert hasattr(self.rag_system, 'tool_manager')
        assert isinstance(self.rag_system.tool_manager, ToolManager)
        
        # Verify tools are registered
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        assert len(tool_definitions) == 2
    
    def test_query_processing_without_tools(self):
        """Test query processing when AI doesn't use tools"""
        # Mock AI response without tool use
        self.mock_ai_generator.generate_response.return_value = "Direct answer without tools"
        
        # Mock empty sources
        self.mock_vector_store.search.return_value = SearchResults([], [], [])
        
        # Process query
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify response
        assert response == "Direct answer without tools"
        assert sources == []  # No tools used, so no sources
        
        # Verify AI generator was called with correct parameters
        call_args = self.mock_ai_generator.generate_response.call_args
        assert "What is Python?" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    def test_query_processing_with_tool_use(self):
        """Test query processing when AI uses tools"""
        # Mock tool execution result
        mock_search_result = "[Course Title]\nSome course content"
        
        # Mock AI response that would use tools
        self.mock_ai_generator.generate_response.return_value = "Answer based on search results"
        
        # Mock tool manager to simulate tool was used
        mock_sources = [{"text": "Course Title", "url": "https://example.com"}]
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        
        # Process query
        response, sources = self.rag_system.query("Search for course content")
        
        # Verify response and sources
        assert response == "Answer based on search results"
        assert sources == mock_sources
        
        # Verify sources were reset after retrieval
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session_management(self):
        """Test query processing with session management"""
        # Mock session manager
        test_session_id = "test_session_123"
        self.mock_session_manager.get_conversation_history.return_value = "Previous conversation"
        
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Response with context"
        
        # Process query with session
        response, sources = self.rag_system.query("Follow up question", session_id=test_session_id)
        
        # Verify session management
        self.mock_session_manager.get_conversation_history.assert_called_once_with(test_session_id)
        self.mock_session_manager.add_exchange.assert_called_once_with(
            test_session_id, "Follow up question", "Response with context"
        )
        
        # Verify AI generator received conversation history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"
    
    def test_tool_execution_error_handling(self):
        """Test error handling during tool execution"""
        # Mock AI generator to raise an exception
        self.mock_ai_generator.generate_response.side_effect = Exception("Tool execution failed")
        
        # Query processing should raise the exception
        with pytest.raises(Exception, match="Tool execution failed"):
            self.rag_system.query("Test query")
    
    def test_course_analytics(self):
        """Test course analytics functionality"""
        # Mock vector store analytics
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        # Get analytics
        analytics = self.rag_system.get_course_analytics()
        
        # Verify analytics
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
    
    def test_tool_manager_functionality(self):
        """Test tool manager operations"""
        tool_manager = self.rag_system.tool_manager
        
        # Test tool execution
        with patch.object(self.rag_system.search_tool, 'execute') as mock_search_execute:
            mock_search_execute.return_value = "Search result"
            
            result = tool_manager.execute_tool("search_course_content", query="test")
            assert result == "Search result"
            mock_search_execute.assert_called_once_with(query="test")
        
        # Test invalid tool execution
        result = tool_manager.execute_tool("nonexistent_tool", param="value")
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_source_tracking_and_reset(self):
        """Test source tracking across tool operations"""
        tool_manager = self.rag_system.tool_manager
        
        # Mock sources on search tool
        self.rag_system.search_tool.last_sources = [{"text": "Test Source"}]
        
        # Get sources
        sources = tool_manager.get_last_sources()
        assert sources == [{"text": "Test Source"}]
        
        # Reset sources
        tool_manager.reset_sources()
        
        # Verify sources were reset
        assert self.rag_system.search_tool.last_sources == []


class TestToolManagerStandalone:
    """Test ToolManager independently"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        
        # Create mock tools
        self.mock_tool_1 = Mock()
        self.mock_tool_1.get_tool_definition.return_value = {
            "name": "test_tool_1",
            "description": "Test tool 1"
        }
        self.mock_tool_1.execute.return_value = "Tool 1 result"
        
        self.mock_tool_2 = Mock()
        self.mock_tool_2.get_tool_definition.return_value = {
            "name": "test_tool_2", 
            "description": "Test tool 2"
        }
        self.mock_tool_2.execute.return_value = "Tool 2 result"
    
    def test_tool_registration(self):
        """Test tool registration"""
        # Register tools
        self.tool_manager.register_tool(self.mock_tool_1)
        self.tool_manager.register_tool(self.mock_tool_2)
        
        # Verify registration
        assert "test_tool_1" in self.tool_manager.tools
        assert "test_tool_2" in self.tool_manager.tools
        
        # Verify tool definitions
        definitions = self.tool_manager.get_tool_definitions()
        assert len(definitions) == 2
        
        tool_names = [def_["name"] for def_ in definitions]
        assert "test_tool_1" in tool_names
        assert "test_tool_2" in tool_names
    
    def test_tool_execution(self):
        """Test tool execution"""
        # Register tool
        self.tool_manager.register_tool(self.mock_tool_1)
        
        # Execute tool
        result = self.tool_manager.execute_tool("test_tool_1", param1="value1", param2="value2")
        
        # Verify execution
        assert result == "Tool 1 result"
        self.mock_tool_1.execute.assert_called_once_with(param1="value1", param2="value2")
    
    def test_invalid_tool_registration(self):
        """Test registration of invalid tool"""
        # Create tool with no name
        invalid_tool = Mock()
        invalid_tool.get_tool_definition.return_value = {"description": "No name"}
        
        # Registration should raise error
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            self.tool_manager.register_tool(invalid_tool)


def test_rag_system_integration():
    """Integration test for RAG system instantiation"""
    try:
        from rag_system import RAGSystem
        from config import Config
        
        # Create mock config
        mock_config = MockConfig()
        
        # This will fail without proper mocking, but tests import structure
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag_system = RAGSystem(mock_config)
            
            # Verify basic attributes
            assert hasattr(rag_system, 'tool_manager')
            assert hasattr(rag_system, 'search_tool')
            assert hasattr(rag_system, 'outline_tool')
            assert hasattr(rag_system, 'query')
            assert hasattr(rag_system, 'get_course_analytics')
        
        print("✓ RAG System integration test passed")
        
    except ImportError as e:
        print(f"✗ Import error in RAG System integration test: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error in RAG System integration test: {e}")
        raise


if __name__ == "__main__":
    # Run integration test when script is executed directly
    test_rag_system_integration()
    print("RAG System integration test completed")
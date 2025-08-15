"""
Tests for AIGenerator tool calling functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class MockAnthropicClient:
    """Mock Anthropic client for testing"""
    
    def __init__(self):
        self.messages = Mock()
    
    def create_mock_response(self, content, stop_reason="end_turn"):
        """Create a mock response object"""
        response = Mock()
        response.content = [Mock()]
        response.content[0].text = content
        response.stop_reason = stop_reason
        return response
    
    def create_tool_use_response(self, tool_name, tool_input, tool_id="test_tool_id"):
        """Create a mock tool use response"""
        response = Mock()
        response.stop_reason = "tool_use"
        
        # Create tool use content block
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_input
        tool_block.id = tool_id
        
        response.content = [tool_block]
        return response


class TestAIGenerator:
    """Test cases for AIGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_client = MockAnthropicClient()
        self.ai_generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        # Replace the real client with our mock
        self.ai_generator.client = self.mock_client
    
    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Mock simple response
        mock_response = self.mock_client.create_mock_response("This is a test response")
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response
        result = self.ai_generator.generate_response("What is Python?")
        
        # Verify result
        assert result == "This is a test response"
        
        # Verify API call parameters
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-sonnet-4-20250514"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        # Mock response
        mock_response = self.mock_client.create_mock_response("Response with history")
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response with history
        history = "Previous conversation context"
        result = self.ai_generator.generate_response("Follow up question", conversation_history=history)
        
        # Verify result
        assert result == "Response with history"
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation context" in system_content
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Mock tools
        mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        # Mock response without tool use
        mock_response = self.mock_client.create_mock_response("Direct answer without using tools")
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response
        result = self.ai_generator.generate_response("General question", tools=mock_tools)
        
        # Verify result
        assert result == "Direct answer without using tools"
        
        # Verify tools were included in API call
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["tools"] == mock_tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_execution(self):
        """Test response generation with tool execution"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Mock tools
        mock_tools = [{"name": "search_course_content"}]
        
        # Mock initial tool use response
        tool_response = self.mock_client.create_tool_use_response(
            "search_course_content", 
            {"query": "test query"},
            "tool_123"
        )
        
        # Mock final response after tool execution
        final_response = self.mock_client.create_mock_response("Final answer based on tool results")
        
        # Set up mock to return tool use first, then final response
        self.mock_client.messages.create.side_effect = [tool_response, final_response]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Search for course content",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify result
        assert result == "Final answer based on tool results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2
    
    def test_tool_execution_with_multiple_tools(self):
        """Test tool execution when multiple tools are called"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Mock tools
        mock_tools = [{"name": "tool1"}, {"name": "tool2"}]
        
        # Create tool use response with multiple tools
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        
        # Create multiple tool blocks
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "tool1"
        tool_block_1.input = {"param": "value1"}
        tool_block_1.id = "tool_1_id"
        
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "tool2"
        tool_block_2.input = {"param": "value2"}
        tool_block_2.id = "tool_2_id"
        
        tool_response.content = [tool_block_1, tool_block_2]
        
        # Mock final response
        final_response = self.mock_client.create_mock_response("Combined results")
        
        self.mock_client.messages.create.side_effect = [tool_response, final_response]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Use multiple tools",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("tool1", param="value1")
        mock_tool_manager.execute_tool.assert_any_call("tool2", param="value2")
    
    def test_tool_execution_error_handling(self):
        """Test error handling in tool execution"""
        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Mock tools
        mock_tools = [{"name": "failing_tool"}]
        
        # Mock tool use response
        tool_response = self.mock_client.create_tool_use_response(
            "failing_tool",
            {"param": "value"}
        )
        
        # Mock final response (should still be called)
        final_response = self.mock_client.create_mock_response("Error handled")
        
        self.mock_client.messages.create.side_effect = [tool_response, final_response]
        
        # With new error handling, this should not raise but handle gracefully
        result = self.ai_generator.generate_response(
            "Use failing tool",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Should get a response even with tool failure
        assert result == "Error handled"
        
        # Verify tool was called
        mock_tool_manager.execute_tool.assert_called_once_with("failing_tool", param="value")
    
    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling across two rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search result",
            "Second search result"
        ]
        
        # Mock tools
        mock_tools = [{"name": "search_course_content"}]
        
        # Mock first round: tool use
        first_tool_response = self.mock_client.create_tool_use_response(
            "search_course_content",
            {"query": "first search"},
            "tool_1"
        )
        
        # Mock second round: tool use
        second_tool_response = self.mock_client.create_tool_use_response(
            "search_course_content", 
            {"query": "second search"},
            "tool_2"
        )
        
        # Mock final response: no tools
        final_response = self.mock_client.create_mock_response("Final answer based on both searches")
        
        # Set up responses in sequence
        self.mock_client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response, 
            final_response
        ]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Complex query needing multiple searches",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify result
        assert result == "Final answer based on both searches"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first search")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="second search")
        
        # Verify three API calls were made (two tool rounds + final)
        assert self.mock_client.messages.create.call_count == 3
    
    def test_sequential_tool_calling_early_termination(self):
        """Test that tool calling terminates early when Claude gives final response"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        # Mock tools
        mock_tools = [{"name": "search_course_content"}]
        
        # Mock first round: tool use
        first_tool_response = self.mock_client.create_tool_use_response(
            "search_course_content",
            {"query": "search"},
            "tool_1"
        )
        
        # Mock second round: direct response (no tools)
        final_response = self.mock_client.create_mock_response("Direct answer without more tools")
        
        # Set up responses
        self.mock_client.messages.create.side_effect = [
            first_tool_response,
            final_response
        ]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Simple query",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify result
        assert result == "Direct answer without more tools"
        
        # Verify only one tool was executed
        assert mock_tool_manager.execute_tool.call_count == 1
        
        # Verify only two API calls were made
        assert self.mock_client.messages.create.call_count == 2
    
    def test_sequential_tool_calling_max_rounds_limit(self):
        """Test that tool calling respects max rounds limit"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First result",
            "Second result"
        ]
        
        # Mock tools
        mock_tools = [{"name": "search_course_content"}]
        
        # Mock two tool use responses
        first_tool_response = self.mock_client.create_tool_use_response(
            "search_course_content",
            {"query": "first"},
            "tool_1"
        )
        
        second_tool_response = self.mock_client.create_tool_use_response(
            "search_course_content",
            {"query": "second"}, 
            "tool_2"
        )
        
        # Mock final response for when max rounds reached
        final_response = self.mock_client.create_mock_response("Final answer after max rounds")
        
        # Set up responses - both want to use tools, then final
        self.mock_client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response,
            final_response
        ]
        
        # Generate response with max_rounds=2
        result = self.ai_generator.generate_response(
            "Query",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Verify result
        assert result == "Final answer after max rounds"
        
        # Should complete after 2 rounds even though second round wanted tools
        # The second round should be processed as the final round
        assert mock_tool_manager.execute_tool.call_count == 2
        assert self.mock_client.messages.create.call_count == 3  # 2 tool rounds + 1 final
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Verify key components
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "Tool Usage Guidelines" in system_prompt
        assert "course outline requests" in system_prompt
        assert "content questions" in system_prompt
        
        # Verify sequential tool calling features
        assert "Sequential tool usage allowed" in system_prompt
        assert "Maximum 2 rounds" in system_prompt
        assert "Complex queries" in system_prompt
        assert "Multi-part questions" in system_prompt
        
        # Verify old restriction is removed
        assert "One tool call per query maximum" not in system_prompt
    
    def test_api_parameters(self):
        """Test that API parameters are set correctly"""
        ai_gen = AIGenerator("test_key", "test_model")
        
        assert ai_gen.model == "test_model"
        assert ai_gen.base_params["model"] == "test_model"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800


def test_ai_generator_integration():
    """Integration test for AIGenerator instantiation"""
    try:
        from ai_generator import AIGenerator
        
        # Test instantiation with mock credentials
        ai_gen = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Verify basic attributes
        assert hasattr(ai_gen, 'client')
        assert hasattr(ai_gen, 'model')
        assert hasattr(ai_gen, 'base_params')
        assert hasattr(ai_gen, 'generate_response')
        
        # Verify system prompt
        assert AIGenerator.SYSTEM_PROMPT is not None
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
        
        print("✓ AIGenerator integration test passed")
        
    except ImportError as e:
        print(f"✗ Import error in AIGenerator integration test: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error in AIGenerator integration test: {e}")
        raise


@patch('anthropic.Anthropic')
def test_ai_generator_with_real_api_mock(mock_anthropic):
    """Test AIGenerator with mocked Anthropic client"""
    # Mock the client creation
    mock_client = Mock()
    mock_anthropic.return_value = mock_client
    
    # Create AIGenerator
    ai_gen = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
    
    # Verify client was created with correct API key
    mock_anthropic.assert_called_once_with(api_key="test_api_key")
    
    # Mock a response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Mocked response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    
    # Test response generation
    result = ai_gen.generate_response("Test query")
    assert result == "Mocked response"


if __name__ == "__main__":
    # Run integration test when script is executed directly
    test_ai_generator_integration()
    print("AIGenerator integration test completed")
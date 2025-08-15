from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic


@dataclass
class RoundData:
    """Data for a single conversation round"""

    round_number: int
    assistant_message: List[Any]
    tool_results: List[Dict[str, Any]]
    response_type: str  # "tool_use" or "final"


@dataclass
class ConversationState:
    """Tracks conversation state across multiple rounds"""

    initial_query: str
    history: Optional[str] = None
    max_rounds: int = 2
    rounds: List[RoundData] = field(default_factory=list)
    final_response: Optional[str] = None

    def build_messages_for_round(self, round_number: int) -> List[Dict[str, Any]]:
        """Build message list for current round including all previous exchanges"""
        messages = []

        # Add initial user query for first round
        if round_number == 1:
            messages.append({"role": "user", "content": self.initial_query})

        # Add all previous round exchanges
        for round_data in self.rounds:
            messages.append(
                {"role": "assistant", "content": round_data.assistant_message}
            )
            if round_data.tool_results:
                messages.append({"role": "user", "content": round_data.tool_results})

        return messages

    def add_tool_round(self, response, tool_results: List[Dict[str, Any]]):
        """Add a tool execution round to the conversation"""
        round_data = RoundData(
            round_number=len(self.rounds) + 1,
            assistant_message=response.content,
            tool_results=tool_results,
            response_type="tool_use",
        )
        self.rounds.append(round_data)

    def add_final_response(self, response):
        """Add the final response to complete the conversation"""
        self.final_response = response.content[0].text

    def get_final_response(self) -> str:
        """Get the final response for the user"""
        return self.final_response or "I wasn't able to generate a response."


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
- **search_course_content**: For searching specific course content and materials
- **get_course_outline**: For getting complete course outlines with lesson lists

Tool Usage Guidelines:
- **Sequential tool usage allowed**: You can make multiple tool calls across rounds to gather comprehensive information
- **Complex queries**: For questions requiring information from multiple courses or comparisons, use multiple searches
- **Multi-part questions**: Break down complex questions and search for different components
- **Maximum 2 rounds**: You have up to 2 rounds to gather information and provide a final answer
- **Tool strategy**: Use the first round to gather initial information, second round to fill gaps or compare information
- For course outline requests: Use get_course_outline to return course title, course link, and complete lesson list with numbers and titles
- For content questions: Use search_course_content for specific educational materials
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Tool Usage Examples:
- Compare courses: Search for one course, then search for another to compare
- Multi-part questions: Search for different aspects of a complex question
- Incomplete results: If initial search doesn't provide enough detail, search with different parameters

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer based on results
- **Complex comparisons**: Use multiple searches to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling across multiple rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """
        # Initialize conversation state
        conversation = ConversationState(
            initial_query=query, history=conversation_history, max_rounds=max_rounds
        )

        # Execute sequential rounds
        for round_num in range(1, max_rounds + 1):
            response = self._execute_round(
                conversation=conversation,
                round_number=round_num,
                tools=tools,
                tool_manager=tool_manager,
            )

            # Check termination conditions
            if self._should_terminate(response, round_num, max_rounds):
                break

        # If we completed max rounds and don't have a final response, make one final call
        if conversation.final_response is None and conversation.rounds:
            final_response = self._make_final_response_call(conversation)
            conversation.add_final_response(final_response)

        return conversation.get_final_response()

    def _execute_round(
        self,
        conversation: ConversationState,
        round_number: int,
        tools: Optional[List],
        tool_manager,
    ) -> Any:
        """
        Execute a single round of the conversation.

        Args:
            conversation: Current conversation state
            round_number: Current round number (1-based)
            tools: Available tools
            tool_manager: Tool execution manager

        Returns:
            API response object
        """
        # Build messages for this round
        messages = conversation.build_messages_for_round(round_number)

        # Build system content with round awareness
        system_content = self._build_system_content_for_round(
            conversation.history, round_number
        )

        # Prepare API parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Make API call
        response = self.client.messages.create(**api_params)

        # Process response
        if response.stop_reason == "tool_use" and tool_manager:
            # Execute tools and add to conversation state
            tool_results = self._execute_tools_with_error_handling(
                response, tool_manager
            )
            conversation.add_tool_round(response, tool_results)
            return response
        else:
            # Final response without tools
            conversation.add_final_response(response)
            return response

    def _should_terminate(self, response, round_num: int, max_rounds: int) -> bool:
        """
        Check if we should terminate the conversation rounds.

        Args:
            response: Current API response
            round_num: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if conversation should terminate
        """
        # Condition 1: Maximum rounds reached
        if round_num >= max_rounds:
            return True

        # Condition 2: No tool use in response
        if response.stop_reason != "tool_use":
            return True

        return False

    def _build_system_content_for_round(
        self, history: Optional[str], round_number: int
    ) -> str:
        """
        Build system content with round-specific context.

        Args:
            history: Conversation history
            round_number: Current round number

        Returns:
            System content string
        """
        base_content = self.SYSTEM_PROMPT

        if round_number > 1:
            base_content += f"\n\nThis is round {round_number} of your tool usage. You have already made tool calls in previous rounds. Use the previous results to inform your next actions."

        if history:
            base_content += f"\n\nPrevious conversation:\n{history}"

        return base_content

    def _execute_tools_with_error_handling(
        self, response, tool_manager
    ) -> List[Dict[str, Any]]:
        """
        Execute tools with comprehensive error handling.

        Args:
            response: API response containing tool calls
            tool_manager: Tool execution manager

        Returns:
            List of tool results with error handling
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                except Exception as e:
                    # Handle tool execution errors gracefully
                    error_message = f"Tool execution failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_message,
                            "is_error": True,
                        }
                    )

        return tool_results

    def _make_final_response_call(self, conversation: ConversationState) -> Any:
        """
        Make a final API call without tools to get the conclusive response.

        Args:
            conversation: Current conversation state

        Returns:
            Final API response
        """
        # Build final messages including all rounds
        messages = []

        # Add initial query
        messages.append({"role": "user", "content": conversation.initial_query})

        # Add all previous exchanges
        for round_data in conversation.rounds:
            messages.append(
                {"role": "assistant", "content": round_data.assistant_message}
            )
            if round_data.tool_results:
                messages.append({"role": "user", "content": round_data.tool_results})

        # Build system content for final response
        system_content = self._build_system_content_for_round(
            conversation.history, len(conversation.rounds) + 1
        )
        system_content += "\n\nFINAL ROUND: Based on all the information gathered above, provide your final comprehensive answer. No more tool calls are available."

        # Make final API call without tools
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        return self.client.messages.create(**api_params)

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

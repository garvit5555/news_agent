import os
import datetime
import sys
import re
import json
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pydantic import TypeAdapter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv(override=True)

# Load API keys directly from .env file
try:
    with open(".env", "r") as f:
        env_content = f.read()
        env_lines = env_content.strip().split('\n')
        env_vars = {}
        for line in env_lines:
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
                
    # Set API keys directly from parsed .env file
    OPENROUTER_API_KEY = env_vars.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
    TAVILY_API_KEY = env_vars.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
    
    print("API Keys loaded:")
    print("OPENROUTER_API_KEY:", OPENROUTER_API_KEY[:5] + "*****" if OPENROUTER_API_KEY else "Not found")
    print("TAVILY_API_KEY:", TAVILY_API_KEY[:5] + "*****" if TAVILY_API_KEY else "Not found")
except Exception as e:
    print(f"Error reading .env file: {e}")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Define available models
AVAILABLE_MODELS = {
    "deepseek-v3": {
        "id": "deepseek/deepseek-v3-base:free",
        "description": "DeepSeek V3 0324 - 685B parameter model, good for general tasks",
        "context_length": 131000,
    },
    "deepseek-r1": {
        "id": "deepseek/deepseek-r1",
        "description": "DeepSeek R1 - Reasoning-focused model on par with OpenAI o1",
        "context_length": 65000,
    },
    "deepseek-r1-free": {
        "id": "deepseek/deepseek-r1:free",
        "description": "Free version of DeepSeek R1 (limited rate)",
        "context_length": 163000,
    },
    "deepseek-distill-llama-8b": {
        "id": "deepseek/deepseek-r1-distill-llama-8b",
        "description": "DeepSeek R1 Distill Llama 8B - Smaller, faster model",
        "context_length": 32000,
    }
}

# Define the ToolCall model
class ToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to call")
    args: dict = Field(..., description="Arguments to pass to the tool")

class ManualToolAgent(Runnable):
    """
    A custom agent that handles tools manually with a graph-like workflow.
    """
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.json_parser = JsonOutputParser(pydantic_object=ToolCall)
        self.max_retries = 3
        self.max_iterations = 5  # Maximum number of tool calls in a chain
        self.last_tool_outputs = []  # Track tool outputs
        
    def convert_messages(self, messages: List[dict]) -> List[SystemMessage | HumanMessage | AIMessage]:
        """Convert dictionary-based messages to LangChain message objects."""
        converted_messages = []
        message_types = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage
        }
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role in message_types:
                MessageClass = message_types[role]
                converted_message = MessageClass(content=content)
                converted_messages.append(converted_message)
        return converted_messages
    
    def is_empty_response(self, response_text: str) -> bool:
        """Check if the response is empty or contains only whitespace."""
        if response_text is None:
            return True
        if not response_text.strip():
            return True
        return False
    
    def extract_json(self, text: str) -> Optional[str]:
        """Extract valid JSON from text, handling nested structures."""
        matches = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                    return match
            except json.JSONDecodeError:
                continue
        return None
    
    def looks_like_tool_call(self, response: str) -> bool:
        """Check if the response looks like it contains a tool call."""
        return bool(re.search(r'\{"tool":', response))
    
    def get_base_response(self, messages: List[SystemMessage | HumanMessage | AIMessage]) -> Optional[str]:
        """Get response from model with error handling."""
        try:
            response = self.model.invoke(messages)
            if response and hasattr(response, 'content'):
                return response.content
        except Exception as e:
            print(f"Error getting model response: {str(e)}")
        return None
    
    def invoke(self, inputs: dict) -> dict:
        """Execute the agent with graph-like workflow."""
        messages = inputs["messages"]
        user_query = messages[-1]["content"]
        converted_messages = self.convert_messages(messages)
        
        iteration_count = 0
        self.last_tool_outputs = []  # Reset tool outputs for new invocation
        
        while iteration_count < self.max_iterations:
            last_response = self.get_base_response(converted_messages)
            
            if not last_response or self.is_empty_response(last_response):
                break
            
            if not self.looks_like_tool_call(last_response):
                if self.last_tool_outputs:
                    final_prompt = {
                        "role": "user",
                        "content": f"""Based on these tool results:
                        {json.dumps(self.last_tool_outputs, indent=2)}
                        
                        Please provide a comprehensive response to the original query: {user_query}
                        Include relevant information from all tool outputs."""
                    }
                    converted_messages.append(HumanMessage(content=final_prompt["content"]))
                    final_response = self.get_base_response(converted_messages)
                    return {"messages": [{"content": final_response or last_response}]}
                return {"messages": [{"content": last_response}]}
            
            json_text = self.extract_json(last_response)
            if json_text:
                try:
                    parsed = json.loads(json_text)
                    tool_call = ToolCall(tool=parsed["tool"], args=parsed["args"])
                    
                    tool_dict = {tool.name: tool for tool in self.tools}
                    if tool_call.tool in tool_dict:
                        raw_result = tool_dict[tool_call.tool].invoke(tool_call.args)
                        self.last_tool_outputs.append({
                            "tool": tool_call.tool,
                            "args": tool_call.args,
                            "result": raw_result
                        })
                        
                        result_prompt = {
                            "role": "system",
                            "content": f"""Tool execution result:
                            {json.dumps(self.last_tool_outputs[-1], indent=2)}
                            
                            Based on this result, either:
                            1. Call another tool if more information is needed (respond with tool call JSON)
                            2. Provide a final response if you have enough information
                            
                            Remember to maintain the structured format of tool calls if choosing option 1."""
                        }
                        converted_messages.append(SystemMessage(content=result_prompt["content"]))
                        iteration_count += 1
                    continue
                        
                except Exception as e:
                    print(f"Error in tool execution: {str(e)}")
            
            break
        
        if self.last_tool_outputs:
            final_prompt = {
                "role": "user",
                "content": f"""Based on all collected information:
                {json.dumps(self.last_tool_outputs, indent=2)}
                
                Please provide a final response to: {user_query}"""
            }
            converted_messages.append(HumanMessage(content=final_prompt["content"]))
            final_response = self.get_base_response(converted_messages)
            if final_response:
                return {"messages": [{"content": final_response}]}
        
        return {"messages": [{"content": "I apologize, but I'm having trouble processing your request."}]}

def create_react_agent_taot(model, tools) -> ManualToolAgent:
    """Create a React agent with manual tool handling."""
    return ManualToolAgent(model, tools)

def create_system_message_taot(system_message: str) -> str:
    """Create a system message with tool instructions and JSON schema."""
    json_parser = JsonOutputParser(pydantic_object=ToolCall)
    
    sys_msg_taot = (f"{system_message}\n\n"
                    f"CRITICAL INSTRUCTIONS:\n"
                    f"1. When you determine an action is needed, output JSON like this without additional text:\n"
                    f'   {{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}\n'
                    f"2. The JSON MUST follow this schema:\n"
                    f"{json_parser.get_format_instructions()}")
    return sys_msg_taot

# Initialize tools
def write_to_file(content: str, filename: str) -> str:
    """Write content to a file"""
    try:
        safe_filename = filename.replace(" ", "_")
        with open(safe_filename, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Verify write was successful
        with open(safe_filename, 'r', encoding='utf-8') as f:
            saved_content = f.read()
            if not saved_content.strip():
                return f"Error: File {safe_filename} was created but is empty"
        return f"Successfully wrote content to {safe_filename}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def search_wrapper(query: str) -> str:
    """Format search results into a readable string."""
    try:
        search = TavilySearchResults(api_key=TAVILY_API_KEY)
        results = search.invoke({"query": query})
        
        formatted_results = []
        for result in results:
            title = result.get('title', 'No Title')
            url = result.get('url', 'No URL')
            snippet = result.get('content', 'No Content')
            formatted_results.append(f"\n## {title}\n\n{snippet}\n\nSource: {url}")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error during search: {str(e)}"

def get_llm(model_id: str) -> ChatOpenAI:
    """Get a language model instance."""
    return ChatOpenAI(
        model=model_id,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=1024
    )

def get_news(topic: str, model_name: str) -> dict:
    """Get news about a topic using the specified model with dynamic tool selection."""
    try:
        # Get model ID from available models
        if model_name not in AVAILABLE_MODELS:
            return {"error": f"Model {model_name} not found in available models"}
        
        model_id = AVAILABLE_MODELS[model_name]["id"]
        print(f"Using model: {model_id}")
        
        # Initialize LLM
        llm = get_llm(model_id)
        
        # Initialize tools with more flexible descriptions
        tools = [
            StructuredTool(
                name="search",
                func=search_wrapper,
                description="Search for news articles or any information. Use this tool when you need to find current information about any topic.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            ),
            StructuredTool(
                name="write_file",
                func=write_to_file,
                description="Write content to a file. Use this tool when you need to save information for later reference or when asked to save content.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The content to write"},
                        "filename": {"type": "string", "description": "The filename to write to"}
                    },
                    "required": ["content", "filename"]
                }
            )
        ]
        
        # Create a simpler system message that doesn't prescribe how to use tools
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_message = (
            "You are a helpful assistant with the ability to find information and save content to files. "
            f"Today's date is {current_date}. "
            "Your goal is to provide helpful and informative responses to the user's queries."
        )
        
        # Create agent
        agent = create_react_agent_taot(llm, tools)
        
        # Simplify the user message to just be the topic without prescribing actions
        user_message = f"I'm interested in: {topic}"
        
        # Let the agent handle the entire process
        messages = [
            {"role": "system", "content": create_system_message_taot(system_message)},
            {"role": "user", "content": user_message}
        ]
        
        # Let agent make all decisions about tool usage
        response = agent.invoke({"messages": messages})
        
        # Extract filename from tool outputs if available
        tool_outputs = []
        if hasattr(agent, 'last_tool_outputs'):
            tool_outputs = agent.last_tool_outputs
        
        filename = None
        for output in tool_outputs:
            if output["tool"] == "write_file":
                filename = output["args"].get("filename")
                if filename:  # Found a valid filename
                    break
        
        # If no filename was found, that's fine - let the model decide if file writing is needed
        
        result = {
            "result": "Success",
            "agent_response": response["messages"][0]["content"],
            "model_used": model_id,
            "tool_outputs": tool_outputs
        }
        
        if filename:
            result["filename"] = filename
            
        return result
        
    except Exception as e:
        return {"error": f"Error in get_news: {str(e)}"}

def main():
    """Main function to run the news search tool."""
    print("\n=== AI Assistant Tool ===\n")
    
    try:
        # Print available models
        print("Available DeepSeek Models:")
        for i, (name, details) in enumerate(AVAILABLE_MODELS.items(), 1):
            print(f"{i}. {name} - {details['description']}")
        
        # Get model choice
        while True:
            choice = input("\nSelect a model (1-4), or press Enter for default (DeepSeek V3): ").strip()
            if not choice:
                model_name = "deepseek-v3"
                break
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(AVAILABLE_MODELS):
                    model_name = list(AVAILABLE_MODELS.keys())[choice_num - 1]
                    break
                else:
                    print("Invalid choice. Please select a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter.")
        
        print(f"Selected model: {model_name} ({AVAILABLE_MODELS[model_name]['description']})\n")
        
        # Menu for search options
        while True:
            print("Search Options:")
            print("1. Ask about a specific topic")
            print("2. Get information on latest technology trends")
            print("3. Get world news")
            print("4. Get business news")
            print("5. Get sports news")
            print("6. Exit")
            
            try:
                choice = int(input("\nEnter your choice (1-6): "))
                
                if choice == 6:
                    print("\nThank you for using the AI Assistant Tool!")
                    break
                
                topic = ""
                if choice == 1:
                    topic = input("\nWhat would you like to know about? ")
                elif choice == 2:
                    topic = "the latest technology trends and innovations"
                elif choice == 3:
                    topic = "the latest world news and global events"
                elif choice == 4:
                    topic = "recent business and economic developments"
                elif choice == 5:
                    topic = "recent sports news and updates"
                else:
                    print("Invalid choice. Please select a number between 1 and 6.")
                    continue
                
                if topic:
                    print(f"\nProcessing request...")
                    result = get_news(topic, model_name)
                    
                    if "error" in result:
                        print(f"\nError: {result['error']}")
                    else:
                        print("\nRequest processed successfully!")
                        
                        # Show response first
                        print("\nResponse:")
                        print("------------------------")
                        print(result["agent_response"])
                        print("------------------------")
                        
                        # Then show file info if a file was created
                        if "filename" in result:
                            print(f"\nContent has been saved to: {result['filename']}")
                            view_file = input("\nWould you like to view the saved file? (y/n): ").lower()
                            if view_file == 'y':
                                try:
                                    with open(result['filename'], 'r', encoding='utf-8') as f:
                                        print("\nFile contents:")
                                        print("------------------------")
                                        print(f.read())
                                        print("------------------------")
                                except Exception as e:
                                    print(f"Error reading file: {str(e)}")
                
                print()  # Add blank line for readability
                
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 6.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()



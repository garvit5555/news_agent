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
    A custom agent that handles tools manually.
    """
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.json_parser = JsonOutputParser(pydantic_object=ToolCall)
        self.base_executor = create_react_agent(model, tools=[])
        self.max_retries = 3
        self.timeout = 30  # Add timeout for API calls
    
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
    
    def format_tool_result(self, tool_name: str, tool_result: str, user_query: str) -> str:
        """Format tool result using LLM to create natural language response."""
        try:
            prompt = f"""Given the following:
                        User query: {user_query}
                        Tool used: {tool_name}
                        Tool result: {tool_result}

                        Create a natural language response that incorporates the result.
                        Keep it concise and direct. Do not mention the tool used."""
            
            response = self.model.invoke([HumanMessage(content=prompt)])
            return response.content if not self.is_empty_response(response.content) else tool_result
        except Exception as e:
            print(f"Error formatting result: {str(e)}")
            return tool_result
    
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
    
    def get_base_response(self, messages: List[SystemMessage | HumanMessage | AIMessage]) -> Optional[str]:
        """Get response from base executor with timeout handling."""
        try:
            response = self.base_executor.invoke({"messages": messages})
            if response and "messages" in response and response["messages"]:
                return response["messages"][-1].content
        except Exception as e:
            print(f"Error getting base response: {str(e)}")
        return None
    
    def invoke(self, inputs: dict) -> dict:
        """Execute the agent with manual tool handling."""
        messages = inputs["messages"]
        user_query = messages[-1]["content"]
        
        # Convert messages to LangChain format
        converted_messages = self.convert_messages(messages)
        
        # Get response from base executor with retry logic
        retry_count = 0
        while retry_count < self.max_retries:
            # Use the base ReAct agent to get the initial response
            last_response = self.get_base_response(converted_messages)
            
            if last_response and not self.is_empty_response(last_response):
                json_text = self.extract_json(last_response)
                if json_text:
                    try:
                        parsed = json.loads(json_text)
                        tool_call = ToolCall(tool=parsed["tool"], args=parsed["args"])
                        
                        tool_dict = {tool.name: tool for tool in self.tools}
                        if tool_call.tool in tool_dict:
                            raw_result = tool_dict[tool_call.tool].invoke(tool_call.args)
                            # Format the result using LLM for natural language
                            formatted_result = self.format_tool_result(tool_call.tool, raw_result, user_query)
                            return {"messages": [{"content": formatted_result}]}
                    except Exception as e:
                        print(f"Error processing tool call (attempt {retry_count + 1}): {str(e)}")
            
            # If we get here, either the response was empty or JSON parsing failed
            retry_count += 1
            if last_response:
                converted_messages.append(AIMessage(content=last_response))
            converted_messages.append(HumanMessage(content="Please respond with a valid JSON object following the schema exactly."))
        
        # If all retries failed, return an error message
        return {"messages": [{"content": "I apologize, but I'm having trouble processing your request. Please try rephrasing your question."}]}

def create_react_agent_taot(model, tools) -> ManualToolAgent:
    """Create a React agent with manual tool handling."""
    return ManualToolAgent(model, tools)

def create_system_message_taot(system_message: str) -> str:
    """Create a system message with tool instructions and JSON schema."""
    json_parser = JsonOutputParser(pydantic_object=ToolCall)
    
    sys_msg_taot = (f"{system_message}\n\n"
                    f"CRITICAL INSTRUCTIONS:\n"
                    f"1. You MUST respond in English only\n"
                    f"2. For search queries, you MUST output JSON exactly like this:\n"
                    f'   {{"tool": "search", "args": {{"query": "search query here"}}}}\n'
                    f"3. For file writing, you MUST output JSON exactly like this:\n"
                    f'   {{"tool": "write_file", "args": {{"content": "content here", "filename": "filename.txt"}}}}\n'
                    f"4. DO NOT include any other text before or after the JSON\n"
                    f"5. DO NOT modify the JSON structure\n\n"
                    f"The JSON MUST follow this schema:\n"
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
    """Get news about a topic using the specified model."""
    try:
        # Get model ID from available models
        if model_name not in AVAILABLE_MODELS:
            return {"error": f"Model {model_name} not found in available models"}
        
        model_id = AVAILABLE_MODELS[model_name]["id"]
        print(f"Using model: {model_id}")
        
        # Initialize LLM
        llm = get_llm(model_id)
        
        # Initialize tools
        tools = [
            StructuredTool(
                name="search",
                func=search_wrapper,
                description="Search for news articles. Takes a query string as argument.",
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
                description="Write content to a file. Takes content and filename as arguments.",
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
        
        # Create system message
        system_message = (
            "You are a helpful news search assistant. Your task is to search for news articles "
            "and save them to a file. When searching for news, use the search tool. "
            "When you have the results, use the write_file tool to save them. "
            "The filename should include the topic and today's date."
        )
        
        # Create agent
        agent = create_react_agent_taot(llm, tools)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": create_system_message_taot(system_message)},
            {"role": "user", "content": f"Search for news articles about: {topic}"}
        ]
        
        # Get search results
        print(f"Searching for news about: {topic}")
        search_response = agent.invoke({"messages": messages})
        
        if "error" in search_response:
            return search_response
        
        search_result = search_response["messages"][0]["content"]
        print("\nFormatted News Articles:")
        print("------------------------")
        print(search_result)
        print("------------------------\n")
        
        # Save results to file
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        safe_topic = topic.lower().replace(" ", "_")
        filename = f"{safe_topic}_{today}.txt"
        
        # Prepare save messages
        save_messages = [
            {"role": "system", "content": create_system_message_taot(system_message)},
            {"role": "user", "content": f'Save the following content to a file named "{filename}":\n\n{search_result}'}
        ]
        
        # Save to file
        save_response = agent.invoke({"messages": save_messages})
        
        if "error" in save_response:
            return save_response
        
        return {
            "result": "Success",
            "filename": filename,
            "agent_response": search_result,
            "model_used": model_id
        }
    
    except Exception as e:
        return {"error": f"Error in get_news: {str(e)}"}

def main():
    """Main function to run the news search tool."""
    print("\n=== News Search Tool ===\n")
    
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
        print("1. Search for a specific topic")
        print("2. Search for latest technology trends")
        print("3. Search for world news")
        print("4. Search for business news")
        print("5. Search for sports news")
        print("6. Exit")
        
        try:
            choice = int(input("\nEnter your choice (1-6): "))
            
            if choice == 6:
                print("\nThank you for using the News Search Tool!")
                break
                
            if choice == 1:
                topic = input("\nEnter a news topic to search for: ")
            elif choice == 2:
                topic = "latest technology trends and innovations"
            elif choice == 3:
                topic = "latest world news and global events"
            elif choice == 4:
                topic = "latest business and economic news"
            elif choice == 5:
                topic = "latest sports news and updates"
            else:
                print("Invalid choice. Please select a number between 1 and 6.")
                continue
    
            print(f"\nFetching news about: {topic}")
            result = get_news(topic, model_name)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nNews articles have been saved to: {result['filename']}")
    
            # Ask if user wants to view the file
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

if __name__ == "__main__":
    main()

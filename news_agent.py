import os
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.file_management import WriteFileTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize LLM with OpenAI configured to use OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o",  
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# Initialize tools
search_tool = TavilySearchResults(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

# Custom function for the WriteFile tool to make it more robust
def write_to_file(file_path_and_text):
    """Write text to a file.
    
    Args:
        file_path_and_text: String in the format "file_path::text_content"
            where file_path is the path to the file and text_content is the text to write.
    """
    try:
        # Split the input string at the first occurrence of "::"
        parts = file_path_and_text.split("::", 1)
        if len(parts) != 2:
            return "Error: Input should be in the format 'file_path::text_content'"
        
        file_path, text = parts
        
        # Clean up the file path
        file_path = file_path.strip()
        
        # Write to the file
        with open(file_path, 'w') as file:
            file.write(text)
        
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

# Define tools in the format expected by the agent
tools = [
    Tool(
        name="TavilySearch",
        func=search_tool.invoke,
        description="Search for information about a topic. Returns the latest news and information. Usage: TavilySearch(\"search query\")"
    ),
    Tool(
        name="WriteFile",
        func=write_to_file,
        description="Write text to a file. Usage: WriteFile(\"file_path::text_content\"). The text before '::' is the file path, and everything after '::' is the content to write to the file."
    )
]

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create a ReAct agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

def get_news_and_save(topic):
    """
    Gets the latest news on a topic and saves it to a file using a ReAct agent.
    
    Args:
        topic (str): The news topic to search for
    """
    # Generate current date for filename
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{topic.replace(' ', '_')}_{current_date}.txt"
    
    # Run the agent
    result = agent.invoke(
        {"input": f"""Search for the latest news about {topic} and save them to a file named {filename}.
        
        Format the results with headlines, sources, and brief summaries.
        
        The text should be formatted in markdown for better readability.
        """}
    )
    
    print(f"Task completed. Check {filename} for the news.")
    return result

if __name__ == "__main__":
    # Example usage
    topic = input("Enter a news topic to search for: ")
    get_news_and_save(topic)
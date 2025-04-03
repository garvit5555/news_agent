# News Agent with LangChain

This is a ReAct agent built with LangChain that can:
1. Search the web for the latest news on any topic using Tavily Search
2. Save the news to a text file in the local filesystem

## Requirements

- Python 3.8+
- OpenRouter API key (to access GPT-4o)
- Tavily API key

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Add your API keys to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

Run the script:
```
python news_agent.py
```

You will be prompted to enter a news topic. The agent will:
1. Search for the latest news about the topic
2. Format the information
3. Save it to a text file in the current directory with the format `topic_YYYY-MM-DD.txt`

## Example

```
Enter a news topic to search for: artificial intelligence
```

This will create a file named `artificial_intelligence_2023-06-15.txt` (with the current date) containing the latest news about AI.

## How It Works

The agent uses:
- GPT-4o via OpenRouter as the LLM
- LangChain's ReAct agent framework
- Tavily Search tool for retrieving information from the web
- FileSystem tools to write files

The agent follows a reasoning and action process to complete the task, explaining its thought process along the way. 
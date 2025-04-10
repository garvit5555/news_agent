# News Search Tool

A powerful news search tool that uses DeepSeek models through OpenRouter to search for and save news articles. The tool uses a ReAct agent for intelligent search and content formatting.

## Features

- Multiple DeepSeek model options:
  - DeepSeek V3 (685B parameters)
  - DeepSeek R1 (Reasoning-focused)
  - DeepSeek R1 Free (Limited rate)
  - DeepSeek R1 Distill Llama 8B (Smaller, faster)
- Intelligent search using Tavily API
- Automatic content formatting and file saving
- Natural language responses
- Error handling and retry logic
- Custom tool integration with ReAct agent

## Prerequisites

- Python 3.9+
- OpenRouter API key
- Tavily API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd news-search-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

Run the tool:
```bash
python news_agent.py
```

The tool provides several options:
1. Search for a specific topic
2. Search for latest technology trends
3. Search for world news
4. Search for business news
5. Search for sports news

Results are saved to a file with the format: `topic_YYYY-MM-DD.txt`

## Architecture

The tool uses a custom ReAct agent implementation that:
1. Takes user input and converts it to a search query
2. Uses Tavily API to search for relevant news
3. Formats the results using the language model
4. Saves the formatted content to a file

### Key Components

- `ManualToolAgent`: Custom agent for tool handling
- `ToolCall`: Pydantic model for structured tool calls
- Search and file writing tools
- Natural language formatting

## Error Handling

The tool includes:
- API error handling
- Retry logic for failed requests
- File writing verification
- Input validation
- Timeout handling

## Contributing

Feel free to submit issues and enhancement requests.

## License

[MIT License](LICENSE)

## Acknowledgments

- DeepSeek for their powerful language models
- OpenRouter for API access
- Tavily for news search capabilities
- LangChain for the agent framework 

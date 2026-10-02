# GenAI Flask App

A Flask app that extends the hands-on activity in the Coursera course [develop-generative-ai-applications-get-started](https://www.coursera.org/learn/develop-generative-ai-applications-get-started/). It manages multiple AI agents, each configured with different system prompts. Users can switch between specialized roles to tackle different tasks.

## Features
- Flask application with AI capabilities
- CRUD operations on agents persisted in SQLite via SQLAlchemy
- Integrate and compare multiple language models (e.g., Llama, Granite, Mistral)
- Agents have memory through LangChain's checkpointers (thread-level persistence)
- Agents within the same chat share the same memory graph
- Agents use additional kwargs for storing metadata in the graph memory (e.g., human and AI request timestamps)

## Technologies used
- Python (Flask, LangChain)
- SQLite
- Flask-SQLAlchemy
- HTML, CSS, JS

## Installation & setup

- Set up API keys
```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
```

- Activate the virtual environment
```bash
source ./.venv/bin/activate
```

- Install the dependencies
```bash
uv sync
```

- Run the project (runs on port 5000)
```bash
uv run genai
```

- Open in Browser
Open `http://127.0.0.1:5000/` in your browser

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Knowledge Graph Builder application that transforms unstructured data (PDFs, DOCs, TXT, YouTube videos, web pages) into structured Knowledge Graphs stored in Neo4j using Large Language Models (LLMs) and the LangChain framework.

## Architecture

The project consists of two main components:

### Frontend (React/TypeScript)
- **Location**: `/frontend/`
- **Technology**: React 18 + TypeScript + Vite
- **UI Framework**: Neo4j Design Language (@neo4j-ndl/react)
- **Key Features**: File upload, graph visualization, chat interface, various data source integrations

### Backend (Python/FastAPI)
- **Location**: `/backend/`
- **Technology**: FastAPI + Python
- **Main Entry**: `score.py` (contains FastAPI app)
- **Key Modules**:
  - `src/main.py`: Core graph generation logic
  - `src/QA_integration.py`: Chat/Q&A functionality
  - `src/graphDB_dataAccess.py`: Neo4j database operations
  - `src/llm.py`: LLM integrations (OpenAI, Gemini, Diffbot, etc.)
  - `src/communities.py`: Graph community detection
  - `src/post_processing.py`: Vector indexes and graph cleanup

## Common Development Commands

### Frontend Commands
```bash
cd frontend
yarn                    # Install dependencies
yarn dev               # Start development server (runs on 0.0.0.0)
yarn build             # Build for production (includes TypeScript compilation)
yarn lint              # Run ESLint
yarn format           # Format code with Prettier
```

### Backend Commands
```bash
cd backend
python -m venv envName                    # Create virtual environment
source envName/bin/activate               # Activate virtual environment (Unix/macOS)
pip install -r requirements.txt          # Install dependencies
uvicorn score:app --reload              # Start development server
```

### Testing
- Backend tests: `test_commutiesqa.py`, `test_integrationqa.py`
- Performance testing: `Performance_test.py`, `locustperf.py`

## Key Integrations

### LLM Models Supported
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini
- Diffbot
- Azure OpenAI
- Anthropic Claude
- Fireworks
- Groq
- Amazon Bedrock
- Ollama (local)
- Other OpenAI-compatible models

### Data Sources
- Local file uploads
- YouTube videos
- Wikipedia
- AWS S3
- Google Cloud Storage
- Web scraping

### Database
- **Neo4j Database**: Version 5.23+ required
- **Connection**: Configurable via environment variables or UI login
- **Features**: APOC procedures, vector search, full-text search

## Environment Configuration

The application uses extensive environment configuration:

### Frontend (.env in /frontend/)
- `VITE_BACKEND_API_URL`: Backend API URL
- `VITE_LLM_MODELS_PROD`: Available LLM models
- `VITE_REACT_APP_SOURCES`: Enabled data sources
- `VITE_CHAT_MODES`: Available chat modes
- Authentication settings (Auth0, Google OAuth)

### Backend (.env in /backend/)
- Database: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- LLM API Keys: `OPENAI_API_KEY`, `DIFFBOT_API_KEY`, etc.
- Processing settings: `MAX_TOKEN_CHUNK_SIZE`, `DUPLICATE_SCORE_VALUE`
- Optional: `LANGCHAIN_API_KEY` for tracing

## Development Workflow

1. **Local Development**: Run frontend and backend separately using the commands above
2. **Docker**: Use `docker-compose.yml` for containerized deployment
3. **Cloud Deployment**: Supports Google Cloud Run deployment

## Code Structure

### Frontend Components
- `src/components/`: Reusable UI components
- `src/components/ChatBot/`: Chat interface components
- `src/components/Graph/`: Graph visualization components
- `src/components/DataSources/`: Data source integration components
- `src/context/`: React context providers

### Backend Modules
- `src/shared/`: Common utilities and constants
- `src/`: Core business logic modules
- Root level: FastAPI application and test files
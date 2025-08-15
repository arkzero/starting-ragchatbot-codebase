# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start using provided script
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- Requires Python 3.13+ and the `uv` package manager

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system for querying course materials using semantic search and AI responses.

### Core Components

**Backend Structure** (`/backend/`):
- `app.py` - FastAPI web server with CORS and static file serving
- `rag_system.py` - Main orchestrator connecting all components
- `vector_store.py` - ChromaDB integration for semantic search
- `ai_generator.py` - Anthropic Claude API integration
- `document_processor.py` - Text chunking and course extraction
- `session_manager.py` - Conversation history management
- `search_tools.py` - Tool-based search capabilities
- `models.py` - Pydantic data models
- `config.py` - Configuration management

**Frontend Structure** (`/frontend/`):
- Simple HTML/CSS/JS interface served as static files
- Communicates with backend via `/api/query` and `/api/courses` endpoints

**Data Flow**:
1. Documents in `/docs/` are processed into chunks on startup
2. Chunks are stored in ChromaDB with embeddings using sentence-transformers
3. User queries trigger semantic search to find relevant chunks
4. Claude AI generates responses using retrieved context and conversation history
5. Tool-based search system allows AI to dynamically query the vector store

### Key Dependencies
- **FastAPI** - Web framework with automatic OpenAPI docs at `/docs`
- **ChromaDB** - Vector database for semantic search
- **Anthropic** - Claude AI for response generation
- **sentence-transformers** - Text embeddings for semantic search
- **uvicorn** - ASGI server for FastAPI

### Configuration
The system loads course documents from the `/docs/` folder automatically on startup. Supported formats: PDF, DOCX, TXT.

### API Endpoints
- `POST /api/query` - Submit questions and get AI responses with sources
- `GET /api/courses` - Get course statistics and analytics
- Web interface served at root `/`
- Always use uv to run the server do not use pip directly
- use uv to manage all depdencies
- use uv to run Python files
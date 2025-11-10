# LLM Graph Builder Pipeline - Detailed Workflow

The pipeline transforms PDFs and other documents into a Neo4j knowledge graph through the following stages:

## 1. Document Ingestion & Source Node Creation

When a PDF is uploaded via `/extract` endpoint (`backend/src/main.py:230-244`), the system:

- Creates a **Document node** in Neo4j with metadata (filename, size, status, timestamps)
- Supports multiple sources: local files, S3, GCS, web pages, YouTube, Wikipedia
- Uses PyMuPDF and unstructured library for PDF parsing (`backend/src/document_sources/local_file.py`)

### Libraries Used for Document Ingestion

**PDF Processing:**
- **PyMuPDF** (`fitz`, `PyMuPDF==1.26.1`): Primary PDF reader
- **PyPDF2** (`PyPDF2==3.0.1`): Fallback PDF reader
- **Unstructured** (`unstructured==0.17.2`): Advanced document parsing with strategies:
  - `partition_pdf()` with strategies: "hi_res", "fast", "ocr_only", "auto"
  - Handles complex layouts, tables, and OCR

**Other Document Types:**
- **LangChain Document Loaders** (`langchain-community==0.3.25`):
  - `WikipediaLoader`: Wikipedia content extraction
  - `WebBaseLoader`: Web page scraping
- **youtube-transcript-api** (`youtube-transcript-api==1.1.0`): YouTube transcript extraction
- **wikipedia** (`wikipedia==1.4.0`): Wikipedia API integration

**Cloud Storage:**
- **boto3** (`boto3==1.38.36`): AWS S3 document fetching
- **google-cloud-core** (`google-cloud-core==2.4.3`): Google Cloud Storage access

## 2. Document Chunking

The `CreateChunksofDocument` class (`backend/src/create_chunks.py:12-55`):

- Uses LangChain's `TokenTextSplitter` for tokenization
- **Hyperparameters:**
  - `token_chunk_size`: Default configurable, typically 200-1000 tokens
  - `chunk_overlap`: Overlap between chunks (default 20 tokens)
  - `MAX_TOKEN_CHUNK_SIZE`: 10000 tokens (environment variable)
- Creates Chunk nodes with relationships:
  - `FIRST_CHUNK`: Document → first chunk
  - `NEXT_CHUNK`: Sequential chunk connections
  - `PART_OF`: All chunks → parent Document

### Libraries Used for Document Chunking

- **langchain-text-splitters** (`langchain-text-splitters==0.3.8`):
  - `TokenTextSplitter` class for consistent tokenization
  ```python
  from langchain_text_splitters import TokenTextSplitter
  text_splitter = TokenTextSplitter(chunk_size=token_chunk_size, chunk_overlap=chunk_overlap)
  ```
- **langchain-core** (`langchain-core==0.3.65`):
  - `Document` class for chunk metadata management

## 3. Embedding Generation

The `create_chunk_embeddings` function (`backend/src/make_relationships.py:41-65`):

- **Embedding models supported:**
  - OpenAI Ada (1536 dimensions)
  - VertexAI textembedding-gecko (768 dimensions)
  - HuggingFace all-MiniLM-L6-v2 (384 dimensions)
  - AWS Bedrock Titan (1536 dimensions)
- Stores embeddings as properties on Chunk nodes
- Creates vector index for similarity search

### Libraries Used for Embedding Creation

Location: `backend/src/shared/common_fn.py:72-93`

- **langchain-openai** (`langchain-openai==0.3.23`):
  - `OpenAIEmbeddings` class
  - Model: text-embedding-ada-002 (1536 dimensions)
  
- **langchain-google-vertexai** (`langchain-google-vertexai==2.0.25`):
  - `VertexAIEmbeddings` class
  - Model: textembedding-gecko@003 (768 dimensions)
  
- **langchain-aws** (`langchain-aws==0.2.25`):
  - `BedrockEmbeddings` class
  - Model: Amazon Titan (1536 dimensions)
  
- **langchain-huggingface** (`langchain-huggingface==0.3.0`) + **sentence-transformers** (`sentence-transformers==4.1.0`):
  - `HuggingFaceEmbeddings` class for local embeddings
  - Model: all-MiniLM-L6-v2 (384 dimensions)

## 4. LLM Entity/Relationship Extraction

The `get_graph_from_llm` function (`backend/src/llm.py:212-251`):

- **Chunk combination:** Combines multiple chunks (default 5, configurable via `chunks_to_combine`)
- **LLM providers supported:**
  - OpenAI (GPT-3.5, GPT-4, GPT-4o)
  - Google Gemini (1.0-pro, 1.5-pro, 1.5-flash)
  - Anthropic Claude
  - Groq, Fireworks, Ollama, AWS Bedrock
  - Diffbot (specialized graph extraction)
- Uses LangChain's `LLMGraphTransformer` (`backend/src/llm.py:196-204`):
  - Extracts entities with properties: `["description"]`
  - Extracts relationships with properties: `["description"]`
  - Supports schema constraints via `allowedNodes` and `allowedRelationships`
  - Temperature: 0 (deterministic extraction)

### Libraries Used for LLM Entity/Relationship Extraction

Location: `backend/src/llm.py:21-129`

**Core Extraction Framework:**
- **langchain-experimental** (`langchain-experimental==0.3.4`):
  - `LLMGraphTransformer` class for structured extraction
  - `DiffbotGraphTransformer` for Diffbot API integration

**LLM Provider Libraries:**
- **langchain-openai** (`langchain-openai==0.3.23`):
  - `ChatOpenAI` for OpenAI models (GPT-3.5, GPT-4, GPT-4o)
  - `AzureChatOpenAI` for Azure OpenAI deployments

- **langchain-google-vertexai** (`langchain-google-vertexai==2.0.25`):
  - `ChatVertexAI` for Gemini models
  - Includes safety settings for content filtering

- **langchain-anthropic** (`langchain-anthropic==0.3.15`):
  - `ChatAnthropic` for Claude models

- **langchain-groq** (`langchain-groq==0.3.2`):
  - `ChatGroq` for Groq-hosted models

- **langchain-fireworks** (`langchain-fireworks==0.3.0`):
  - `ChatFireworks` for Fireworks AI

- **langchain-aws** (`langchain-aws==0.2.25`):
  - `ChatBedrock` for AWS Bedrock models

- **langchain-community** (`langchain-community==0.3.25`):
  - `ChatOllama` for local Ollama models
  - `GraphDocument` class for structured graph representation

## 5. Graph Construction in Neo4j

The `save_graphDocuments_in_neo4j` function (`backend/src/shared/common_fn.py:95-109`):

- Creates Entity nodes with labels from extracted types
- Creates relationships between entities
- Handles deadlock retries (max 3 attempts)
- Creates `HAS_ENTITY` relationships: Chunk → Entity
- Uses batch processing with `UNWIND` for performance

### Libraries Used for Neo4j Integration

- **langchain-neo4j** (`langchain-neo4j==0.4.0`):
  - `Neo4jGraph` class for database operations
  - `add_graph_documents()` method for batch insertion
  - `Neo4jVector` for vector similarity operations
- **neo4j-rust-ext** (`neo4j-rust-ext==5.28.1.0`):
  - Performance optimizations for Neo4j operations

## 6. Processing Flow Control

Batch processing (`backend/src/main.py:369-411`):

- `UPDATE_GRAPH_CHUNKS_PROCESSED`: Process 20 chunks at a time (default)
- Updates Document node with progress metrics:
  - `processed_chunk`: Current progress
  - `total_chunks`: Total chunks to process
  - `nodeCount`: Total entities extracted
  - `relationshipCount`: Total relationships created

## 7. Post-Processing Features

- **Graph cleanup:** Optional model for post-processing refinement
- **Community detection:** Creates hierarchical communities (up to `MAX_COMMUNITY_LEVELS`)
- **Similarity relationships:** KNN-based `SIMILAR` edges between chunks

## Key Configuration Parameters

### Environment Variables
- `MAX_TOKEN_CHUNK_SIZE`: Maximum tokens per chunk (default: 10000)
- `UPDATE_GRAPH_CHUNKS_PROCESSED`: Batch size for processing (default: 20)
- `NUMBER_OF_CHUNKS_TO_COMBINE`: Chunks to combine for LLM (default: 5)
- `EMBEDDING_MODEL`: Choice of embedding model (openai/vertexai/titan/default)
- `IS_EMBEDDING`: Enable/disable embedding generation (TRUE/FALSE)

### API Parameters
- `token_chunk_size`: Size of each text chunk in tokens
- `chunk_overlap`: Number of overlapping tokens between chunks
- `chunks_to_combine`: Number of chunks to process together
- `allowedNodes`: Comma-separated list of allowed entity types
- `allowedRelationships`: Triplets of (source, relation, target)

## Performance Optimizations

- Deadlock handling with retry logic (max 3 retries, 2-second delays)
- Batch processing with `UNWIND` queries
- Vector indexing for similarity search
- Configurable chunk processing batches
- Connection pooling for Neo4j
- Asynchronous processing with `asyncio`
- Duplicate query execution for Neo4j consistency

## API Framework

- **FastAPI** (`fastapi==0.115.12`): Modern async web framework
- **Uvicorn** (`uvicorn==0.34.3`): ASGI server
- **Gunicorn** (`gunicorn==23.0.0`): Production WSGI server
- **SSE-Starlette** (`sse-starlette==2.3.6`): Server-sent events support

## Pipeline Orchestration

The entire pipeline is orchestrated through the `processing_source` function (`backend/src/main.py:297-464`), which handles the complete workflow from document ingestion to graph storage, with detailed latency tracking at each stage:

1. **Document Upload** → Source Node Creation
2. **Text Extraction** → Document parsing with appropriate library
3. **Chunking** → TokenTextSplitter creates overlapping chunks
4. **Embedding** → Vector representations stored on chunks
5. **LLM Processing** → Entity/relationship extraction from combined chunks
6. **Graph Storage** → Batch insertion into Neo4j with relationships
7. **Post-Processing** → Optional community detection and similarity links

Each stage includes error handling, retry logic, and progress tracking to ensure robust processing of documents into knowledge graphs.
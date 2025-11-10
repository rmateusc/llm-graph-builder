# Simplified PDF to Knowledge Graph Pipeline

A streamlined, modular pipeline for converting PDF documents into knowledge graphs stored in Neo4j using Large Language Models (LLMs).

## Pipeline Overview

This simplified pipeline follows these key steps:

1. **PDF Processing**: Load and extract text from PDF files
2. **Text Chunking**: Split documents into manageable chunks
3. **Entity & Relationship Extraction**: Use LLMs to extract entities and relationships
4. **Graph Construction**: Build the knowledge graph in Neo4j
5. **Post-Processing**: Enhance the graph with deduplication, communities, and embeddings

## Project Structure

```
simplified_pipeline/
├── extractors/
│   ├── pdf_processor.py      # PDF loading and text extraction
│   └── entity_extractor.py   # LLM-based entity/relationship extraction
├── processors/
│   ├── text_chunker.py       # Document chunking strategies
│   └── post_processor.py     # Graph enhancements and cleanup
├── graph/
│   └── graph_builder.py      # Neo4j graph construction
├── pipeline.py               # Main orchestrator
├── example_usage.py          # Usage examples
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# LLM Configuration (choose your provider)
OPENAI_API_KEY=your-openai-key
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from pipeline import PDFToGraphPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    chunk_size=2000,
    chunk_overlap=200
)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Create pipeline
pipeline = PDFToGraphPipeline(config=config, llm=llm)

# Process PDF
results = pipeline.process_pdf("path/to/document.pdf", clear_existing=True)
print(f"Created {results['final_node_count']} nodes and {results['final_relationship_count']} relationships")
```

## Module Details

### 1. PDF Processor (`extractors/pdf_processor.py`)
- Uses PyMuPDF for PDF parsing
- Extracts text while preserving page structure
- Adds metadata (page numbers, source file)

### 2. Text Chunker (`processors/text_chunker.py`)
- Configurable chunk size and overlap
- Recursive character splitting for optimal chunks
- Preserves context between chunks

### 3. Entity Extractor (`extractors/entity_extractor.py`)
- LLM-based extraction using LangChain's LLMGraphTransformer
- Configurable entity and relationship types
- Combines extractions from multiple chunks

### 4. Graph Builder (`graph/graph_builder.py`)
- Neo4j connection management
- Node and relationship creation
- Constraint and index management
- Graph statistics

### 5. Post Processor (`processors/post_processor.py`)
- Duplicate node merging
- Community detection
- Vector embeddings for similarity search
- Graph cleanup and metadata enrichment

## Customization Points

### 1. Modify Entity Extraction
Edit `extractors/entity_extractor.py` to:
- Change extraction prompts
- Add custom entity types
- Implement different extraction strategies

### 2. Adjust Chunking Strategy
Edit `processors/text_chunker.py` to:
- Use different splitting methods
- Implement semantic chunking
- Add chunk preprocessing

### 3. Enhance Graph Construction
Edit `graph/graph_builder.py` to:
- Add custom node/relationship properties
- Implement different merge strategies
- Add graph validation

### 4. Extend Post-Processing
Edit `processors/post_processor.py` to:
- Add custom graph algorithms
- Implement different deduplication logic
- Add more metadata enrichment

## Configuration Options

```python
PipelineConfig(
    # Neo4j settings
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",

    # Chunking settings
    chunk_size: int = 2000,
    chunk_overlap: int = 200,

    # Entity extraction settings
    allowed_nodes: List[str] = None,  # e.g., ["Person", "Organization"]
    allowed_relationships: List[str] = None,  # e.g., ["WORKS_FOR"]
    node_properties: List[str] = None,
    relationship_properties: List[str] = None,

    # Post-processing settings
    enable_embeddings: bool = False,
    enable_communities: bool = True,
    merge_duplicates: bool = True,
    similarity_threshold: float = 0.95
)
```

## Processing Multiple PDFs

```python
pipeline = PDFToGraphPipeline(config=config, llm=llm)

pdf_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = pipeline.process_multiple_pdfs(pdf_paths, clear_existing=True)
```

## Neo4j Requirements

- Neo4j 5.0+
- Optional: APOC procedures for advanced features
- Optional: Graph Data Science library for community detection

## Tips for Modification

1. **Start with the pipeline.py**: This is the main orchestrator that ties everything together
2. **Test modules independently**: Each module can be imported and tested separately
3. **Add logging**: Modules use Python's logging for debugging
4. **Monitor costs**: Entity extraction uses LLM calls which incur costs
5. **Batch processing**: Process multiple chunks together to reduce API calls

## Common Modifications

### Using Different LLMs
```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Local Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### Custom Entity Types
```python
config = PipelineConfig(
    allowed_nodes=["Person", "Company", "Product", "Technology"],
    allowed_relationships=["WORKS_FOR", "PRODUCES", "USES"],
    ...
)
```

### Disable Post-Processing
```python
config = PipelineConfig(
    enable_embeddings=False,
    enable_communities=False,
    merge_duplicates=False,
    ...
)
```

## Troubleshooting

- **Memory issues**: Reduce chunk_size or process PDFs sequentially
- **Slow extraction**: Use a faster/cheaper LLM model
- **Neo4j connection**: Ensure Neo4j is running and credentials are correct
- **Missing dependencies**: Install optional dependencies as needed

## License

This is a simplified version of the main llm-graph-builder project, provided for educational and customization purposes.
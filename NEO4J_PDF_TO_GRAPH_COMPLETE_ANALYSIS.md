# Complete Pipeline Analysis: PDF to Neo4j Knowledge Graph

**Date**: 2025-11-09
**Repository**: llm-graph-builder
**Scope**: Backend pipeline analysis (excludes simplified_pipeline/)

---

## Table of Contents

1. [Overview](#overview)
2. [PDF Parsing](#1-pdf-parsing)
3. [Text Chunking](#2-text-chunking)
4. [Entity & Relationship Extraction](#3-entity--relationship-extraction)
5. [Data Standardization & Deduplication](#4-data-standardization--deduplication)
6. [Embeddings Usage](#5-embeddings-usage)
7. [Graph Traversal & RAG](#6-graph-traversal--rag)
8. [Complete Pipeline Flow](#complete-pipeline-flow)
9. [Key Configuration Parameters](#key-configuration-parameters)

---

## Overview

This document provides a comprehensive analysis of how the LLM Graph Builder transforms unstructured PDF documents into structured Neo4j knowledge graphs, and how those graphs are queried using Retrieval-Augmented Generation (RAG).

**Technology Stack:**
- **Backend**: Python, FastAPI
- **LLM Framework**: LangChain
- **Database**: Neo4j 5.23+
- **PDF Processing**: PyMuPDF, PyPDF2
- **Embeddings**: HuggingFace, OpenAI, VertexAI, Bedrock

---

## 1. PDF Parsing

### Libraries Used

**File**: `backend/requirements.txt`

- **PyMuPDF (fitz)** version 1.26.1 - Primary PDF parser
- **PyPDF2** version 3.0.1 - Backup PDF parser
- **Unstructured** version 0.17.2 - General document loader with PDF support

### Implementation

**File**: `backend/src/document_sources/local_file.py:23-61`

```python
def load_document_content(file_path):
    file_extension = Path(file_path).suffix.lower()
    encoding_flag = False

    if file_extension == '.pdf':
        loader = PyMuPDFLoader(file_path)  # Uses PyMuPDF as primary
        return loader, encoding_flag
    elif file_extension == ".txt":
        # Text file handling with encoding detection
        encoding = detect_encoding(file_path)
        if encoding.lower() == "utf-8":
            loader = UnstructuredFileLoader(file_path, mode="elements",
                                          autodetect_encoding=True)
        else:
            with open(file_path, encoding=encoding, errors="replace") as f:
                content = f.read()
            loader = ListLoader([Document(page_content=content,
                                        metadata={"source": file_path})])
            encoding_flag = True
        return loader, encoding_flag
    else:
        loader = UnstructuredFileLoader(file_path, mode="elements",
                                       autodetect_encoding=True)
        return loader, encoding_flag
```

### Key Features

- **PyMuPDFLoader** extracts text page-by-page with metadata
- Returns LangChain `Document` objects with `page_content` and `metadata`
- Preserves page numbers for reference tracking
- Handles various document formats (PDF, TXT, DOCX, etc.)

---

## 2. Text Chunking

### Configuration

**File**: `backend/src/create_chunks.py:17-55`

**Environment Variables:**
- `MAX_TOKEN_CHUNK_SIZE`: Default 10,000 tokens (maximum total tokens from file)
- `token_chunk_size`: Configurable per-request (typical: 1000-2000 tokens)
- `chunk_overlap`: Configurable overlap between chunks (typical: 100-200 tokens)

### Chunking Algorithm

**File**: `backend/src/create_chunks.py:17-55`

```python
class CreateChunksofDocument:
    def split_file_into_chunks(self, token_chunk_size, chunk_overlap):
        text_splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=chunk_overlap
        )
        MAX_TOKEN_CHUNK_SIZE = int(os.getenv('MAX_TOKEN_CHUNK_SIZE', 10000))
        chunk_to_be_created = int(MAX_TOKEN_CHUNK_SIZE / token_chunk_size)

        # For PDFs with page metadata
        if 'page' in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                if len(chunks) >= chunk_to_be_created:
                    break
                for chunk in text_splitter.split_documents([document]):
                    chunks.append(Document(
                        page_content=chunk.page_content,
                        metadata={'page_number': page_number}
                    ))

        # For YouTube videos with timestamps
        elif 'length' in self.pages[0].metadata:
            chunks_without_time_range = text_splitter.split_documents([self.pages[0]])
            chunks = get_calculated_timestamps(
                chunks_without_time_range[:chunk_to_be_created],
                youtube_id
            )

        # For other document types
        else:
            chunks = text_splitter.split_documents(self.pages)

        chunks = chunks[:chunk_to_be_created]
        return chunks
```

### Chunking Strategy

1. **Token-based splitting**: Uses LangChain's `TokenTextSplitter` (not character-based)
2. **Sentence boundary respect**: Attempts to split at natural boundaries
3. **Configurable overlap**: Maintains context between chunks (100-200 tokens typical)
4. **Chunk limit**: Enforces maximum total chunks based on `MAX_TOKEN_CHUNK_SIZE`
5. **Metadata preservation**: Maintains page numbers, timestamps, or other source metadata

### Chunk Graph Structure

**File**: `backend/src/make_relationships.py:67-155`

```python
def create_relation_between_chunks(graph, file_name, chunks):
    # Creates Chunk nodes with properties:
    # - id: SHA1 hash of content
    # - text: chunk content
    # - position: sequential position (1, 2, 3...)
    # - length: character count
    # - content_offset: byte offset in original document
    # - page_number: (if PDF)
    # - embedding: (added later)

    # Relationships created:
    # Document-[:FIRST_CHUNK]->Chunk  (first chunk only)
    # Chunk-[:NEXT_CHUNK]->Chunk      (sequential chunks)
    # Chunk-[:PART_OF]->Document      (all chunks)
```

**Graph Structure:**
```
Document
  ├─[:FIRST_CHUNK]→ Chunk[0]
  │                    ├─[:NEXT_CHUNK]→ Chunk[1]
  │                    │                    ├─[:NEXT_CHUNK]→ Chunk[2]
  │                    │                    │                    └─...
  │                    ├─[:PART_OF]→ Document
  │                    ├─[:HAS_ENTITY]→ Entity
  │                    └─[:SIMILAR]→ Chunk (optional KNN-based)
```

---

## 3. Entity & Relationship Extraction

### LLM-Based Extraction

**File**: `backend/src/llm.py:212-251`

The system uses LangChain's `LLMGraphTransformer` for entity extraction:

```python
async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes,
                             allowedRelationship, chunks_to_combine,
                             additional_instructions=None):
    llm, model_name = get_llm(model)

    # Combine multiple chunks for better context
    combined_chunk_document_list = get_combined_chunks(
        chunkId_chunkDoc_list,
        chunks_to_combine
    )

    # Parse allowed nodes and relationships
    allowed_nodes = [node.strip() for node in allowedNodes.split(',')
                    if node.strip()]

    allowed_relationships = []
    if allowedRelationship:
        items = [item.strip() for item in allowedRelationship.split(',')
                if item.strip()]
        # Format: source, relation, target (triplets)
        for i in range(0, len(items), 3):
            source, relation, target = items[i:i + 3]
            allowed_relationships.append((source, relation, target))

    # Create transformer with configuration
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["description"],
        relationship_properties=["description"],
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        ignore_tool_usage=ignore_tool_usage,
        additional_instructions=ADDITIONAL_INSTRUCTIONS +
            (additional_instructions if additional_instructions else "")
    )

    # Extract entities and relationships
    graph_document_list = await llm_transformer.aconvert_to_graph_documents(
        combined_chunk_document_list
    )
    return graph_document_list
```

### Extraction Prompts

**File**: `backend/src/shared/constants.py:891-893`

```python
ADDITIONAL_INSTRUCTIONS = """Your goal is to identify and categorize entities
while ensuring that specific data types such as dates, numbers, revenues, and
other non-entity information are not extracted as separate nodes. Instead,
treat these as properties associated with the relevant entities."""
```

### LLMGraphTransformer Behavior

- Uses **structured output / function calling** for entity extraction
- Extracts nodes with: `id`, `type` (label), `description` (optional)
- Extracts relationships with: `source`, `target`, `type`, `description` (optional)
- Validates against `allowed_nodes` and `allowed_relationships`
- Uses the LLM to infer semantic connections between entities

### Chunk Combination Strategy

**File**: `backend/src/llm.py:140-164`

```python
def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    # Combines N sequential chunks into single documents for extraction
    # Default chunks_to_combine: typically 3-5

    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    # Tracks which original chunks contributed to combined chunk
    combined_chunks_ids = [
        [document["chunk_id"]
         for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    # ... creates combined documents ...
```

**Purpose**: Combining chunks provides the LLM with more context, reducing entity fragmentation and improving relationship detection across chunk boundaries.

**Trade-off**: Larger context windows but fewer total extractions per document (due to token limits).

---

## 4. Data Standardization & Deduplication

The system implements a **multi-layered deduplication strategy** to ensure entity consistency.

### Layer 1: Write-Time Deduplication (MERGE)

**File**: `backend/src/make_relationships.py:17-38`

```python
def merge_relationship_between_chunk_and_entites(graph, graph_documents_chunk_chunk_Id):
    # Uses APOC merge to deduplicate entities
    unwind_query = """
        UNWIND $batch_data AS data
        MATCH (c:Chunk {id: data.chunk_id})
        CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
        MERGE (c)-[:HAS_ENTITY]->(n)
    """
```

**Key Points:**
- `apoc.merge.node()` creates or matches existing nodes by ID and type
- Entities with same ID and type are automatically merged
- No explicit deduplication logic needed - Neo4j handles it via MERGE

### Layer 2: Cleaning Extracted Data

**File**: `backend/src/shared/common_fn.py:111-129`

```python
def handle_backticks_nodes_relationship_id_type(graph_document_list):
    for graph_document in graph_document_list:
        # Clean node id and types
        cleaned_nodes = []
        for node in graph_document.nodes:
            if node.type.strip() and node.id.strip():
                node.type = node.type.replace('`', '')  # Remove backticks
                cleaned_nodes.append(node)

        # Clean relationship types
        cleaned_relationships = []
        for rel in graph_document.relationships:
            if rel.type.strip() and rel.source.id.strip() and rel.target.id.strip():
                rel.type = rel.type.replace('`', '')
                rel.source.type = rel.source.type.replace('`', '')
                rel.target.type = rel.target.type.replace('`', '')
                cleaned_relationships.append(rel)
```

### Layer 3: Post-Processing Duplicate Detection

**File**: `backend/src/graphDB_dataAccess.py:398-446`

```python
def get_duplicate_nodes_list(self):
    score_value = float(os.environ.get('DUPLICATE_SCORE_VALUE'))  # Default: 0.8-0.9
    text_distance = int(os.environ.get('DUPLICATE_TEXT_DISTANCE'))  # Default: 3

    query_duplicate_nodes = """
        MATCH (n:!Chunk&!Session&!Document&!`__Community__`)
        WHERE n.embedding is not null and n.id is not null
        WITH n ORDER BY count { (n)--() } DESC, size(toString(n.id)) DESC
        WITH collect(n) as nodes
        UNWIND nodes as n
        WITH n, [other in nodes
            WHERE elementId(n) < elementId(other) and labels(n) = labels(other)
            AND (
                -- Strategy 1: String containment
                (size(toString(other.id)) > 2 AND
                 toLower(toString(n.id)) CONTAINS toLower(toString(other.id))) OR
                (size(toString(n.id)) > 2 AND
                 toLower(toString(other.id)) CONTAINS toLower(toString(n.id)))

                -- Strategy 2: Edit distance (typos/variations)
                OR (size(toString(n.id))>5 AND
                    apoc.text.distance(toLower(toString(n.id)),
                                      toLower(toString(other.id))) < $duplicate_text_distance)

                -- Strategy 3: Embedding similarity
                OR vector.similarity.cosine(other.embedding, n.embedding) > $duplicate_score_value
            )] as similar
        WHERE size(similar) > 0
        ...
    """
```

**Deduplication Strategies:**

1. **String Containment**: "Apple Inc" contains "Apple"
2. **Edit Distance**: Levenshtein distance < 3 (catches typos/variations)
3. **Embedding Similarity**: Cosine similarity > 0.8-0.9 (semantic equivalence)
4. **Label Matching**: Only compares entities with same labels

**Priority System**: Keeps nodes with:
- More relationships (higher degree)
- Longer IDs (more specific names)

### Layer 4: Node Merging

**File**: `backend/src/graphDB_dataAccess.py:448-466`

```python
def merge_duplicate_nodes(self, duplicate_nodes_list):
    query = """
        UNWIND $rows AS row
        CALL { with row
            MATCH (first) WHERE elementId(first) = row.firstElementId
            MATCH (rest) WHERE elementId(rest) IN row.similarElementIds
            WITH first, collect(rest) as rest
            WITH [first] + rest as nodes
            CALL apoc.refactor.mergeNodes(nodes,
                {properties:"discard", mergeRels:true, produceSelfRel:false,
                 preserveExistingSelfRels:false, singleElementAsArray:true})
            YIELD node
            RETURN size(nodes) as mergedCount
        }
        RETURN sum(mergedCount) as totalMerged
    """
```

**Merge Behavior:**
- Combines multiple entity nodes into one
- Merges all relationships (incoming and outgoing)
- Discards duplicate properties (keeps first node's properties)
- Prevents self-relationships

### Layer 5: Schema Consolidation

**File**: `backend/src/post_processing.py:199-236`

```python
def graph_schema_consolidation(graph):
    # Uses LLM to consolidate similar node labels and relationship types
    # Example: "Person", "Human", "People" -> "Person"
    #          "WORKS_AT", "WORKS_FOR" -> "WORKS_AT"

    node_labels, relation_labels = graphDb_data_Access.get_nodelabels_relationships()

    # Prompt with examples
    chain = prompt | llm | parser
    nodes_relations_input = {'nodes': node_labels, 'relationships': relation_labels}
    mappings = chain.invoke({'input': nodes_relations_input})

    # Apply mappings
    for old_label, new_label in node_mapping.items():
        query = f"""
            MATCH (n:`{old_label}`)
            SET n:`{new_label}`
            REMOVE n:`{old_label}`
        """
        graph.query(query)
```

**Purpose**: Normalizes schema variations that emerge from LLM extraction inconsistencies.

---

## 5. Embeddings Usage

Embeddings serve **three critical purposes** in the system:

1. **Vector similarity search during RAG**
2. **Duplicate entity detection**
3. **Adaptive graph traversal decisions**

### Embedding Models

**File**: `backend/src/shared/common_fn.py:72-93`

```python
def load_embedding_model(embedding_model_name: str):
    if embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
    elif embedding_model_name == "vertexai":
        embeddings = VertexAIEmbeddings(model="textembedding-gecko@003")
        dimension = 768
    elif embedding_model_name == "titan":
        embeddings = get_bedrock_embeddings()
        dimension = 1536
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        dimension = 384  # DEFAULT
```

**Default Model**: HuggingFace `all-MiniLM-L6-v2` (384 dimensions)

### Chunk Embeddings

**File**: `backend/src/make_relationships.py:41-65`

```python
def create_chunk_embeddings(graph, chunkId_chunkDoc_list, file_name):
    isEmbedding = os.getenv('IS_EMBEDDING')  # Default: TRUE
    embeddings, dimension = EMBEDDING_FUNCTION, EMBEDDING_DIMENSION

    data_for_query = []
    for row in chunkId_chunkDoc_list:
        if isEmbedding.upper() == "TRUE":
            embeddings_arr = embeddings.embed_query(row['chunk_doc'].page_content)
            data_for_query.append({
                "chunkId": row['chunk_id'],
                "embeddings": embeddings_arr
            })

    query = """
        UNWIND $data AS row
        MATCH (d:Document {fileName: $fileName})
        MERGE (c:Chunk {id: row.chunkId})
        SET c.embedding = row.embeddings
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query, params={"data": data_for_query, "fileName": file_name})
```

**What's Embedded**: Full chunk text content

### Entity Embeddings

**File**: `backend/src/post_processing.py:172-197`

```python
def create_entity_embedding(graph):
    rows = fetch_entities_for_embedding(graph)
    for i in range(0, len(rows), 1000):
        update_embeddings(rows[i:i+1000], graph)

def fetch_entities_for_embedding(graph):
    query = """
        MATCH (e)
        WHERE NOT (e:Chunk OR e:Document OR e:`__Community__`)
          AND e.embedding IS NULL
          AND e.id IS NOT NULL
        RETURN elementId(e) AS elementId,
               e.id + " " + coalesce(e.description, "") AS text
    """
    # Embeds: entity ID + description (if available)
```

**What's Embedded**: `entity.id + " " + entity.description`

**Purpose**: Creates semantic representation for entity-based retrieval and duplicate detection.

### Vector Indexes

**File**: `backend/src/post_processing.py:40-75`

```cypher
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (c:Chunk) ON c.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }
}
```

**Created Indexes:**
- `vector`: Chunk embeddings (for RAG retrieval)
- `entity_vector`: Entity embeddings (for entity-based search)
- `community_vector`: Community embeddings (for global search - if community detection enabled)

---

## 6. Graph Traversal & RAG

### RAG Architecture Overview

**File**: `backend/src/QA_integration.py:656-691`

The system supports **6 different chat modes** with different retrieval strategies:

```python
CHAT_MODE_CONFIG_MAP = {
    "vector": {
        retrieval_query: VECTOR_SEARCH_QUERY,
        index_name: "vector",
        top_k: 5
    },
    "fulltext": {
        retrieval_query: VECTOR_SEARCH_QUERY,
        index_name: "vector",
        keyword_index: "keyword",  # Hybrid: vector + fulltext
        top_k: 5
    },
    "graph_vector": {
        retrieval_query: VECTOR_GRAPH_SEARCH_QUERY,  # Includes graph traversal
        index_name: "vector",
        top_k: 5
    },
    "graph_vector_fulltext": {  # DEFAULT MODE
        retrieval_query: VECTOR_GRAPH_SEARCH_QUERY,
        index_name: "vector",
        keyword_index: "keyword",
        top_k: 5
    },
    "entity_vector": {
        retrieval_query: LOCAL_COMMUNITY_SEARCH_QUERY,
        index_name: "entity_vector",
        top_k: 10
    },
    "global_vector": {
        retrieval_query: GLOBAL_VECTOR_SEARCH_QUERY,
        index_name: "community_vector",
        top_k: 10
    }
}
```

### Mode 1: Pure Vector Search

**File**: `backend/src/shared/constants.py:310-332`

```cypher
VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d,
     collect(distinct {chunk: chunk, score: score}) AS chunks,
     avg(score) AS avg_score

WITH d, avg_score,
     [c IN chunks | c.chunk.text] AS texts,
     [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails

WITH d, avg_score, chunkdetails,
     apoc.text.join(texts, "\n----\n") AS text

RETURN text,
       avg_score AS score,
       {source: d.fileName, chunkdetails: chunkdetails} AS metadata
"""
```

**Process:**
1. Vector similarity search on Chunk embeddings
2. Group chunks by document
3. Average similarity scores
4. Return combined chunk text

**Use Case**: Simple text-based retrieval without graph structure.

### Mode 2: Vector + Adaptive Graph Traversal (DEFAULT)

**File**: `backend/src/shared/constants.py:341-513`

This is the most sophisticated mode. Abbreviated version:

```cypher
VECTOR_GRAPH_SEARCH_QUERY = """
-- Step 1: Vector search on chunks
WITH node as chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score

-- Step 2: Fetch entities from chunks
CALL { WITH chunks
    UNWIND chunks as chunkScore
    WITH chunkScore.chunk as chunk
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks
    ORDER BY numChunks DESC
    LIMIT 40  -- Top 40 entities

    -- Step 3: ADAPTIVE GRAPH TRAVERSAL based on embedding similarity
    WITH
    CASE
        -- Medium similarity: traverse 1 hop
        WHEN e.embedding IS NULL OR
             (0.3 <= cosine(query_embedding, e.embedding) <= 0.9) THEN
            collect {
                MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk)
                RETURN path LIMIT 20
            }

        -- High similarity: traverse 2 hops
        WHEN cosine(query_embedding, e.embedding) > 0.9 THEN
            collect {
                MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk)
                RETURN path LIMIT 40
            }

        -- Low similarity: no traversal
        ELSE
            collect { MATCH path=(e) RETURN path }
    END AS paths, e

    -- Step 4: Deduplicate and collect nodes/relationships
    WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
         collect(DISTINCT e) AS entities
    RETURN
        collect {
            UNWIND paths AS p UNWIND relationships(p) AS r
            RETURN DISTINCT r
        } AS rels,
        collect {
            UNWIND paths AS p UNWIND nodes(p) AS n
            RETURN DISTINCT n
        } AS nodes,
        entities
}

-- Step 5: Format output with entities and relationships
WITH d, avg_score, chunks, nodes, rels, entities
WITH d, avg_score,
    [c IN chunks | c.chunk.text] AS texts,
    [n IN nodes | n.id + " (" + coalesce(n.description, "") + ")"] AS nodeTexts,
    [r IN rels | startNode(r).id + " " + type(r) + " " + endNode(r).id] AS relTexts

WITH d, avg_score,
    "Text Content:\n" + apoc.text.join(texts, "\n----\n") +
    "\n----\nEntities:\n" + apoc.text.join(nodeTexts, "\n") +
    "\n----\nRelationships:\n" + apoc.text.join(relTexts, "\n") AS text

RETURN text, avg_score AS score, {source: d.fileName, ...} AS metadata
"""
```

**Key Innovation: Adaptive Traversal Depth**

| Embedding Similarity | Traversal Strategy | Purpose |
|---------------------|-------------------|---------|
| < 0.3 (Low) | No traversal (entity only) | Avoid irrelevant graph exploration |
| 0.3 - 0.9 (Medium) | 1-hop traversal | Fetch immediate neighbors |
| > 0.9 (High) | 2-hop traversal | Extended neighborhood exploration |

**This adaptive approach provides richer context without overwhelming the LLM with irrelevant information.**

**Output Format:**
```
Text Content:
[Chunk text 1]
----
[Chunk text 2]
----
Entities:
Apple Inc (A multinational technology company)
Steve Jobs (Co-founder of Apple Inc)
----
Relationships:
Steve Jobs FOUNDED Apple Inc
Apple Inc LOCATED_IN Cupertino
```

### Mode 3: Entity-Based Search

**File**: `backend/src/shared/constants.py:521-613`

Abbreviated version:

```cypher
LOCAL_COMMUNITY_SEARCH_QUERY = """
WITH collect(node) AS nodes, avg(score) AS score

WITH score, nodes,
     -- Get top 3 chunks mentioning these entities
     collect {
         UNWIND nodes AS n
         MATCH (n)<-[:HAS_ENTITY]->(c:Chunk)
         WITH c, count(distinct n) AS freq
         RETURN c ORDER BY freq DESC LIMIT 3
     } AS chunks,

     -- Get top 3 communities these entities belong to
     collect {
         UNWIND nodes AS n
         OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
         WITH c, c.community_rank AS rank, c.weight AS weight
         RETURN c ORDER BY rank, weight DESC LIMIT 3
     } AS communities,

     -- Get relationships between entities
     collect {
         UNWIND nodes AS n
         UNWIND nodes AS m
         MATCH (n)-[r]->(m)
         RETURN DISTINCT r
     } AS rels,

     -- Get top 10 outside relationships
     collect {
         UNWIND nodes AS n
         MATCH path = (n)-[r]-(m:__Entity__)
         WHERE NOT m IN nodes
         WITH m, collect(distinct r) AS rels, count(*) AS freq
         ORDER BY freq DESC LIMIT 10
         RETURN {nodes: collect(m), rels: apoc.coll.flatten(collect(rels))}
     } AS outside

RETURN {
  chunks: [c IN chunks | c.text],
  communities: [c IN communities | c.summary],
  entities: [n IN nodes | n.id + " " + coalesce(n.description, "")],
  relationships: [r IN rels | startNode(r).id + " " + type(r) + " " + endNode(r).id],
  outside: {...}
} AS text, score, {entities: metadata} AS metadata
"""
```

**Process:**
1. Vector search on entity embeddings (not chunks)
2. Fetch chunks mentioning those entities
3. Fetch community summaries (if available)
4. Extract relationships between entities
5. Include top outside connections

**Use Case**: When query is about specific entities rather than general concepts.

### Question Processing Chain

**File**: `backend/src/QA_integration.py:293-333`

```python
def create_document_retriever_chain(llm, retriever):
    # Step 1: Transform question based on chat history
    query_transform_prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_TRANSFORM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Step 2: Split and filter retrieved documents
    splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0)
    embeddings_filter = EmbeddingsFilter(
        embeddings=EMBEDDING_FUNCTION,
        similarity_threshold=0.10  # Filter out low-relevance chunks
    )

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, embeddings_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=retriever
    )

    # Step 3: Chain: transform question -> retrieve -> compress
    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,  # First message
            (lambda x: x["messages"][-1].content) | compression_retriever,
        ),
        # Subsequent messages: transform with history
        query_transform_prompt | llm | output_parser | compression_retriever,
    )

    return query_transforming_retriever_chain
```

**Question Transform Template:**
```
Given the below conversation, generate a search query to look up in order to
get information relevant to the conversation. Only respond with the query,
nothing else.
```

**Example Transformation:**
```
User: "What did Apple announce?"
[...conversation...]
User: "When was it released?"

Transformed Query: "When was Apple's announcement released?"
```

### RAG Response Generation

**File**: `backend/src/QA_integration.py:228-270`

```python
def process_documents(docs, question, messages, llm, model, chat_mode_settings):
    # Format retrieved documents
    formatted_docs, sources, entitydetails, communities = format_documents(
        docs, model, chat_mode_settings
    )

    # Create RAG chain with system template
    rag_chain = get_rag_chain(llm=llm)

    # Generate response
    ai_response = rag_chain.invoke({
        "messages": messages[:-1],  # Chat history
        "context": formatted_docs,   # Retrieved context
        "input": question            # Current question
    })

    return ai_response.content, result, total_tokens, formatted_docs
```

### System Prompt Template

**File**: `backend/src/shared/constants.py:263-303`

```
You are an AI-powered question-answering agent. Your task is to provide
accurate and comprehensive responses based on the given context, chat history,
and available resources.

Response Guidelines:
1. Direct Answers: Provide clear and thorough answers without headers unless requested
2. Utilize History and Context: Leverage previous interactions and provided context
3. Admit Unknowns: Clearly state if an answer is unknown
4. Avoid Hallucination: Only provide information based on context provided
5. Keep responses concise (4-5 sentences unless more detail requested)

Context:
{context}

IMPORTANT: DO NOT ANSWER FROM YOUR KNOWLEDGE BASE. USE THE CONTEXT PROVIDED.
```

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      1. PDF UPLOAD & PARSING                        │
│  PyMuPDFLoader → LangChain Documents (page_content + metadata)      │
│  Location: backend/src/document_sources/local_file.py               │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      2. TEXT CHUNKING                               │
│  TokenTextSplitter(chunk_size=1000-2000, overlap=100-200)           │
│  • Creates Chunk nodes with SHA1 IDs                                │
│  • Relationships: FIRST_CHUNK, NEXT_CHUNK, PART_OF                  │
│  Location: backend/src/create_chunks.py                             │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  3. CHUNK EMBEDDING GENERATION                      │
│  HuggingFace all-MiniLM-L6-v2 (384 dim) or OpenAI/VertexAI          │
│  • Embeds each chunk's text content                                 │
│  • Stores in Chunk.embedding property                               │
│  • Creates vector index for similarity search                       │
│  Location: backend/src/make_relationships.py:41-65                  │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 4. ENTITY & RELATIONSHIP EXTRACTION                 │
│  Batch Processing: Combine 3-5 chunks → LLMGraphTransformer         │
│  • Extracts entities (nodes with id, type, description)             │
│  • Extracts relationships (source, target, type, description)       │
│  • Validates against allowed_nodes/allowed_relationships            │
│  Location: backend/src/llm.py:212-251                               │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    5. ENTITY DEDUPLICATION                          │
│  Layer 1 - MERGE: apoc.merge.node([type], {id})                     │
│            → Same ID + type = automatic merge                       │
│  Layer 2 - Cleaning: Remove backticks, validate IDs                 │
│  Layer 3 - Similarity: String containment, edit distance,           │
│            embedding similarity (cosine > 0.8)                      │
│  Layer 4 - Merging: apoc.refactor.mergeNodes()                      │
│  Layer 5 - Schema: LLM-based label normalization                    │
│            → "Person"/"Human"/"People" → "Person"                   │
│  Locations: backend/src/make_relationships.py:17-38                 │
│             backend/src/graphDB_dataAccess.py:398-466               │
│             backend/src/post_processing.py:199-236                  │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 6. ENTITY EMBEDDING GENERATION                      │
│  For each entity: embed(entity.id + " " + entity.description)       │
│  • Used for duplicate detection (Layer 3 above)                     │
│  • Used for entity-based vector search                              │
│  • Creates entity_vector index                                      │
│  Location: backend/src/post_processing.py:172-197                   │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 7. GRAPH STRUCTURE FINALIZATION                     │
│  Neo4j Graph:                                                       │
│    Document ←[PART_OF]─ Chunk ─[HAS_ENTITY]→ Entity                │
│             ←[FIRST_CHUNK]─ Chunk                                   │
│             Chunk ←[NEXT_CHUNK]→ Chunk                              │
│             Chunk ←[SIMILAR]→ Chunk (optional, KNN-based)           │
│             Entity ─[various]→ Entity                               │
│             Entity ─[IN_COMMUNITY]→ Community (if computed)         │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     8. RAG QUERY PROCESSING                         │
│  User Question → Transform with chat history                        │
│      ↓                                                              │
│  Multi-Mode Retrieval (6 modes):                                    │
│    • vector: Top-K chunks by similarity                             │
│    • fulltext: Hybrid vector + keyword search                       │
│    • graph_vector: Vector + adaptive graph traversal (DEFAULT)      │
│    • graph_vector_fulltext: Hybrid graph traversal                  │
│    • entity_vector: Entity-based search + communities               │
│    • global_vector: Community-level search                          │
│      ↓                                                              │
│  Contextual Compression:                                            │
│    • Re-split to 3000 tokens                                        │
│    • Filter by embedding similarity > 0.10                          │
│      ↓                                                              │
│  LLM Response Generation:                                           │
│    • Context: Retrieved chunks + entities + relationships           │
│    • History: Previous conversation                                 │
│    • Instruction: Answer from context only                          │
│  Location: backend/src/QA_integration.py                            │
│            backend/src/shared/constants.py (retrieval queries)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Configuration Parameters

### Chunking Configuration
```python
MAX_TOKEN_CHUNK_SIZE = 10000        # Maximum total tokens processed per document
token_chunk_size = 1000-2000        # Individual chunk size (configurable)
chunk_overlap = 100-200             # Overlap between chunks
```

### Extraction Configuration
```python
chunks_to_combine = 3-5             # Chunks combined for entity extraction
ADDITIONAL_INSTRUCTIONS = "..."     # Prompt guidance for extraction
allowed_nodes = ["Person", "Organization", ...]  # Allowed entity types
allowed_relationships = [("Person", "WORKS_AT", "Organization"), ...]
```

### Embedding Configuration
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default HuggingFace model
EMBEDDING_DIMENSION = 384              # Default dimension
IS_EMBEDDING = "TRUE"                  # Enable/disable embeddings
```

### Deduplication Configuration
```python
DUPLICATE_SCORE_VALUE = 0.8-0.9     # Embedding similarity threshold
DUPLICATE_TEXT_DISTANCE = 3         # Levenshtein distance threshold
```

### RAG Configuration
```python
# Retrieval
VECTOR_SEARCH_TOP_K = 5             # Number of chunks retrieved
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 40  # Max entities in graph search

# Adaptive traversal thresholds
LOW_SIMILARITY_THRESHOLD = 0.3      # Below: no traversal
HIGH_SIMILARITY_THRESHOLD = 0.9     # Above: 2-hop traversal

# Compression
CHAT_DOC_SPLIT_SIZE = 3000          # Re-split size for compression
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD = 0.10  # Filter threshold
```

### Environment Variables (backend/.env)

```bash
# Database
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM
OPENAI_API_KEY=sk-...
DIFFBOT_API_KEY=...

# Chunking
MAX_TOKEN_CHUNK_SIZE=10000

# Embeddings
IS_EMBEDDING=TRUE

# Deduplication
DUPLICATE_SCORE_VALUE=0.85
DUPLICATE_TEXT_DISTANCE=3

# LangChain (optional)
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

---

## Key Architectural Insights

### Why This Design Works Well

1. **Chunk Combination Strategy**
   - Combining 3-5 chunks during extraction reduces entity fragmentation
   - Provides LLM with sufficient context to detect cross-chunk relationships
   - Trade-off: Fewer extractions but higher quality

2. **Multi-Layer Deduplication**
   - Write-time MERGE catches exact duplicates
   - Post-processing catches semantic duplicates (embeddings)
   - Schema consolidation normalizes LLM inconsistencies
   - Result: Clean, consistent knowledge graph

3. **Adaptive Graph Traversal**
   - Only explores graph when semantically relevant (similarity > 0.3)
   - Depth increases with relevance (1-hop vs 2-hop)
   - Prevents context overload while maintaining richness
   - Key innovation for graph-based RAG

4. **Triple-Purpose Embeddings**
   - Same embeddings serve retrieval, deduplication, and traversal
   - Efficient use of embedding computation
   - Consistent semantic representation across pipeline

5. **Contextual Compression**
   - Re-filters retrieved context before LLM generation
   - Removes low-relevance sections (similarity < 0.10)
   - Maximizes signal-to-noise ratio in final prompt
   - Critical for maintaining response quality

6. **Multi-Mode Retrieval**
   - Different modes for different query types
   - graph_vector_fulltext (default) balances coverage and precision
   - entity_vector mode for entity-centric queries
   - Flexibility without mode-switching complexity

---

## Performance Considerations

### Bottlenecks

1. **Entity Extraction**: Most expensive operation (LLM calls per combined chunk)
2. **Embedding Generation**: O(n) for chunks and entities
3. **Graph Traversal**: Can be expensive for highly connected entities (mitigated by adaptive depth)
4. **Duplicate Detection**: O(n²) comparisons (optimized with embeddings)

### Optimizations

1. **Batch Processing**: Entities and embeddings processed in batches (1000 at a time)
2. **Vector Indexes**: Efficient similarity search (HNSW algorithm)
3. **Entity Limit**: Max 40 entities in graph traversal
4. **Path Limits**: Max 20-40 paths per entity depending on similarity
5. **Chunk Limit**: MAX_TOKEN_CHUNK_SIZE prevents unbounded processing

### Scalability

- **Small documents (< 100 pages)**: Processes in seconds
- **Large documents (> 500 pages)**: May take minutes (extraction bottleneck)
- **Graph queries**: Sub-second for most retrieval modes (thanks to vector indexes)

---

## Common Use Cases

### 1. Technical Documentation
- **Best Mode**: `graph_vector_fulltext`
- **Why**: Technical docs have many interconnected concepts (APIs, classes, functions)
- **Graph Value**: HIGH (relationships between components critical)

### 2. Legal Documents
- **Best Mode**: `entity_vector`
- **Why**: Focus on specific entities (parties, clauses, dates)
- **Graph Value**: MEDIUM (relationships important but entity-centric)

### 3. Research Papers
- **Best Mode**: `graph_vector_fulltext`
- **Why**: Citations, methodologies, and findings are interconnected
- **Graph Value**: HIGH (methodology flows and citation networks)

### 4. News Articles
- **Best Mode**: `vector` or `fulltext`
- **Why**: Standalone articles with minimal cross-references
- **Graph Value**: LOW (simple keyword/vector search sufficient)

### 5. Knowledge Bases
- **Best Mode**: `global_vector` (if communities computed)
- **Why**: Hierarchical organization benefits from community detection
- **Graph Value**: VERY HIGH (community-level summaries powerful)

---

## Future Improvements

### Potential Enhancements

1. **Hierarchical Chunking**: Multi-level chunks (paragraphs → sections → documents)
2. **Incremental Updates**: Update graph without full reprocessing
3. **Entity Linking**: Link to external knowledge bases (Wikidata, DBpedia)
4. **Relation Extraction**: Use specialized relation extraction models
5. **Query Routing**: Automatically select best chat mode per query
6. **Caching**: Cache embeddings and LLM responses
7. **Parallel Processing**: Process chunks/entities in parallel (asyncio)

### Known Limitations

1. **LLM Hallucination**: Entity extraction can hallucinate relationships (partially mitigated by validation)
2. **Context Window**: Large combined chunks may exceed model limits
3. **Language Support**: Optimized for English (embeddings/prompts)
4. **Schema Drift**: User-defined allowed_nodes can lead to schema inconsistency
5. **Cold Start**: Initial processing slow for large documents

---

## References

### Key Files

- **Document Loading**: `backend/src/document_sources/local_file.py`
- **Chunking**: `backend/src/create_chunks.py`
- **Entity Extraction**: `backend/src/llm.py`
- **Graph Operations**: `backend/src/make_relationships.py`
- **Deduplication**: `backend/src/graphDB_dataAccess.py`
- **Post-Processing**: `backend/src/post_processing.py`
- **RAG Integration**: `backend/src/QA_integration.py`
- **Constants/Queries**: `backend/src/shared/constants.py`
- **Common Functions**: `backend/src/shared/common_fn.py`

### External Dependencies

- **LangChain**: LLM orchestration framework
- **Neo4j**: Graph database (5.23+)
- **PyMuPDF**: PDF parsing
- **HuggingFace**: Embedding models
- **APOC**: Neo4j procedures library

---

**Document End**

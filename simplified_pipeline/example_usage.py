"""
Example usage of the simplified PDF to Graph pipeline
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pipeline import PDFToGraphPipeline, PipelineConfig

# Load environment variables
load_dotenv()


def main():
    """Run the PDF to Graph pipeline"""

    # Configure the pipeline
    config = PipelineConfig(
        # Neo4j connection settings
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),

        # Chunking settings
        chunk_size=2000,
        chunk_overlap=200,

        # Entity extraction settings (None means extract all types)
        allowed_nodes=None,  # e.g., ["Person", "Organization", "Location"]
        allowed_relationships=None,  # e.g., ["WORKS_FOR", "LOCATED_IN"]
        node_properties=["description", "type"],
        relationship_properties=["description", "weight"],

        # Post-processing settings
        enable_embeddings=True,  # Set to True if you want vector search
        enable_communities=True,
        merge_duplicates=True,
        similarity_threshold=0.95
    )

    # Initialize LLM (you can use any LangChain-compatible LLM)
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize embedding model (optional, for vector search)
    embedding_model = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    ) if config.enable_embeddings else None

    # Create pipeline
    pipeline = PDFToGraphPipeline(
        config=config,
        llm=llm,
        embedding_model=embedding_model
    )

    # Process a single PDF
    pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path

    try:
        print(f"Processing PDF: {pdf_path}")
        results = pipeline.process_pdf(
            pdf_path=pdf_path,
            clear_existing=True  # Clear existing graph data
        )

        # Print results
        print("\n=== Processing Results ===")
        print(f"Status: {results['status']}")
        print(f"Pages loaded: {results.get('pages_loaded', 0)}")
        print(f"Chunks created: {results.get('chunks_created', 0)}")
        print(f"Chunks processed: {results.get('chunks_processed', 0)}")
        print(f"Total nodes extracted: {results.get('total_nodes', 0)}")
        print(f"Total relationships extracted: {results.get('total_relationships', 0)}")
        print(f"Final nodes in graph: {results.get('final_node_count', 0)}")
        print(f"Final relationships in graph: {results.get('final_relationship_count', 0)}")

        if results['status'] == 'failed':
            print(f"Error: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Pipeline failed: {e}")


def process_multiple_pdfs_example():
    """Example of processing multiple PDFs"""

    config = PipelineConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        chunk_size=1500,
        chunk_overlap=150,
        enable_embeddings=False,  # Disable for faster processing
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    pipeline = PDFToGraphPipeline(config=config, llm=llm)

    # List of PDFs to process
    pdf_paths = [
        "path/to/document1.pdf",
        "path/to/document2.pdf",
        "path/to/document3.pdf"
    ]

    results = pipeline.process_multiple_pdfs(
        pdf_paths=pdf_paths,
        clear_existing=True
    )

    for i, result in enumerate(results):
        print(f"\nPDF {i+1}: {pdf_paths[i]}")
        print(f"  Status: {result['status']}")
        print(f"  Nodes: {result.get('final_node_count', 0)}")
        print(f"  Relationships: {result.get('final_relationship_count', 0)}")


if __name__ == "__main__":
    main()

    # Uncomment to run multiple PDFs example
    # process_multiple_pdfs_example()
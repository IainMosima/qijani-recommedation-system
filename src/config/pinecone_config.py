import os
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone(index_name: str) -> None:
    """Initialize Pinecone and create index if it doesn't exist."""
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # List existing indexes
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")

def delete_pinecone_index(index_name: str = "recommendation-index"):
    """
    Delete a Pinecone index.
    
    Args:
        index_name (str): Name of the index to delete
    """
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Delete the index
    if index_name in [index_info["name"] for index_info in pc.list_indexes()]:
        print(f"Deleting Pinecone index: {index_name}")
        pc.delete_index(index_name)
        print(f"Index {index_name} deleted successfully")
    else:
        print(f"Index {index_name} does not exist")

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import Optional

load_dotenv()

def initialize_pinecone(dimension: int = 1536, index_name: str = "recommendation-index"):
    """
    Initialize Pinecone with the specified index.
    
    Args:
        dimension (int): Dimension of vectors to store (default for OpenAI embeddings is 1536)
        index_name (str): Name of the Pinecone index to use
        
    Returns:
        The initialized Pinecone index
    """
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Check if index already exists
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        # Create index if it doesn't exist
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")
    
    # Return the index
    return pc.Index(index_name)


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

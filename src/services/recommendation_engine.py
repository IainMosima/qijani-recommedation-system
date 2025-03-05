import os
import uuid
from typing import List, Dict, Any, Optional

from pinecone import Pinecone
from langchain.vectorstores import PineconeVectorStore

from src.services.embedding_cache import EmbeddingCache


class RecommendationEngine:
    """
    Recommendation engine using Pinecone vector database and cached embeddings.
    """

    def __init__(self, index_name: str = "recommendation-index"):
        """
        Initialize the recommendation engine.
        
        Args:
            index_name (str): Name of the Pinecone index to use
        """
        self.index_name = index_name
        self.embedding_cache = EmbeddingCache()
        
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(self.index_name)
        
    def add_item(self, content: str, metadata: Dict[str, Any], item_type: str) -> str:
        """
        Add a single item to the recommendation engine.
        
        Args:
            content (str): The text content to embed
            metadata (Dict[str, Any]): Metadata associated with the content
            item_type (str): Type of the item (e.g., 'article', 'product')
            
        Returns:
            str: ID of the added item
        """
        # Add content to metadata for retrieval
        metadata["content"] = content
        metadata["item_type"] = item_type
        
        # Generate embedding using the cache
        embedding = self.embedding_cache.get_embedding(content)
        
        # Generate a unique ID
        item_id = str(uuid.uuid4())
        
        # Upsert into Pinecone
        self.index.upsert(vectors=[
            {
                "id": item_id,
                "values": embedding,
                "metadata": metadata
            }
        ])
        
        return item_id
    
    def bulk_add_items(
        self, 
        contents: List[str], 
        metadatas: List[Dict[str, Any]], 
        item_types: List[str]
    ) -> List[str]:
        """
        Add multiple items to the recommendation engine.
        
        Args:
            contents (List[str]): List of text contents to embed
            metadatas (List[Dict[str, Any]]): List of metadata dicts for each content
            item_types (List[str]): List of item types
            
        Returns:
            List[str]: List of IDs for the added items
        """
        if not (len(contents) == len(metadatas) == len(item_types)):
            raise ValueError("Contents, metadatas, and item_types must have the same length")
        
        # Generate embeddings in bulk using the cache
        embeddings = self.embedding_cache.get_embeddings(contents)
        
        # Prepare data for Pinecone
        item_ids = [str(uuid.uuid4()) for _ in range(len(contents))]
        vectors = []
        
        for i, (content, metadata, item_type, embedding) in enumerate(
            zip(contents, metadatas, item_types, embeddings)
        ):
            # Add content and item_type to metadata
            metadata = metadata.copy()
            metadata["content"] = content
            metadata["item_type"] = item_type
            
            # Create vector record
            vectors.append({
                "id": item_ids[i],
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches to avoid exceeding request size limits
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return item_ids
    
    def get_recommendations(
        self, 
        query: str, 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on a query.
        
        Args:
            query (str): The query text
            top_k (int): Number of recommendations to return
            filter_criteria (Dict[str, Any], optional): Criteria to filter results
            
        Returns:
            List[Dict[str, Any]]: Recommended items with scores and metadata
        """
        # Generate embedding for query using cache
        query_embedding = self.embedding_cache.get_embedding(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_criteria
        )
        
        # Format results
        recommendations = []
        for match in results["matches"]:
            recommendations.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            })
        
        return recommendations
    
    def delete_item(self, item_id: str) -> None:
        """
        Delete an item from the recommendation engine.
        
        Args:
            item_id (str): ID of the item to delete
        """
        self.index.delete(ids=[item_id])

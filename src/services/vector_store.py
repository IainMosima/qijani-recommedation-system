import hashlib
import json
from typing import List, Dict, Any, Optional, Union
from src.config.pinecone_config import get_pinecone_index

class PineconeVectorStore:
    def __init__(self):
        """Initialize Pinecone vector store"""
        self.index = get_pinecone_index()
    
    def _generate_id(self, content: str, namespace: Optional[str] = None) -> str:
        """Generate a deterministic ID based on content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if namespace:
            return f"{namespace}_{content_hash}"
        return content_hash
    
    def store_embedding(
        self, 
        content: str, 
        embedding: List[float], 
        metadata: Dict[str, Any], 
        namespace: str = "default"
    ) -> str:
        """
        Store an embedding in Pinecone
        
        Args:
            content: The original content
            embedding: Vector embedding
            metadata: Additional information
            namespace: Optional namespace for organizing embeddings
            
        Returns:
            vector_id: ID of the stored vector
        """
        vector_id = self._generate_id(content, namespace)
        
        # Add content to metadata for future reference
        metadata["content"] = content
        
        # Upsert the vector into Pinecone
        self.index.upsert(
            vectors=[(vector_id, embedding, metadata)],
            namespace=namespace
        )
        
        return vector_id
    
    def get_embedding_by_id(self, vector_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve a specific embedding by ID"""
        result = self.index.fetch(ids=[vector_id], namespace=namespace)
        
        if not result.get('vectors', {}):
            return None
            
        vector_data = result['vectors'][vector_id]
        return {
            "id": vector_id,
            "embedding": vector_data.values,
            "metadata": vector_data.metadata
        }
    
    def get_embedding_by_content(self, content: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """Check if content already has an embedding and retrieve it"""
        vector_id = self._generate_id(content, namespace)
        return self.get_embedding_by_id(vector_id, namespace)
    
    def find_similar(
        self, 
        embedding: List[float], 
        namespace: str = "default",
        top_k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find similar vectors in Pinecone
        
        Args:
            embedding: Query vector
            namespace: Namespace to search in
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            
        Returns:
            List of similar vectors with metadata and similarity scores
        """
        query_result = self.index.query(
            vector=embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter
        )
        
        results = []
        for match in query_result.matches:
            result = {
                "id": match.id,
                "score": match.score,
            }
            if include_metadata and match.metadata:
                result["metadata"] = match.metadata
            results.append(result)
            
        return results
    
    def delete_embedding(self, vector_id: str, namespace: str = "default") -> bool:
        """Delete an embedding by ID"""
        self.index.delete(ids=[vector_id], namespace=namespace)
        return True
        
    def delete_embeddings_by_filter(self, filter: Dict[str, Any], namespace: str = "default") -> bool:
        """Delete embeddings matching a metadata filter"""
        self.index.delete(filter=filter, namespace=namespace)
        return True
        
    def get_total_vector_count(self, namespace: str = "default") -> int:
        """Get the total number of vectors in the index"""
        stats = self.index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        if namespace in namespaces:
            return namespaces[namespace].get('vector_count', 0)
        return 0

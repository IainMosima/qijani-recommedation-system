import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import hashlib
import pickle

from langchain_openai import OpenAIEmbeddings
from langchain_nomic import NomicEmbeddings


class EmbeddingCache:
    """
    Cache for storing and retrieving text embeddings to avoid redundant API calls.
    """
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir (str): Directory to store the cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.cache: Dict[str, List[float]] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
        # Try to initialize OpenAI embeddings first
        if os.getenv("OPENAI_API_KEY"):
            self.embeddings_model = OpenAIEmbeddings()
        else:
            # Fallback to local embeddings model
            try:
                self.embeddings_model = NomicEmbeddings()
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize embeddings. Please set OPENAI_API_KEY "
                    "environment variable or ensure local embedding model is available."
                ) from e
    
    def _load_cache(self) -> None:
        """Load the cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings")
            else:
                print("No existing cache found, starting fresh")
                self.cache = {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save the cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _generate_key(self, text: str) -> str:
        """Generate a unique key for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text, either from cache or by generating a new one.
        
        Args:
            text (str): The text to get embedding for
            
        Returns:
            List[float]: The embedding vector
        """
        key = self._generate_key(text)
        
        if key in self.cache:
            print(f"Using cached embedding for: {text[:30]}...")
            return self.cache[key]
        
        # Generate new embedding if not in cache
        print(f"Generating new embedding for: {text[:30]}...")
        embedding = self.embeddings_model.embed_query(text)
        
        # Store in cache
        self.cache[key] = embedding
        self._save_cache()
        
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts, using cache when possible.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        indices = []
        
        # Check which texts need new embeddings
        for i, text in enumerate(texts):
            key = self._generate_key(text)
            if key in self.cache:
                print(f"Using cached embedding for: {text[:30]}...")
                embeddings.append(self.cache[key])
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        # If there are texts that need embedding
        if texts_to_embed:
            # Generate new embeddings
            print(f"Generating {len(texts_to_embed)} new embeddings...")
            new_embeddings = self.embeddings_model.embed_documents(texts_to_embed)
            
            # Insert new embeddings in the right positions and update cache
            for idx, text, embedding in zip(indices, texts_to_embed, new_embeddings):
                key = self._generate_key(text)
                self.cache[key] = embedding
                embeddings.insert(idx, embedding)
            
            # Save updated cache
            self._save_cache()
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache = {}
        self._save_cache()
        print("Embedding cache cleared.")

import os
from typing import List
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """
    Validate if the URL is properly formatted.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_appropriate_loader(url: str) -> BaseLoader:
    """
    Get the appropriate document loader based on the URL type.
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL format: {url}")
        
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    if path.endswith('.pdf'):
        return PyPDFLoader(url)
    else:
        return WebBaseLoader(url)

def load_documents_from_urls(urls: List[str]) -> List[Document]:
    """
    Load documents from a list of URLs, handling different file types appropriately.
    """
    documents = []
    
    for url in urls:
        try:
            if not url.strip():
                continue
                
            loader = get_appropriate_loader(url)
            docs = loader.load()
            documents.extend(docs)
            print(f"Successfully loaded document from {url}")
        except Exception as e:
            print(f"Error loading document from {url}: {e}")
            continue
    
    return documents

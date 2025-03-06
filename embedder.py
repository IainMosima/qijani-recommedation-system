import os
import sys
from typing import List, Literal, Optional

import dotenv
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_nomic.embeddings import NomicEmbeddings  # local
from langchain_openai import OpenAIEmbeddings  # api
from pydantic import BaseModel, Field

from src.services.retrival_engine import RetrivalEngine
from src.utils.document_loader import load_documents_from_urls



def main():
    os.environ.clear()
    dotenv.load_dotenv()
    # Read the Excel file path from command-line argument
    if len(sys.argv) < 2:
        print("Usage: python embedder.py <path_to_excel_file>")
        sys.exit(1)

    excel_file = sys.argv[1]
    
    # Load the Excel file and parse the 'urls' column
    df = pd.read_excel(excel_file)
    urls = df['urls'].tolist()

    # Load documents from the URLs
    docs = [PyPDFLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=100
    )

    # Create a retrival engine
    retrival_engine = RetrivalEngine()

    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    contents = [doc.page_content for doc in doc_splits]
    metadatas = []

    for doc in doc_splits:
        metadata = dict(doc.metadata)
        metadata["content"] = "nutrition_article"
        metadatas.append(metadata)

    # Add items to retrival engine in smaller batches
    batch_size = 10
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        # This will use the cache for any existing embeddings
        ids = retrival_engine.bulk_add_items(
            contents=batch_contents,
            metadatas=batch_metadatas,
            item_types=["nutrition_document"] * len(batch_contents)
        )

    print("All documents stored in Pinecone!")


if __name__ == "__main__":
    main()

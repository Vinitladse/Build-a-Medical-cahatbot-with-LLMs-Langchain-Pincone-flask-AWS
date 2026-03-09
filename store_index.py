from dotenv import load_dotenv

import os
from src.helpers import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# loading the env file
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("GROQ_API_KEY")

# setting as environmental variable
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = OPENAI_API_KEY

# loading the dara
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)  # filterig the minimal data
text_chunks = text_split(filter_data)  # perform the chunking

embeddings = download_hugging_face_embeddings()  # downloading the embedding the model

# Initializing the pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


# creating the index name
index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# it will store all of the embedding in the pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

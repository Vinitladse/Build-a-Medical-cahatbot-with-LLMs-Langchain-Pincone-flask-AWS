from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# extract text from pdf file


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,  # path of the pdf
        glob="*.pdf",  # bcz i want load only pdf file
        loader_cls=PyPDFLoader,  # coz of pdf docs
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects
    containing only 'source' in metadata and the orginal page_content.

    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# embedding


def download_hugging_face_embeddings():
    """
    Download and return the hugging face embeddings model.

    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )

    return embeddings


embeddings = download_hugging_face_embeddings()

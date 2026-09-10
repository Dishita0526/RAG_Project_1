from openai import OpenAI

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# NVIDIA OpenAI-compatible API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1"
)


# NVIDIA embedding model
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"

# Embedding dimension of the model
EMBED_DIM = 2048


# Split PDF text into chunks
splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF and split its text into smaller chunks.
    """

    reader = PDFReader()

    docs = reader.load_data(
        file=path
    )

    texts = [
        doc.text
        for doc in docs
        if getattr(doc, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(
            splitter.split_text(text)
        )

    return chunks


def embed_text(
    texts: list[str],
    input_type: str = "passage"
) -> list[list[float]]:
    """
    Convert text into embeddings using NVIDIA.

    input_type:
        passage -> used when embedding PDF chunks
        query   -> used when embedding user questions
    """

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={
            "input_type": input_type
        }
    )

    return [
        item.embedding
        for item in response.data
    ]
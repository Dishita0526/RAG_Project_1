import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    """
    Chunks extracted from a document
    and the original source.
    """

    chunks: list[str]
    source_id: str | None = None


class RAGUpsertResult(pydantic.BaseModel):
    """
    Result of inserting chunks into Qdrant.
    """

    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    """
    Search results returned from Qdrant.
    """

    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    """
    Final answer returned by the RAG pipeline.
    """

    answer: str
    sources: list[str]
    num_contexts: int
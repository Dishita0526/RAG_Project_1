import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    """A Chunk and its original source filename"""
    chunk :list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    inngested:int 
    
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    ans: str
    contexts : list[str]
    num_contexts : int 
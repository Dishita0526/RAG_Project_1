


import logging
import uuid
import os
import datetime

from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
from openai import OpenAI
from data_loader import load_and_chunk_pdf,embed_text
from vector_db import QdrantStorage
from custom_types import RAGQueryResult,RAGSearchResult,RAGChunkAndSrc,RAGUpsertResult


load_dotenv()


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1"
)

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger =logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
    
)

@inngest_client.create_function(
    fn_id = "RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)

    
    def _upsert(chunck_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
        payloads = [{"source":source_id,"texts": chunks[i]} for i in range(len(chunks))]
        QdrantStorage.upsert(vecs,ids,payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.modal_dump()

@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger = inngest.TriggerEvent(event="rag/query_pdf_ai")

)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question:str , top_k :int =5):
        query_vec= embed_text([question])[0]
        hits= Qdrant

app = FastAPI()
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf]
)
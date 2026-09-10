import logging
import os
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import inngest
import inngest.fast_api

from dotenv import load_dotenv
from openai import OpenAI

from data_loader import (
    load_and_chunk_pdf,
    embed_text,
)

from vector_db import QdrantStorage

from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# NVIDIA OPENAI-COMPATIBLE CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)


# ============================================================
# INNGEST
# ============================================================

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ============================================================
# PDF INGESTION FUNCTION
# ============================================================

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(
        event="rag/ingest_pdf"
    ),
)
async def rag_ingest_pdf(
    ctx: inngest.Context,
):

    # --------------------------------------------------------
    # STEP 1: LOAD AND CHUNK PDF
    # --------------------------------------------------------

    def _load(
        ctx: inngest.Context,
    ) -> RAGChunkAndSrc:

        pdf_path = ctx.event.data["pdf_path"]

        source_id = ctx.event.data.get(
            "source_id",
            pdf_path,
        )

        chunks = load_and_chunk_pdf(
            pdf_path
        )

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id,
        )


    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )


    # --------------------------------------------------------
    # STEP 2: EMBED AND UPSERT INTO QDRANT
    # --------------------------------------------------------

    def _upsert(
        chunks_and_src: RAGChunkAndSrc,
    ) -> RAGUpsertResult:

        chunks = chunks_and_src.chunks

        source_id = chunks_and_src.source_id

        # Create embeddings for PDF chunks
        vectors = embed_text(
            chunks,
            input_type="passage",
        )

        # Create unique IDs
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}_{i}",
                )
            )
            for i in range(len(chunks))
        ]

        # Create payloads
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        # Create Qdrant storage
        store = QdrantStorage()

        # IMPORTANT:
        # upsert expects:
        # ids, vectors, payloads
        store.upsert(
            ids,
            vectors,
            payloads,
        )

        return RAGUpsertResult(
            ingested=len(chunks),
        )


    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )


    return ingested.model_dump()


# ============================================================
# QUERY PDF FUNCTION
# ============================================================

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    ),
)
async def rag_query_pdf_ai(
    ctx: inngest.Context,
) -> RAGQueryResult:

    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = ctx.event.data["question"]

    top_k = int(
        ctx.event.data.get(
            "top_k",
            5,
        )
    )


    # --------------------------------------------------------
    # STEP 1: SEARCH QDRANT
    # --------------------------------------------------------

    def _search(
        question: str,
        top_k: int,
    ) -> RAGSearchResult:

        # Embed the question as a QUERY
        query_vector = embed_text(
            [question],
            input_type="query",
        )[0]

        # Connect to Qdrant
        store = QdrantStorage()

        # Search for similar chunks
        found = store.search(
            query_vector,
            top_k,
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )


    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(
            question,
            top_k,
        ),
        output_type=RAGSearchResult,
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_block = "\n\n".join(
        f"- {context}"
        for context in found.contexts
    )


    user_content = (
        "Use ONLY the following context to answer "
        "the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "If the answer cannot be found in the context, "
        "say: I don't have enough information in the "
        "provided documents."
    )


    # --------------------------------------------------------
    # STEP 2: GENERATE ANSWER WITH NVIDIA
    # --------------------------------------------------------

    def _generate_answer():

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a RAG assistant. "
                        "Answer the question using ONLY "
                        "the supplied context. "
                        "Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=False,
        )

        return response.choices[0].message.content


    answer = await ctx.step.run(
        "llm-answer",
        _generate_answer,
    )


    # --------------------------------------------------------
    # RETURN FINAL RESULT
    # --------------------------------------------------------

    return RAGQueryResult(
        answer=answer.strip(),
        sources=found.sources,
        num_contexts=len(found.contexts),
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# --------------------------------------------------------
# CORS — allow the Vite dev server to talk to FastAPI
# --------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[
        rag_ingest_pdf,
        rag_query_pdf_ai,
    ],
)


# ============================================================
# HTTP API ROUTES  (used by the frontend)
# ============================================================


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class IngestRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None


@app.post("/api/query")
def api_query(req: QueryRequest):
    """
    Synchronous HTTP wrapper around the RAG query pipeline.
    Reuses embed_text, QdrantStorage, and the OpenAI client
    already defined above — no logic duplication.
    """

    # --- embed question ---
    query_vector = embed_text(
        [req.question],
        input_type="query",
    )[0]

    # --- search Qdrant ---
    store = QdrantStorage()
    found = store.search(query_vector, req.top_k)

    contexts = found["contexts"]
    sources = found["sources"]

    if not contexts:
        return {
            "answer": "No relevant documents found. Please ingest a PDF first.",
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(
        f"- {ctx}" for ctx in contexts
    )

    user_content = (
        "Use ONLY the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n\n"
        "If the answer cannot be found in the context, "
        "say: I don't have enough information in the provided documents."
    )

    # --- call NVIDIA LLM ---
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant. "
                    "Answer the question using ONLY the supplied context. "
                    "Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=False,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "num_contexts": len(contexts),
    }


@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """
    Synchronous HTTP wrapper for PDF ingestion.
    Reuses load_and_chunk_pdf, embed_text, and QdrantStorage.
    """

    source_id = req.source_id or req.pdf_path

    chunks = load_and_chunk_pdf(req.pdf_path)

    vectors = embed_text(chunks, input_type="passage")

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}"))
        for i in range(len(chunks))
    ]

    payloads = [
        {"source": source_id, "text": chunks[i]}
        for i in range(len(chunks))
    ]

    store = QdrantStorage()
    store.upsert(ids, vectors, payloads)

    return {"ingested": len(chunks), "source": source_id}
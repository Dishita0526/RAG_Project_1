from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)


class QdrantStorage:

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 2048,
    ):
        self.client = QdrantClient(
            url=url,
            timeout=30,
        )

        self.collection = collection

        # Create collection if it doesn't exist
        if not self.client.collection_exists(
            self.collection
        ):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )


    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ):
        """
        Insert embeddings and their payloads into Qdrant.
        """

        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )


    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ):
        """
        Search Qdrant for the most similar vectors.
        """

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        contexts = []
        sources = set()

        for result in results:

            payload = (
                getattr(result, "payload", None)
                or {}
            )

            text = payload.get(
                "text",
                ""
            )

            source = payload.get(
                "source",
                ""
            )

            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }
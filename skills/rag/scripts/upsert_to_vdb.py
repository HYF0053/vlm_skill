import sys
import os
import argparse
import uuid

# Add project root to path for imports
sys.path.append("/home/ubuntu/ai-agent-platform")

from adapters.outbound.rag.qdrant_rag_adapter import QdrantRAGAdapter
from adapters.outbound.embeddings.vllm_embedding_adapter import VLLMEmbeddingAdapter
from infrastructure.config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Upsert a fact or document chunk into Qdrant VDB.")
    parser.add_argument("content", help="The text content to store.")
    parser.add_argument("-c", "--collection", default="user_memory", help="Collection name (default: user_memory)")
    parser.add_argument("-s", "--source", default="manual_entry", help="Source metadata (e.g. filename or 'user_input')")
    parser.add_argument("--metadata", help="JSON string for additional metadata", default="{}")
    
    args = parser.parse_args()

    # Init Embedding
    embedding_port = VLLMEmbeddingAdapter(
        base_url=settings.EMBEDDING_BASE_URL,
        model=settings.EMBEDDING_MODEL
    )
    
    # Init RAG Adapter
    # Note: We might need to ensure the collection exists. 
    # Current adapter doesn't auto-create in _connect, so we handle it here or in adapter.
    
    rag_adapter = QdrantRAGAdapter(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection=args.collection,
        embedding_port=embedding_port
    )
    
    # Check if connected (collection might not exist yet)
    if not rag_adapter.client:
        print(f"Collection '{args.collection}' does not exist. Attempting to create it...")
        # Re-init with a dummy collection to get the client, or just use qdrant_client directly
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        dim = embedding_port.get_dimension()
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"Created collection: {args.collection}")
        # Re-init adapter
        rag_adapter = QdrantRAGAdapter(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection=args.collection,
            embedding_port=embedding_port
        )

    import json
    from datetime import datetime, timezone
    
    try:
        extra_meta = json.loads(args.metadata)
    except:
        extra_meta = {}

    # 2026 Standard Metadata Schema
    meta = {
        "source": args.source,
        "type": extra_meta.get("type", "knowledge"),
        "importance": extra_meta.get("importance", 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": os.environ.get("SESSION_ID", "default"),
    }
    # Merge remaining keys from extra_meta
    for k, v in extra_meta.items():
        if k not in meta:
            meta[k] = v
    
    success = rag_adapter.add_documents([args.content], [meta])
    
    if success:
        print(f"Successfully added to {args.collection} with metadata: {meta}")
    else:
        print(f"Failed to add to {args.collection}")
        sys.exit(1)

if __name__ == "__main__":
    # Add project root to path for imports
    sys.path.append("/home/ubuntu/ai-agent-platform")
    main()

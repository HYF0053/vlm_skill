---
name: rag
description: Use this skill for Retrieval-Augmented Generation (RAG) capabilities, searching internal vector databases (knowledge bases) regarding product documents, specifications, roadmaps, and guidelines.
---

# RAG (Retrieval-Augmented Generation) Skill

This skill allows you to retrieve documents from the internal Knowledge Base (Vector Database) maintained by the `ai-agent-platform`. The knowledge base contains information about Leadtek's GPU products, systems, workstations, and company documents.

## Available Scripts

The scripts are located in `/home/ubuntu/vlm_skill/skills/rag/scripts/`.

### 1. `search_vdb.py`

This script searches the vector database using hybrid search (Keyword + Vector) and reranking. It provides a detailed, human-readable list of search results including scores and metadata. Use this when you want to explore the data structure or visually inspect search outcomes.

**Usage:**
```bash
python3 /home/ubuntu/vlm_skill/skills/rag/scripts/search_vdb.py "your search query" [--limit LIMIT] [-c COLLECTION_NAME]
```

**Options:**
- `query` (required): The search string (e.g., "PRO 6000 specifications").
- `--limit` (optional): Maximum number of results to return (default is 5).
- `-c` or `--collection` (optional): Restrict the search to a specific collection. Can specify multiple. If omitted, it searches all available collections natively.

**Available Collections:**
- `test_docs`: AIDMS product intro, FAQs.
- `hw_gpu_kb`: Leadtek standalone GPUs, specs, roadmaps, benchmarks.
- `hw_system_kb`: Leadtek Workstations and Servers, specs, compatibility.

**Example:**
```bash
# Search across all collections for GPU specifications
python3 /home/ubuntu/vlm_skill/skills/rag/scripts/search_vdb.py "Leadtek RTX A6000 specs" --limit 3

# Search strictly within the system KB collection
python3 /home/ubuntu/vlm_skill/skills/rag/scripts/search_vdb.py "WS650 supported GPU" -c hw_system_kb
```

### 2. `retrieve_context.py`

This script behaves similarly to `search_vdb.py`, but its output is formatted optimally for immediate use as context by Large Language Models. It concatenates the retrieved documents into a single block of text and intelligently truncates them to fit within a specified token limit. Use this when you are dynamically feeding retrieved documents into an LLM block.

**Usage:**
```bash
python3 /home/ubuntu/vlm_skill/skills/rag/scripts/retrieve_context.py "query" [--max_tokens MAX] [-c COLLECTION_NAME]
```

### 3. Ingesting New Documents

If you have a collection of Markdown documents that you need to ingest into the Vector Database, you can run the master ingestion script natively provided by the `ai-agent-platform`:

```bash
cd /home/ubuntu/ai-agent-platform
python3 ingest_markdown_to_qdrant.py
```
*Note: Make sure your new `.md` files are placed within the target directories configured by the platform (e.g., `/home/ubuntu/ai-agent-platform/hw_product_md/GPU/`) before running the ingest script.*

## Implementation Details

If you need deeper integration or customization for RAG, the source modules are securely loaded from:
- **Core Orchestration:** `/home/ubuntu/ai-agent-platform/domain/services/rag_service.py`
- **Qdrant Vector Adapter:** `/home/ubuntu/ai-agent-platform/adapters/outbound/rag/qdrant_rag_adapter.py`
- **Document Chunking & Ingest:** `/home/ubuntu/ai-agent-platform/ingest_markdown_to_qdrant.py`

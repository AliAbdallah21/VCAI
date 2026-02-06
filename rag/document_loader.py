# rag/document_loader.py
"""
Document loader for RAG pipeline.
Loads documents from data/documents folder and chunks them for embedding.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from rag.config import DOCS_DIR


@dataclass
class RawChunk:
    """A chunk of text with metadata."""
    id: str
    content: str
    source: str
    metadata: Optional[dict] = None


def _generate_chunk_id(content: str, source: str, chunk_idx: int) -> str:
    """Generate unique ID for a chunk."""
    text = f"{source}:{chunk_idx}:{content[:100]}"
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with overlap.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            for sep in ['. ', '.\n', '؟ ', '؟\n', '! ', '!\n', '\n\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def load_json_document(file_path: Path) -> List[RawChunk]:
    """Load a JSON document and create chunks."""
    chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    source = file_path.name
    
    # Handle list of properties
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                # Convert dict to readable text
                text_parts = []
                for key, value in item.items():
                    text_parts.append(f"{key}: {value}")
                content = "\n".join(text_parts)
                
                chunk_id = _generate_chunk_id(content, source, idx)
                chunks.append(RawChunk(
                    id=chunk_id,
                    content=content,
                    source=source,
                    metadata={"index": idx, "type": "property"}
                ))
    
    # Handle dict with nested data
    elif isinstance(data, dict):
        # Check for common patterns
        if "properties" in data:
            for idx, prop in enumerate(data["properties"]):
                text_parts = []
                for key, value in prop.items():
                    text_parts.append(f"{key}: {value}")
                content = "\n".join(text_parts)
                
                chunk_id = _generate_chunk_id(content, source, idx)
                chunks.append(RawChunk(
                    id=chunk_id,
                    content=content,
                    source=source,
                    metadata={"index": idx, "type": "property"}
                ))
        else:
            # Single document - chunk the whole thing
            text_parts = []
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            
            content = "\n".join(text_parts)
            for idx, chunk_text in enumerate(_chunk_text(content)):
                chunk_id = _generate_chunk_id(chunk_text, source, idx)
                chunks.append(RawChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source=source,
                    metadata={"chunk_index": idx}
                ))
    
    return chunks


def load_text_document(file_path: Path) -> List[RawChunk]:
    """Load a text document and create chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    source = file_path.name
    chunks = []
    
    for idx, chunk_text in enumerate(_chunk_text(content)):
        chunk_id = _generate_chunk_id(chunk_text, source, idx)
        chunks.append(RawChunk(
            id=chunk_id,
            content=chunk_text,
            source=source,
            metadata={"chunk_index": idx}
        ))
    
    return chunks


def load_chunks() -> List[RawChunk]:
    """
    Load all documents from DOCS_DIR and return chunks.
    Supports: .json, .txt, .md
    """
    all_chunks = []
    
    if not DOCS_DIR.exists():
        print(f"[RAG] Documents directory not found: {DOCS_DIR}")
        return []
    
    # Supported extensions
    extensions = ['.json', '.txt', '*.md']
    
    for ext in extensions:
        for file_path in DOCS_DIR.glob(ext):
            try:
                if file_path.suffix == '.json':
                    chunks = load_json_document(file_path)
                else:
                    chunks = load_text_document(file_path)
                
                all_chunks.extend(chunks)
                print(f"[RAG] Loaded {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                print(f"[RAG] Error loading {file_path}: {e}")
    
    print(f"[RAG] Total chunks loaded: {len(all_chunks)}")
    return all_chunks


# For testing
if __name__ == "__main__":
    chunks = load_chunks()
    for c in chunks[:5]:
        print(f"\n{'='*50}")
        print(f"ID: {c.id}")
        print(f"Source: {c.source}")
        print(f"Content: {c.content[:200]}...")
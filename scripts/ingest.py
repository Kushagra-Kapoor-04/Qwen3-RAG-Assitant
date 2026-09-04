

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.constants import get_success_message, get_error_message
from ingestion.loader import load_documents, Document
from ingestion.splitter import split_documents
from ingestion.embedder import embed_chunks, create_embedder
from vectorstore.faiss_store import create_faiss_store
from services.logging_service import get_logger

logger = get_logger(__name__)

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=settings.raw_data_path,
        help="Source directory or file path"
    )
    
    parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to metadata.json file (default: data/metadata/metadata.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=settings.faiss_index_path,
        help="Output path for FAISS index"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk size for splitting"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Overlap between chunks"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Recursively search directories"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingestion"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_metadata(metadata_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    
    if metadata_path is None:
        metadata_path = Path(settings.metadata_path) / "metadata.json"
    else:
        metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        logger.info(f"No metadata file found at {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        metadata_dict = {}
        for item in metadata_list:
            filename = item.get("filename")
            if filename:
                metadata_dict[filename] = {
                    "doc_id": item.get("doc_id", ""),
                    "source": item.get("source", ""),
                    "tags": item.get("tags", []),
                    **{k: v for k, v in item.items() 
                       if k not in ("filename", "doc_id", "source", "tags")}
                }
        
        logger.info(f"Loaded metadata for {len(metadata_dict)} documents")
        return metadata_dict
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return {}

def enrich_documents_with_metadata(
    documents: List[Document],
    metadata: Dict[str, Dict[str, Any]],
    verbose: bool = False
) -> List[Document]:
    
    if not metadata:
        return documents
    
    enriched_count = 0
    
    for doc in documents:
        source_path = Path(doc.metadata.get("source", ""))
        filename = source_path.name
        
        if filename in metadata:
            doc_meta = metadata[filename]
            
            doc.metadata["doc_id"] = doc_meta.get("doc_id", "")
            doc.metadata["dataset_source"] = doc_meta.get("source", "")
            doc.metadata["tags"] = doc_meta.get("tags", [])
            
            for key, value in doc_meta.items():
                if key not in ("doc_id", "source", "tags"):
                    doc.metadata[key] = value
            
            enriched_count += 1
            
            if verbose:
                logger.info(f"  Enriched: {filename} -> {doc_meta.get('doc_id', 'N/A')}")
    
    logger.info(f"Enriched {enriched_count}/{len(documents)} documents with metadata")
    return documents

def ingest(
    source_path: str,
    output_path: str,
    metadata_path: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    recursive: bool = True,
    clear: bool = False,
    verbose: bool = False
) -> bool:
    
    try:
        source = Path(source_path)
        if not source.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return False
        
        logger.info(f"Starting ingestion from: {source_path}")
        
        logger.info("Loading metadata...")
        metadata = load_metadata(metadata_path)
        
        logger.info("Loading documents...")
        documents = load_documents(source_path, recursive=recursive)
        
        if not documents:
            logger.warning("No documents found to process")
            return False
        
        logger.info(f"Loaded {len(documents)} documents")
        
        if verbose:
            for doc in documents:
                logger.info(f"  - {doc.source}")
        
        if metadata:
            logger.info("Enriching documents with metadata...")
            documents = enrich_documents_with_metadata(documents, metadata, verbose)
        
        logger.info("Splitting documents into chunks...")
        chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        logger.info("Generating embeddings...")
        embedder = create_embedder()
        embedded_chunks = embedder.embed_chunks(chunks, show_progress=verbose)
        
        logger.info(f"Generated {len(embedded_chunks)} embeddings")
        
        logger.info("Building FAISS index...")
        store = create_faiss_store(
            index_path=output_path,
            dimension=embedder.dimension
        )
        
        if clear:
            logger.info("Clearing existing index...")
            store.clear()
        
        store.add_embeddings(embedded_chunks)
        
        logger.info("Saving index...")
        store.save()
        
        logger.info(get_success_message("ingestion_complete", count=len(documents)))
        logger.info(f"Index saved to: {output_path}")
        
        print("\n" + "=" * 50)
        print("INGESTION COMPLETE")
        print("=" * 50)
        print(f"Documents processed: {len(documents)}")
        print(f"Metadata enriched:   {len(metadata)} documents")
        print(f"Chunks created:      {len(chunks)}")
        print(f"Index location:      {output_path}")
        print("=" * 50)
        
        if metadata and verbose:
            print("\nSample document metadata:")
            for i, doc in enumerate(documents[:3]):
                print(f"  {doc.metadata.get('doc_id', 'N/A')}: {Path(doc.source).name}")
                if doc.metadata.get('tags'):
                    print(f"    Tags: {', '.join(doc.metadata['tags'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    
    args = parse_args()
    
    settings.ensure_directories()
    
    success = ingest(
        source_path=args.source,
        output_path=args.output,
        metadata_path=args.metadata,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recursive=args.recursive,
        clear=args.clear,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
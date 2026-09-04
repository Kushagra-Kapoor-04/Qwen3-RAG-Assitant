
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from vectorstore.faiss_store import load_faiss_store

def check_db():
    print("Loading FAISS store...")
    store = load_faiss_store()
    if not store:
        print("No index found.")
        return

    print(f"Total Chunks: {store.num_documents}")
    print("-" * 50)
    
    # Aggregate by source
    sources = {}
    for doc in store._documents:
        src = doc.get('metadata', {}).get('source', 'Unknown')
        if src not in sources:
            sources[src] = 0
        sources[src] += 1
        
    print(f"Unique Documents: {len(sources)}")
    print("Sources encountered:")
    for src, count in sources.items():
        print(f"  - {Path(src).name} ({count} chunks)")
        
if __name__ == "__main__":
    check_db()

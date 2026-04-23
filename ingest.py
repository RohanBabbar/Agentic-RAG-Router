import os
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
DB_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def chunk_text_with_overlap(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Slices text into overlapping chunks without splitting words in half.
    """
    words = text.split()
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1 # +1 for the space
        if current_length + word_len > chunk_size and current_chunk:
            # We reached the size limit, save the chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap: keep the last few words that fit into the overlap size
            overlap_chunk = []
            overlap_length = 0
            for w in reversed(current_chunk):
                if overlap_length + len(w) + 1 <= overlap:
                    overlap_chunk.insert(0, w)
                    overlap_length += len(w) + 1
                else:
                    break
            
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(word)
        current_length += word_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def ingest_documents():
    """Reads documents from data directory, chunks them, and stores embeddings in ChromaDB."""
    # 1. Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # We delete the collection if it exists to ensure a fresh ingestion run
    try:
        client.delete_collection(name="ai_regulation")
    except Exception:
        pass
        
    collection = client.create_collection(name="ai_regulation")
    
    # 2. Initialize open-source embedding model
    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Process each document
    documents = []
    metadatas = []
    ids = []
    
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found!")
        return
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    if not files:
        logger.warning(f"No text files found in {DATA_DIR}/")
        return
        
    logger.info(f"Found {len(files)} files to ingest.")
    
    global_chunk_idx = 0
    
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        if not text:
            continue
            
        chunks = chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Document '{filename}' split into {len(chunks)} chunks.")
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"chunk_{global_chunk_idx}")
            global_chunk_idx += 1
            
    # 4. Embed and store in ChromaDB
    logger.info(f"Computing embeddings for {len(documents)} total chunks...")
    embeddings = embedding_model.encode(documents).tolist()
    
    logger.info("Storing embeddings in ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info("Ingestion Pipeline completed successfully!")

if __name__ == "__main__":
    ingest_documents()

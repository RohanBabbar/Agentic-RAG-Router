import chromadb
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        logger.info("Initializing Router and loading models...")
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_collection(name="ai_regulation")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def route_query(self, query: str) -> dict:
        """
        Classifies a query as Factual, Synthesis, or Out of Scope.
        Returns a dictionary containing the routing decision and the retrieved chunks.
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query ChromaDB (returns L2 distances by default. Lower is more similar)
        # 0.0 is perfect match. Values > 1.2 generally mean no semantic match.
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )
        
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        if not distances:
            return {"route": "Out of scope", "chunks": []}
            
        best_distance = distances[0]
        
        # 1. Out of Scope Check
        # If the closest document is too far away semantically, it's out of scope.
        OUT_OF_SCOPE_THRESHOLD = 1.3 
        if best_distance > OUT_OF_SCOPE_THRESHOLD:
            logger.info(f"Route: Out of scope (Best distance {best_distance:.2f} > {OUT_OF_SCOPE_THRESHOLD})")
            return {
                "route": "Out of scope", 
                "chunks": []
            }
            
        # 2. Synthesis vs Factual Check
        # We look at the top 3 results to see if the information spans multiple documents.
        top_3_sources = [meta['source'] for meta in metadatas[:3]]
        unique_sources = set(top_3_sources)
        
        # If the top chunks come from multiple different documents, it's a Synthesis query
        if len(unique_sources) > 1:
            logger.info(f"Route: Synthesis (Found in {len(unique_sources)} distinct documents)")
            return {
                "route": "Synthesis",
                "chunks": documents[:5], # Return more context for synthesis
                "sources": list(unique_sources)
            }
            
        # Otherwise, it's a Factual query located in a single document
        logger.info(f"Route: Factual (Found entirely within {list(unique_sources)[0]})")
        return {
            "route": "Factual",
            "chunks": [documents[0]], # Only need the single best chunk for a factual answer
            "sources": list(unique_sources)
        }

if __name__ == "__main__":
    # A quick local test
    router = QueryRouter()
    
    print("\n--- Testing Router ---")
    
    q1 = "What is the capital of France?" # Should be Out of Scope
    print(f"\nQuery: {q1}")
    router.route_query(q1)
    
    q2 = "What are the core technical requirements for the AI registry?" # Should be Factual (Document 4)
    print(f"\nQuery: {q2}")
    router.route_query(q2)
    
    q3 = "How do the technical requirements differ from the stakeholder's concerns regarding the registry?" # Should be Synthesis (Docs 3 & 4)
    print(f"\nQuery: {q3}")
    router.route_query(q3)

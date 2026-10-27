import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import BlockedPromptException, StopCandidateException
from dotenv import load_dotenv
import logging
from router import QueryRouter

logger = logging.getLogger(__name__)

def _failure(error_type: str, exc: Exception) -> dict:
    """A marked failure: no answer, plus an error the evaluation can detect and exclude."""
    logger.error(f"LLM call failed ({error_type}): {exc}")
    return {"answer": None, "error": {"type": error_type, "message": str(exc)}}

class RAGGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.router = QueryRouter()

    def _complete(self, prompt: str, temperature: float) -> dict:
        """Calls the LLM. Returns {"answer": str, "error": None} or {"answer": None, "error": {...}}."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
        except google_exceptions.TooManyRequests as e:
            # HTTP 429 (includes ResourceExhausted): per-minute and daily quota limits.
            return _failure("rate_limited", e)
        except google_exceptions.ServerError as e:
            # HTTP 5xx and timeouts: the service is down or overloaded.
            return _failure("unavailable", e)
        except google_exceptions.GoogleAPICallError as e:
            # Any other API error, e.g. an invalid API key, a bad request or a retired model name.
            return _failure("api_error", e)
        except (BlockedPromptException, StopCandidateException) as e:
            return _failure("blocked", e)

        try:
            return {"answer": response.text.strip(), "error": None}
        except ValueError as e:
            # .text raises ValueError when the response has no text, e.g. it was safety-blocked.
            return _failure("blocked", e)
        
    def generate_answer(self, query: str) -> dict:
        """
        End-to-end pipeline: Routes the query and generates an answer.
        Returns a dict with the route taken, the generated answer, and source chunks.
        If the LLM call fails, "answer" is None and "error" holds {"type", "message"}; otherwise "error" is None.
        """
        logger.info(f"Processing query: '{query}'")
        
        # 1. Explicit Routing
        routing_decision = self.router.route_query(query)
        route_type = routing_decision["route"]
        chunks = routing_decision["chunks"]
        sources = routing_decision.get("sources", [])
        
        # 2. Hardcoded Out of Scope (Guaranteed zero hallucination)
        if route_type == "Out of scope":
            return {
                "route": route_type,
                "answer": "I'm sorry, but the provided documents do not contain enough information to answer this query.",
                "sources": [],
                "error": None
            }
            
        # 3. Factual Query Generation
        elif route_type == "Factual":
            context = chunks[0] # Use only the best chunk
            prompt = f"""
            You are a strict, factual AI regulation assistant.
            Use ONLY the following context to answer the user's query. If the context does not fully answer the query, say so. Do not use outside knowledge.
            
            CONTEXT:
            {context}
            
            QUERY: {query}
            """
            
            result = self._complete(prompt, temperature=0.0) # 0.0 for strict factuality
            return {"route": route_type, "sources": sources, **result}
            
        # 4. Synthesis Query Generation
        elif route_type == "Synthesis":
            # Combine all retrieved chunks
            context = "\n\n---\n\n".join(chunks)
            prompt = f"""
            You are an expert AI regulation analyst. 
            The user has asked a complex question that requires synthesizing information from multiple documents.
            Read the provided context chunks below and provide a comprehensive answer.
            If the documents contain conflicting or contradictory information regarding the query, you MUST explicitly point out the contradiction and explain the different perspectives.
            Do not use outside knowledge.
            
            CONTEXT CHUNKS:
            {context}
            
            QUERY: {query}
            """
            
            result = self._complete(prompt, temperature=0.2) # Slight temp for synthesis creativity, but still grounded
            return {"route": route_type, "sources": sources, **result}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generator = RAGGenerator()
    
    # Let's test the 3 types
    queries = [
        "What is the capital of France?",
        "What are the core technical requirements for the AI registry?",
        "Do the stakeholders agree with the technical requirements for the registry?"
    ]
    
    for q in queries:
        print("\n" + "="*50)
        result = generator.generate_answer(q)
        print(f"ROUTE TAKEN: {result['route']}")
        print(f"SOURCES USED: {result['sources']}")
        if result["error"]:
            print(f"ERROR ({result['error']['type']}): {result['error']['message']}")
        else:
            print(f"ANSWER:\n{result['answer']}")

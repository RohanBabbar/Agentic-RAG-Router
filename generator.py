import os
import re
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import BlockedPromptException, StopCandidateException
from dotenv import load_dotenv
import logging
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from router import QueryRouter

logger = logging.getLogger(__name__)

# Retry policy for transient API failures. When Google's error says how long to wait (it does for
# rate limits), that delay is used, capped at MAX_RETRY_DELAY_SECONDS. Otherwise the waits grow
# exponentially: about 4, 8, 16 and 32 seconds, plus jitter.
MAX_ATTEMPTS = 5
MAX_RETRY_DELAY_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 60

def _is_transient(exc: BaseException) -> bool:
    """Rate limits (429), server errors and timeouts are worth retrying; a daily quota is not."""
    if isinstance(exc, google_exceptions.TooManyRequests):
        # Waiting a minute cannot fix an exhausted daily quota, so fail fast on it.
        return "PerDay" not in str(exc)
    return isinstance(exc, google_exceptions.ServerError)

def _server_retry_delay(exc: BaseException) -> float | None:
    """The wait in seconds that Google asks for in the error, or None if it gives none."""
    for detail in getattr(exc, "details", None) or []:
        if hasattr(detail, "retry_delay"):  # google.rpc.RetryInfo (the default gRPC transport)
            return detail.retry_delay.seconds + detail.retry_delay.nanos / 1e9
        if isinstance(detail, dict) and "retryDelay" in detail:  # REST transport, e.g. "33s"
            try:
                return float(detail["retryDelay"].rstrip("s"))
            except ValueError:
                pass
    # Fall back to the human-readable message: "Please retry in 33.1s."
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) if match else None

_exponential_wait = wait_exponential_jitter(initial=4, max=60)

def _retry_wait(state: RetryCallState) -> float:
    delay = _server_retry_delay(state.outcome.exception())
    if delay is None:
        return _exponential_wait(state)
    # One extra second so the retry lands after the quota window has reset.
    return min(delay + 1, MAX_RETRY_DELAY_SECONDS)

def _log_retry(state: RetryCallState) -> None:
    # One line per retry: the API's full error message runs to about 30 lines of quota details.
    exc = state.outcome.exception()
    first_line = str(exc).splitlines()[0][:150]
    logger.warning(f"LLM call failed (attempt {state.attempt_number}/{MAX_ATTEMPTS}), retrying in {state.next_action.sleep:.0f}s: {first_line}")

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

    @retry(
        retry=retry_if_exception(_is_transient),
        wait=_retry_wait,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _generate_with_retry(self, prompt: str, temperature: float):
        return self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
            request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
        )

    def _complete(self, prompt: str, temperature: float) -> dict:
        """Calls the LLM. Returns {"answer": str, "error": None} or {"answer": None, "error": {...}}.

        Transient errors are retried first (see _is_transient); these handlers see what is left.
        """
        try:
            response = self._generate_with_retry(prompt, temperature)
        except google_exceptions.TooManyRequests as e:
            # HTTP 429 (includes ResourceExhausted): per-minute limits after all retries, or a daily quota.
            return _failure("rate_limited", e)
        except google_exceptions.ServerError as e:
            # HTTP 5xx and timeouts, after all retries: the service is down or overloaded.
            return _failure("unavailable", e)
        except google_exceptions.GoogleAPICallError as e:
            # Not retried: any other API error, e.g. an invalid API key, a bad request or a retired model name.
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

# System Failure Analysis

As part of the evaluation framework, the following three system failures or underperformances were identified. 

### 1. API Rate Limiting & Quota Exhaustion
- **The Failure:** During the evaluation of the 15 queries, the system encountered a `429 ResourceExhausted` error from the Google Gemini API (hitting the daily Free Tier limit of 20 requests). 
- **The Root Cause:** The system relies entirely on a synchronous, third-party public LLM endpoint with strict rate limits. 
- **What I would do differently:** We implemented a `try/except` block to gracefully handle the failure by returning a hardcoded fallback string (preventing a complete pipeline crash). However, a true production solution would involve implementing an asynchronous message queue (like Celery or RabbitMQ) with exponential backoff (e.g., the `tenacity` library) to retry the requests over time, or migrating to a dedicated provisioned-throughput model rather than a shared free-tier endpoint.

### 2. Keyword Overlap Triggering False "Synthesis" Routes
- **The Failure:** When testing the query *"What are the core technical requirements for the AI registry?"*, the system routed it to **Synthesis** instead of **Factual**.
- **The Root Cause:** Our explicit routing logic states that if the top 5 retrieved chunks come from multiple documents, the query is "Synthesis." Because all 4 documents are highly focused on "AI", "technical", and "requirements", chunks from multiple documents were retrieved, triggering the Synthesis route even if the primary answer only lived in one document.
- **What I would do differently:** Instead of a simple `len(unique_sources) > 1` check, I would implement a "Distance Margin" check. If the #1 retrieved chunk has a cosine similarity score that is vastly superior to the #2 chunk (e.g., a margin > 0.2), the system should confidently route to Factual, regardless of where the other 4 chunks came from.

### 3. Brittle "Out of Scope" Distance Thresholds
- **The Failure:** The system uses a hardcoded L2 distance threshold (`> 1.3`) to determine if a query is Out of Scope. 
- **The Root Cause:** Semantic distance thresholds are mathematically brittle. An Out of Scope query that happens to use a lot of policy or technology jargon (e.g., *"What are the data privacy laws for quantum computing hardware?"*) could mathematically slip below the 1.3 threshold, tricking the router into sending it to the LLM, risking hallucination.
- **What I would do differently:** Instead of relying on raw vector distance, I would implement a lightweight Intent Classifier. We could train a simple Logistic Regression model on top of the embeddings (using a small dataset of "In Domain" vs "Out of Domain" queries) to explicitly predict the scope of the query with higher statistical confidence.

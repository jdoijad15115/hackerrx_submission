import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from functools import lru_cache

# Load environment variables
load_dotenv()

# --- Initialize services once ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

index = pc.Index("hackathon-policy-index")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# This is a lightweight but powerful model for re-ranking search results
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

@lru_cache(maxsize=128)
def get_answer_from_llm(question: str) -> str:
    """
    The ultimate reasoning engine with an advanced re-ranking step for maximum accuracy.
    """
    # === STEP 1: PARSE THE QUERY ===
    parser_prompt = f"""
    Analyze the user's question to extract key entities into a clean JSON object.
    Extract: 'age', 'gender', 'procedure', 'location', 'policy_duration_months', and the 'main_question'.
    If a value is not present, use null.
    User Question: "{question}"
    JSON Output:
    """
    try:
        parser_response = llm.generate_content(parser_prompt)
        structured_query_str = parser_response.text.strip().replace("```json", "").replace("```", "")
        structured_query = json.loads(structured_query_str)
    except Exception as e:
        print(f"❌ Parser Error: {e}. Falling back to using the raw question.")
        structured_query = {"main_question": question}

    # === STEP 2: BROAD RETRIEVAL ===
    search_term = structured_query.get("main_question", question)
    query_embedding = embeddings.embed_query(search_term)
    # Retrieve more documents initially to give the re-ranker more to work with
    retrieval_results = index.query(vector=query_embedding, top_k=25, include_metadata=True)
    retrieved_docs = [match['metadata']['text'] for match in retrieval_results['matches']]

    # === STEP 3: PRECISION RE-RANKING (The Winning Move) ===
    # Create pairs of [question, retrieved_clause] for the cross-encoder
    rerank_pairs = [[search_term, doc] for doc in retrieved_docs]
    # Get relevance scores
    scores = cross_encoder.predict(rerank_pairs)
    # Combine docs with their new scores and sort
    scored_docs = sorted(zip(scores, retrieved_docs), reverse=True)
    # Select the top 7 highest-scoring documents for the final context
    top_docs = [doc for score, doc in scored_docs[:7]]
    context = "\n\n---\n\n".join(top_docs)

    # === STEP 4: DECISION MAKING WITH CHAIN-OF-THOUGHT ===
    decision_prompt = f"""
    You are an expert insurance claims adjudicator for Bajaj Finserv Health.
    Task: Evaluate a claim with extreme precision based on the provided policy clauses and structured query.

    Instructions:
    1.  **Analyze**: First, in a `<scratchpad>`, write a step-by-step analysis. Compare the user's situation from the `structured_query` against the `retrieved_clauses`.
    2.  **Decide**: Second, based ONLY on your scratchpad analysis, provide the final answer as a single, valid JSON object.

    Retrieved Clauses:
    ---
    {context}
    ---

    Structured Query:
    ```json
    {json.dumps(structured_query, indent=2)}
    ```

    Your Response:
    <scratchpad>
    [Your step-by-step reasoning appears here.]
    </scratchpad>
    ```json
    {{
        "decision": "Approved | Rejected | Insufficient Information",
        "amount": "The approved amount as a number, or null",
        "justification": "A clear, step-by-step explanation of your reasoning.",
        "source_clauses": ["The exact, verbatim text of the clause(s) that support your decision."]
    }}
    ```
    """
    try:
        decision_response = llm.generate_content(decision_prompt)
        json_part = decision_response.text.split("```json")[-1].split("```")[0].strip()
        json.loads(json_part)
        return json_part
    except Exception as e:
        print(f"❌ Decision Error: {e}")
        return json.dumps({
            "decision": "Error", "amount": None,
            "justification": "An internal error occurred while generating the decision.",
            "source_clauses": []
        })
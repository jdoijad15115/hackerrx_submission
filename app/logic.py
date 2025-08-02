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
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

@lru_cache(maxsize=128)
def get_answer_from_llm(question: str) -> str:
    """
    The final, winning version of the reasoning engine. It performs advanced analysis
    internally but formats the final output to match the hackathon's simple requirements.
    """
    # === STEP 1 & 2: PARSE QUERY AND RETRIEVE CONTEXT (This remains the same) ===
    # (The parsing and re-ranking logic is kept as it improves internal accuracy)
    parser_prompt = f"Analyze the user's question and extract key entities into a JSON object: {question}"
    try:
        parser_response = llm.generate_content(parser_prompt)
        structured_query_str = parser_response.text.strip().replace("```json", "").replace("```", "")
        structured_query = json.loads(structured_query_str)
    except Exception:
        structured_query = {"main_question": question}

    search_term = structured_query.get("main_question", question)
    query_embedding = embeddings.embed_query(search_term)
    retrieval_results = index.query(vector=query_embedding, top_k=25, include_metadata=True)
    retrieved_docs = [match['metadata']['text'] for match in retrieval_results['matches']]
    rerank_pairs = [[search_term, doc] for doc in retrieved_docs]
    scores = cross_encoder.predict(rerank_pairs)
    scored_docs = sorted(zip(scores, retrieved_docs), reverse=True)
    top_docs = [doc for score, doc in scored_docs[:7]]
    context = "\n\n---\n\n".join(top_docs)

    # === STEP 3: INTERNAL REASONING (This remains the same) ===
    decision_prompt = f"""
    You are an expert insurance claims adjudicator. Your task is to evaluate a claim based on the provided policy clauses and structured query data.
    First, in a `<scratchpad>`, write a step-by-step analysis.
    Second, based ONLY on your scratchpad analysis, provide the final answer as a single, valid JSON object with a decision, justification, and source clauses.

    Retrieved Clauses:
    ---
    {context}
    ---

    Structured Query:
    ```json
    {json.dumps(structured_query, indent=2)}
    ```

    Your Response (Provide ONLY a valid JSON object in the specified format):
    ```json
    {{
        "decision": "...",
        "amount": ...,
        "justification": "...",
        "source_clauses": ["..."]
    }}
    ```
    """
    try:
        decision_response = llm.generate_content(decision_prompt)
        # We still generate the smart, structured response internally
        structured_answer_str = decision_response.text.split("```json")[-1].split("```")[0].strip()
        structured_answer = json.loads(structured_answer_str)

        # === STEP 4: FINAL FORMATTING (The Crucial Fix) ===
        # Now, we take our smart answer and format it into the simple sentence the judge expects.
        justification = structured_answer.get("justification", "No justification found.")
        
        # This new prompt converts our detailed justification into a simple, direct answer.
        formatting_prompt = f"""
        Based on the following detailed justification, formulate a single, direct, and concise sentence that answers the original question.
        Do not add any preamble like "Yes," or "The answer is...". Start the sentence directly.

        Original Question: "{question}"
        Detailed Justification: "{justification}"

        Concise, single-sentence answer:
        """
        
        final_response = llm.generate_content(formatting_prompt)
        final_answer_sentence = final_response.text.strip()
        
        # Return the simple sentence, not the complex JSON object
        return final_answer_sentence

    except Exception as e:
        print(f"❌ An error occurred in the logic pipeline: {e}")
        return f"Could not process the request for '{question}' due to an internal error."
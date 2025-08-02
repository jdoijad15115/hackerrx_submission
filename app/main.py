import os
import asyncio
from fastapi import FastAPI, Depends, HTTPException, Header
from dotenv import load_dotenv
from .schemas import HackathonRequest, HackathonResponse
from .logic import get_answer_from_llm

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Bajaj Finserv Health HackRx - AI Decision Engine",
    description="Processes natural language queries against insurance documents to provide structured, reasoned decisions.",
    version="1.0.0"
)

# --- Security: Verify the Bearer Token ---
API_KEY = os.getenv("BEARER_TOKEN")
async def verify_api_key(Authorization: str = Header(...)):
    """Verifies that the request includes the correct bearer token."""
    if Authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or Missing Authorization Bearer Token")

# --- Helper for Concurrent Processing ---
async def get_answer_async(question: str):
    """Runs the synchronous LLM logic in a separate thread to avoid blocking the server."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_answer_from_llm, question)

# --- API Endpoints ---
@app.post("/hackrx/run", response_model=HackathonResponse, dependencies=[Depends(verify_api_key)])
async def run_hackathon_query(request: HackathonRequest):
    """
    Accepts a document URL and a list of questions.
    Processes all questions concurrently for maximum speed and returns a list of structured JSON answers.
    """
    try:
        # Create a task for each question to be processed in parallel
        tasks = [get_answer_async(q) for q in request.questions]
        # Wait for all tasks to complete
        answers = await asyncio.gather(*tasks)

        # The LLM returns JSON strings, so we parse them before returning
        parsed_answers = [json.loads(ans) for ans in answers]

        return HackathonResponse(answers=parsed_answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/", summary="API Status")
def root():
    """A simple endpoint to confirm that the API is running."""
    return {"status": "ok", "message": "AI Decision Engine is online."}
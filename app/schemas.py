# app/schemas.py
from pydantic import BaseModel, HttpUrl
from typing import List, Any

# We use 'Any' because the answer is now a complex JSON object (as a string)
# A more advanced solution would create a Pydantic model for the decision object itself.
class HackathonResponse(BaseModel):
    answers: List[Any]

class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
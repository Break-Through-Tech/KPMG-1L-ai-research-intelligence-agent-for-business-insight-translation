import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.chain import respond_to_query

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

def event_stream(query: str):
    for event in respond_to_query(query):
        yield f"data: {json.dumps(event)}\n\n"

@router.post('/chat')
def chat(request: QueryRequest):
    return StreamingResponse(
        event_stream(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
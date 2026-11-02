# from rag.chain import respond_to_query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router

app = FastAPI(
    title="KPMG Research Intelligence API",
    version="1.0.0"
)

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_methods=['POST', 'GET'],
    allow_headers=['Content-Type'],
)


app.include_router(
    chat_router,
    prefix="/api"
)

@app.get('/')
def root():
    return({"message": "KPMG Research API is running"})

# # # Response all at once
# # response = rag_chain.invoke(input("Enter Query: "))
# # print(response)

# # Streamed response
# for chunk in respond_to_query(input("Enter Query: ")):

#     if chunk["type"] == "status":
#         print(f"💭 {chunk['message']}")

#     elif chunk["type"] == "chunk":
#         print(chunk["content"], end="", flush=True)

#     elif chunk["type"] == "done":
#         print("\n✅ All done!")
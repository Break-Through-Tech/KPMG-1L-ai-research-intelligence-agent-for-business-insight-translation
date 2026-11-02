from rag.chain import respond_to_query



# # Response all at once
# response = rag_chain.invoke(input("Enter Query: "))
# print(response)

# Streamed response
for chunk in respond_to_query(input("Enter Query: ")):

    if chunk["type"] == "status":
        print(f"💭 {chunk['message']}")

    elif chunk["type"] == "chunk":
        print(chunk["content"], end="", flush=True)

    elif chunk["type"] == "done":
        print("\n✅ All done!")
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import graph

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Message":"Hello there!"}

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user" #Acts as the session ID for memory

@app.post("/chatbot")
async def chatbot(req: ChatRequest):
    # Graph requires a config to save memory per thread_id
    config = {"configurable": {"thread_id": req.thread_id}}

    # ainvoke is used so the server doesn't block
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config
    )

    # return only the final AI response content
    final_response = result["messages"][-1].content

    return {"response": final_response}

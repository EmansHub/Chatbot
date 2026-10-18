from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from pydantic import BaseModel

load_dotenv()
app = FastAPI()

llm = init_chat_model(
    model= "gpt-5.6-luna",
    temperature=1.0,
    )

agent = create_agent(model=llm)

messages = []

@app.get("/")
async def read_root():
    return {"Message":"Hello there!"}


class ChatRequest(BaseModel):
    message: str

@app.post("/chatbot")
async def chatbot(req: ChatRequest):
    messages.append(req.message)
    return {"response": f"Received message: {req.message}", "messages": messages}

'''
@app.post("/chatbot")
async def chatbot(message: str):
    messages.append(message)
    return {"response": f"Received message: {message}", "messages": messages}
'''

@app.get("/chatbot")
async def chatbot():
    response = await llm.ainvoke("How to fly a car?")
    # 'invoke' blocks the server while waiting
    return response.content

@app.get("/agent")
async def chatbot():
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="How to fly a car?")]}
    )
    # 'invoke' blocks the server while waiting
    return response['messages'][-1].content

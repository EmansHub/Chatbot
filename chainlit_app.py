import uuid
import chainlit as cl
from numpy import random
from graph import graph
from langchain_core.messages import HumanMessage


@cl.on_chat_start
async def start():
     # 1. Generate a new random ID for this session
    new_id = random.randint(1, 20)
    
    # 2. Update the variable in the imported graph module
    graph.current_investorID = new_id
    
    # 3. Set a fresh thread_id
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    
    await cl.Message(content=f"Hi investor {new_id}, how can I help you today?").send()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    
    # Call the graph
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message.content)]},
        config
    )
    
    # Send response back to Chainlit UI
    await cl.Message(content=result["messages"][-1].content).send()
